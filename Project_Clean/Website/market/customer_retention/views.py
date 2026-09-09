from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import ExpressionWrapper, F, FloatField
from django.shortcuts import render

from dunnhumby.models import ChurnExperiment, CustomerSegment

@login_required
def retention_dashboard(request):
    """Prioritize at-risk customers and hand off to the existing hybrid recommender."""
    try:
        minimum_probability = min(max(float(request.GET.get('minimum_probability', .50)), 0), 1)
    except (TypeError, ValueError):
        minimum_probability = .50

    customer_query = request.GET.get('customer', '').strip()
    maximum_customers_raw = request.GET.get('maximum_customers', '').strip()
    try:
        maximum_customers = int(maximum_customers_raw) if maximum_customers_raw else None
        if maximum_customers is not None and maximum_customers < 1:
            maximum_customers = None
    except (TypeError, ValueError):
        maximum_customers = None
    probability_order = request.GET.get('probability_order', 'descending')
    if probability_order not in {'ascending', 'descending'}:
        probability_order = 'descending'
    customers = CustomerSegment.objects.filter(churn_probability__gte=minimum_probability).annotate(
        churn_percentage=ExpressionWrapper(F('churn_probability') * 100.0, output_field=FloatField())
    )
    if customer_query:
        # Household keys are numeric, so this is an exact customer-number lookup.
        if customer_query.isdigit():
            customers = customers.filter(household_key=int(customer_query))
        else:
            customers = customers.none()

    churn_ordering = 'churn_probability' if probability_order == 'ascending' else '-churn_probability'
    customers = customers.order_by(churn_ordering, '-total_spend')
    # Apply the limit after every filter and the risk/value ranking.  This means
    # a request for 100 customers fills from the next risk category when fewer
    # than 100 customers exist in the highest-risk category.
    if maximum_customers is not None:
        customers = customers[:maximum_customers]
    page_obj = Paginator(customers, 20).get_page(request.GET.get('page'))
    return render(request, 'site/customer_retention/customer_retention.html', {
        'page_obj': page_obj,
        'minimum_probability': minimum_probability,
        'customer_query': customer_query,
        'maximum_customers': maximum_customers,
        'probability_order': probability_order,
        'active_experiment': ChurnExperiment.objects.filter(is_active=True).first(),
    })
