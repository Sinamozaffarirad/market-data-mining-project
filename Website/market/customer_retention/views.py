from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import ExpressionWrapper, F, FloatField
from django.shortcuts import render

from dunnhumby.models import ChurnExperiment, CustomerSegment

@login_required
def retention_dashboard(request):
    """Prioritize at-risk customers and hand off to the existing hybrid recommender."""
    minimum_probability = min(max(float(request.GET.get('minimum_probability', .50)), 0), 1)
    customers = CustomerSegment.objects.filter(churn_probability__gte=minimum_probability).annotate(
        churn_percentage=ExpressionWrapper(F('churn_probability') * 100.0, output_field=FloatField())
    ).order_by('-churn_probability', '-total_spend')
    page_obj = Paginator(customers, 20).get_page(request.GET.get('page'))
    return render(request, 'site/customer_retention/customer_retention.html', {
        'page_obj': page_obj,
        'minimum_probability': minimum_probability,
        'active_experiment': ChurnExperiment.objects.filter(is_active=True).first(),
    })
