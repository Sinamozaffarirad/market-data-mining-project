# customers/views.py
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from .models import CustomerProfile, Transaction, Product
from collections import defaultdict
from django.core.paginator import Paginator

def customer_search(request):
    household_key = request.GET.get("household_key")

    if household_key:
        if CustomerProfile.objects.filter(household_key=household_key).exists():
            return redirect("customers:detail", pk=household_key)
        else:
            messages.error(request, "No household found with this key.")

    return render(request, "site/customers/search.html")


def customer_detail(request, pk):
    household = get_object_or_404(CustomerProfile, household_key=pk)
    return render(request, "site/customers/detail.html", {"household": household})


def customer_recommendations(request, pk):
    household = get_object_or_404(CustomerProfile, household_key=pk)
    return render(request, "site/customers/recommendations.html", {"household": household})


def customer_churn(request, pk):
    household = get_object_or_404(CustomerProfile, household_key=pk)
    return render(request, "site/customers/churn.html", {"household": household})


FILTER_OPTIONS = {
    "3m": (622, 711),
    "6m": (532, 711),
    "9m": (442, 711),
    "12m": (347, 711),
    "15m": (257, 711),
    "18m": (167, 711),
    "all": (1, 711),
}

def customer_purchases(request, pk):
    household = get_object_or_404(CustomerProfile, household_key=pk)

    period = request.GET.get("period", "all")
    start_day, end_day = FILTER_OPTIONS.get(period, (1, 711))

    transactions = Transaction.objects.filter(
        household_key=pk,
        day__gte=start_day,
        day__lte=end_day
    ).order_by("-day")

    product_map = {
        p.product_id: p for p in Product.objects.filter(
            product_id__in=[tr.product_id for tr in transactions]
        )
    }

    grouped_purchases = defaultdict(list)
    for tr in transactions:
        product = product_map.get(tr.product_id)
        grouped_purchases[tr.basket_id].append({
            "day": tr.day,
            "trans_time": tr.trans_time,
            "product_id": tr.product_id,
            "quantity": tr.quantity,
            "sales_value": tr.sales_value,
            "brand": product.brand if product else "Unknown",
            "department": product.department if product else "Unknown",
            "commodity": product.commodity_desc if product else "Unknown",
            "size": product.curr_size_of_product if product else "",
        })

    grouped_purchases = dict(sorted(
        grouped_purchases.items(),
        key=lambda x: min(p["day"] for p in x[1]),
        reverse=True
    ))

    basket_list = list(grouped_purchases.items())  
    paginator = Paginator(basket_list, 10)  
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    return render(request, "site/customers/purchases.html", {
        "household": household,
        "page_obj": page_obj,  
        "selected_period": period,
        "transaction_count": len(grouped_purchases), 
    })