# customers/views.py
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from .models import CustomerProfile

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


def customer_purchases(request, pk):
    household = get_object_or_404(CustomerProfile, household_key=pk)
    return render(request, "site/customers/purchases.html", {"household": household})