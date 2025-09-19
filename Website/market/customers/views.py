# customers/views.py
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from .models import CustomerProfile

def customer_search(request):
    household_key = request.GET.get("household_key")

    if household_key:
        try:
            profile = CustomerProfile.objects.get(household_key=household_key)
            return redirect("customers:detail", pk=profile.household_key)
        except CustomerProfile.DoesNotExist:
            messages.error(request, "No household found with this key.")

    return render(request, "site/customers/search.html")

def customer_detail(request, pk):
    profile = get_object_or_404(CustomerProfile, household_key=pk)
    return render(request, "site/customers/detail.html", {"household": profile})

def customer_recommendations(request, pk):
    profile = get_object_or_404(CustomerProfile, household_key=pk)
    return render(request, "site/customers/recommendations.html", {"household": profile})

def customer_churn(request, pk):
    profile = get_object_or_404(CustomerProfile, household_key=pk)
    return render(request, "site/customers/churn.html", {"household": profile})

def customer_purchases(request, pk):
    profile = get_object_or_404(CustomerProfile, household_key=pk)
    return render(request, "site/customers/purchases.html", {"household": profile})