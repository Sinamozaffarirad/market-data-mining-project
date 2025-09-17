# customers/views.py
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from .models import CustomerProfile

def customer_search(request):
    household_key = request.GET.get("household_key")

    if household_key:
        try:
            # Search is now performed on the CustomerProfile model.
            profile = CustomerProfile.objects.get(household_key=household_key)
            return redirect("customers:detail", pk=profile.household_key)
        except CustomerProfile.DoesNotExist:
            messages.error(request, "No household found with this key.")

    return render(request, "site/customers/search.html")

def customer_detail(request, pk):
    # The detail view now fetches a CustomerProfile object.
    profile = get_object_or_404(CustomerProfile, household_key=pk)
    # The context variable is renamed for clarity.
    return render(request, "site/customers/detail.html", {"household": profile})