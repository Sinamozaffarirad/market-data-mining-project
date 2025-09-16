from django.shortcuts import render, get_object_or_404
from .models import CustomerProfile

def customer_search(request):
    return render(request, "site/customers/search.html")

def customer_detail(request, pk):
    customer = get_object_or_404(CustomerProfile, pk=pk)
    return render(request, "site/customers/detail.html", {"customer": customer})

