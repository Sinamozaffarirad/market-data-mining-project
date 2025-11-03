from django.shortcuts import render

def retention_dashboard(request):
    return render(request, 'site/customer_retention/customer_retention.html')
