from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect


def home(request):
    return redirect("dunnhumby_site:index")


@login_required
def dashboard(request):
    return redirect("dunnhumby_site:bi_dashboard")


@login_required
def analytics(request):
    return redirect("dunnhumby_site:basket_analysis")


@login_required
def reports(request):
    return redirect("dunnhumby_site:bi_dashboard")
