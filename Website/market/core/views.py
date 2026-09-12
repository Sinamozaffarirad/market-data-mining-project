from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.db.models import Sum, Count, Avg
from dunnhumby.models import Transaction, DunnhumbyProduct, Household, CustomerSegment, BasketAnalysis


def home(request):
    """Redirect root to the new public/staff analytics site."""
    return redirect('/analysis/')


@login_required
def dashboard(request):
    """Main dashboard for authenticated users"""
    top_products = Transaction.objects.values(
        'product_id'
    ).annotate(
        total_sales=Sum('sales_value'),
        frequency=Count('id')
    ).order_by('-total_sales')[:10]

    segments = CustomerSegment.objects.values('rfm_segment').annotate(
        count=Count('id'),
        avg_spend=Avg('total_spend')
    ).order_by('-count')
    
    recent_baskets = BasketAnalysis.objects.order_by('-created_at')[:10]
    
    context = {
        'top_products': top_products,
        'segments': segments,
        'recent_baskets': recent_baskets,
    }
    return render(request, 'core/dashboard.html', context)


@login_required
def analytics(request):
    """Analytics page with charts and insights"""
    dept_analysis = Transaction.objects.values(
        'product_id'  
    ).annotate(
        total_sales=Sum('sales_value'),
        transaction_count=Count('id')
    ).order_by('-total_sales')[:15]

    weekly_trends = Transaction.objects.values(
        'week_no'
    ).annotate(
        total_sales=Sum('sales_value'),
        transaction_count=Count('id')
    ).order_by('week_no')[:12]
    
    context = {
        'dept_analysis': dept_analysis,
        'weekly_trends': weekly_trends,
    }
    return render(request, 'core/analytics.html', context)


@login_required
def reports(request):
    """Reports page with downloadable reports"""
    context = {
        'available_reports': [
            {
                'name': 'Customer Segmentation Report',
                'description': 'RFM analysis and customer segments',
                'url': '#'
            },
            {
                'name': 'Product Performance Report',
                'description': 'Top performing products by sales and frequency',
                'url': '#'
            },
            {
                'name': 'Basket Analysis Report',
                'description': 'Shopping basket insights and patterns',
                'url': '#'
            },
            {
                'name': 'Campaign Effectiveness Report',
                'description': 'Marketing campaign ROI and performance',
                'url': '#'
            },
        ]
    }
    return render(request, 'core/reports.html', context)
