from django.shortcuts import render

def recommend_customers_view(request):
    """
    صفحه اصلی ماژول پیشنهاد مشتری برای کالا
    """
    return render(request, 'site/product_recommender/recommend_customers.html')
