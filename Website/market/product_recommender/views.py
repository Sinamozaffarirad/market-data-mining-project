# product_recommender/views.py

from django.shortcuts import render, get_object_or_404
from django.db.models import Q
from django.core.paginator import Paginator
# Import models from dunnhumby app
from dunnhumby.models import Transaction, DunnhumbyProduct, AssociationRule, CustomerSegment
from .forms import ProductSearchForm

def recommend_home(request):
    """
    صفحه اصلی جستجوی محصول
    """
    form = ProductSearchForm(request.GET or None)
    query = None
    products = None

    if form.is_valid():
        query = form.cleaned_data.get("query")

        if query:
            # جستجو در مدل DunnhumbyProduct
            products = DunnhumbyProduct.objects.filter(
                Q(brand__icontains=query) |
                Q(commodity_desc__icontains=query) |
                Q(sub_commodity_desc__icontains=query)
            ).order_by('brand', 'commodity_desc')[:50] # محدود کردن نتایج

    return render(request, "site/product_recommender/recommend_customers.html", {
        "form": form,
        "query": query,
        "products": products,
    })


def recommend_customers(request, product_id):
    """
    صفحه اصلی پیشنهاد مشتری برای یک محصول خاص
    """

    # -----------------------------
    # 1) دریافت اطلاعات محصول
    # -----------------------------
    product = get_object_or_404(DunnhumbyProduct, product_id=product_id)

    # -----------------------------
    # 2) Direct Buyers (Seed Group)
    # -----------------------------
    direct_buyers = set(
        Transaction.objects.filter(product_id=product_id)
        .values_list("household_key", flat=True)
    )

    # -----------------------------
    # 3) Similar Products (Content-Based)
    # -----------------------------
    similar_products = DunnhumbyProduct.objects.filter(
        Q(sub_commodity_desc=product.sub_commodity_desc) |
        Q(commodity_desc=product.commodity_desc)
    ).values_list("product_id", flat=True)

    similar_buyers = set(
        Transaction.objects.filter(product_id__in=similar_products)
        .exclude(product_id=product_id) 
        .values_list("household_key", flat=True)
    )

    # -----------------------------
    # 4) Collaborative Filtering (Similar Customers)
    # -----------------------------
    if direct_buyers:
        cf_segments = CustomerSegment.objects.filter(
            household_key__in=direct_buyers
        ).values_list("rfm_segment", flat=True)

        cf_customers = set(
            CustomerSegment.objects.filter(
                rfm_segment__in=cf_segments
            ).values_list("household_key", flat=True)
        )
    else:
        cf_customers = set()

    # -----------------------------
    # 5) ترکیب همه مشتری‌ها با وزن‌دهی
    # -----------------------------
    scores = {}
    W_DIRECT = 1.0
    W_SIMILAR_PRODUCTS = 0.7
    W_CF = 0.5
    W_CF_EXCLUDE_DIRECT = 0.3 

    for h in direct_buyers:
        scores[h] = scores.get(h, 0) + W_DIRECT

    for h in similar_buyers:
        if h not in direct_buyers: 
            scores[h] = scores.get(h, 0) + W_SIMILAR_PRODUCTS

    for h in cf_customers:
        if h not in direct_buyers and h not in similar_buyers: 
            scores[h] = scores.get(h, 0) + W_CF
        elif h in direct_buyers: 
            scores[h] = scores.get(h, 0) + W_CF_EXCLUDE_DIRECT 


    # -----------------------------
    # 6) مرتب‌سازی مشتری‌ها طبق امتیاز و حذف خریداران مستقیم
    # -----------------------------
    for h in direct_buyers:
        if h in scores:
            del scores[h]

    sorted_customers = sorted(scores.items(), key=lambda x: -x[1])[:100] 
    household_keys = [h for h, score in sorted_customers]

    # -----------------------------
    # 7) دریافت اطلاعات پروفایل مشتریان
    # -----------------------------
    try:
        from customers.models import CustomerProfile
        customers_data = CustomerProfile.objects.filter(
            household_key__in=household_keys
        )
        customer_segments = {
            s.household_key: s.rfm_segment 
            for s in CustomerSegment.objects.filter(household_key__in=household_keys)
        }
        
        final_customers = []
        # مپ کردن دیتاها برای حفظ ترتیب امتیازدهی
        # (توجه: این بخش ممکن است کند باشد اگر household_keys زیاد باشد، اما برای 100 مورد مناسب است)
        household_profile_map = {c.household_key: c for c in customers_data}
        for h_key in household_keys:
            if h_key in household_profile_map:
                c = household_profile_map[h_key]
                c.rfm_segment = customer_segments.get(h_key, 'N/A')
                final_customers.append(c)

    except ImportError:
        # Fallback (ترتیب ممکن است حفظ نشود)
        final_customers = CustomerSegment.objects.filter(
            household_key__in=household_keys
        )

    # -----------------------------
    # ۸. صفحه‌بندی (Pagination)  <-- تغییر در اینجا
    # -----------------------------
    paginator = Paginator(final_customers, 10) # ۱۰ مشتری در هر صفحه
    page_number = request.GET.get('page')
    customers_page = paginator.get_page(page_number)


    # -----------------------------
    # نتیجه نهایی
    # -----------------------------
    return render(request, "site/product_recommender/recommendations_list.html", {
        "product": product,
        "customers_page": customers_page, # <-- ۲. ارسال آبجکت صفحه به تمپلیت
        "scores": scores,
        "direct_buyers_count": len(direct_buyers),
        "similar_buyers_count": len(similar_buyers - direct_buyers),
        "cf_customers_count": len(cf_customers - direct_buyers - similar_buyers),
        "total_recommendations": len(final_customers), # <-- تعداد کل ثابت می‌ماند
    })

# ... (view product_detail بدون تغییر باقی می‌ماند) ...

def product_detail(request, product_id):
    """
    این View اطلاعات تکمیلی محصول را نشان می‌دهد (بدون تغییر)
    """
    # 1) محصول
    product = get_object_or_404(DunnhumbyProduct, product_id=product_id)

    # 2) مشتریانی که این محصول را خریده‌اند
    households = Transaction.objects.filter(product_id=product_id) \
        .values_list("household_key", flat=True).distinct()

    segments = CustomerSegment.objects.filter(household_key__in=households)

    # 3) قوانین مرتبط با محصول
    rules = AssociationRule.objects.filter(
        Q(antecedent__contains=str(product_id)) |
        Q(consequent__contains=str(product_id))
    )

    return render(request, "site/product_recommender/product_detail.html", {
        "product": product,
        "segments": segments,
        "rules": rules,
    })