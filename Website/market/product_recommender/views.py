from django.shortcuts import render, get_object_or_404
from django.db.models import Q, Count, TextField
from django.db.models.functions import Cast
from django.core.paginator import Paginator
from django.db import connection
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Import models from dunnhumby app
from dunnhumby.models import Transaction, DunnhumbyProduct, AssociationRule, CustomerSegment
from customers.models import CustomerProfile
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

def _get_cf_leads(direct_buyers_set, limit_users=2000):
    """
    Helper function for Collaborative Filtering (User-Based)
    Finds customers similar to the direct_buyers group.
    """
    # 1. Fetch data for User-Item Matrix
    query = """
        SELECT household_key, product_id, COUNT(*) as cnt 
        FROM transactions 
        GROUP BY household_key, product_id
    """
    with connection.cursor() as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()
    
    if not rows:
        return {}

    df = pd.DataFrame(rows, columns=['household_key', 'product_id', 'cnt'])
    
    # ساخت ماتریس کاربر-کالا
    user_item = df.pivot_table(index='household_key', columns='product_id', values='cnt', fill_value=0)
    
    # محاسبه شباهت کسینوسی
    sim_matrix = cosine_similarity(user_item)
    sim_df = pd.DataFrame(sim_matrix, index=user_item.index, columns=user_item.index)
    
    cf_scores = {}
    
    # فقط کاربرانی را بررسی می‌کنیم که در user_item هستند
    valid_direct_buyers = [h for h in direct_buyers_set if h in sim_df.index]
    
    if not valid_direct_buyers:
        return {}

    # جمع زدن ستون‌های مربوط به خریداران مستقیم برای یافتن شبیه‌ترین‌ها
    raw_scores = sim_df[valid_direct_buyers].sum(axis=1)
    
    for hh_key, score in raw_scores.items():
        if score > 0:
            cf_scores[hh_key] = float(score)
            
    return cf_scores

def recommend_customers(request, product_id):
    """
    صفحه اصلی پیشنهاد مشتری برای یک محصول خاص
    با استفاده از AR و CF (User-Based)
    """
    product = get_object_or_404(DunnhumbyProduct, product_id=product_id)

    # -----------------------------
    # 1) Direct Buyers (Seed Group)
    # -----------------------------
    direct_buyers = set(
        Transaction.objects.filter(product_id=product_id)
        .values_list("household_key", flat=True)
    )

    scores = {}

    # وزن‌دهی (قابل تنظیم)
    W_CONTENT = 1.0
    W_ASSOC = 2.0  # وزن بالاتر برای قوانین
    W_CF = 1.5     # وزن فیلترینگ مشارکتی

    # -----------------------------
    # 2) Content-Based (Similar Products)
    # مشتریانی که محصولات مشابه خریده‌اند
    # -----------------------------
    similar_products_qs = DunnhumbyProduct.objects.filter(
        Q(sub_commodity_desc=product.sub_commodity_desc) |
        Q(commodity_desc=product.commodity_desc)
    ).exclude(product_id=product_id)
    
    similar_products_ids = list(similar_products_qs.values_list("product_id", flat=True)[:500])

    content_buyers = Transaction.objects.filter(product_id__in=similar_products_ids)\
        .values('household_key').annotate(cnt=Count('id'))
    
    for row in content_buyers:
        h = row['household_key']
        # امتیاز بر اساس تعداد خرید کالای مشابه
        s = np.log1p(row['cnt']) * W_CONTENT
        scores[h] = scores.get(h, 0) + s

    # -----------------------------
    # 3) Association Rules (Rules-Based) - اصلاح شده برای MSSQL
    # -----------------------------
    
    # تبدیل فیلد JSON به متن برای جستجو (چون JSON lookup در MSSQL پشتیبانی نمی‌شود)
    # ابتدا کاندیداهایی که شامل شماره محصول هستند را می‌گیریم (ممکن است شامل موارد مشابه مثل 199401 هم باشد)
    candidates = AssociationRule.objects.annotate(
        consequent_text=Cast('consequent', TextField())
    ).filter(
        consequent_text__contains=str(product_id)
    ).order_by('-lift')[:100]  # تعداد بیشتری می‌گیریم تا بعدا فیلتر کنیم

    relevant_rules = []
    target_pid_str = str(product_id)

    # فیلتر دقیق در پایتون
    for rule in candidates:
        # rule.consequent یک لیست است، مثلا ["99401"] یا [99401]
        consequents_list = [str(x) for x in rule.consequent]
        if target_pid_str in consequents_list:
            relevant_rules.append(rule)
            if len(relevant_rules) >= 20:
                break

    for rule in relevant_rules:
        antecedents = rule.antecedent
        clean_antecedents = [int(x) for x in antecedents if str(x).isdigit()]
        
        if not clean_antecedents:
            continue

        rule_buyers = Transaction.objects.filter(product_id__in=clean_antecedents)\
            .values('household_key').distinct()
        
        rule_score = (rule.confidence * rule.lift) * W_ASSOC
        
        for rb in rule_buyers:
            h = rb['household_key']
            scores[h] = scores.get(h, 0) + rule_score


    # -----------------------------
    # 4) Collaborative Filtering (User-Based)
    # -----------------------------
    if direct_buyers:
        try:
            cf_leads = _get_cf_leads(direct_buyers)
            max_cf = max(cf_leads.values()) if cf_leads else 1
            
            for h, raw_score in cf_leads.items():
                norm_score = (raw_score / max_cf) * 5.0
                scores[h] = scores.get(h, 0) + (norm_score * W_CF)
                
        except Exception as e:
            print(f"CF Error: {e}")

    # -----------------------------
    # 5) فیلترینگ و آماده‌سازی نهایی
    # -----------------------------
    
    # حذف خریداران مستقیم
    for h in direct_buyers:
        if h in scores:
            del scores[h]

    # مرتب‌سازی
    sorted_customers = sorted(scores.items(), key=lambda x: -x[1])[:100]
    household_keys = [h for h, score in sorted_customers]

    max_score = sorted_customers[0][1] if sorted_customers else 1.0

    # دریافت اطلاعات پروفایل
    customers_data_map = {
        c.household_key: c 
        for c in CustomerProfile.objects.filter(household_key__in=household_keys)
    }

    final_customers = []
    for h_key, score in sorted_customers:
        c_obj = customers_data_map.get(h_key)
        if not c_obj:
            c_obj = CustomerProfile(household_key=h_key)
            c_obj.age_desc = "-"
        
        c_obj.recommender_score = score
        final_customers.append(c_obj)

    # صفحه‌بندی
    paginator = Paginator(final_customers, 10)
    page_number = request.GET.get('page')
    customers_page = paginator.get_page(page_number)

    return render(request, "site/product_recommender/recommendations_list.html", {
        "product": product,
        "customers_page": customers_page,
        "direct_buyers_count": len(direct_buyers),
        "total_recommendations": len(final_customers),
        "max_score": max_score,
    })

def product_detail(request, product_id):
    """
    این View اطلاعات تکمیلی محصول را نشان می‌دهد
    """
    # 1) محصول
    product = get_object_or_404(DunnhumbyProduct, product_id=product_id)

    # 2) مشتریانی که این محصول را خریده‌اند
    households = Transaction.objects.filter(product_id=product_id) \
        .values_list("household_key", flat=True).distinct()

    segments = CustomerSegment.objects.filter(household_key__in=households)

    # 3) قوانین مرتبط با محصول - اصلاح شده برای MSSQL
    # استفاده از همان تکنیک Cast و contains
    rules_qs = AssociationRule.objects.annotate(
        antecedent_text=Cast('antecedent', TextField()),
        consequent_text=Cast('consequent', TextField())
    ).filter(
        Q(antecedent_text__contains=str(product_id)) |
        Q(consequent_text__contains=str(product_id))
    )
    
    # فیلتر دقیق پایتونی برای نمایش
    rules = []
    target_pid_str = str(product_id)
    for r in rules_qs:
        ants = [str(x) for x in r.antecedent]
        cons = [str(x) for x in r.consequent]
        if target_pid_str in ants or target_pid_str in cons:
            rules.append(r)

    return render(request, "site/product_recommender/product_detail.html", {
        "product": product,
        "segments": segments,
        "rules": rules,
    })