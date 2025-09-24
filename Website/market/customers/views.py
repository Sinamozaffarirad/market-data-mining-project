# customers/views.py
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from .models import CustomerProfile, Transaction, Product, CustomerRecommendationCache
from collections import defaultdict
from django.core.paginator import Paginator
from dunnhumby.models import AssociationRule  
from dunnhumby.collab_filter import get_cf_recommendations
from django.utils import timezone
from django.db import models

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


# ---------------------------
# تابع تولید Hybrid Recommender
# ---------------------------
def generate_hybrid_recommendations(household_key, alpha=0.6, top_n=20):
    # ... (this function remains the same as your provided code) ...
    # --- AssociationRule-based ---
    recent_transactions = Transaction.objects.filter(
        household_key=household_key
    ).order_by("-day")[:20]

    purchased_items = set(str(tr.product_id) for tr in recent_transactions)
    rules = AssociationRule.objects.filter(rule_type="product").order_by("-lift")[:200]

    association_recommendations = {}
    for rule in rules:
        antecedent = set(map(str, rule.antecedent))
        consequent = set(map(str, rule.consequent))
        if antecedent & purchased_items:
            for product_id_str in consequent:
                if product_id_str not in purchased_items:
                    product = Product.objects.filter(product_id=product_id_str).first()
                    if product:
                        score = rule.confidence * rule.lift
                        association_recommendations[product.product_id] = {
                            "product": product,
                            "assoc_score": score,
                            "confidence": round(rule.confidence, 2),
                            "lift": round(rule.lift, 2),
                            "support": round(rule.support, 4),
                        }

    # --- Collaborative Filtering ---
    cf_list = get_cf_recommendations(household_key, top_n=50)
    cf_recommendations = {rec["product"].product_id: rec for rec in cf_list}

    # --- Hybrid Merge ---
    hybrid = {}
    all_pids = set(association_recommendations.keys()) | set(cf_recommendations.keys())

    for pid in all_pids:
        assoc_rec = association_recommendations.get(pid)
        cf_rec = cf_recommendations.get(pid)

        assoc_score = assoc_rec["assoc_score"] if assoc_rec else 0
        cf_score = cf_rec["score"] if cf_rec else 0
        
        product_obj = (assoc_rec and assoc_rec["product"]) or (cf_rec and cf_rec["product"])
        
        if product_obj:
            hybrid[pid] = {
                "product": product_obj,
                "hybrid_score": round(alpha * assoc_score + (1 - alpha) * cf_score, 3),
                "confidence": assoc_rec["confidence"] if assoc_rec else None,
                "lift": assoc_rec["lift"] if assoc_rec else None,
                "support": assoc_rec["support"] if assoc_rec else None,
                "cf_score": cf_rec["score"] if cf_rec else 0,
            }

    # --- Sort and Limit ---
    hybrid_recommendations = sorted(
        hybrid.values(), key=lambda x: x["hybrid_score"], reverse=True
    )[:top_n]

    # --- Prepare for JSON serialization ---
    recommendations_for_cache = []
    for rec in hybrid_recommendations:
        prod = rec["product"]
        recommendations_for_cache.append({
            "product_id": prod.product_id,
            "brand": prod.brand or "N/A",
            "department": prod.department or "N/A",
            "commodity_desc": prod.commodity_desc or "N/A",
            "curr_size_of_product": prod.curr_size_of_product or "N/A",
            "hybrid_score": rec["hybrid_score"] or 0,
            "confidence": rec["confidence"] or 0,
            "lift": rec["lift"] or 0,
            "support": rec["support"] or 0,
            "cf_score": rec["cf_score"] or 0,
        })


    return hybrid_recommendations, recommendations_for_cache

# ---------------------------
# Main View with Caching
# ---------------------------
def customer_recommendations(request, pk):
    household = get_object_or_404(CustomerProfile, household_key=pk)

    # --- آلفا رو از GET یا session بگیر ---
    if "alpha" in request.GET:
        try:
            alpha = float(request.GET.get("alpha"))
            if not (0.0 <= alpha <= 1.0):
                alpha = 0.6
        except (ValueError, TypeError):
            alpha = 0.6
        request.session["global_alpha"] = alpha  # ذخیره در session
    else:
        alpha = request.session.get("global_alpha", 0.6)  # اگر نبود، مقدار پیش‌فرض

    recommendations = None
    if 'alpha' in request.GET:
        # کاربر آلفا رو تغییر داده → مستقیم ریکام جدید بساز
        recommendations, _ = generate_hybrid_recommendations(pk, alpha=alpha)
    else:
        latest_rule_timestamp = AssociationRule.objects.aggregate(
            models.Max("created_at")
        )["created_at__max"] or timezone.now()
        cache = CustomerRecommendationCache.objects.filter(household_key=pk).first()

        if cache and cache.alpha == alpha and cache.rules_version >= latest_rule_timestamp:
            cached_recs = cache.recommendations
            product_ids = [rec['product_id'] for rec in cached_recs]
            products = {p.product_id: p for p in Product.objects.filter(product_id__in=product_ids)}
            
            recommendations = []
            for rec in cached_recs:
                if products.get(rec['product_id']):
                    rec['product'] = products[rec['product_id']]
                    recommendations.append(rec)
        else:
            live_recs, cache_recs = generate_hybrid_recommendations(pk, alpha=alpha)
            recommendations = live_recs
            CustomerRecommendationCache.objects.update_or_create(
                household_key=pk,
                defaults={
                    "recommendations": cache_recs,
                    "alpha": alpha,
                    "rules_version": latest_rule_timestamp,
                }
            )

    # Group recommendations by commodity for the card layout
    grouped_recs = defaultdict(list)
    if recommendations:
        for rec in recommendations:
            commodity = rec['product'].commodity_desc or "Uncategorized"
            grouped_recs[commodity].append(rec)

    # --- مرتب‌سازی کارت‌ها بر اساس تعداد محصولات (descending) ---
    grouped_recs = dict(
        sorted(grouped_recs.items(), key=lambda x: len(x[1]), reverse=True)
    )

    return render(request, "site/customers/recommendations.html", {
        "household": household,
        "grouped_recommendations": grouped_recs,
        "current_alpha": alpha,
    })



def customer_churn(request, pk):
    household = get_object_or_404(CustomerProfile, household_key=pk)
    return render(request, "site/customers/churn.html", {"household": household})

FILTER_OPTIONS = {
    "3m": (622, 711),
    "6m": (532, 711),
    "9m": (442, 711),
    "12m": (347, 711),
    "15m": (257, 711),
    "18m": (167, 711),
    "all": (1, 711),
}

def customer_purchases(request, pk):
    household = get_object_or_404(CustomerProfile, household_key=pk)

    period = request.GET.get("period", "all")
    start_day, end_day = FILTER_OPTIONS.get(period, (1, 711))

    transactions = Transaction.objects.filter(
        household_key=pk,
        day__gte=start_day,
        day__lte=end_day
    ).order_by("-day")

    product_map = {
        p.product_id: p for p in Product.objects.filter(
            product_id__in=[tr.product_id for tr in transactions]
        )
    }

    grouped_purchases = defaultdict(list)
    for tr in transactions:
        product = product_map.get(tr.product_id)
        grouped_purchases[tr.basket_id].append({
            "day": tr.day,
            "trans_time": tr.trans_time,
            "product_id": tr.product_id,
            "quantity": tr.quantity,
            "sales_value": tr.sales_value,
            "brand": product.brand if product else "Unknown",
            "department": product.department if product else "Unknown",
            "commodity": product.commodity_desc if product else "Unknown",
            "size": product.curr_size_of_product if product else "",
        })

    grouped_purchases = dict(sorted(
        grouped_purchases.items(),
        key=lambda x: min(p["day"] for p in x[1]),
        reverse=True
    ))

    basket_list = list(grouped_purchases.items())  
    paginator = Paginator(basket_list, 10)  
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    return render(request, "site/customers/purchases.html", {
        "household": household,
        "page_obj": page_obj,  
        "selected_period": period,
        "transaction_count": len(grouped_purchases), 
    })
