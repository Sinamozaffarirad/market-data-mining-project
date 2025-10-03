# customers/views.py
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .models import CustomerProfile, Transaction, Product, CustomerRecommendationCache
from collections import defaultdict
from django.core.paginator import Paginator
from dunnhumby.models import AssociationRule, CustomerSegment, BasketAnalysis
from dunnhumby.collab_filter import get_cf_recommendations
from django.utils import timezone
from django.db import models
import re
import logging

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
logger = logging.getLogger(__name__)

def _normalize_label(s):
    """Normalize a label for reliable matching: lower, remove punctuation, collapse spaces."""
    if not s:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()
    # replace non-alnum chars with space, collapse multiple spaces
    s = re.sub(r'[^0-9a-z]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def generate_hybrid_recommendations(household_key, alpha=0.6, top_n=20, levels_order=None):
    if levels_order is None:
        levels_order = ['product', 'commodity', 'department']

    recent_pids = list(
        Transaction.objects.filter(household_key=household_key)
        .order_by("-day")
        .values_list("product_id", flat=True)[:250]
    )
    purchased_product_ids = {str(pid) for pid in recent_pids}
    
    # ✅ دیکشنری‌های جداگانه برای جمع‌آوری نتایج هر الگوریتم
    all_assoc_recs = {}
    all_cf_recs = {}

    latest_rule_timestamp = AssociationRule.objects.aggregate(models.Max("created_at"))["created_at__max"]

    # --- مرحله ۱: جمع‌آوری تمام توصیه‌ها از همه سطوح ---
    for level in levels_order:
        # --- بخش قوانین انجمنی (Association Rules) ---
        rules_qs = AssociationRule.objects.filter(rule_type=level).order_by('-lift')[:500]
        
        # ساخت مجموعه آیتم‌های خریداری شده بر اساس سطح فعلی
        if level == 'product':
            purchased_items_level = purchased_product_ids
        else:
            prod_meta = {
                str(p.product_id): (p.commodity_desc or '', p.department or '')
                for p in Product.objects.filter(product_id__in=purchased_product_ids)
            }
            if level == 'commodity':
                purchased_items_level = { _normalize_label(v[0]) for v in prod_meta.values() if v[0] }
            else: # department
                purchased_items_level = { _normalize_label(v[1]) for v in prod_meta.values() if v[1] }

        for rule in rules_qs:
            antecedent_items = rule.antecedent if isinstance(rule.antecedent, list) else [rule.antecedent]
            antecedent_candidates = {_normalize_label(a) for a in (antecedent_items or [])}
            
            if not antecedent_candidates.isdisjoint(purchased_items_level):
                consequent_items = rule.consequent if isinstance(rule.consequent, list) else [rule.consequent]
                raw_consequents = [c for c in consequent_items if c]
                if not raw_consequents: continue

                product_candidates = []
                if level == "product":
                    pids_to_fetch = [int(c) for c in raw_consequents if c.isdigit()]
                    if pids_to_fetch:
                        product_candidates = list(Product.objects.filter(product_id__in=pids_to_fetch))
                else:
                    from django.db.models import Q
                    q_objects = Q()
                    field = "commodity_desc" if level == "commodity" else "department"
                    for cons in raw_consequents:
                        q_objects |= Q(**{f"{field}__iexact": cons})
                    if q_objects:
                        product_candidates = list(Product.objects.filter(q_objects).order_by('?')[:5])
                
                for prod_obj in product_candidates:
                    pid = prod_obj.product_id
                    if str(pid) not in purchased_product_ids and pid not in all_assoc_recs:
                        score_multiplier = 1.0 if level == "product" else 0.9
                        score = (float(rule.confidence or 0) * float(rule.lift or 0)) * score_multiplier
                        all_assoc_recs[pid] = {
                            "product": prod_obj, "assoc_score": score,
                            "confidence": round(rule.confidence or 0, 3), "lift": round(rule.lift or 0, 3),
                            "support": round(rule.support or 0, 4), "source_level": level
                        }

        # --- بخش فیلترینگ مشارکتی (Collaborative Filtering) ---
        cf_list = get_cf_recommendations(household_key, top_n=(top_n * 2), level=level)
        for rec in cf_list:
            pid = rec['product'].product_id
            if str(pid) not in purchased_product_ids and pid not in all_cf_recs:
                all_cf_recs[pid] = {
                    'product': rec['product'], 'cf_score': rec['score'], 'source_level': level
                }

    # --- مرحله ۲: ترکیب، نرمال‌سازی و مرتب‌سازی نهایی ---
    final_recs = {}
    all_pids = set(all_assoc_recs.keys()) | set(all_cf_recs.keys())

    if not all_pids:
        return [], [], latest_rule_timestamp

    max_assoc = max((rec['assoc_score'] for rec in all_assoc_recs.values()), default=1.0)
    max_cf = max((rec['cf_score'] for rec in all_cf_recs.values()), default=1.0)
    max_assoc = max(max_assoc, 1.0)
    max_cf = max(max_cf, 1.0)

    for pid in all_pids:
        assoc_data = all_assoc_recs.get(pid, {})
        cf_data = all_cf_recs.get(pid, {})
        
        norm_assoc = assoc_data.get('assoc_score', 0) / max_assoc
        norm_cf = cf_data.get('cf_score', 0) / max_cf
        
        hybrid_score = (alpha * norm_assoc) + ((1 - alpha) * norm_cf)
        
        if hybrid_score > 0:
            origin_parts = []
            if assoc_data: origin_parts.append('rule')
            if cf_data: origin_parts.append('cf')

            final_recs[pid] = {
                'product': assoc_data.get('product') or cf_data.get('product'),
                'hybrid_score': hybrid_score,
                'assoc_score': assoc_data.get('assoc_score', 0),
                'cf_score': cf_data.get('cf_score', 0),
                'confidence': assoc_data.get('confidence'),
                'lift': assoc_data.get('lift'),
                'support': assoc_data.get('support'),
                'source_level': assoc_data.get('source_level') or cf_data.get('source_level'),
                'origin': '+'.join(origin_parts)
            }

    # مرتب‌سازی نهایی بر اساس امتیاز ترکیبی
    sorted_recs = sorted(final_recs.values(), key=lambda x: x['hybrid_score'], reverse=True)
    final_list = sorted_recs[:top_n]

    # آماده‌سازی داده برای ذخیره در کش
    cache_for_store = []
    for rec in final_list:
        prod = rec['product']
        cache_for_store.append({
            'product_id': prod.product_id,
            'brand': prod.brand or "N/A", 'department': prod.department or "N/A",
            'commodity_desc': prod.commodity_desc or "N/A", 'curr_size_of_product': prod.curr_size_of_product or "N/A",
            'hybrid_score': round(rec.get('hybrid_score', 0), 4), 'assoc_score': round(rec.get('assoc_score', 0), 4),
            'cf_score': round(rec.get('cf_score', 0), 4), 'confidence': rec.get('confidence') or 0,
            'lift': rec.get('lift') or 0, 'support': rec.get('support') or 0,
            'source_level': rec.get('source_level'), 'origin': rec.get('origin'),
        })
        
    return final_list, cache_for_store, latest_rule_timestamp

# ---------------------------
# Main View with Caching
# ---------------------------
def customer_recommendations(request, pk):
    household = get_object_or_404(CustomerProfile, household_key=pk)

    # Get alpha from the request, falling back to the session or a default value
    try:
        alpha = float(request.GET.get("alpha", request.session.get("global_alpha", 0.6)))
        if not (0.0 <= alpha <= 1.0):
            alpha = 0.6
    except (ValueError, TypeError):
        alpha = 0.6

    # Save the current alpha to the session for persistence
    request.session["global_alpha"] = alpha

    # Check cache validity
    latest_rule_timestamp = AssociationRule.objects.aggregate(models.Max("created_at"))["created_at__max"]
    cache = CustomerRecommendationCache.objects.filter(household_key=pk).first()

    recalculate = (
        "alpha" in request.GET
        or not cache
        or cache.alpha != alpha
        or cache.rules_version != latest_rule_timestamp
    )

    if recalculate:
        live_recs, cache_recs, latest_rule_timestamp = generate_hybrid_recommendations(pk, alpha=alpha)
        recommendations = live_recs
        CustomerRecommendationCache.objects.update_or_create(
            household_key=pk,
            defaults={
                "recommendations": cache_recs,
                "alpha": alpha,
                "rules_version": latest_rule_timestamp,
            },
        )
    else:
        # Load valid recommendations from the cache
        cached_recs = cache.recommendations
        product_ids = [rec["product_id"] for rec in cached_recs]
        products = {
            p.product_id: p
            for p in Product.objects.filter(product_id__in=product_ids)
        }
        recommendations = []
        for rec in cached_recs:
            if products.get(rec["product_id"]):
                rec["product"] = products[rec["product_id"]]
                recommendations.append(rec)

    # Group recommendations for card layout
    grouped_recs = defaultdict(list)
    if recommendations:
        for rec in recommendations:
            commodity = rec["product"].commodity_desc or "Uncategorized"
            grouped_recs[commodity].append(rec)
    
    grouped_recs = dict(sorted(grouped_recs.items(), key=lambda item: len(item[1]), reverse=True))

    return render(
        request,
        "site/customers/recommendations.html",
        {
            "household": household,
            "grouped_recommendations": grouped_recs,
            "current_alpha": alpha,
        },
    )


@login_required
def customer_churn(request, pk):
    household = get_object_or_404(CustomerProfile, household_key=pk)
    segment = CustomerSegment.objects.filter(household_key=pk).first()

    # Manually assign the churn_risk label if it doesn't exist on the model
    if segment and hasattr(segment, 'churn_probability'):
        prob = segment.churn_probability
        if prob is None:
            segment.churn_risk = "N/A"
        elif prob > 0.75:
            segment.churn_risk = "Very High Risk"
        elif prob > 0.50:
            segment.churn_risk = "High Risk"
        elif prob > 0.25:
            segment.churn_risk = "Medium Risk"
        else:
            segment.churn_risk = "Low Risk"

    context = {
        "household": household,
        "segment": segment,
    }
    return render(request, "site/customers/churn.html", context)


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
