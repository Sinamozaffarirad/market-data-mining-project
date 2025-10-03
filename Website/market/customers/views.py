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
def generate_hybrid_recommendations(household_key, alpha=0.6, top_n=20, levels_order=None):
    if levels_order is None:
        levels_order = ['product', 'commodity', 'department']

    recent_transactions = Transaction.objects.filter(household_key=household_key).order_by("-day")[:50]
    purchased_product_ids = [str(tr.product_id) for tr in recent_transactions]

    final_recs = {}
    needed = top_n

    latest_rule_timestamp = AssociationRule.objects.aggregate(models.Max("created_at"))["created_at__max"]

    for level in levels_order:
        if needed <= 0:
            continue

        # --- association rules ---
        rules_qs = AssociationRule.objects.filter(rule_type=level).order_by('-lift')[:500]
        assoc_recs = {}

        if level == 'product':
            purchased_items_level = set(purchased_product_ids)
        else:
            prod_meta = {str(p.product_id): (p.commodity_desc, p.department)
                         for p in Product.objects.filter(product_id__in=set(purchased_product_ids))}
            if level == 'commodity':
                purchased_items_level = {v[0] for v in prod_meta.values() if v[0]}
            else:
                purchased_items_level = {v[1] for v in prod_meta.values() if v[1]}

        for rule in rules_qs:
            antecedent = set(map(str, rule.antecedent))
            consequent = [str(x) for x in rule.consequent]

            if antecedent & purchased_items_level:
                for cons in consequent:
                    if level == 'product':
                        try:
                            pid = int(cons)
                        except ValueError:
                            continue
                        prod_obj = Product.objects.filter(product_id=pid).first()
                        if prod_obj:
                            score = float(rule.confidence) * float(rule.lift)
                            assoc_recs[prod_obj.product_id] = {
                                'product': prod_obj,
                                'assoc_score': score,
                                'confidence': round(rule.confidence, 3),
                                'lift': round(rule.lift, 3),
                                'support': round(rule.support, 4),
                                'level': level,
                                'origin': 'rule'
                            }

                    else:
                        # normalize cons
                        if level == 'commodity':
                            qs = Product.objects.filter(commodity_desc__iexact=cons.strip()).order_by('-product_id')[:10]
                        else:
                            qs = Product.objects.filter(department__iexact=cons.strip()).order_by('-product_id')[:10]

                        for prod_obj in qs:
                            score = float(rule.confidence) * float(rule.lift) * 0.9
                            assoc_recs[prod_obj.product_id] = {
                                'product': prod_obj, 'assoc_score': score,
                                'confidence': round(rule.confidence, 3),
                                'lift': round(rule.lift, 3),
                                'support': round(rule.support, 4),
                                'level': level
                            }

        # --- basket analysis (فقط برای commodity/department) ---
        if level in ['commodity', 'department']:
            basket_qs = BasketAnalysis.objects.filter(household_key=household_key).first()
            if basket_qs and basket_qs.department_mix:
                dept_mix = basket_qs.department_mix  # این خودش dict هست
                for item_key, weight in dept_mix.items():
                    if level == 'department':
                        qs = Product.objects.filter(department__iexact=item_key.strip())[:5]
                    else:
                        qs = Product.objects.filter(commodity_desc__iexact=item_key.strip())[:5]

                    for prod_obj in qs:
                        assoc_recs[prod_obj.product_id] = {
                            'product': prod_obj,
                            'assoc_score': float(weight),
                            'confidence': None,
                            'lift': None,
                            'support': None,
                            'level': level,
                            'origin': 'basket'
                        }


        # --- collaborative filtering ---
        cf_list = get_cf_recommendations(household_key, top_n=top_n*2, level=level)
        cf_recs = {rec['product'].product_id: {
            'product': rec['product'], 'cf_score': rec['score'], 'level': level
        } for rec in cf_list}

        # --- merge ---
        merged_level = {}
        for pid in set(list(assoc_recs.keys()) + list(cf_recs.keys())):
            assoc_score = assoc_recs.get(pid, {}).get('assoc_score', 0)
            cf_score = cf_recs.get(pid, {}).get('cf_score', 0)
            product_obj = assoc_recs.get(pid, {}).get('product') or cf_recs.get(pid, {}).get('product')
            hybrid_score = alpha * assoc_score + (1 - alpha) * cf_score
            
            origin_parts = []
            if pid in assoc_recs:
                origin_parts.append('rule' if assoc_recs[pid].get('origin') != 'basket' else 'basket')
            if pid in cf_recs:
                origin_parts.append('cf')
            
            merged_level[pid] = {
                'product': product_obj,
                'hybrid_score': hybrid_score,
                'assoc_score': assoc_score,
                'cf_score': cf_score,
                'confidence': assoc_recs.get(pid, {}).get('confidence'),
                'lift': assoc_recs.get(pid, {}).get('lift'),
                'support': assoc_recs.get(pid, {}).get('support'),
                'source_level': level,
                'origin': '+'.join(origin_parts)  # rule/cf/basket یا ترکیب
            }

        sorted_level = sorted(merged_level.items(), key=lambda x: x[1]['hybrid_score'], reverse=True)
        for pid, rec in sorted_level:
            if needed <= 0:
                break
            if pid in final_recs:
                continue
            if str(pid) in purchased_product_ids:
                continue
            final_recs[pid] = rec
            needed -= 1

    final_list = list(final_recs.values())[:top_n]

    cache_for_store = []
    for rec in final_list:
        prod = rec['product']
        cache_for_store.append({
            'product_id': prod.product_id,
            'brand': prod.brand or "N/A",
            'department': prod.department or "N/A",
            'commodity_desc': prod.commodity_desc or "N/A",
            'curr_size_of_product': prod.curr_size_of_product or "N/A",
            'hybrid_score': round(rec.get('hybrid_score', 0), 4),
            'assoc_score': round(rec.get('assoc_score', 0), 4),
            'cf_score': round(rec.get('cf_score', 0), 4),
            'confidence': rec.get('confidence') or 0,
            'lift': rec.get('lift') or 0,
            'support': rec.get('support') or 0,
            'source_level': rec.get('source_level'),
            'origin': rec.get('origin'),
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
        segment.churn_probability_percent = prob * 100
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
