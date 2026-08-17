# product_recommender/views.py
from django.shortcuts import render, get_object_or_404
from django.db.models import Q, Count, TextField
from django.db.models.functions import Cast
from django.core.paginator import Paginator
import re

from dunnhumby.models import Transaction, DunnhumbyProduct, AssociationRule, CustomerSegment
from customers.models import CustomerProfile, Product
from customers.ml.cf_cache import get_similar_households
from customers.ml.feature_engineering import (
    compute_product_popularity,
    compute_commodity_repurchase_cycles,
    build_household_features_for_target,
)
from customers.ml.recommender_model import HybridRecommenderModel
from .forms import ProductSearchForm

# Lazily-loaded singleton, shared logic with customers app (same trained model).
_ML_MODEL_CACHE = {"model": None, "loaded": False}


def _get_ml_model():
    if not _ML_MODEL_CACHE["loaded"]:
        _ML_MODEL_CACHE["model"] = HybridRecommenderModel.load()
        _ML_MODEL_CACHE["loaded"] = True
    return _ML_MODEL_CACHE["model"]


def _normalize_label(s):
    if not s:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()
    s = re.sub(r"[^0-9a-z]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------
# صفحه‌ی جستجو (انتخاب سطح + مقدار هدف)
# ---------------------------
def recommend_home(request):
    """
    صفحه‌ی جستجو با سه سطح: Product / Commodity / Department.
    کاربر یه سطح رو انتخاب می‌کنه، بعد یه عبارت جستجو می‌زنه، و لیستی از
    گزینه‌های مطابق (با یه دکمه‌ی «Get Recommendations» برای هرکدوم) می‌بینه.
    """
    level = request.GET.get("level", "product")
    if level not in ("product", "commodity", "department"):
        level = "product"

    form = ProductSearchForm(request.GET or None)
    query = None
    products = None
    commodities = None
    departments = None

    if form.is_valid():
        query = form.cleaned_data.get("query")

        if query:
            if level == "product":
                products = DunnhumbyProduct.objects.filter(
                    Q(brand__icontains=query) |
                    Q(commodity_desc__icontains=query) |
                    Q(sub_commodity_desc__icontains=query)
                ).order_by("brand", "commodity_desc")[:50]

            elif level == "commodity":
                commodities = (
                    DunnhumbyProduct.objects
                    .filter(commodity_desc__icontains=query)
                    .exclude(commodity_desc__isnull=True)
                    .values_list("commodity_desc", flat=True)
                    .distinct()
                    .order_by("commodity_desc")[:50]
                )

            else:  # department
                departments = (
                    DunnhumbyProduct.objects
                    .filter(department__icontains=query)
                    .exclude(department__isnull=True)
                    .values_list("department", flat=True)
                    .distinct()
                    .order_by("department")[:50]
                )

    return render(request, "site/product_recommender/recommend_customers.html", {
        "form": form,
        "query": query,
        "level": level,
        "products": products,
        "commodities": commodities,
        "departments": departments,
    })


# ---------------------------
# تولید کاندید خانوار برای هر سطح
# ---------------------------
def _target_product_ids(level, value):
    if level == "product":
        return [value]
    if level == "commodity":
        return list(Product.objects.filter(commodity_desc=value).values_list("product_id", flat=True))
    return list(Product.objects.filter(department=value).values_list("product_id", flat=True))


def _get_direct_buyers(level, value, target_pids):
    if level == "product":
        return set(Transaction.objects.filter(product_id=value).values_list("household_key", flat=True))
    return set(Transaction.objects.filter(product_id__in=target_pids).values_list("household_key", flat=True))


def _get_content_candidates(level, value):
    """فقط در سطح Product معنا داره: مشتریانی که محصولات هم‌دسته خریده‌اند."""
    if level != "product":
        return set()
    product = DunnhumbyProduct.objects.filter(product_id=value).first()
    if not product:
        return set()
    similar_ids = list(
        DunnhumbyProduct.objects.filter(
            Q(sub_commodity_desc=product.sub_commodity_desc) | Q(commodity_desc=product.commodity_desc)
        ).exclude(product_id=value).values_list("product_id", flat=True)[:500]
    )
    return set(
        Transaction.objects.filter(product_id__in=similar_ids)
        .values_list("household_key", flat=True)
    )


def _get_assoc_candidates(level, value, target_pids, max_rules=100):
    """
    خانوارهایی که قوانین انجمنی (در همون سطح: product/commodity/department)
    اونا رو به سمت این محصول/دسته هدایت می‌کنن.
    """
    rule_type = level  # 'product' | 'commodity' | 'department'
    candidates = set()

    if level == "product":
        target_pid_str = str(value)
        rules_qs = AssociationRule.objects.filter(rule_type="product").annotate(
            consequent_text=Cast("consequent", TextField())
        ).filter(consequent_text__contains=target_pid_str).order_by("-lift")[:max_rules]

        for rule in rules_qs:
            consequents = [str(x) for x in rule.consequent]
            if target_pid_str not in consequents:
                continue
            antecedent_pids = [int(x) for x in rule.antecedent if str(x).isdigit()]
            if antecedent_pids:
                candidates |= set(
                    Transaction.objects.filter(product_id__in=antecedent_pids)
                    .values_list("household_key", flat=True)
                )
    else:
        target_norm = _normalize_label(value)
        rules_qs = AssociationRule.objects.filter(rule_type=rule_type).order_by("-lift")[:max_rules]
        for rule in rules_qs:
            consequents_norm = {_normalize_label(c) for c in (rule.consequent or [])}
            if target_norm not in consequents_norm:
                continue
            antecedent_labels = {_normalize_label(a) for a in (rule.antecedent or [])}
            field = "commodity_desc" if level == "commodity" else "department"
            antecedent_pids = list(
                Product.objects.filter(**{f"{field}__in": list(antecedent_labels)})
                .values_list("product_id", flat=True)
            ) if antecedent_labels else []
            if antecedent_pids:
                candidates |= set(
                    Transaction.objects.filter(product_id__in=antecedent_pids)
                    .values_list("household_key", flat=True)
                )

    return candidates


def recommend_customers(request):
    """
    خانوارهایی که به احتمال زیاد این محصول/دسته/دپارتمان رو خریداری می‌کنن،
    رتبه‌بندی‌شده با همون مدل ML آموزش‌دیده‌ی Hybrid Recommender.
    """
    level = request.GET.get("level", "product")
    if level not in ("product", "commodity", "department"):
        level = "product"

    raw_value = request.GET.get("value")
    if not raw_value:
        return render(request, "site/product_recommender/recommendations_list.html", {
            "level": level, "target_label": None, "error": "No target selected.",
        })

    if level == "product":
        product = get_object_or_404(DunnhumbyProduct, product_id=raw_value)
        value = int(raw_value)
        target_label = f"{product.brand} — {product.commodity_desc}"
        target_meta = product
    else:
        product = None
        value = raw_value
        target_label = raw_value
        target_meta = None

    target_pids = _target_product_ids(level, value)
    direct_buyers = _get_direct_buyers(level, value, target_pids)
    candidate_households = set()
    candidate_households |= _get_content_candidates(level, value)
    candidate_households |= _get_assoc_candidates(level, value, target_pids)

    cf_scores = get_similar_households(direct_buyers, top_n=300) if direct_buyers else {}
    if cf_scores is None:
        cf_scores = {}
    candidate_households |= set(cf_scores.keys())

    # کسانی که مستقیم قبلاً خریده‌اند، لید جدید نیستن
    candidate_households -= direct_buyers

    if not candidate_households:
        return render(request, "site/product_recommender/recommendations_list.html", {
            "level": level,
            "target_label": target_label,
            "value": raw_value,
            "product": product,
            "direct_buyers_count": len(direct_buyers),
            "customers_page": None,
        })

    ml_model = _get_ml_model()
    if ml_model is not None:
        popularity_map = ml_model.popularity_map or compute_product_popularity()
        cycle_map = ml_model.cycle_map or compute_commodity_repurchase_cycles()
    else:
        popularity_map = compute_product_popularity()
        cycle_map = compute_commodity_repurchase_cycles()

    as_of_day = None
    from django.db.models import Max
    as_of_day = Transaction.objects.aggregate(Max("day"))["day__max"] or 0

    features_df = build_household_features_for_target(
        list(candidate_households), level, value, as_of_day,
        popularity_map, cycle_map,
        assoc_scores={},  # پیشنهاد association قبلاً در انتخاب کاندید لحاظ شده
        cf_scores=cf_scores,
    )

    if ml_model is not None:
        ml_scores = ml_model.predict_scores(features_df)
    else:
        # نبود مدل؟ برگرد به رتبه‌بندی ساده بر اساس شباهت CF خام
        ml_scores = features_df["cf_score"]

    scored = []
    for hh in candidate_households:
        base_score = float(ml_scores.loc[hh]) if hh in ml_scores.index else 0.0
        target_count = int(features_df.loc[hh, "household_commodity_count"]) if hh in features_df.index else 0
        favorite_boost = 1.0 + min(target_count, 10) * 0.02
        scored.append((hh, base_score * favorite_boost))

    scored.sort(key=lambda kv: kv[1], reverse=True)
    scored = scored[:100]
    max_score = scored[0][1] if scored else 1.0

    household_keys = [hh for hh, _ in scored]
    profiles = {c.household_key: c for c in CustomerProfile.objects.filter(household_key__in=household_keys)}

    final_customers = []
    for hh, score in scored:
        c_obj = profiles.get(hh)
        if not c_obj:
            c_obj = CustomerProfile(household_key=hh)
            c_obj.age_desc = "-"
        c_obj.recommender_score = round(score, 4)
        c_obj.score_percent = round((score / max_score) * 100, 1) if max_score else 0  # ← جدید
        final_customers.append(c_obj)
        
    paginator = Paginator(final_customers, 10)
    customers_page = paginator.get_page(request.GET.get("page"))

    return render(request, "site/product_recommender/recommendations_list.html", {
        "level": level,
        "target_label": target_label,
        "value": raw_value,
        "product": product,
        "customers_page": customers_page,
        "direct_buyers_count": len(direct_buyers),
        "total_recommendations": len(final_customers),
        "max_score": max_score,
    })


def product_detail(request, product_id):
    """اطلاعات تکمیلی محصول: مشتریانی که خریده‌اند و قوانین انجمنی مرتبط."""
    product = get_object_or_404(DunnhumbyProduct, product_id=product_id)

    households = Transaction.objects.filter(product_id=product_id) \
        .values_list("household_key", flat=True).distinct()

    segments = CustomerSegment.objects.filter(household_key__in=households)

    rules_qs = AssociationRule.objects.annotate(
        antecedent_text=Cast("antecedent", TextField()),
        consequent_text=Cast("consequent", TextField())
    ).filter(
        Q(antecedent_text__contains=str(product_id)) |
        Q(consequent_text__contains=str(product_id))
    )

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