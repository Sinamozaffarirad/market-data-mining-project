# customers/views.py
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .models import CustomerProfile, Transaction, Product, CustomerRecommendationCache
from collections import defaultdict
from django.core.paginator import Paginator
from dunnhumby.models import (
    AssociationRule,
    CustomerSegment,
    BasketAnalysis,
    ChurnExperiment,
    CustomerWindowHistory,
    CachedCustomerWindow,
    ChurnExperimentWindowPrediction,
)
from dunnhumby.collab_filter import get_cf_recommendations
from django.utils import timezone
from django.db import models
from django.db.models import Max
from types import SimpleNamespace
import re
import logging

from .ml.feature_engineering import (
    compute_product_popularity,
    compute_commodity_repurchase_cycles,
    build_candidate_features,
)
from .ml.recommender_model import HybridRecommenderModel

logger = logging.getLogger(__name__)

# How many days of "typical repurchase cycle" makes a commodity a "staple"
# (bought in bulk / lasts a while) rather than a "consumable" (bought often).
STAPLE_CYCLE_THRESHOLD_DAYS = 10

# Lazily-loaded singleton so we don't unpickle the model on every request.
_ML_MODEL_CACHE = {"model": None, "loaded": False}


def _get_ml_model():
    if not _ML_MODEL_CACHE["loaded"]:
        _ML_MODEL_CACHE["model"] = HybridRecommenderModel.load()
        _ML_MODEL_CACHE["loaded"] = True
    return _ML_MODEL_CACHE["model"]


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


def _normalize_label(s):
    """Normalize a label for reliable matching: lower, remove punctuation, collapse spaces."""
    if not s:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()
    s = re.sub(r"[^0-9a-z]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _inject_brand_exploration_candidates(
    household_key, purchased_product_ids, existing_pids, max_new=3
):
    """
    If the household is strongly loyal to one brand within a commodity they
    buy often, add a couple of candidates for the SAME commodity from a
    DIFFERENT brand, so the recommender occasionally nudges them to try
    something new (e.g. always buys Pizza brand A -> also surface brand B).
    """
    history = list(
        Transaction.objects.filter(household_key=household_key)
        .order_by("-day")
        .values_list("product_id", flat=True)[:250]
    )
    if not history:
        return []

    products = {p.product_id: p for p in Product.objects.filter(product_id__in=history)}
    commodity_brand_counts = defaultdict(lambda: defaultdict(int))
    commodity_counts = defaultdict(int)

    for pid in history:
        prod = products.get(pid)
        if not prod or not prod.commodity_desc:
            continue
        commodity_counts[prod.commodity_desc] += 1
        commodity_brand_counts[prod.commodity_desc][prod.brand] += 1

    loyal_commodities = []
    for commodity, count in commodity_counts.items():
        if count < 3:
            continue
        brand_counts = commodity_brand_counts[commodity]
        top_brand, top_count = max(brand_counts.items(), key=lambda kv: kv[1])
        if top_count / count >= 0.8:
            loyal_commodities.append((commodity, top_brand))

    new_candidates = []
    for commodity, dominant_brand in loyal_commodities[:max_new]:
        alt_product = (
            Product.objects.filter(commodity_desc=commodity)
            .exclude(brand=dominant_brand)
            .exclude(product_id__in=purchased_product_ids)
            .exclude(product_id__in=existing_pids)
            .order_by("?")
            .first()
        )
        if alt_product:
            new_candidates.append(alt_product)

    return new_candidates


def generate_hybrid_recommendations(household_key, top_n=20, levels_order=None):
    if levels_order is None:
        levels_order = ["product", "commodity", "department"]

    recent_pids = list(
        Transaction.objects.filter(household_key=household_key)
        .order_by("-day")
        .values_list("product_id", flat=True)[:250]
    )
    purchased_product_ids = {str(pid) for pid in recent_pids}

    all_assoc_recs = {}
    all_cf_recs = {}

    latest_rule_timestamp = AssociationRule.objects.aggregate(models.Max("created_at"))[
        "created_at__max"
    ]

    # --- مرحله ۱: جمع‌آوری کاندیدها از قوانین انجمنی + CF در همه سطوح ---
    for level in levels_order:
        rules_qs = AssociationRule.objects.filter(rule_type=level).order_by("-lift")[
            :500
        ]

        if level == "product":
            purchased_items_level = purchased_product_ids
        else:
            prod_meta = {
                str(p.product_id): (p.commodity_desc or "", p.department or "")
                for p in Product.objects.filter(product_id__in=purchased_product_ids)
            }
            if level == "commodity":
                purchased_items_level = {
                    _normalize_label(v[0]) for v in prod_meta.values() if v[0]
                }
            else:
                purchased_items_level = {
                    _normalize_label(v[1]) for v in prod_meta.values() if v[1]
                }

        for rule in rules_qs:
            antecedent_items = (
                rule.antecedent
                if isinstance(rule.antecedent, list)
                else [rule.antecedent]
            )
            antecedent_candidates = {
                _normalize_label(a) for a in (antecedent_items or [])
            }

            if not antecedent_candidates.isdisjoint(purchased_items_level):
                consequent_items = (
                    rule.consequent
                    if isinstance(rule.consequent, list)
                    else [rule.consequent]
                )
                raw_consequents = [c for c in consequent_items if c]
                if not raw_consequents:
                    continue

                product_candidates = []
                if level == "product":
                    pids_to_fetch = [int(c) for c in raw_consequents if c.isdigit()]
                    if pids_to_fetch:
                        product_candidates = list(
                            Product.objects.filter(product_id__in=pids_to_fetch)
                        )
                else:
                    from django.db.models import Q

                    q_objects = Q()
                    field = "commodity_desc" if level == "commodity" else "department"
                    for cons in raw_consequents:
                        q_objects |= Q(**{f"{field}__iexact": cons})
                    if q_objects:
                        product_candidates = list(
                            Product.objects.filter(q_objects).order_by("?")[:5]
                        )

                for prod_obj in product_candidates:
                    pid = prod_obj.product_id
                    if (
                        str(pid) not in purchased_product_ids
                        and pid not in all_assoc_recs
                    ):
                        score_multiplier = 1.0 if level == "product" else 0.9
                        score = (
                            float(rule.confidence or 0) * float(rule.lift or 0)
                        ) * score_multiplier
                        all_assoc_recs[pid] = {
                            "product": prod_obj,
                            "assoc_score": score,
                            "confidence": round(rule.confidence or 0, 3),
                            "lift": round(rule.lift or 0, 3),
                            "support": round(rule.support or 0, 4),
                            "source_level": level,
                        }

        cf_list = get_cf_recommendations(household_key, top_n=(top_n * 2), level=level)
        for rec in cf_list:
            pid = rec["product"].product_id
            if str(pid) not in purchased_product_ids and pid not in all_cf_recs:
                all_cf_recs[pid] = {
                    "product": rec["product"],
                    "cf_score": rec["score"],
                    "source_level": level,
                }

    # --- مرحله ۲: ادغام کاندیدهای association + CF (بدون بلند کردن با alpha) ---
    final_recs = {}
    all_pids = set(all_assoc_recs.keys()) | set(all_cf_recs.keys())

    if not all_pids:
        ml_model = _get_ml_model()
        return (
            [],
            [],
            latest_rule_timestamp,
            (ml_model.trained_at if ml_model else None),
        )

    max_assoc = max(
        (rec["assoc_score"] for rec in all_assoc_recs.values()), default=1.0
    )
    max_cf = max((rec["cf_score"] for rec in all_cf_recs.values()), default=1.0)
    max_assoc = max(max_assoc, 1.0)
    max_cf = max(max_cf, 1.0)

    for pid in all_pids:
        assoc_data = all_assoc_recs.get(pid, {})
        cf_data = all_cf_recs.get(pid, {})

        # فقط برای fallback (وقتی مدل ML موجود نباشه) استفاده می‌شه
        norm_assoc = assoc_data.get("assoc_score", 0) / max_assoc
        norm_cf = cf_data.get("cf_score", 0) / max_cf
        fallback_score = 0.5 * norm_assoc + 0.5 * norm_cf

        if fallback_score > 0:
            origin_parts = []
            if assoc_data:
                origin_parts.append("rule")
            if cf_data:
                origin_parts.append("cf")

            final_recs[pid] = {
                "product": assoc_data.get("product") or cf_data.get("product"),
                "fallback_score": fallback_score,
                "assoc_score": assoc_data.get("assoc_score", 0),
                "cf_score": cf_data.get("cf_score", 0),
                "confidence": assoc_data.get("confidence"),
                "lift": assoc_data.get("lift"),
                "support": assoc_data.get("support"),
                "source_level": assoc_data.get("source_level")
                or cf_data.get("source_level"),
                "origin": "+".join(origin_parts),
            }

    # --- مرحله ۳: تزریق کاندیدهای «کشف برند جدید» ---
    brand_candidates = _inject_brand_exploration_candidates(
        household_key, purchased_product_ids, set(final_recs.keys())
    )
    for prod_obj in brand_candidates:
        final_recs[prod_obj.product_id] = {
            "product": prod_obj,
            "fallback_score": 0.0,
            "assoc_score": 0,
            "cf_score": 0,
            "confidence": None,
            "lift": None,
            "support": None,
            "source_level": "brand_exploration",
            "origin": "brand_exploration",
        }

    # --- مرحله ۴: امتیازدهی نهایی با مدل ML + قوانین کسب‌وکار ---
    ml_model = _get_ml_model()
    max_day = Transaction.objects.aggregate(Max("day"))["day__max"] or 0
    candidate_ids = list(final_recs.keys())

    if ml_model is not None:
        popularity_map = ml_model.popularity_map or compute_product_popularity()
        cycle_map = ml_model.cycle_map or compute_commodity_repurchase_cycles()
    else:
        popularity_map = compute_product_popularity()
        cycle_map = compute_commodity_repurchase_cycles()

    features_df = build_candidate_features(
        household_key,
        candidate_ids,
        max_day,
        popularity_map,
        cycle_map,
        assoc_scores={pid: r["assoc_score"] for pid, r in final_recs.items()},
        cf_scores={pid: r["cf_score"] for pid, r in final_recs.items()},
    )

    ml_scores = ml_model.predict_scores(features_df) if ml_model is not None else None

    default_gap = cycle_map.get("__default__", 14.0)
    surviving_recs = []
    for pid, rec in final_recs.items():
        feat = features_df.loc[pid] if pid in features_df.index else None
        popularity = popularity_map.get(pid, 0.0)
        gap = cycle_map.get(
            getattr(rec["product"], "commodity_desc", None), default_gap
        )
        days_since = (
            feat["days_since_last_household_commodity_purchase"]
            if feat is not None
            else 9999
        )
        household_commodity_count = (
            feat["household_commodity_count"] if feat is not None else 0
        )

        # --- قانون ۱: سرکوب کالاهای اساسی (staple) که تازه خریداری شده‌اند ---
        is_staple = gap > STAPLE_CYCLE_THRESHOLD_DAYS
        recently_bought = days_since < gap
        if is_staple and recently_bought and rec["source_level"] != "brand_exploration":
            continue

        # --- امتیاز پایه: مدل ML، یا در نبودش fallback_score ---
        if ml_scores is not None and pid in ml_scores.index:
            base_score = float(ml_scores.loc[pid])
        else:
            base_score = rec["fallback_score"]

        # --- قانون ۲: بوست علاقه‌مندی شخصی ---
        favorite_boost = 1.0 + min(household_commodity_count, 10) * 0.02

        final_score = base_score * favorite_boost
        if rec["source_level"] == "brand_exploration":
            final_score = max(final_score, 0.15 * popularity)

        rec["ml_score"] = round(base_score, 4)
        rec["popularity_score"] = round(popularity, 4)
        rec["final_score"] = final_score
        surviving_recs.append(rec)

    sorted_recs = sorted(surviving_recs, key=lambda x: x["final_score"], reverse=True)
    final_list = sorted_recs[:top_n]

    cache_for_store = []
    for rec in final_list:
        prod = rec["product"]
        cache_for_store.append(
            {
                "product_id": prod.product_id,
                "brand": prod.brand or "N/A",
                "department": prod.department or "N/A",
                "commodity_desc": prod.commodity_desc or "N/A",
                "curr_size_of_product": prod.curr_size_of_product or "N/A",
                "ml_score": rec.get("ml_score", 0),
                "popularity_score": rec.get("popularity_score", 0),
                "assoc_score": round(rec.get("assoc_score", 0), 4),
                "cf_score": round(rec.get("cf_score", 0), 4),
                "confidence": rec.get("confidence") or 0,
                "lift": rec.get("lift") or 0,
                "support": rec.get("support") or 0,
                "source_level": rec.get("source_level"),
                "origin": rec.get("origin"),
            }
        )

    return (
        final_list,
        cache_for_store,
        latest_rule_timestamp,
        (ml_model.trained_at if ml_model else None),
    )


# ---------------------------
# Main View with Caching
# ---------------------------
def customer_recommendations(request, pk):
    household = get_object_or_404(CustomerProfile, household_key=pk)

    latest_rule_timestamp = AssociationRule.objects.aggregate(models.Max("created_at"))[
        "created_at__max"
    ]
    ml_model = _get_ml_model()
    current_model_trained_at = ml_model.trained_at if ml_model else None

    cache = CustomerRecommendationCache.objects.filter(household_key=pk).first()

    recalculate = (
        request.GET.get("refresh") == "1"
        or not cache
        or cache.rules_version != latest_rule_timestamp
        or cache.model_trained_at != current_model_trained_at
    )

    if recalculate:
        live_recs, cache_recs, latest_rule_timestamp, model_trained_at = (
            generate_hybrid_recommendations(pk)
        )
        recommendations = live_recs
        CustomerRecommendationCache.objects.update_or_create(
            household_key=pk,
            defaults={
                "recommendations": cache_recs,
                "rules_version": latest_rule_timestamp,
                "model_trained_at": model_trained_at,
            },
        )
    else:
        cached_recs = cache.recommendations
        product_ids = [rec["product_id"] for rec in cached_recs]
        products = {
            p.product_id: p for p in Product.objects.filter(product_id__in=product_ids)
        }
        recommendations = []
        for rec in cached_recs:
            if products.get(rec["product_id"]):
                rec["product"] = products[rec["product_id"]]
                recommendations.append(rec)

    grouped_recs = defaultdict(list)
    if recommendations:
        for rec in recommendations:
            commodity = rec["product"].commodity_desc or "Uncategorized"
            grouped_recs[commodity].append(rec)

    grouped_recs = dict(
        sorted(grouped_recs.items(), key=lambda item: len(item[1]), reverse=True)
    )

    return render(
        request,
        "site/customers/recommendations.html",
        {
            "household": household,
            "grouped_recommendations": grouped_recs,
        },
    )


@login_required
def customer_churn(request, pk):
    household = get_object_or_404(CustomerProfile, household_key=pk)
    segment = CustomerSegment.objects.filter(household_key=pk).first()

    if segment and hasattr(segment, "churn_probability"):
        prob = segment.churn_probability
        segment.churn_probability_percent = prob * 100 if prob is not None else None
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

    # One selector exposes every saved rule, including rules that are not the
    # dashboard's active rule.  Only the latest copy of an identical rule is
    # offered, which also keeps older duplicate runs out of the UI.
    all_experiments = list(ChurnExperiment.objects.order_by("-created_at"))
    available_history_experiments = []
    seen_rules = set()
    for candidate in all_experiments:
        rule_key = (
            candidate.method,
            candidate.observation_window_days,
            candidate.prediction_horizon_days,
            candidate.step_size_days,
        )
        if rule_key not in seen_rules:
            available_history_experiments.append(candidate)
            seen_rules.add(rule_key)

    selected_experiment_id = request.GET.get("history_experiment")
    if selected_experiment_id:
        request.session["selected_history_experiment"] = selected_experiment_id
    else:
        selected_experiment_id = request.session.get("selected_history_experiment")
    selected_experiment = next(
        (
            item
            for item in available_history_experiments
            if str(item.pk) == str(selected_experiment_id)
        ),
        None,
    )
    if selected_experiment is None:
        selected_experiment = next(
            (item for item in available_history_experiments if item.is_active),
            available_history_experiments[0] if available_history_experiments else None,
        )

    history = None
    if selected_experiment:
        experiment = selected_experiment
        rows = []
        previous_health_score = None
        # New experiments read RFM/outcomes from the reusable cache. Older
        # experiments continue to work through the legacy per-experiment rows.
        if experiment.window_cache_id:
            records = CachedCustomerWindow.objects.filter(
                cache=experiment.window_cache, household_key=pk
            ).order_by("cutoff_day")
            probability_by_window = dict(
                ChurnExperimentWindowPrediction.objects.filter(
                    experiment=experiment,
                    window__household_key=pk,
                ).values_list("window_id", "churn_probability")
            )
        else:
            records = CustomerWindowHistory.objects.filter(
                experiment=experiment, household_key=pk
            ).order_by("cutoff_day")
            probability_by_window = None
        for record in records:
            probability = (
                probability_by_window.get(record.id)
                if probability_by_window is not None
                else record.churn_probability
            )
            health_score = int(record.r_score + record.f_score + record.m_score)
            if health_score >= 12:
                health_label, health_class = "Strong", "success"
            elif health_score >= 9:
                health_label, health_class = "Healthy", "primary"
            elif health_score >= 6:
                health_label, health_class = "Weak", "warning"
            else:
                health_label, health_class = "Critical", "danger"

            if probability is None:
                risk_label, risk_class = "Not scored", "secondary"
            elif probability > 0.75:
                risk_label, risk_class = "Very high", "danger"
            elif probability > 0.50:
                risk_label, risk_class = "High", "warning"
            elif probability > 0.25:
                risk_label, risk_class = "Medium", "info"
            else:
                risk_label, risk_class = "Low", "success"

            if previous_health_score is None:
                health_change = "First checkpoint"
            elif health_score >= previous_health_score + 2:
                health_change = "Improving"
            elif health_score <= previous_health_score - 2:
                health_change = "Worsening"
            else:
                health_change = "Stable"
            previous_health_score = health_score
            prediction_is_correct = (
                None
                if probability is None
                else (probability >= 0.50) == bool(record.is_churn)
            )
            rows.append(
                SimpleNamespace(
                    record=record,
                    probability_percent=(
                        probability * 100 if probability is not None else None
                    ),
                    actual_outcome=(
                        "No purchase in horizon"
                        if record.is_churn
                        else "Purchased in horizon"
                    ),
                    prediction_is_correct=prediction_is_correct,
                    health_score=health_score,
                    health_label=health_label,
                    health_class=health_class,
                    risk_label=risk_label,
                    risk_class=risk_class,
                    health_change=health_change,
                )
            )
        scored_count = sum(row.probability_percent is not None for row in rows)
        purchased_count = sum(
            row.actual_outcome == "Purchased in horizon" for row in rows
        )
        no_purchase_count = sum(
            row.actual_outcome == "No purchase in horizon" for row in rows
        )
        health_change = (
            rows[-1].health_score - rows[0].health_score if len(rows) > 1 else None
        )
        rows.reverse()
        paginator = Paginator(rows, 10)
        history = {
            "experiment": experiment,
            "cache_available": bool(experiment.window_cache_id),
            "page": paginator.get_page(request.GET.get("history_page", 1)),
            "summary": {
                "snapshot_count": len(rows),
                "scored_count": scored_count,
                "purchased_count": purchased_count,
                "no_purchase_count": no_purchase_count,
                "latest_health_label": rows[0].health_label if rows else "No snapshot",
                "latest_health_score": rows[0].health_score if rows else None,
                "health_change": health_change,
            },
        }

    context = {
        "household": household,
        "segment": segment,
        "history": history,
        "available_history_experiments": available_history_experiments,
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
        household_key=pk, day__gte=start_day, day__lte=end_day
    ).order_by("-day")

    product_map = {
        p.product_id: p
        for p in Product.objects.filter(
            product_id__in=[tr.product_id for tr in transactions]
        )
    }

    grouped_purchases = defaultdict(list)
    for tr in transactions:
        product = product_map.get(tr.product_id)
        grouped_purchases[tr.basket_id].append(
            {
                "day": tr.day,
                "trans_time": tr.trans_time,
                "product_id": tr.product_id,
                "quantity": tr.quantity,
                "sales_value": tr.sales_value,
                "brand": product.brand if product else "Unknown",
                "department": product.department if product else "Unknown",
                "commodity": product.commodity_desc if product else "Unknown",
                "size": product.curr_size_of_product if product else "",
            }
        )

    grouped_purchases = dict(
        sorted(
            grouped_purchases.items(),
            key=lambda x: min(p["day"] for p in x[1]),
            reverse=True,
        )
    )

    basket_list = list(grouped_purchases.items())
    paginator = Paginator(basket_list, 10)
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    return render(
        request,
        "site/customers/purchases.html",
        {
            "household": household,
            "page_obj": page_obj,
            "selected_period": period,
            "transaction_count": len(grouped_purchases),
        },
    )
