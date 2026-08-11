# customers/ml/feature_engineering.py
"""
Feature engineering for the Hybrid Recommender ML model.

CAVEAT: AssociationRule rows are computed once (not re-derived per cutoff
day), so assoc_score is not strictly "as of cutoff day" — rules describe
stable co-purchase patterns, so this is a reasonable simplification.
Household-level features (recency, frequency, repurchase gaps) ARE
strictly filtered to day <= cutoff_day, which is what actually matters
for preventing label leakage.
"""
import numpy as np
import pandas as pd
from django.db import connection

from customers.models import Transaction, Product

FEATURE_COLUMNS = [
    "assoc_score", "cf_score", "content_score", "product_popularity",
    "household_product_count", "household_commodity_count",
    "days_since_last_household_commodity_purchase",
    "commodity_median_gap_days", "is_new_brand_for_household",
]


def compute_product_popularity(as_of_day=None):
    """product_id -> popularity score = log1p(total quantity sold)."""
    query = "SELECT product_id, SUM(quantity) AS qty FROM transactions"
    params = []
    if as_of_day is not None:
        query += " WHERE day <= %s"
        params.append(as_of_day)
    query += " GROUP BY product_id"
    with connection.cursor() as cursor:
        cursor.execute(query, params)
        rows = cursor.fetchall()
    return {int(pid): float(np.log1p(qty or 0)) for pid, qty in rows}


def compute_commodity_repurchase_cycles(as_of_day=None):
    """
    commodity_desc -> median days between consecutive purchases of that
    commodity, across ALL households. Small median (soda, bought every
    few days) = always suggestible. Large median (rice/meat, bought
    monthly) = don't suggest right after a recent purchase.
    """
    query = """
        SELECT t.household_key, p.commodity_desc, t.day
        FROM transactions t
        JOIN product p ON t.product_id = p.product_id
        WHERE p.commodity_desc IS NOT NULL
    """
    params = []
    if as_of_day is not None:
        query += " AND t.day <= %s"
        params.append(as_of_day)

    with connection.cursor() as cursor:
        cursor.execute(query, params)
        rows = cursor.fetchall()
    if not rows:
        return {"__default__": 14.0}

    df = pd.DataFrame(rows, columns=["household_key", "commodity_desc", "day"])
    df = df.drop_duplicates().sort_values(["household_key", "commodity_desc", "day"])
    df["gap"] = df.groupby(["household_key", "commodity_desc"])["day"].diff()
    medians = df.groupby("commodity_desc")["gap"].median()
    default_gap = float(medians.median()) if not medians.empty else 14.0
    result = medians.fillna(default_gap).to_dict()
    result["__default__"] = default_gap
    return result


def build_candidate_features(household_key, candidate_product_ids, as_of_day,
                              popularity_map, cycle_map,
                              assoc_scores=None, cf_scores=None):
    """
    Returns a DataFrame indexed by product_id with FEATURE_COLUMNS, for one
    household and a list of candidate product_ids, using only data on or
    before as_of_day.
    """
    assoc_scores = assoc_scores or {}
    cf_scores = cf_scores or {}

    history = list(Transaction.objects.filter(
        household_key=household_key, day__lte=as_of_day
    ).values("product_id", "day"))
    history_df = pd.DataFrame(history) if history else pd.DataFrame(columns=["product_id", "day"])

    product_meta = {
        p.product_id: p for p in Product.objects.filter(
            product_id__in=set(candidate_product_ids) | set(history_df["product_id"])
        )
    }

    hist = history_df.copy()
    hist["commodity"] = hist["product_id"].map(lambda pid: getattr(product_meta.get(pid), "commodity_desc", None))
    hist["brand"] = hist["product_id"].map(lambda pid: getattr(product_meta.get(pid), "brand", None))

    product_counts = hist.groupby("product_id").size().to_dict()
    commodity_counts = hist.groupby("commodity").size().to_dict()
    last_commodity_day = hist.groupby("commodity")["day"].max().to_dict()
    dominant_brand = (
        hist.groupby(["commodity", "brand"]).size().reset_index(name="cnt")
        .sort_values("cnt", ascending=False).drop_duplicates("commodity")
        .set_index("commodity")["brand"].to_dict()
    ) if not hist.empty else {}

    default_gap = cycle_map.get("__default__", 14.0)
    rows = []
    for pid in candidate_product_ids:
        prod = product_meta.get(pid)
        commodity = getattr(prod, "commodity_desc", None)
        brand = getattr(prod, "brand", None)
        days_since = as_of_day - last_commodity_day.get(commodity, -10_000)
        gap = cycle_map.get(commodity, default_gap)
        rows.append({
            "product_id": pid,
            "assoc_score": assoc_scores.get(pid, 0.0),
            "cf_score": cf_scores.get(pid, 0.0),
            "content_score": 1.0 if commodity in commodity_counts else 0.0,
            "product_popularity": popularity_map.get(pid, 0.0),
            "household_product_count": product_counts.get(pid, 0),
            "household_commodity_count": commodity_counts.get(commodity, 0),
            "days_since_last_household_commodity_purchase": min(days_since, 9999),
            "commodity_median_gap_days": gap,
            "is_new_brand_for_household": int(
                dominant_brand.get(commodity) is not None and dominant_brand.get(commodity) != brand
            ),
        })
    return pd.DataFrame(rows).set_index("product_id")