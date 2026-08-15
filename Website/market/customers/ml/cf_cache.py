# customers/ml/cf_cache.py
"""
Precomputed, on-disk cache of the collaborative-filtering similarity
matrix. dunnhumby.collab_filter.get_cf_recommendations rebuilds the whole
matrix from 2.6M transactions and recomputes cosine_similarity on every
single call - that's why recommendation pages were taking minutes.
This module builds it ONCE (via `python manage.py build_cf_cache`) and
every request afterward just loads the cached file into memory.
"""
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from django.db import connection

from customers.models import Product

CF_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "ml_models_cache" / "hybrid_recommender"
CF_CACHE_PATH = CF_CACHE_DIR / "cf_matrix_cache.pkl"


def build_and_save():
    """Rebuild the household x product matrix + similarity matrix and save to disk."""
    query = "SELECT household_key, product_id, COUNT(*) as cnt FROM transactions GROUP BY household_key, product_id"
    with connection.cursor() as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()

    df = pd.DataFrame(rows, columns=["household_key", "product_id", "cnt"])
    user_item = df.pivot_table(index="household_key", columns="product_id", values="cnt", fill_value=0)
    similarity = cosine_similarity(user_item)
    similarity_df = pd.DataFrame(similarity, index=user_item.index, columns=user_item.index)

    product_meta = {
        p.product_id: {
            "commodity": p.commodity_desc or f"commodity_{p.product_id}",
            "department": p.department or f"department_{p.product_id}",
        }
        for p in Product.objects.all()
    }

    CF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with CF_CACHE_PATH.open("wb") as f:
        pickle.dump(
            {"user_item": user_item, "similarity_df": similarity_df, "product_meta": product_meta},
            f,
        )
    return user_item.shape, len(product_meta)


_CF_CACHE_SINGLETON = {"data": None, "loaded": False}


def _load():
    if not _CF_CACHE_SINGLETON["loaded"]:
        if CF_CACHE_PATH.exists():
            with CF_CACHE_PATH.open("rb") as f:
                _CF_CACHE_SINGLETON["data"] = pickle.load(f)
        _CF_CACHE_SINGLETON["loaded"] = True
    return _CF_CACHE_SINGLETON["data"]


def get_cf_candidates(household_key, level="product", top_n=30):
    """
    Same output shape as dunnhumby.collab_filter.get_cf_recommendations:
    a list of {'product': Product instance, 'score': float, 'level': level}.
    Returns None (not []) if the cache file doesn't exist yet, so the
    caller can fall back to the slow live version just this once.
    """
    data = _load()
    if data is None:
        return None

    user_item = data["user_item"]
    similarity_df = data["similarity_df"]
    product_meta = data["product_meta"]

    if household_key not in similarity_df.index:
        return []

    similar = similarity_df[household_key].drop(household_key).sort_values(ascending=False)
    similar = similar[similar > 0].head(200)
    if similar.empty:
        return []

    row = user_item.loc[household_key]
    purchased_products = set(row[row > 0].index)

    if level == "product":
        item_key_of = lambda pid: pid
    elif level == "commodity":
        item_key_of = lambda pid: product_meta.get(pid, {}).get("commodity", f"commodity_{pid}")
    else:
        item_key_of = lambda pid: product_meta.get(pid, {}).get("department", f"department_{pid}")

    purchased_items_level = {item_key_of(pid) for pid in purchased_products}

    candidate_scores = {}
    for other_hh, sim in similar.items():
        other_row = user_item.loc[other_hh]
        bought = other_row[other_row > 0]
        for pid, cnt in bought.items():
            item_key = item_key_of(pid)
            if item_key in purchased_items_level:
                continue
            candidate_scores[item_key] = candidate_scores.get(item_key, 0.0) + sim * cnt

    if not candidate_scores:
        return []

    sorted_items = sorted(candidate_scores.items(), key=lambda kv: kv[1], reverse=True)[: top_n * 3]

    if level == "product":
        pids = [pid for pid, _ in sorted_items][:top_n]
        products = {p.product_id: p for p in Product.objects.filter(product_id__in=pids)}
        return [
            {"product": products[pid], "score": float(score), "level": "product"}
            for pid, score in sorted_items if pid in products
        ][:top_n]

    field = "commodity_desc" if level == "commodity" else "department"
    results = []
    for item_key, score in sorted_items:
        prod = Product.objects.filter(**{field: item_key}).first()
        if prod:
            results.append({"product": prod, "score": float(score), "level": level})
        if len(results) >= top_n:
            break
    return results