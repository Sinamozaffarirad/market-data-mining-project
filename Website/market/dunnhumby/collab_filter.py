# dunnhumby/collab_filter.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from django.db import connection
from customers.models import Product

def _get_product_meta_map():
    """
    بازگرداندن mapping از product_id -> (commodity_desc, department)
    """
    metas = Product.objects.values_list('product_id', 'commodity_desc', 'department')
    prod_to_meta = {}
    for pid, comm, dept in metas:
        prod_to_meta[str(pid)] = {
            'commodity': comm or f"commodity_{pid}",
            'department': dept or f"department_{pid}"
        }
    return prod_to_meta

def get_cf_recommendations(household_key, top_n=10, level='product', sample_limit=None):
    """
    Collaborative Filtering که از سه سطح پشتیبانی می‌کند:
    level in {'product', 'commodity', 'department'}
    خروجی: لیستی از دیکشنری‌ها: { 'product': Product instance, 'score': float, 'level': level }
    """
    # 1. load aggregated purchase counts
    query = "SELECT household_key, product_id, COUNT(*) as purchase_count FROM transactions GROUP BY household_key, product_id"
    with connection.cursor() as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()

    df = pd.DataFrame(rows, columns=["household_key", "product_id", "purchase_count"])
    if df.empty:
        return []

    prod_meta = _get_product_meta_map()
    df['product_id'] = df['product_id'].astype(str)

    # map product to requested level
    if level == 'product':
        df['item_key'] = df['product_id']
    elif level == 'commodity':
        df['item_key'] = df['product_id'].map(lambda p: prod_meta.get(str(p), {}).get('commodity', f"commodity_{p}"))
    elif level == 'department':
        df['item_key'] = df['product_id'].map(lambda p: prod_meta.get(str(p), {}).get('department', f"department_{p}"))
    else:
        raise ValueError("level must be 'product', 'commodity' or 'department'")

    # build user-item matrix at requested granularity
    user_item = df.groupby(['household_key', 'item_key'])['purchase_count'].sum().reset_index()
    user_item_matrix = user_item.pivot_table(index='household_key', columns='item_key', values='purchase_count', fill_value=0)

    if household_key not in user_item_matrix.index:
        return []

    # similarity
    similarity_matrix = cosine_similarity(user_item_matrix)
    similarity_df = pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)
    similar_customers = similarity_df[household_key].drop(household_key).sort_values(ascending=False)
    if similar_customers.empty:
        return []

    # candidate scoring on item_key
    purchased_items = set(user_item_matrix.loc[household_key][user_item_matrix.loc[household_key] > 0].index)
    candidate_scores = {}
    for other, sim in similar_customers.items():
        if sim <= 0:
            continue
        other_row = user_item_matrix.loc[other]
        for item_key, cnt in other_row.items():
            if cnt > 0 and item_key not in purchased_items:
                candidate_scores[item_key] = candidate_scores.get(item_key, 0.0) + sim * cnt

    if not candidate_scores:
        return []

    # If level == product => directly map to Product objects
    recommendations = []
    if level == 'product':
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        for pid_str, score in sorted_candidates:
            try:
                pid = int(pid_str)
                product = Product.objects.filter(product_id=pid).first()
                if product:
                    recommendations.append({'product': product, 'score': float(score), 'level': 'product'})
            except:
                continue
    else:
        # For commodity/department: for each recommended item_key, pick top products in that category by popularity
        sorted_items = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        # get top products per item_key (popularity by total households)
        item_to_products = {}
        # Query popularity once
        pop_q = Product.objects.all().values('product_id', 'commodity_desc', 'department')
        pop_map = {}
        for p in pop_q:
            key = p['commodity_desc'] if level == 'commodity' else p['department']
            pop_map.setdefault(key, []).append(p['product_id'])

        for item_key, score in sorted_items:
            pids = pop_map.get(item_key, [])
            # fallback: if no products found, skip
            for pid in pids[:max(1, top_n)]:
                prod = Product.objects.filter(product_id=pid).first()
                if prod:
                    recommendations.append({'product': prod, 'score': float(score), 'level': level})
                    if len(recommendations) >= top_n:
                        break
            if len(recommendations) >= top_n:
                break

    return recommendations
