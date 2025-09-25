# dunnhumby/collab_filter.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from django.db import connection
from customers.models import Product

def get_cf_recommendations(household_key, top_n=10, min_overlap=2):
    """
    Collaborative Filtering recommendations for a given household
    """

    # --- ۱. بارگذاری داده‌ها ---
    query = """
        SELECT household_key, product_id, COUNT(*) as purchase_count
        FROM transactions
        GROUP BY household_key, product_id
    """
    with connection.cursor() as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()

    df = pd.DataFrame(rows, columns=["household_key", "product_id", "purchase_count"])

    if df.empty:
        return []

    # --- ۲. ساخت ماتریس User-Item ---
    user_item_matrix = df.pivot_table(
        index="household_key",
        columns="product_id",
        values="purchase_count",
        fill_value=0
    )

    if household_key not in user_item_matrix.index:
        return []

    # --- ۳. محاسبه شباهت مشتری‌ها ---
    similarity_matrix = cosine_similarity(user_item_matrix)
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )

    # مشتری‌های شبیه
    similar_customers = similarity_df[household_key].drop(household_key).sort_values(ascending=False)

    if similar_customers.empty:
        return []

    # --- ۴. محصولات پیشنهادی ---
    purchased_products = set(user_item_matrix.loc[household_key][user_item_matrix.loc[household_key] > 0].index)

    candidate_products = {}
    for other_cust, similarity in similar_customers.items():
        if similarity <= 0:
            continue

        other_purchases = user_item_matrix.loc[other_cust]
        for product_id, count in other_purchases.items():
            if count > 0 and product_id not in purchased_products:
                candidate_products[product_id] = candidate_products.get(product_id, 0) + similarity * count

    # مرتب‌سازی محصولات پیشنهادی
    sorted_candidates = sorted(candidate_products.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # گرفتن اطلاعات محصول
    recommendations = []
    for product_id, score in sorted_candidates:
        product = Product.objects.filter(product_id=product_id).first()
        if product:
            recommendations.append({
                "product": product,
                "score": round(score, 2)
            })

    return recommendations
