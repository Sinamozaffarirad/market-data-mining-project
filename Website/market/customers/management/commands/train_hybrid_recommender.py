# customers/management/commands/train_hybrid_recommender.py
import random
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from django.core.management.base import BaseCommand
from django.db import connection
from django.db.models import Max

from customers.models import Transaction
from dunnhumby.models import AssociationRule
from customers.ml.feature_engineering import (
    compute_product_popularity,
    compute_commodity_repurchase_cycles,
    build_candidate_features,
)
from customers.ml.recommender_model import HybridRecommenderModel


def _precompute_cf_matrix(as_of_day):
    """Build the household x product purchase-count matrix and cosine
    similarity ONCE, reused for every household during training instead of
    being rebuilt from scratch per household (that was the slow part)."""
    query = "SELECT household_key, product_id, COUNT(*) as cnt FROM transactions"
    params = []
    if as_of_day is not None:
        query += " WHERE day <= %s"
        params.append(as_of_day)
    query += " GROUP BY household_key, product_id"

    with connection.cursor() as cursor:
        cursor.execute(query, params)
        rows = cursor.fetchall()

    df = pd.DataFrame(rows, columns=["household_key", "product_id", "cnt"])
    user_item = df.pivot_table(index="household_key", columns="product_id", values="cnt", fill_value=0)
    similarity = cosine_similarity(user_item)
    similarity_df = pd.DataFrame(similarity, index=user_item.index, columns=user_item.index)
    return user_item, similarity_df


def _cf_scores_from_precomputed(household_key, user_item, similarity_df, top_n=30):
    if household_key not in similarity_df.index:
        return {}
    similar = similarity_df[household_key].drop(household_key).sort_values(ascending=False)
    similar = similar[similar > 0].head(200)  # only look at the closest neighbors
    if similar.empty:
        return {}

    purchased = set(user_item.loc[household_key][user_item.loc[household_key] > 0].index)
    scores = {}
    for other_hh, sim in similar.items():
        other_row = user_item.loc[other_hh]
        bought = other_row[other_row > 0]
        for pid, cnt in bought.items():
            if pid in purchased:
                continue
            scores[pid] = scores.get(pid, 0.0) + sim * cnt
    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return dict(top)


def _get_raw_candidates(household_key, purchased_ids, user_item, similarity_df, rules_list, top_n=30):
    """Product-level association + CF candidates for one household."""
    assoc_scores = {}

    for rule in rules_list:
        antecedents = {str(a) for a in (rule.antecedent or [])}
        if antecedents.isdisjoint(purchased_ids):
            continue
        consequents = [c for c in (rule.consequent or []) if c]
        pids = [int(c) for c in consequents if str(c).isdigit()]
        for pid in pids:
            if str(pid) in purchased_ids:
                continue
            score = float(rule.confidence or 0) * float(rule.lift or 0)
            if pid not in assoc_scores or score > assoc_scores[pid]:
                assoc_scores[pid] = score

    cf_scores = _cf_scores_from_precomputed(household_key, user_item, similarity_df, top_n=top_n)

    return assoc_scores, cf_scores


class Command(BaseCommand):
    help = "Trains the Hybrid Recommender ML model (product-purchase-likelihood classifier)."

    def add_arguments(self, parser):
        parser.add_argument("--horizon-days", type=int, default=30,
                            help="Future window (days) that defines a positive label.")
        parser.add_argument("--sample-households", type=int, default=1500,
                            help="How many households to sample for training.")

    def handle(self, *args, **options):
        horizon_days = options["horizon_days"]
        sample_size = options["sample_households"]

        self.stdout.write(self.style.SUCCESS("🚀 Building Hybrid Recommender training data..."))

        max_day = Transaction.objects.aggregate(Max("day"))["day__max"]
        cutoff_day = max_day - horizon_days
        if cutoff_day <= 0:
            self.stdout.write(self.style.ERROR("Not enough history for this horizon."))
            return
        self.stdout.write(f"  - cutoff_day={cutoff_day}, future window=({cutoff_day}, {max_day}]")

        popularity_map = compute_product_popularity(as_of_day=cutoff_day)
        cycle_map = compute_commodity_repurchase_cycles(as_of_day=cutoff_day)

        self.stdout.write("  - precomputing collaborative-filtering similarity matrix (this is the heavy step, runs ONCE)...")
        user_item, similarity_df = _precompute_cf_matrix(as_of_day=cutoff_day)
        self.stdout.write(f"  - matrix ready: {user_item.shape[0]} households x {user_item.shape[1]} products")

        # بارگیری قوانین انجمنی از دیتابیس
        rules_list = list(AssociationRule.objects.all())

        all_household_keys = list(user_item.index)
        random.seed(42)
        sampled = random.sample(all_household_keys, min(sample_size, len(all_household_keys)))

        rows = []
        for i, hh in enumerate(sampled, 1):
            purchased_ids = {str(pid) for pid in user_item.loc[hh][user_item.loc[hh] > 0].index}
            if not purchased_ids:
                continue

            # ارسال rules_list به عنوان آرگومان پنجم
            assoc_scores, cf_scores = _get_raw_candidates(hh, purchased_ids, user_item, similarity_df, rules_list)
            candidate_ids = list(set(assoc_scores) | set(cf_scores))
            if not candidate_ids:
                continue

            features = build_candidate_features(
                hh, candidate_ids, cutoff_day, popularity_map, cycle_map,
                assoc_scores=assoc_scores, cf_scores=cf_scores,
            )

            future_ids = set(
                Transaction.objects.filter(
                    household_key=hh, day__gt=cutoff_day, day__lte=max_day
                ).values_list("product_id", flat=True)
            )
            features["label"] = [1 if pid in future_ids else 0 for pid in features.index]
            rows.append(features.reset_index())

            if i % 200 == 0:
                self.stdout.write(f"  - processed {i}/{len(sampled)} households")

        if not rows:
            self.stdout.write(self.style.ERROR("No training rows produced."))
            return

        full_df = pd.concat(rows, ignore_index=True)
        self.stdout.write(
            f"  - total candidate rows: {len(full_df)} "
            f"(positive rate: {full_df['label'].mean():.3f})"
        )

        full_df = full_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        split_idx = int(len(full_df) * 0.8)
        train_df, test_df = full_df.iloc[:split_idx], full_df.iloc[split_idx:]

        model = HybridRecommenderModel()
        model.popularity_map = popularity_map
        model.cycle_map = cycle_map
        metrics = model.train(train_df, test_df)
        model.save()

        self.stdout.write(self.style.SUCCESS(f"✅ Trained. Metrics: {metrics}"))