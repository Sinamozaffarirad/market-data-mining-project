"""Leakage-safe, time-windowed churn dataset generation and training."""
from dataclasses import dataclass
from enum import Enum
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

from .models import DunnhumbyProduct, Transaction


class WindowMethod(str, Enum):
    SLIDING = "sliding"
    NON_OVERLAPPING = "non_overlapping"


@dataclass(frozen=True)
class ChurnWindowConfig:
    method: WindowMethod
    observation_window_days: int
    prediction_horizon_days: int
    sliding_step_days: int | None = None

    def step_size(self) -> int:
        if self.method == WindowMethod.NON_OVERLAPPING:
            return self.observation_window_days + self.prediction_horizon_days
        if not self.sliding_step_days or self.sliding_step_days <= 0:
            raise ValueError("A positive sliding step size is required.")
        return self.sliding_step_days


def generate_time_windows(minimum_day: int, maximum_day: int, config: ChurnWindowConfig):
    """Yield complete observation/label windows; the label never extends beyond the data."""
    start = int(minimum_day)
    while True:
        cutoff = start + config.observation_window_days - 1
        label_end = cutoff + config.prediction_horizon_days
        if label_end > maximum_day:
            return
        yield {
            "observation_start": start,
            "observation_end": cutoff,
            "cutoff_day": cutoff,
            "label_start": cutoff + 1,
            "label_end": label_end,
        }
        start += config.step_size()


def _score(series: pd.Series, ascending: bool) -> pd.Series:
    """Create stable 1-5 scores without qcut failures on tied retail data."""
    ranks = series.rank(method="average", pct=True, ascending=ascending)
    return np.clip(np.ceil(ranks * 5), 1, 5).astype(int)


def _segment(row: pd.Series) -> str:
    if row.r_score >= 4 and row.f_score >= 4 and row.m_score >= 4:
        return "Champions"
    if row.f_score >= 4 and row.m_score >= 3:
        return "Loyal Customers"
    if row.m_score >= 4:
        return "Big Spenders"
    if row.r_score >= 4 and row.f_score >= 3:
        return "Potential Loyalists"
    if row.r_score >= 4:
        return "New Customers"
    if row.r_score <= 2 and row.f_score >= 3:
        return "At Risk"
    return "Regular Customers"


def _load_transactions() -> pd.DataFrame:
    fields = ["household_key", "day", "basket_id", "product_id", "quantity", "sales_value", "retail_disc", "coupon_disc", "coupon_match_disc"]
    frame = pd.DataFrame.from_records(Transaction.objects.values(*fields))
    if frame.empty:
        return frame
    numeric = ["day", "basket_id", "product_id", "quantity", "sales_value", "retail_disc", "coupon_disc", "coupon_match_disc"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    products = pd.DataFrame.from_records(DunnhumbyProduct.objects.values("product_id", "department", "commodity_desc"))
    if not products.empty:
        frame = frame.merge(products, on="product_id", how="left")
    else:
        frame["department"] = "Unknown"
        frame["commodity_desc"] = "Unknown"
    return frame


def build_customer_features(observation: pd.DataFrame, cutoff_day: int) -> pd.DataFrame:
    """Build all features strictly from one observation window."""
    if observation.empty:
        return pd.DataFrame()
    observation = observation.copy()
    observation["discount_value"] = observation[["retail_disc", "coupon_disc", "coupon_match_disc"]].abs().sum(axis=1)
    base = observation.groupby("household_key").agg(
        last_purchase_day=("day", "max"),
        frequency=("basket_id", "nunique"),
        monetary=("sales_value", "sum"),
        transaction_count=("product_id", "size"),
        total_quantity=("quantity", "sum"),
        total_sales_value=("sales_value", "sum"),
        unique_products=("product_id", "nunique"),
        unique_departments=("department", "nunique"),
        unique_commodities=("commodity_desc", "nunique"),
        total_discount=("discount_value", "sum"),
    )
    basket_values = observation.groupby(["household_key", "basket_id"])["sales_value"].sum()
    base["average_basket_value"] = basket_values.groupby(level=0).mean()
    base["recency_days"] = cutoff_day - base.pop("last_purchase_day")
    base["discount_ratio"] = base["total_discount"] / base["total_sales_value"].abs().replace(0, np.nan)
    coupon_use = observation.loc[(observation["coupon_disc"] != 0) | (observation["coupon_match_disc"] != 0)].groupby("household_key")["basket_id"].nunique()
    base["coupon_usage_count"] = coupon_use
    base["coupon_usage_rate"] = base["coupon_usage_count"] / base["frequency"].replace(0, np.nan)

    purchase_days = observation[["household_key", "day"]].drop_duplicates().sort_values(["household_key", "day"])
    purchase_days["gap"] = purchase_days.groupby("household_key")["day"].diff()
    gaps = purchase_days.groupby("household_key")["gap"].agg(["mean", "max", "min"]).rename(columns={"mean": "average_purchase_gap", "max": "maximum_purchase_gap", "min": "minimum_purchase_gap"})
    base = base.join(gaps)
    base["days_since_previous_purchase"] = base["recency_days"]

    midpoint = observation["day"].min() + (cutoff_day - observation["day"].min() + 1) // 2
    old = observation[observation["day"] < midpoint].groupby("household_key").agg(old_frequency=("basket_id", "nunique"), old_monetary=("sales_value", "sum"), old_basket=("basket_id", "nunique"))
    recent = observation[observation["day"] >= midpoint].groupby("household_key").agg(recent_frequency=("basket_id", "nunique"), recent_monetary=("sales_value", "sum"), recent_basket=("basket_id", "nunique"))
    base = base.join(old).join(recent)
    base["frequency_change"] = base["recent_frequency"] - base["old_frequency"]
    base["monetary_change"] = base["recent_monetary"] - base["old_monetary"]
    base["frequency_change_ratio"] = base["frequency_change"] / base["old_frequency"].replace(0, np.nan)
    base["monetary_change_ratio"] = base["monetary_change"] / base["old_monetary"].replace(0, np.nan)
    base = base.drop(columns=["old_frequency", "old_monetary", "old_basket", "recent_frequency", "recent_monetary", "recent_basket"])
    base["r_score"] = _score(base["recency_days"], ascending=False)
    base["f_score"] = _score(base["frequency"], ascending=True)
    base["m_score"] = _score(base["monetary"], ascending=True)
    base["rfm_segment"] = base.apply(_segment, axis=1)
    return base.reset_index().replace([np.inf, -np.inf], np.nan).fillna(0)


def build_training_dataset(transactions: pd.DataFrame, config: ChurnWindowConfig) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame()
    transactions = transactions.sort_values("day").reset_index(drop=True)
    transaction_days = transactions["day"].to_numpy()
    rows = []
    for window in generate_time_windows(transactions.day.min(), transactions.day.max(), config):
        observation_left = np.searchsorted(transaction_days, window["observation_start"], side="left")
        observation_right = np.searchsorted(transaction_days, window["observation_end"], side="right")
        observed = transactions.iloc[observation_left:observation_right]
        features = build_customer_features(observed, window["cutoff_day"])
        if features.empty:
            continue
        label_left = np.searchsorted(transaction_days, window["label_start"], side="left")
        label_right = np.searchsorted(transaction_days, window["label_end"], side="right")
        future = transactions.iloc[label_left:label_right]
        returning = set(future.household_key.unique())
        features["is_churn"] = (~features.household_key.isin(returning)).astype(int)
        for key, value in window.items():
            features[key] = value
        features["window_method"] = config.method.value
        features["observation_window_days"] = config.observation_window_days
        features["prediction_horizon_days"] = config.prediction_horizon_days
        features["step_size_days"] = config.step_size()
        rows.append(features)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _metrics(y_true, probabilities) -> dict:
    predicted = (probabilities >= 0.5).astype(int)
    result = {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
    }
    result["roc_auc"] = float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) > 1 else None
    return result


def _new_model() -> XGBClassifier:
    return XGBClassifier(n_estimators=100, max_depth=6, learning_rate=.05, subsample=.8, colsample_bytree=.8, objective="binary:logistic", eval_metric="logloss", tree_method="hist", n_jobs=-1, random_state=42)


def train_and_score(config: ChurnWindowConfig):
    """Train, score current customers, and create leakage-safe walk-forward history."""
    started = perf_counter()
    transactions = _load_transactions()
    dataset = build_training_dataset(transactions, config)
    if dataset.empty or dataset.cutoff_day.nunique() < 3:
        raise ValueError("Not enough complete time windows to create train, validation, and test periods.")
    cutoffs = sorted(dataset.cutoff_day.unique())
    train_end, validation_end = int(len(cutoffs) * .70), int(len(cutoffs) * .85)
    train_end = max(1, train_end)
    validation_end = max(train_end + 1, validation_end)
    if validation_end >= len(cutoffs):
        validation_end = len(cutoffs) - 1
    train_cutoffs, validation_cutoffs, test_cutoffs = cutoffs[:train_end], cutoffs[train_end:validation_end], cutoffs[validation_end:]
    if not test_cutoffs:
        raise ValueError("Not enough cutoff dates for a time-based test split.")
    excluded = {"household_key", "is_churn", "observation_start", "observation_end", "cutoff_day", "label_start", "label_end", "window_method", "observation_window_days", "prediction_horizon_days", "step_size_days"}
    feature_columns = [column for column in dataset.columns if column not in excluded]
    encoded = pd.get_dummies(dataset[feature_columns], columns=["rfm_segment"], dtype=float)
    train_mask = dataset.cutoff_day.isin(train_cutoffs)
    test_mask = dataset.cutoff_day.isin(test_cutoffs)
    model = _new_model()
    fit_started = perf_counter()
    model.fit(encoded.loc[train_mask], dataset.loc[train_mask, "is_churn"])
    training_seconds = perf_counter() - fit_started
    prediction_started = perf_counter()
    test_probabilities = model.predict_proba(encoded.loc[test_mask])[:, 1]
    prediction_seconds = perf_counter() - prediction_started
    metrics = _metrics(dataset.loc[test_mask, "is_churn"], test_probabilities)

    # Blocked expanding-window scoring keeps every prediction leakage-safe while
    # limiting historical model fits to five instead of fitting once per cutoff.
    historical_rows = []
    prediction_cutoffs = np.asarray(cutoffs[3:])
    cutoff_blocks = np.array_split(prediction_cutoffs, min(5, len(prediction_cutoffs))) if len(prediction_cutoffs) else []
    for cutoff_block in cutoff_blocks:
        if len(cutoff_block) == 0:
            continue
        block_start = cutoff_block[0]
        history_mask = dataset.cutoff_day < block_start
        cutoff_mask = dataset.cutoff_day.isin(cutoff_block.tolist())
        if dataset.loc[history_mask, "is_churn"].nunique() < 2:
            continue
        fold_model = _new_model()
        fold_model.fit(encoded.loc[history_mask], dataset.loc[history_mask, "is_churn"])
        fold = dataset.loc[cutoff_mask, ["household_key", "cutoff_day", "observation_window_days"]].copy()
        fold["churn_probability"] = fold_model.predict_proba(encoded.loc[cutoff_mask])[:, 1]
        historical_rows.append(fold)
    historical_predictions = pd.concat(historical_rows, ignore_index=True) if historical_rows else pd.DataFrame(columns=["household_key", "cutoff_day", "observation_window_days", "churn_probability"])

    current_cutoff = int(transactions.day.max())
    current_observation = transactions[transactions.day >= current_cutoff - config.observation_window_days + 1]
    current_features = build_customer_features(current_observation, current_cutoff)
    current_encoded = pd.get_dummies(current_features[feature_columns], columns=["rfm_segment"], dtype=float).reindex(columns=encoded.columns, fill_value=0)
    # Retrain on every labelled historical sample before making the current forecast.
    production_model = _new_model()
    production_model.fit(encoded, dataset["is_churn"])
    current_features["churn_probability"] = production_model.predict_proba(current_encoded)[:, 1]
    current_features["cutoff_day"] = current_cutoff
    current_features["observation_window_days"] = config.observation_window_days
    metrics.update({
        "training_samples": int(train_mask.sum()), "validation_samples": int(dataset.cutoff_day.isin(validation_cutoffs).sum()),
        "test_samples": int(test_mask.sum()), "churn_rate": float(dataset.loc[test_mask, "is_churn"].mean()),
        "training_time_seconds": training_seconds, "prediction_time_seconds": prediction_seconds,
        "total_time_seconds": perf_counter() - started, "current_cutoff_day": current_cutoff,
    })
    snapshot_columns = ["household_key", "cutoff_day", "observation_window_days", "recency_days", "frequency", "monetary", "r_score", "f_score", "m_score", "rfm_segment", "is_churn"]
    historical_snapshots = dataset[snapshot_columns].copy()
    current_snapshots = current_features[[column for column in snapshot_columns if column != "is_churn"]].copy()
    return metrics, current_features[["household_key", "churn_probability"]], historical_snapshots, historical_predictions, current_snapshots
