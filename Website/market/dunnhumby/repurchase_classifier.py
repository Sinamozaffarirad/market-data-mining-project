"""Leakage-safe household-department repurchase classification.

The prediction grain is one household and one department at an as-of day.  Every
feature is aggregated from transactions on or before that day; the binary target
is whether the same household buys from the same department in the subsequent
forecast window.  Chronological validation is purged so no training target
overlaps the final test snapshot.
"""
from __future__ import annotations

import json
import logging
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from django.db import connection
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.svm import SVC


logger = logging.getLogger(__name__)
MODEL_DIR = Path(__file__).resolve().parent.parent / "ml_models_cache"
ARTIFACT_VERSION = 4
# A horizon needs two non-overlapping outcome windows inside the 711-day
# calendar: one to train on and a later one to test against.  Twelve months
# cannot supply that - 711 days holds only one 360-day outcome - so it was
# always reported as "validation unavailable" and has been dropped rather than
# offered as a setting that cannot produce an honest metric.  Ten months leaves
# a single training origin, which is too thin to fit on, so the grid stops at
# nine and the origin count is reported alongside every result.
VALID_HORIZON_MONTHS = (1, 2, 3, 4, 5, 6, 7, 8, 9)


def horizon_key(months):
    """Cache key for a horizon; "1month" is kept singular for compatibility."""
    months = int(months)
    return "1month" if months == 1 else f"{months}months"


HORIZON_DAYS = {horizon_key(m): m * 30 for m in VALID_HORIZON_MONTHS}
HORIZON_MONTHS = {horizon_key(m): m for m in VALID_HORIZON_MONTHS}
REQUIRED_MODEL_NAMES = ("neural_network", "random_forest", "gradient_boost", "svm")
MIN_HISTORY_DAYS = 90
SNAPSHOT_STEP_DAYS = 30

CATEGORICAL_FEATURES = [
    "department",
    "age_desc",
    "income_desc",
    "household_size_desc",
    "kid_category_desc",
]
NUMERIC_FEATURES = [
    "origin_day",
    "recency_days",
    "tenure_days",
    "transaction_count",
    "basket_count",
    "unique_products",
    "total_sales",
    "avg_sales",
    "total_quantity",
    "avg_quantity",
    "shopping_days",
    "customer_transactions",
    "customer_departments",
    "customer_total_sales",
    "department_households",
    "department_transactions",
    "department_total_sales",
]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


class PredictiveMarketBasketAnalyzer:
    """Train and serve household-department repurchase probability models."""

    def __init__(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.models = {horizon: {} for horizon in HORIZON_DAYS}
        self.model_metrics = {}
        self._load_cached_models()

    @staticmethod
    def _size_tag(training_size):
        return f"tr{int(round(float(training_size) * 100)):02d}"

    @classmethod
    def _model_path(cls, horizon, model_name, training_size):
        return MODEL_DIR / (
            f"repurchase_v{ARTIFACT_VERSION}_{horizon}_"
            f"{cls._size_tag(training_size)}_{model_name}.pkl"
        )

    @staticmethod
    def _metrics_path():
        return MODEL_DIR / f"repurchase_v{ARTIFACT_VERSION}_model_metrics.json"

    @classmethod
    def _metric_key(cls, horizon, model_name, training_size):
        """Metrics are keyed by training size too, so each size keeps its own."""
        return f"{horizon}_{cls._size_tag(training_size)}_{model_name}"

    def _load_cached_models(self):
        # Keyed by (horizon, size tag) so sizes do not overwrite one another.
        self.models = {}
        metrics_path = self._metrics_path()
        if metrics_path.exists():
            try:
                with metrics_path.open(encoding="utf-8") as handle:
                    self.model_metrics = json.load(handle)
            except (OSError, ValueError):
                logger.exception("Could not read corrected repurchase metrics cache")
                self.model_metrics = {}
        for path in MODEL_DIR.glob(f"repurchase_v{ARTIFACT_VERSION}_*_*_*.pkl"):
            try:
                with path.open("rb") as handle:
                    artifact = pickle.load(handle)
            except (OSError, ValueError, pickle.UnpicklingError):
                logger.exception("Could not load repurchase artifact %s", path.name)
                continue
            if artifact.get("artifact_version") != ARTIFACT_VERSION:
                continue
            metadata = artifact.get("metadata", {})
            horizon = metadata.get("horizon")
            model_name = metadata.get("model_name")
            size = metadata.get("training_size")
            if horizon and model_name and size is not None:
                bucket = self.models.setdefault((horizon, self._size_tag(size)), {})
                bucket[model_name] = artifact

    def refresh_cached_models(self):
        self._load_cached_models()

    def _is_horizon_ready(self, horizon, training_size=0.8):
        bucket = self.models.get((horizon, self._size_tag(training_size)), {})
        return (
            horizon in HORIZON_DAYS
            and all(name in bucket for name in REQUIRED_MODEL_NAMES)
            and all(
                self._metric_key(horizon, name, training_size) in self.model_metrics
                for name in REQUIRED_MODEL_NAMES
            )
        )

    def has_cached_models(self, horizons=None, refresh=False, training_size=0.8):
        if refresh:
            self._load_cached_models()
        requested = list(horizons or HORIZON_DAYS)
        return bool(requested) and all(
            self._is_horizon_ready(horizon, training_size) for horizon in requested
        )

    @staticmethod
    def _dataset_bounds():
        with connection.cursor() as cursor:
            cursor.execute("SELECT MIN(day), MAX(day), COUNT(*) FROM transactions")
            minimum, maximum, rows = cursor.fetchone()
        if minimum is None or maximum is None:
            raise ValueError("No transaction history is available.")
        return int(minimum), int(maximum), int(rows)

    @staticmethod
    def _candidate_origins(max_day, horizon_days):
        maximum_labelled_origin = max_day - horizon_days
        if maximum_labelled_origin < MIN_HISTORY_DAYS:
            return []
        origins = list(range(MIN_HISTORY_DAYS, maximum_labelled_origin + 1, SNAPSHOT_STEP_DAYS))
        if origins and origins[-1] != maximum_labelled_origin:
            origins.append(maximum_labelled_origin)
        return origins

    @classmethod
    def _split_origins(cls, max_day, horizon_days, training_size):
        candidates = cls._candidate_origins(max_day, horizon_days)
        if not candidates:
            return [], None, [], "unavailable_insufficient_history"
        test_origin = candidates[-1]
        purged_train = [origin for origin in candidates[:-1] if origin + horizon_days <= test_origin]
        if not purged_train:
            return [], None, candidates, "unavailable_insufficient_independent_windows"
        keep = max(1, int(math.ceil(len(purged_train) * float(training_size))))
        return purged_train[-keep:], test_origin, candidates, "chronological_holdout"

    @staticmethod
    def _snapshot_query(include_target, limit):
        top = f"TOP {int(limit)}" if limit else ""
        future_cte = """
            , future AS (
                SELECT t.household_key, p.department
                FROM transactions t
                JOIN product p ON p.product_id = t.product_id
                WHERE t.day BETWEEN %s AND %s AND p.department IS NOT NULL
                GROUP BY t.household_key, p.department
            )
        """ if include_target else ""
        future_join = (
            "LEFT JOIN future f ON f.household_key = hist.household_key AND f.department = hist.department"
            if include_target else ""
        )
        target_sql = "CASE WHEN f.household_key IS NULL THEN 0 ELSE 1 END" if include_target else "CAST(NULL AS INT)"
        return f"""
            ;WITH hist AS (
                SELECT t.household_key, p.department,
                       MAX(t.day) AS last_day, MIN(t.day) AS first_day,
                       COUNT_BIG(*) AS transaction_count,
                       COUNT(DISTINCT t.basket_id) AS basket_count,
                       COUNT(DISTINCT t.product_id) AS unique_products,
                       SUM(CAST(t.sales_value AS FLOAT)) AS total_sales,
                       AVG(CAST(t.sales_value AS FLOAT)) AS avg_sales,
                       SUM(CAST(t.quantity AS FLOAT)) AS total_quantity,
                       AVG(CAST(t.quantity AS FLOAT)) AS avg_quantity,
                       COUNT(DISTINCT t.day) AS shopping_days
                FROM transactions t
                JOIN product p ON p.product_id = t.product_id
                WHERE t.day <= %s AND p.department IS NOT NULL
                GROUP BY t.household_key, p.department
            )
            {future_cte}
            , customer_hist AS (
                SELECT household_key, SUM(transaction_count) AS customer_transactions,
                       COUNT(*) AS customer_departments, SUM(total_sales) AS customer_total_sales
                FROM hist GROUP BY household_key
            )
            , department_hist AS (
                SELECT department, COUNT(*) AS department_households,
                       SUM(transaction_count) AS department_transactions,
                       SUM(total_sales) AS department_total_sales
                FROM hist GROUP BY department
            )
            SELECT {top}
                   hist.household_key, hist.department,
                   h.age_desc, h.income_desc, h.household_size_desc, h.kid_category_desc,
                   %s AS origin_day, %s - hist.last_day AS recency_days,
                   %s - hist.first_day AS tenure_days,
                   hist.transaction_count, hist.basket_count, hist.unique_products,
                   hist.total_sales, hist.avg_sales, hist.total_quantity,
                   hist.avg_quantity, hist.shopping_days,
                   ch.customer_transactions, ch.customer_departments, ch.customer_total_sales,
                   dh.department_households, dh.department_transactions, dh.department_total_sales,
                   {target_sql} AS target
            FROM hist
            JOIN customer_hist ch ON ch.household_key = hist.household_key
            JOIN department_hist dh ON dh.department = hist.department
            {future_join}
            LEFT JOIN household h ON h.household_key = hist.household_key
            ORDER BY CHECKSUM(hist.household_key, hist.department)
        """

    def _load_snapshot(self, origin_day, horizon_days=None, limit=None):
        include_target = horizon_days is not None
        params = [int(origin_day)]
        if include_target:
            params.extend([int(origin_day) + 1, int(origin_day) + int(horizon_days)])
        params.extend([int(origin_day), int(origin_day), int(origin_day)])
        columns = [
            "household_key", "department", "age_desc", "income_desc",
            "household_size_desc", "kid_category_desc", "origin_day", "recency_days",
            "tenure_days", "transaction_count", "basket_count", "unique_products",
            "total_sales", "avg_sales", "total_quantity", "avg_quantity", "shopping_days",
            "customer_transactions", "customer_departments", "customer_total_sales",
            "department_households", "department_transactions", "department_total_sales", "target",
        ]
        with connection.cursor() as cursor:
            cursor.execute(self._snapshot_query(include_target, limit), params)
            frame = pd.DataFrame(cursor.fetchall(), columns=columns)
        if frame.empty:
            return frame
        for column in CATEGORICAL_FEATURES:
            frame[column] = frame[column].fillna("Unknown").astype(str)
        for column in NUMERIC_FEATURES:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if include_target:
            frame["target"] = frame["target"].astype(int)
        return frame

    def _load_training_frames(self, horizon, training_size, sample_size):
        horizon_days = HORIZON_DAYS[horizon]
        min_day, max_day, transaction_rows = self._dataset_bounds()
        train_origins, test_origin, candidates, validation_status = self._split_origins(
            max_day, horizon_days, training_size
        )
        if validation_status == "chronological_holdout":
            fit_origins = train_origins
        else:
            keep = max(1, int(math.ceil(len(candidates) * float(training_size))))
            fit_origins = candidates[-keep:]
        if not fit_origins:
            raise ValueError(f"Not enough labelled history to train the {horizon} classifier.")
        per_origin = max(250, int(math.ceil(sample_size / len(fit_origins))))
        fit_frames = [self._load_snapshot(origin, horizon_days, per_origin) for origin in fit_origins]
        fit_frame = pd.concat(fit_frames, ignore_index=True)
        test_frame = (
            self._load_snapshot(test_origin, horizon_days, None)
            if validation_status == "chronological_holdout" else pd.DataFrame()
        )
        metadata = {
            "artifact_version": ARTIFACT_VERSION,
            "prediction_unit": "household x previously purchased department x as-of day",
            "target": f"repurchase from the same department within the next {HORIZON_MONTHS[horizon]} month(s)",
            "output": "probability of department repurchase; not revenue",
            "feature_policy": "all aggregates use transactions on or before each origin day",
            "validation": "purged chronological holdout; every fit target ends on or before the test origin",
            "validation_status": validation_status,
            "horizon": horizon,
            "horizon_months": HORIZON_MONTHS[horizon],
            "horizon_days": horizon_days,
            "dataset_min_day": min_day,
            "dataset_max_day": max_day,
            "transaction_rows": transaction_rows,
            "fit_origins": fit_origins,
            "test_origin": test_origin,
            "candidate_origins": candidates,
            "fit_rows": int(len(fit_frame)),
            "test_rows": int(len(test_frame)),
            "training_size": float(training_size),
            "feature_names": MODEL_FEATURES,
        }
        return fit_frame, test_frame, metadata

    @staticmethod
    def _preprocessor():
        categorical = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        numerical = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        return ColumnTransformer([
            ("categorical", categorical, CATEGORICAL_FEATURES),
            ("numerical", numerical, NUMERIC_FEATURES),
        ])

    @staticmethod
    def _classifier(model_name):
        if model_name == "neural_network":
            return MLPClassifier(
                hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=180,
                learning_rate_init=0.001, random_state=42,
            )
        if model_name == "random_forest":
            return RandomForestClassifier(
                n_estimators=140, max_depth=15, min_samples_leaf=3,
                class_weight="balanced_subsample", n_jobs=-1, random_state=42,
            )
        if model_name == "gradient_boost":
            return GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.05, max_depth=3,
                min_samples_leaf=10, random_state=42,
            )
        if model_name == "svm":
            return SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
        raise ValueError(f"Unsupported model: {model_name}")

    @classmethod
    def _pipeline(cls, model_name):
        return Pipeline([
            ("preprocess", cls._preprocessor()),
            ("classifier", cls._classifier(model_name)),
        ])

    @staticmethod
    def _deterministic_svm_sample(frame, maximum=5000):
        if len(frame) <= maximum:
            return frame
        parts = []
        for target, group in frame.groupby("target"):
            take = max(1, int(round(maximum * len(group) / len(frame))))
            parts.append(group.sample(n=min(take, len(group)), random_state=42 + int(target)))
        return pd.concat(parts).sort_index().head(maximum)

    @staticmethod
    def _metrics(y_true, probabilities, metadata):
        probabilities = np.asarray(probabilities, dtype=float)
        predictions = (probabilities >= 0.5).astype(int)
        y_true = np.asarray(y_true, dtype=int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        both_classes = len(np.unique(y_true)) == 2
        return {
            **metadata,
            "accuracy": round(float(accuracy_score(y_true, predictions)), 6),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_true, predictions)), 6),
            "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 6),
            "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 6),
            "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 6),
            "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 6) if both_classes else None,
            "pr_auc": round(float(average_precision_score(y_true, probabilities)), 6) if both_classes else None,
            "brier_score": round(float(brier_score_loss(y_true, probabilities)), 6),
            "positive_rate": round(float(y_true.mean()), 6),
            "mean_predicted_probability": round(float(probabilities.mean()), 6),
            "calibration_gap": round(float(probabilities.mean() - y_true.mean()), 6),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }

    @staticmethod
    def _unavailable_metrics(metadata, positive_rate):
        return {
            **metadata,
            "accuracy": None, "balanced_accuracy": None, "precision": None,
            "recall": None, "f1": None, "roc_auc": None, "pr_auc": None,
            "brier_score": None, "positive_rate": round(float(positive_rate), 6),
            "mean_predicted_probability": None, "calibration_gap": None,
            "confusion_matrix": None,
            "validation_note": (
                "The 711-day dataset cannot contain two non-overlapping 360-day outcomes. "
                "The model is trained for scoring, but no leakage-free holdout metric is claimed."
            ),
        }

    def train_models(self, training_size=0.8, time_horizon=None, force_retrain=False, sample_size=60000):
        training_size = max(0.5, min(float(training_size), 0.95))
        horizons = [time_horizon] if time_horizon else list(HORIZON_DAYS)
        if any(horizon not in HORIZON_DAYS for horizon in horizons):
            raise ValueError(
                "time_horizon must be one of " + ", ".join(HORIZON_DAYS) + "."
            )
        for horizon in horizons:
            if not force_retrain and self._is_horizon_ready(horizon, training_size):
                continue
            logger.info("Building leakage-safe repurchase snapshots for %s", horizon)
            fit_frame, test_frame, metadata = self._load_training_frames(
                horizon, training_size, int(sample_size)
            )
            if fit_frame["target"].nunique() < 2:
                raise ValueError(f"The {horizon} training data contains only one target class.")
            for model_name in REQUIRED_MODEL_NAMES:
                model_frame = (
                    self._deterministic_svm_sample(fit_frame)
                    if model_name == "svm" else fit_frame
                )
                pipeline = self._pipeline(model_name)
                logger.info("Training corrected %s %s on %s rows", horizon, model_name, len(model_frame))
                pipeline.fit(model_frame[MODEL_FEATURES], model_frame["target"])
                metric_metadata = {
                    **metadata,
                    "model_name": model_name,
                    "estimator": pipeline.named_steps["classifier"].__class__.__name__,
                    "model_fit_rows": int(len(model_frame)),
                }
                if metadata["validation_status"] == "chronological_holdout":
                    probabilities = pipeline.predict_proba(test_frame[MODEL_FEATURES])[:, 1]
                    metrics = self._metrics(test_frame["target"], probabilities, metric_metadata)
                else:
                    metrics = self._unavailable_metrics(metric_metadata, fit_frame["target"].mean())
                artifact = {
                    "artifact_version": ARTIFACT_VERSION,
                    "pipeline": pipeline,
                    "metadata": metric_metadata,
                }
                path = self._model_path(horizon, model_name, training_size)
                with path.open("wb") as handle:
                    pickle.dump(artifact, handle)
                bucket = self.models.setdefault(
                    (horizon, self._size_tag(training_size)), {}
                )
                bucket[model_name] = artifact
                self.model_metrics[
                    self._metric_key(horizon, model_name, training_size)
                ] = metrics
            self._persist_metrics()
        return True

    def _persist_metrics(self):
        """Merge this run's metrics into the file rather than replacing it.

        The metrics for every horizon, training size and algorithm share one
        JSON file. Writing the in-memory dict wholesale meant a process holding
        a stale copy - the web server, say, while a training script ran - erased
        entries it had never seen. Re-reading immediately before the write keeps
        both sets.
        """
        metrics_path = self._metrics_path()
        merged = {}
        if metrics_path.exists():
            try:
                with metrics_path.open(encoding="utf-8") as handle:
                    merged = json.load(handle)
            except (OSError, ValueError):
                logger.exception("Could not read repurchase metrics before writing")
        merged.update(self.model_metrics)
        self.model_metrics = merged
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(merged, handle, indent=2)

    def get_model_performance(self):
        metrics_path = self._metrics_path()
        if metrics_path.exists():
            try:
                with metrics_path.open(encoding="utf-8") as handle:
                    self.model_metrics = json.load(handle)
            except (OSError, ValueError):
                logger.exception("Could not refresh repurchase metrics cache")
        return self.model_metrics

    def predict_future_purchases(self, model_name="neural_network", time_horizon=3,
                                 top_n=10, training_size=0.8):
        horizon = horizon_key(time_horizon)
        if horizon not in HORIZON_DAYS:
            raise ValueError(
                "time_horizon must be one of " + ", ".join(str(m) for m in VALID_HORIZON_MONTHS) + " months."
            )
        if model_name not in REQUIRED_MODEL_NAMES:
            raise ValueError("Unknown classifier.")
        key = (horizon, self._size_tag(training_size))
        if model_name not in self.models.get(key, {}):
            self._load_cached_models()
        artifact = self.models.get(key, {}).get(model_name)
        if not artifact:
            raise ValueError(
                "No cached model exists for this horizon, algorithm and training size. Train it first."
            )
        _, max_day, _ = self._dataset_bounds()
        snapshot = self._load_snapshot(max_day, None, None)
        probabilities = artifact["pipeline"].predict_proba(snapshot[MODEL_FEATURES])[:, 1]
        scored = snapshot[["household_key", "department"]].copy()
        scored["repurchase_probability"] = probabilities
        grouped = scored.groupby("department", dropna=False)["repurchase_probability"].agg(
            average_probability="mean",
            median_probability="median",
            expected_repurchase_households="sum",
            households_scored="size",
        )
        grouped["high_likelihood_households"] = scored.assign(
            high=(scored["repurchase_probability"] >= 0.5).astype(int)
        ).groupby("department")["high"].sum()
        grouped = grouped.reset_index().sort_values(
            ["expected_repurchase_households", "department"], ascending=[False, True]
        )
        metrics = self.model_metrics.get(f"{horizon}_{model_name}", {})
        requested_top_n = int(top_n)
        selected_departments = grouped if requested_top_n <= 0 else grouped.head(min(requested_top_n, 100))
        result = []
        for _, row in selected_departments.iterrows():
            result.append({
                "department": str(row.department),
                "average_repurchase_probability": round(float(row.average_probability), 6),
                "median_repurchase_probability": round(float(row.median_probability), 6),
                "expected_repurchase_households": round(float(row.expected_repurchase_households), 2),
                "high_likelihood_households": int(row.high_likelihood_households),
                "households_scored": int(row.households_scored),
                "as_of_day": max_day,
                "time_horizon_months": int(time_horizon),
                "model_used": model_name,
                "output_type": "household-department repurchase probability",
                "validation_status": metrics.get("validation_status"),
            })
        return result

    def get_department_predictions(self, model_name, time_horizon=3):
        return self.predict_future_purchases(model_name, time_horizon, top_n=10)

    def predict_customer_preferences(self, model_name, customer_id=None, top_n=10, time_horizon=3):
        """Deprecated: the corrected classifier has no defensible Product-ID output."""
        return []
