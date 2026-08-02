"""Leakage-safe product-level revenue forecasting for Predictive Basket Analysis.

The existing project classifiers predict customer repurchase.  This module keeps
those models intact and adds a separate, comparable revenue-forecasting task:

* one Product ID by one complete 30-day period is the analytical grain;
* ordered prior-period revenue is supplied through overlapping sliding windows;
* the final forecast horizon is a strictly out-of-time holdout;
* a one-period model is rolled forward recursively for multi-step forecasts; and
* revenue errors and Top-K product-ranking metrics are reported against the same
  independent recent-average baseline and the same evaluation population.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from django.db import connection
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


logger = logging.getLogger(__name__)
MODEL_DIR = Path(__file__).resolve().parent.parent / "ml_models_cache" / "time_series"
ARTIFACT_VERSION = 2
PERIOD_DAYS = 30
VALID_HORIZONS = {1, 3, 6, 12}
VALID_WINDOWS = {3, 6, 12}
VALID_STEPS = {1, 2, 3}
RANKING_CUTOFFS = (5, 10, 20)


class ProductRevenueTimeSeriesForecaster:
    """Global recursive forecaster trained from each product's revenue sequence."""

    def __init__(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _artifact_path(horizon: int, window_size: int, sliding_step: int) -> Path:
        return MODEL_DIR / (
            f"product_revenue_v{ARTIFACT_VERSION}_h{horizon}_w{window_size}_s{sliding_step}.pkl"
        )

    @staticmethod
    def _metrics_path(horizon: int, window_size: int, sliding_step: int) -> Path:
        return MODEL_DIR / (
            f"product_revenue_v{ARTIFACT_VERSION}_h{horizon}_w{window_size}_s{sliding_step}_metrics.json"
        )

    @staticmethod
    def _validate_configuration(horizon: int, window_size: int, sliding_step: int, training_size: float):
        if horizon not in VALID_HORIZONS:
            raise ValueError("horizon must be 1, 3, 6, or 12 months.")
        if window_size not in VALID_WINDOWS:
            raise ValueError("window_size must be 3, 6, or 12 months.")
        if sliding_step not in VALID_STEPS:
            raise ValueError("sliding_step must be 1, 2, or 3 months.")
        if not 0.5 <= training_size <= 0.95:
            raise ValueError("training_size must be between 0.50 and 0.95.")

    def load_product_panels(self):
        """Return equal-length revenue/unit panels ending on the dataset's last day.

        Anchoring backwards from MAX(day) avoids treating the final 21 days of the
        711-day dataset as a complete month.  Days 22-711 form 23 complete and
        directly comparable 30-day periods; days 1-21 are disclosed as excluded.
        """
        with connection.cursor() as cursor:
            cursor.execute("SELECT MIN(day), MAX(day), COUNT(*) FROM transactions")
            min_day, max_day, transaction_rows = cursor.fetchone()
        if min_day is None or max_day is None:
            raise ValueError("No transaction data is available for time-series forecasting.")

        complete_periods = int((max_day - min_day + 1) // PERIOD_DAYS)
        if complete_periods < 1:
            raise ValueError("The transaction history does not contain one complete 30-day period.")
        anchor_day = int(max_day - complete_periods * PERIOD_DAYS + 1)

        monthly_query = """
            SELECT t.product_id,
                   ((t.day - %s) / %s) + 1 AS period_no,
                   SUM(CAST(t.sales_value AS FLOAT)) AS revenue,
                   SUM(CASE WHEN t.quantity > 0 THEN CAST(t.quantity AS FLOAT) ELSE 0 END) AS units
            FROM transactions t
            WHERE t.product_id IS NOT NULL
              AND t.day BETWEEN %s AND %s
              AND t.sales_value IS NOT NULL
            GROUP BY t.product_id, ((t.day - %s) / %s) + 1
        """
        product_query = """
            SELECT product_id, department, commodity_desc, sub_commodity_desc,
                   manufacturer, brand, curr_size_of_product
            FROM product
        """
        with connection.cursor() as cursor:
            params = [anchor_day, PERIOD_DAYS, anchor_day, max_day, anchor_day, PERIOD_DAYS]
            cursor.execute(monthly_query, params)
            monthly = pd.DataFrame(
                cursor.fetchall(), columns=["product_id", "period_no", "revenue", "units"]
            )
            cursor.execute(product_query)
            products = pd.DataFrame(
                cursor.fetchall(),
                columns=[
                    "product_id", "department", "commodity", "sub_commodity",
                    "manufacturer", "brand", "size",
                ],
            )
        if monthly.empty:
            raise ValueError("No usable transaction rows remain after period alignment.")

        monthly = monthly.astype(
            {"product_id": int, "period_no": int, "revenue": float, "units": float}
        )
        products["product_id"] = products["product_id"].astype(int)
        periods = range(1, complete_periods + 1)

        def make_panel(value_column: str) -> pd.DataFrame:
            panel = (
                monthly.pivot_table(
                    index="product_id", columns="period_no", values=value_column, aggfunc="sum"
                )
                .reindex(columns=periods, fill_value=0.0)
                .fillna(0.0)
                .sort_index()
            )
            panel.columns = [int(period) for period in panel.columns]
            return panel

        revenue_panel = make_panel("revenue")
        unit_panel = make_panel("units").reindex(revenue_panel.index, fill_value=0.0)
        data_profile = {
            "source": "transactions joined to product metadata",
            "grain": "Product ID x complete 30-day period",
            "transaction_rows": int(transaction_rows),
            "products_with_transactions": int(len(revenue_panel)),
            "source_min_day": int(min_day),
            "as_of_day": int(max_day),
            "period_days": PERIOD_DAYS,
            "complete_periods": complete_periods,
            "analysis_start_day": anchor_day,
            "excluded_leading_days": int(anchor_day - min_day),
            "zero_revenue_cells_rate": round(float((revenue_panel.to_numpy() == 0).mean()), 6),
        }
        return revenue_panel, unit_panel, products.drop_duplicates("product_id"), data_profile

    @staticmethod
    def _feature_matrix(lag_values: np.ndarray, target_period_index: int) -> np.ndarray:
        """Create ordered lag and summary features using only observed/predicted history."""
        lag_values = np.maximum(np.asarray(lag_values, dtype=float), 0.0)
        recent_width = min(3, lag_values.shape[1])
        recent = lag_values[:, -recent_width:]
        trend = lag_values[:, -1] - lag_values[:, 0]
        nonzero_share = (lag_values > 0).mean(axis=1)
        summary = np.column_stack(
            (
                np.log1p(lag_values.sum(axis=1)),
                np.log1p(lag_values.mean(axis=1)),
                np.log1p(recent.mean(axis=1)),
                np.log1p(lag_values.std(axis=1)),
                np.sign(trend) * np.log1p(np.abs(trend)),
                nonzero_share,
                np.full(len(lag_values), np.sin(2 * np.pi * (target_period_index + 1) / 12)),
                np.full(len(lag_values), np.cos(2 * np.pi * (target_period_index + 1) / 12)),
            )
        )
        return np.hstack((np.log1p(lag_values), summary))

    @staticmethod
    def _one_step_baseline(lag_values: np.ndarray) -> np.ndarray:
        """Independent recent-average revenue baseline for the next period."""
        recent_width = min(3, lag_values.shape[1])
        return np.maximum(lag_values[:, -recent_width:].mean(axis=1), 0.0)

    @staticmethod
    def _training_origins(
        period_count: int,
        horizon: int,
        window_size: int,
        sliding_step: int,
        training_size: float,
    ):
        """Choose fitting origins strictly before the final-horizon holdout.

        The target of each one-step fitting sample is the period at its origin.
        Consequently every fitting target is earlier than ``test_origin`` and no
        train target can overlap the final ``horizon`` test periods.
        """
        test_origin = period_count - horizon
        if test_origin < window_size:
            raise ValueError(
                f"Not enough complete history: {period_count} periods cannot support "
                f"window={window_size} and a final {horizon}-period holdout."
            )
        candidates = list(range(window_size, test_origin, sliding_step))
        if candidates and candidates[-1] != test_origin - 1:
            candidates.append(test_origin - 1)
        if len(candidates) < 2:
            raise ValueError(
                "Not enough pre-test forecast origins for leakage-safe training; "
                "use a shorter window/horizon or a smaller sliding step."
            )
        keep = max(2, int(np.ceil(len(candidates) * training_size)))
        return candidates[-keep:], test_origin, candidates

    def _build_training_samples(self, panel: pd.DataFrame, origins: list[int], window_size: int):
        values = panel.to_numpy(dtype=float)
        features, residual_targets, sample_ids = [], [], []
        product_ids = panel.index.to_numpy(dtype=int)
        origin_counts = {}
        for origin in origins:
            lags = values[:, origin - window_size:origin]
            active = lags.sum(axis=1) > 0
            if not np.any(active):
                continue
            baseline = self._one_step_baseline(lags[active])
            actual = np.maximum(values[active, origin], 0.0)
            features.append(self._feature_matrix(lags[active], origin))
            residual_targets.append(np.log1p(actual) - np.log1p(baseline))
            sample_ids.append(product_ids[active])
            origin_counts[str(origin + 1)] = int(active.sum())
        if not features:
            raise ValueError("No active product histories are available for model fitting.")
        return (
            np.vstack(features),
            np.concatenate(residual_targets),
            np.concatenate(sample_ids),
            origin_counts,
        )

    def _recursive_predict(
        self, model, observed_history: np.ndarray, horizon: int, window_size: int, first_target_index: int
    ) -> np.ndarray:
        """Forecast one period at a time, feeding each prediction into the next step."""
        history = np.maximum(np.asarray(observed_history, dtype=float), 0.0).copy()
        predictions = []
        for step in range(horizon):
            lags = history[:, -window_size:]
            active = lags.sum(axis=1) > 0
            next_values = np.zeros(len(lags), dtype=float)
            if np.any(active):
                baseline = self._one_step_baseline(lags[active])
                correction = model.predict(
                    self._feature_matrix(lags[active], first_target_index + step)
                )
                next_values[active] = np.maximum(
                    np.expm1(np.log1p(baseline) + correction), 0.0
                )
            predictions.append(next_values)
            history = np.column_stack((history, next_values))
        return np.column_stack(predictions)

    @staticmethod
    def _stable_descending_order(product_ids: np.ndarray, values: np.ndarray) -> np.ndarray:
        return np.lexsort((np.asarray(product_ids), -np.asarray(values, dtype=float)))

    @classmethod
    def _ranking_metrics(cls, product_ids, actual, predicted, cutoffs=RANKING_CUTOFFS):
        product_ids = np.asarray(product_ids)
        actual = np.maximum(np.asarray(actual, dtype=float), 0.0)
        predicted = np.maximum(np.asarray(predicted, dtype=float), 0.0)
        if len(actual) < 2:
            return {
                "spearman": 0.0,
                "kendall_tau": 0.0,
                "ranking_at_k": {},
            }

        spearman = spearmanr(actual, predicted).statistic
        kendall = kendalltau(actual, predicted).statistic
        actual_order = cls._stable_descending_order(product_ids, actual)
        predicted_order = cls._stable_descending_order(product_ids, predicted)
        ranking_at_k = {}
        for requested_k in cutoffs:
            k = min(int(requested_k), len(actual))
            actual_top_ids = product_ids[actual_order[:k]]
            predicted_top_indices = predicted_order[:k]
            predicted_top_ids = product_ids[predicted_top_indices]
            actual_set, predicted_set = set(actual_top_ids), set(predicted_top_ids)
            overlap = len(actual_set & predicted_set)
            union = len(actual_set | predicted_set)
            discounts = np.log2(np.arange(2, k + 2))
            dcg = float((actual[predicted_top_indices] / discounts).sum())
            ideal_dcg = float((actual[actual_order[:k]] / discounts).sum())
            ranking_at_k[str(requested_k)] = {
                "k_used": k,
                "overlap_count": overlap,
                "precision_at_k": round(overlap / k, 6) if k else 0.0,
                "recall_at_k": round(overlap / k, 6) if k else 0.0,
                "jaccard_at_k": round(overlap / union, 6) if union else 0.0,
                "ndcg_at_k": round(dcg / ideal_dcg, 6) if ideal_dcg else 0.0,
            }
        return {
            "spearman": round(float(np.nan_to_num(spearman)), 6),
            "kendall_tau": round(float(np.nan_to_num(kendall)), 6),
            "ranking_at_k": ranking_at_k,
        }

    @classmethod
    def _evaluate(cls, product_ids, actual, predicted, primary_top_k=20):
        actual = np.maximum(np.asarray(actual, dtype=float), 0.0)
        predicted = np.maximum(np.asarray(predicted, dtype=float), 0.0)
        total_actual = float(actual.sum())
        absolute_error = np.abs(actual - predicted)
        denominator = np.abs(actual) + np.abs(predicted)
        ranking = cls._ranking_metrics(product_ids, actual, predicted)
        primary = ranking["ranking_at_k"].get(str(primary_top_k), {})
        metrics = {
            "evaluation_products": int(len(actual)),
            "actual_revenue": round(total_actual, 2),
            "predicted_revenue": round(float(predicted.sum()), 2),
            "mae": round(float(mean_absolute_error(actual, predicted)), 6),
            "rmse": round(float(mean_squared_error(actual, predicted) ** 0.5), 6),
            "wmape": round(float(absolute_error.sum() / total_actual), 6) if total_actual else 0.0,
            "smape": round(float(np.mean(np.divide(
                2 * absolute_error,
                denominator,
                out=np.zeros_like(absolute_error),
                where=denominator > 0,
            ))), 6),
            "r2": round(float(r2_score(actual, predicted)), 6) if len(actual) > 1 else 0.0,
            "top_k": int(primary_top_k),
            "spearman": ranking["spearman"],
            "kendall_tau": ranking["kendall_tau"],
            "ranking_at_k": ranking["ranking_at_k"],
        }
        metrics.update({key: primary.get(key, 0.0) for key in (
            "precision_at_k", "recall_at_k", "jaccard_at_k", "ndcg_at_k"
        )})
        return metrics

    @classmethod
    def _ranking_similarity(cls, product_ids, first, second):
        metrics = cls._ranking_metrics(product_ids, first, second)
        return {
            "spearman": metrics["spearman"],
            "kendall_tau": metrics["kendall_tau"],
            "top_k_overlap": metrics["ranking_at_k"],
            "interpretation": "Similarity compares the two model rankings; it is not accuracy against actual revenue.",
        }

    def train(self, horizon=3, window_size=6, sliding_step=1, training_size=0.8, top_k=20):
        """Fit the recursive model, run the strict final-horizon test, and persist it."""
        horizon, window_size, sliding_step = int(horizon), int(window_size), int(sliding_step)
        training_size = float(training_size)
        self._validate_configuration(horizon, window_size, sliding_step, training_size)

        revenue_panel, _, _, data_profile = self.load_product_panels()
        origins, test_origin, all_candidates = self._training_origins(
            revenue_panel.shape[1], horizon, window_size, sliding_step, training_size
        )
        X_train, residual_target, _, origin_counts = self._build_training_samples(
            revenue_panel, origins, window_size
        )

        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=220,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=42,
        )
        model.fit(X_train, residual_target)

        values = revenue_panel.to_numpy(dtype=float)
        observed = values[:, :test_origin]
        step_forecasts = self._recursive_predict(
            model, observed, horizon, window_size, test_origin
        )
        time_series_forecast = step_forecasts.sum(axis=1)
        actual = values[:, test_origin:test_origin + horizon].sum(axis=1)
        baseline = self._one_step_baseline(observed[:, -window_size:]) * horizon

        lookback_active = observed[:, -window_size:].sum(axis=1) > 0
        actual_active = actual > 0
        evaluation_mask = lookback_active | actual_active
        evaluation_ids = revenue_panel.index.to_numpy(dtype=int)[evaluation_mask]
        actual_eval = actual[evaluation_mask]
        time_series_eval = time_series_forecast[evaluation_mask]
        baseline_eval = baseline[evaluation_mask]

        model_metrics = self._evaluate(evaluation_ids, actual_eval, time_series_eval, top_k)
        baseline_metrics = self._evaluate(evaluation_ids, actual_eval, baseline_eval, top_k)
        ranking_similarity = self._ranking_similarity(
            evaluation_ids, time_series_eval, baseline_eval
        )
        lower_is_better = ("mae", "rmse", "wmape", "smape")
        model_error_wins = sum(
            model_metrics[key] < baseline_metrics[key] for key in lower_is_better
        )
        baseline_error_wins = sum(
            baseline_metrics[key] < model_metrics[key] for key in lower_is_better
        )
        if model_error_wins > baseline_error_wins:
            overall_error_assessment = "time_series_model"
        elif baseline_error_wins > model_error_wins:
            overall_error_assessment = "independent_recent_average_baseline"
        else:
            overall_error_assessment = "mixed_no_clear_winner"

        test_start_day = data_profile["analysis_start_day"] + test_origin * PERIOD_DAYS
        test_end_day = test_start_day + horizon * PERIOD_DAYS - 1
        report = {
            "artifact_version": ARTIFACT_VERSION,
            "forecast_method": "recursive_multi_step_monthly_revenue",
            "model": "HistGradientBoostingRegressor one-step residual model (scikit-learn)",
            "target": f"cumulative Product ID revenue for the next {horizon} complete 30-day period(s)",
            "horizon_months": horizon,
            "window_size_months": window_size,
            "sliding_step_months": sliding_step,
            "training_size": training_size,
            "validation": (
                "strict final-horizon chronological holdout; fitting targets end before the test starts; "
                "no random split and no future-revenue features"
            ),
            "fit_origin_periods": [origin + 1 for origin in origins],
            "available_pretest_origins": [origin + 1 for origin in all_candidates],
            "test_origin_period": test_origin + 1,
            "test_period": {
                "start_day": int(test_start_day),
                "end_day": int(test_end_day),
                "periods": list(range(test_origin + 1, test_origin + horizon + 1)),
            },
            "samples": {
                "train": int(len(X_train)),
                "train_by_origin": origin_counts,
                "products_total": int(len(revenue_panel)),
                "evaluation_products": int(evaluation_mask.sum()),
                "lookback_active_products": int(lookback_active.sum()),
                "cold_start_products_in_test": int((~lookback_active & actual_active).sum()),
            },
            "data_profile": data_profile,
            "time_series_model": model_metrics,
            "independent_recent_average_baseline": baseline_metrics,
            "model_ranking_similarity": ranking_similarity,
            "selection_guidance": {
                "best_for_revenue_error": (
                    "time_series_model"
                    if model_metrics["wmape"] < baseline_metrics["wmape"]
                    else "independent_recent_average_baseline"
                ),
                "best_for_revenue_error_basis": "WMAPE only",
                "overall_error_assessment": overall_error_assessment,
                "error_metric_wins": {
                    "time_series_model": int(model_error_wins),
                    "independent_recent_average_baseline": int(baseline_error_wins),
                    "metrics_compared": list(lower_is_better),
                },
                "best_for_top_k_ranking": (
                    "time_series_model"
                    if model_metrics["ndcg_at_k"] >= baseline_metrics["ndcg_at_k"]
                    else "independent_recent_average_baseline"
                ),
            },
            "required_caveats": [
                "The dataset provides 23 complete aligned periods, so the final holdout is one horizon block rather than many independent years.",
                "Products with no revenue in the lookback are cold starts; the revenue-only model predicts zero for them and reports their count.",
                "The legacy customer-repurchase classifier has a different target and its classification Accuracy must not be compared with these revenue metrics.",
                "Revenue per unit is historical sales_value divided by positive quantity; quantity units vary by product and are not documented as a universal physical unit.",
            ],
        }
        artifact = {
            "artifact_version": ARTIFACT_VERSION,
            "model": model,
            "horizon": horizon,
            "window_size": window_size,
            "sliding_step": sliding_step,
            "data_profile": data_profile,
            "report": report,
        }
        artifact_path = self._artifact_path(horizon, window_size, sliding_step)
        metrics_path = self._metrics_path(horizon, window_size, sliding_step)
        with artifact_path.open("wb") as handle:
            pickle.dump(artifact, handle)
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        return report

    def get_report(self, horizon=3, window_size=6, sliding_step=1):
        path = self._metrics_path(int(horizon), int(window_size), int(sliding_step))
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def forecast(
        self,
        horizon=3,
        window_size=6,
        sliding_step=1,
        top_n=20,
        revenue_threshold=None,
    ):
        """Forecast beyond the data as-of day and return ranked products and totals."""
        horizon, window_size, sliding_step = int(horizon), int(window_size), int(sliding_step)
        self._validate_configuration(horizon, window_size, sliding_step, 0.8)
        path = self._artifact_path(horizon, window_size, sliding_step)
        if not path.exists():
            raise ValueError("No trained time-series model for this horizon/window/step. Train it first.")
        with path.open("rb") as handle:
            artifact = pickle.load(handle)
        if artifact.get("artifact_version") != ARTIFACT_VERSION:
            raise ValueError("The saved model is obsolete. Retrain it with the corrected validation pipeline.")

        revenue_panel, unit_panel, products, data_profile = self.load_product_panels()
        trained_profile = artifact.get("data_profile", {})
        if (
            trained_profile.get("as_of_day") != data_profile["as_of_day"]
            or trained_profile.get("complete_periods") != data_profile["complete_periods"]
        ):
            raise ValueError("Transaction data changed after training. Retrain before forecasting.")
        if revenue_panel.shape[1] < window_size:
            raise ValueError("Not enough complete history for the requested window.")

        observed = revenue_panel.to_numpy(dtype=float)
        step_forecasts = self._recursive_predict(
            artifact["model"], observed, horizon, window_size, observed.shape[1]
        )
        predicted_revenue = step_forecasts.sum(axis=1)
        latest_lags = observed[:, -window_size:]
        baseline_revenue = self._one_step_baseline(latest_lags) * horizon
        eligible = latest_lags.sum(axis=1) > 0

        recent_units = unit_panel.iloc[:, -window_size:].sum(axis=1).to_numpy(dtype=float)
        recent_revenue = revenue_panel.iloc[:, -window_size:].sum(axis=1).to_numpy(dtype=float)
        revenue_per_unit = np.divide(
            recent_revenue,
            recent_units,
            out=np.full_like(recent_revenue, np.nan),
            where=recent_units > 0,
        )
        predicted_units = np.divide(
            predicted_revenue,
            revenue_per_unit,
            out=np.full_like(predicted_revenue, np.nan),
            where=np.isfinite(revenue_per_unit) & (revenue_per_unit > 0),
        )

        result = pd.DataFrame(
            {
                "product_id": revenue_panel.index.astype(int),
                "predicted_revenue": predicted_revenue,
                "baseline_revenue": baseline_revenue,
                "revenue_per_unit": revenue_per_unit,
                "predicted_units": predicted_units,
                "eligible": eligible,
            }
        ).merge(products, on="product_id", how="left")
        result = result[result["eligible"]].copy()
        if revenue_threshold is not None:
            threshold = float(revenue_threshold)
            if threshold < 0:
                raise ValueError("revenue_threshold cannot be negative.")
            result = result[result["predicted_revenue"] >= threshold]

        result["department"] = result["department"].fillna("Unknown")
        department_forecast = (
            result.groupby("department", dropna=False)["predicted_revenue"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        result = result.sort_values(
            ["predicted_revenue", "product_id"], ascending=[False, True]
        )
        top_n = max(1, min(int(top_n), 500))
        top_result = result.head(top_n).copy()
        top_result["forecast_rank"] = np.arange(1, len(top_result) + 1)
        top_result["revenue_change_vs_baseline"] = (
            top_result["predicted_revenue"] - top_result["baseline_revenue"]
        )
        top_result = top_result.fillna(
            {
                "department": "Unknown",
                "commodity": "Unknown",
                "sub_commodity": "Unknown",
                "brand": "Unknown",
                "size": "N/A",
            }
        )

        def json_scalar(value, default=None):
            if pd.isna(value):
                return default
            if isinstance(value, (np.integer, int)):
                return int(value)
            if isinstance(value, (np.floating, float)):
                return float(value)
            return str(value)

        predictions = []
        for _, row in top_result.iterrows():
            predictions.append(
                {
                    "product_id": int(row.product_id),
                    "forecast_rank": int(row.forecast_rank),
                    "department": str(row.department),
                    "commodity": str(row.commodity),
                    "sub_commodity": str(row.sub_commodity),
                    "manufacturer": json_scalar(row.manufacturer, "Unknown"),
                    "brand": str(row.brand),
                    "size": str(row["size"]),
                    "predicted_revenue": round(float(row.predicted_revenue), 2),
                    "baseline_revenue": round(float(row.baseline_revenue), 2),
                    "revenue_change_vs_baseline": round(float(row.revenue_change_vs_baseline), 2),
                    "revenue_per_unit": (
                        round(float(row.revenue_per_unit), 4)
                        if pd.notna(row.revenue_per_unit) else None
                    ),
                    "predicted_units": (
                        round(float(row.predicted_units), 2)
                        if pd.notna(row.predicted_units) else None
                    ),
                    "monthly_predictions": [round(float(value), 2) for value in step_forecasts[
                        revenue_panel.index.get_loc(int(row.product_id))
                    ]],
                }
            )

        forecast_start_day = data_profile["as_of_day"] + 1
        return {
            "horizon_months": horizon,
            "window_size_months": window_size,
            "sliding_step_months": sliding_step,
            "forecast_method": "recursive_multi_step_monthly_revenue",
            "forecast_period": {
                "start_day": int(forecast_start_day),
                "end_day": int(forecast_start_day + horizon * PERIOD_DAYS - 1),
            },
            "eligible_products": int(eligible.sum()),
            "products_after_threshold": int(len(result)),
            "model_report": artifact["report"],
            "department_forecast": [
                {
                    "department": str(row.department),
                    "predicted_revenue": round(float(row.predicted_revenue), 2),
                }
                for _, row in department_forecast.iterrows()
            ],
            "predictions": predictions,
        }


product_revenue_forecaster = ProductRevenueTimeSeriesForecaster()
