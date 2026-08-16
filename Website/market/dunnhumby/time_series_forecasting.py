"""Leakage-safe product-level revenue forecasting for Predictive Basket Analysis.

The existing project classifiers predict customer repurchase.  This module keeps
those models intact and adds a separate, comparable revenue-forecasting task:

* one Product ID by one complete 30-day period is the analytical grain;
* ordered prior-period revenue is supplied through overlapping sliding windows;
* the final forecast horizon is a strictly out-of-time holdout;
* an independent direct gradient-boosting model is preserved as a comparison;
* an encoder-decoder RNN recursively feeds each predicted period into the next;
* a joint multi-step loss is propagated through the complete supervised rollout;
* long-horizon forecasts are reconciled to a separately estimated aggregate
  revenue path; and
* revenue errors and Top-K product-ranking metrics are reported against the same
  independent recent-average baseline and the same evaluation population.
"""
from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from django.db import connection
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .autoregressive_rnn import AutoregressiveRevenueRNN


logger = logging.getLogger(__name__)
MODEL_DIR = Path(__file__).resolve().parent.parent / "ml_models_cache" / "time_series"
ARTIFACT_VERSION = 8
PERIOD_DAYS = 30
VALID_HORIZONS = {1, 3, 6, 12}
VALID_WINDOWS = {3, 6, 12}
VALID_STEPS = {1, 2, 3}
RANKING_CUTOFFS = (5, 10, 20)
AUTO_HIDDEN_UNITS = 16
AUTO_FEEDBACK_RATE = 0.5
AUTO_EPOCHS = 10
VALID_PARAMETER_MODES = {"auto", "custom"}
VALID_HIDDEN_UNITS = {8, 16, 32, 64}
VALID_EPOCHS = {5, 10, 15, 20, 30}


class ProductRevenueTimeSeriesForecaster:
    """Compare independent direct and recursive neural sequence forecasts."""

    def __init__(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _refresh_report_interpretation(report):
        """Refresh explanatory metadata when loading an otherwise valid artifact."""
        if report and report.get("bias_diagnostics") is not None:
            report["bias_diagnostics"]["interpretation"] = (
                "Negative values mean underprediction and positive values mean "
                "overprediction. Sparse products, cold starts, and log-target "
                "shrinkage can create negative aggregate bias; long recursive "
                "rollouts can also accumulate feedback error in either direction."
            )
        return report

    @staticmethod
    def _resolve_architecture(parameter_mode="auto", hidden_units=None, feedback_rate=None, epochs=None):
        mode = str(parameter_mode or "auto").lower()
        if mode not in VALID_PARAMETER_MODES:
            raise ValueError("parameter_mode must be auto or custom.")
        if mode == "auto":
            hidden, feedback, epoch_count = (
                AUTO_HIDDEN_UNITS, AUTO_FEEDBACK_RATE, AUTO_EPOCHS
            )
        else:
            try:
                hidden = int(hidden_units)
                feedback = float(feedback_rate)
                epoch_count = int(epochs)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Custom mode requires numeric hidden_units, feedback_rate, and epochs."
                ) from exc
            if hidden not in VALID_HIDDEN_UNITS:
                raise ValueError("hidden_units must be 8, 16, 32, or 64.")
            if not 0.1 <= feedback <= 1.0:
                raise ValueError("feedback_rate must be between 0.10 and 1.00.")
            if epoch_count not in VALID_EPOCHS:
                raise ValueError("epochs must be 5, 10, 15, 20, or 30.")
        return {
            "parameter_mode": mode,
            "hidden_units": hidden,
            "feedback_rate": round(feedback, 4),
            "epochs": epoch_count,
            "recursive_feedback": True,
            "joint_multi_step_loss": True,
            "backpropagation_through_time": True,
        }

    @staticmethod
    def _configuration_key(horizon, window_size, sliding_step, training_size, architecture):
        return (
            f"h{int(horizon)}_w{int(window_size)}_s{int(sliding_step)}"
            f"_tr{int(round(float(training_size) * 100)):02d}"
            f"_{architecture['parameter_mode']}_hu{architecture['hidden_units']}"
            f"_fb{int(round(architecture['feedback_rate'] * 100)):03d}"
            f"_e{architecture['epochs']}"
        )

    @classmethod
    def _artifact_path(
        cls, horizon, window_size, sliding_step, training_size=0.8,
        parameter_mode="auto", hidden_units=None, feedback_rate=None, epochs=None,
    ) -> Path:
        architecture = cls._resolve_architecture(
            parameter_mode, hidden_units, feedback_rate, epochs
        )
        key = cls._configuration_key(
            horizon, window_size, sliding_step, training_size, architecture
        )
        return MODEL_DIR / (
            f"product_revenue_v{ARTIFACT_VERSION}_{key}.pkl"
        )

    @classmethod
    def _metrics_path(
        cls, horizon, window_size, sliding_step, training_size=0.8,
        parameter_mode="auto", hidden_units=None, feedback_rate=None, epochs=None,
    ) -> Path:
        architecture = cls._resolve_architecture(
            parameter_mode, hidden_units, feedback_rate, epochs
        )
        key = cls._configuration_key(
            horizon, window_size, sliding_step, training_size, architecture
        )
        return MODEL_DIR / (
            f"product_revenue_v{ARTIFACT_VERSION}_{key}_metrics.json"
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
        """Create ordered lag and summary features using only observed history."""
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
        features, residual_targets, sample_ids, sample_actuals = [], [], [], []
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
            sample_actuals.append(actual)
            origin_counts[str(origin + 1)] = int(active.sum())
        if not features:
            raise ValueError("No active product histories are available for model fitting.")
        return (
            np.vstack(features),
            np.concatenate(residual_targets),
            np.concatenate(sample_ids),
            np.concatenate(sample_actuals),
            origin_counts,
        )

    @staticmethod
    def _build_sequence_samples(panel, origins, window_size, horizon, training_end):
        """Build masked multi-step sequences without crossing the training boundary."""
        values = panel.to_numpy(dtype=float)
        lag_batches, target_batches, mask_batches, period_batches = [], [], [], []
        origin_counts, supervised_by_origin = {}, {}
        for origin in origins:
            available_steps = min(int(horizon), int(training_end) - int(origin))
            if available_steps <= 0:
                continue
            lags = values[:, origin - window_size:origin]
            active = lags.sum(axis=1) > 0
            if not np.any(active):
                continue
            targets = np.zeros((int(active.sum()), int(horizon)), dtype=float)
            mask = np.zeros_like(targets, dtype=bool)
            targets[:, :available_steps] = values[
                active, origin:origin + available_steps
            ]
            mask[:, :available_steps] = True
            lag_batches.append(lags[active])
            target_batches.append(targets)
            mask_batches.append(mask)
            period_batches.append(np.full(int(active.sum()), origin, dtype=int))
            origin_counts[str(origin + 1)] = int(active.sum())
            supervised_by_origin[str(origin + 1)] = int(available_steps)
        if not lag_batches:
            raise ValueError("No active multi-step sequences are available for recurrent training.")
        return (
            np.vstack(lag_batches),
            np.vstack(target_batches),
            np.vstack(mask_batches),
            np.concatenate(period_batches),
            origin_counts,
            supervised_by_origin,
        )

    @classmethod
    def _fit_sequence_model(
        cls, lags, targets, target_mask, start_periods, architecture=None
    ):
        architecture = architecture or cls._resolve_architecture()
        model = AutoregressiveRevenueRNN(
            hidden_size=architecture["hidden_units"],
            epochs=architecture["epochs"],
            batch_size=8192,
            learning_rate=0.008,
            huber_delta=0.75,
            gradient_clip=5.0,
            l2=1e-5,
            feedback_rate=architecture["feedback_rate"],
            random_state=42,
        )
        return model.fit(lags, targets, target_mask, start_periods)

    @staticmethod
    def _production_origins(period_count, window_size, sliding_step, training_size):
        """Use every fully observed target when refitting the future-production model."""
        candidates = list(range(window_size, period_count, sliding_step))
        if candidates and candidates[-1] != period_count - 1:
            candidates.append(period_count - 1)
        if len(candidates) < 2:
            raise ValueError("Not enough fully observed origins for production refitting.")
        keep = max(2, int(np.ceil(len(candidates) * training_size)))
        return candidates[-keep:], candidates

    @staticmethod
    def _new_regressor():
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=220,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=42,
        )

    @staticmethod
    def _fit_retransformation_bins(log_level, actual, n_bins=10):
        """Correct log-space retransformation bias, conditional on forecast size.

        The estimators are fitted on ``log1p(actual) - log1p(baseline)`` and
        inverted with ``expm1``.  Exponentiating a conditional mean of logs
        yields a geometric mean, which understates the arithmetic mean of a
        right-skewed revenue distribution, so revenue is reported too low.
        Duan's (1983) smearing factor ``E[exp(residual)]`` is the classical
        remedy but assumes a multiplicative ``log`` model with homoscedastic
        residuals.  Neither holds here: the ``log1p`` inverse carries a ``-1``
        that breaks the decomposition, and residual spread varies sharply with
        product size because 75% of the Product ID x period cells are zero.

        A single global factor therefore fixes the aggregate bias but inflates
        the many near-zero products and degrades per-product error.  Instead a
        separate factor is solved per decile of predicted level, so that within
        each bin the retransformed training total matches the observed total:

            sum[exp(level) * f - 1] = sum(actual)

        Bins and factors are estimated on the training fold only, so the
        holdout is never used to calibrate the forecast.
        """
        log_level = np.asarray(log_level, dtype=float)
        actual = np.maximum(np.asarray(actual, dtype=float), 0.0)
        if log_level.size == 0:
            return np.array([-np.inf, np.inf]), np.array([1.0])
        edges = np.unique(np.quantile(log_level, np.linspace(0.0, 1.0, n_bins + 1)))
        if edges.size < 2:
            edges = np.array([float(log_level.min()), float(log_level.max()) + 1e-9])
        factors = np.ones(edges.size - 1, dtype=float)
        index = np.clip(
            np.searchsorted(edges, log_level, side="right") - 1, 0, edges.size - 2
        )
        for position in range(edges.size - 1):
            mask = index == position
            if not mask.any():
                continue
            denominator = float(np.exp(log_level[mask]).sum())
            if not np.isfinite(denominator) or denominator <= 0.0:
                continue
            factor = (float(actual[mask].sum()) + float(mask.sum())) / denominator
            if np.isfinite(factor) and factor > 0.0:
                factors[position] = float(np.clip(factor, 0.25, 6.0))
        return edges, factors

    @staticmethod
    def _apply_retransformation_bins(log_level, edges, factors):
        factors = np.asarray(factors, dtype=float)
        if factors.size == 0:
            return np.ones_like(np.asarray(log_level, dtype=float))
        index = np.clip(
            np.searchsorted(edges, np.asarray(log_level, dtype=float), side="right") - 1,
            0,
            factors.size - 1,
        )
        return factors[index]

    @classmethod
    def _fit_estimators(cls, features, target, sample_actual):
        """Fit equal-weight and revenue-aware estimators for a stable ensemble."""
        unweighted = cls._new_regressor()
        unweighted.fit(features, target)
        weights = 1.0 + np.log1p(np.maximum(sample_actual, 0.0))
        weights = np.minimum(weights, np.quantile(weights, 0.995))
        weights = weights / weights.mean()
        revenue_weighted = cls._new_regressor()
        revenue_weighted.fit(features, target, sample_weight=weights)
        # target == log1p(actual) - log1p(baseline), so the fitting baseline is
        # recoverable exactly and no extra state has to be threaded through.
        log1p_baseline = np.log1p(np.maximum(sample_actual, 0.0)) - target
        for model in (unweighted, revenue_weighted):
            level = log1p_baseline + model.predict(features)
            model.smearing_edges_, model.smearing_factors_ = (
                cls._fit_retransformation_bins(level, sample_actual)
            )
        return {
            "equal_product_weight": unweighted,
            "log_revenue_weight": revenue_weighted,
        }

    @classmethod
    def _direct_predict(
        cls, estimators, observed_history, horizon, window_size, first_target_index
    ):
        """Predict every lead from observed history, avoiding recursive error compounding."""
        observed = np.maximum(np.asarray(observed_history, dtype=float), 0.0)
        lags = observed[:, -window_size:]
        active = lags.sum(axis=1) > 0
        forecasts = np.zeros((len(lags), horizon), dtype=float)
        if not np.any(active):
            return forecasts
        baseline = cls._one_step_baseline(lags[active])
        for step in range(horizon):
            features = cls._feature_matrix(lags[active], first_target_index + step)
            component_forecasts = []
            selected_estimators = (
                [estimators["equal_product_weight"]]
                if horizon <= 1 else list(estimators.values())
            )
            for model in selected_estimators:
                level = np.log1p(baseline) + model.predict(features)
                # Size-conditional retransformation correction; identity when
                # the estimator predates the calibrated artifact version.
                edges = getattr(model, "smearing_edges_", None)
                factor = 1.0 if edges is None else cls._apply_retransformation_bins(
                    level, edges, model.smearing_factors_
                )
                component_forecasts.append(np.maximum(
                    np.exp(level) * factor - 1.0, 0.0
                ))
            forecasts[active, step] = np.mean(component_forecasts, axis=0)
        return forecasts

    @staticmethod
    def _aggregate_total_forecast(observed_history, horizon):
        """Damped robust trend for total revenue, using only observed periods."""
        totals = np.maximum(np.asarray(observed_history, dtype=float), 0.0).sum(axis=0)
        recent = totals[-min(6, len(totals)):]
        recent_level = recent[-min(3, len(recent)):]
        level = float(np.average(recent_level, weights=np.arange(1, len(recent_level) + 1)))
        trend = float(np.median(np.diff(recent))) if len(recent) > 1 else 0.0
        trend = float(np.clip(trend, -0.03 * level, 0.03 * level))
        damping = 0.7
        return np.array([
            max(level + trend * sum(damping ** power for power in range(step + 1)), 0.0)
            for step in range(horizon)
        ])

    @staticmethod
    def _reconciliation_power(horizon):
        """Apply only a light drift guard without erasing model-specific paths."""
        if horizon >= 12 or 1 < horizon <= 3:
            return 0.25
        return 0.0

    @classmethod
    def _reconcile_to_aggregate(cls, forecasts, observed_history, horizon):
        """Align product forecasts to a separately forecast aggregate total."""
        forecasts = np.maximum(np.asarray(forecasts, dtype=float), 0.0)
        if horizon <= 1:
            return forecasts
        target_totals = cls._aggregate_total_forecast(observed_history, horizon)
        raw_totals = forecasts.sum(axis=0)
        full_scale = np.divide(
            target_totals,
            raw_totals,
            out=np.ones_like(target_totals),
            where=raw_totals > 0,
        )
        reconciliation_power = cls._reconciliation_power(horizon)
        return forecasts * np.power(full_scale, reconciliation_power)

    @staticmethod
    def _aggregate_period_metrics(actual_steps, predicted_steps, mask):
        actual_totals = np.asarray(actual_steps, dtype=float)[mask].sum(axis=0)
        predicted_totals = np.asarray(predicted_steps, dtype=float)[mask].sum(axis=0)
        error = predicted_totals - actual_totals
        total_actual = float(actual_totals.sum())
        return {
            "periods": int(len(actual_totals)),
            "mae": round(float(np.mean(np.abs(error))), 6),
            "rmse": round(float(np.sqrt(np.mean(error ** 2))), 6),
            "wmape": round(float(np.abs(error).sum() / total_actual), 6) if total_actual else 0.0,
            "bias_percent": round(float(error.sum() / total_actual), 6) if total_actual else 0.0,
            "actual_total": round(total_actual, 2),
            "predicted_total": round(float(predicted_totals.sum()), 2),
        }

    @staticmethod
    def _stable_descending_order(product_ids: np.ndarray, values: np.ndarray) -> np.ndarray:
        return np.lexsort((np.asarray(product_ids), -np.asarray(values, dtype=float)))

    @staticmethod
    def _rank_biased_overlap(first_ids, second_ids, persistence=0.9):
        """Finite extrapolated RBO, emphasizing agreement near the top."""
        first_ids, second_ids = list(first_ids), list(second_ids)
        depth = min(len(first_ids), len(second_ids))
        if depth == 0:
            return 0.0
        weighted_overlap = 0.0
        first_seen, second_seen = set(), set()
        agreement = 0.0
        for index in range(depth):
            first_seen.add(first_ids[index])
            second_seen.add(second_ids[index])
            agreement = len(first_seen & second_seen) / float(index + 1)
            weighted_overlap += (1.0 - persistence) * agreement * persistence ** index
        return float(weighted_overlap + agreement * persistence ** depth)

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
                "rank_biased_overlap": round(cls._rank_biased_overlap(
                    actual_top_ids, predicted_top_ids
                ), 6),
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
            "bias_percent": round(
                float((predicted.sum() - total_actual) / total_actual), 6
            ) if total_actual else 0.0,
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
            "precision_at_k", "recall_at_k", "jaccard_at_k", "ndcg_at_k",
            "rank_biased_overlap",
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

    @classmethod
    def _top_rank_comparison(cls, product_ids, actual, sequence, independent, limit=3):
        product_ids = np.asarray(product_ids)
        actual_order = cls._stable_descending_order(product_ids, actual)
        sequence_order = cls._stable_descending_order(product_ids, sequence)
        independent_order = cls._stable_descending_order(product_ids, independent)
        sequence_positions = {
            int(product_ids[index]): position + 1
            for position, index in enumerate(sequence_order)
        }
        independent_positions = {
            int(product_ids[index]): position + 1
            for position, index in enumerate(independent_order)
        }
        return [
            {
                "actual_rank": rank + 1,
                "product_id": int(product_ids[index]),
                "sequence_rank": int(sequence_positions[int(product_ids[index])]),
                "independent_rank": int(independent_positions[int(product_ids[index])]),
            }
            for rank, index in enumerate(actual_order[:int(limit)])
        ]

    def train(
        self, horizon=3, window_size=6, sliding_step=1, training_size=0.8,
        top_k=20, parameter_mode="auto", hidden_units=None,
        feedback_rate=None, epochs=None,
    ):
        """Validate recursive RNN and independent direct forecasts, then refit both."""
        horizon, window_size, sliding_step = int(horizon), int(window_size), int(sliding_step)
        training_size = float(training_size)
        self._validate_configuration(horizon, window_size, sliding_step, training_size)
        architecture_config = self._resolve_architecture(
            parameter_mode, hidden_units, feedback_rate, epochs
        )
        configuration_key = self._configuration_key(
            horizon, window_size, sliding_step, training_size, architecture_config
        )

        revenue_panel, _, _, data_profile = self.load_product_panels()
        origins, test_origin, all_candidates = self._training_origins(
            revenue_panel.shape[1], horizon, window_size, sliding_step, training_size
        )
        direct_features, direct_target, _, direct_actual, direct_counts = (
            self._build_training_samples(revenue_panel, origins, window_size)
        )
        validation_direct = self._fit_estimators(
            direct_features, direct_target, direct_actual
        )
        (
            sequence_lags,
            sequence_targets,
            sequence_mask,
            sequence_periods,
            sequence_counts,
            supervised_by_origin,
        ) = self._build_sequence_samples(
            revenue_panel, origins, window_size, horizon, test_origin
        )
        validation_sequence = self._fit_sequence_model(
            sequence_lags, sequence_targets, sequence_mask, sequence_periods,
            architecture_config,
        )

        values = revenue_panel.to_numpy(dtype=float)
        observed = values[:, :test_origin]
        lookback = observed[:, -window_size:]
        lookback_active = lookback.sum(axis=1) > 0
        raw_sequence_steps = np.zeros((len(values), horizon), dtype=float)
        raw_sequence_steps[lookback_active] = validation_sequence.predict(
            lookback[lookback_active], test_origin, horizon
        )
        raw_independent_steps = self._direct_predict(
            validation_direct, observed, horizon, window_size, test_origin
        )
        sequence_steps = self._reconcile_to_aggregate(
            raw_sequence_steps, observed, horizon
        )
        independent_steps = self._reconcile_to_aggregate(
            raw_independent_steps, observed, horizon
        )
        baseline_one_step = self._one_step_baseline(lookback)
        baseline_steps = np.repeat(baseline_one_step[:, None], horizon, axis=1)

        actual_steps = values[:, test_origin:test_origin + horizon]
        actual = actual_steps.sum(axis=1)
        sequence_forecast = sequence_steps.sum(axis=1)
        independent_forecast = independent_steps.sum(axis=1)
        baseline_forecast = baseline_steps.sum(axis=1)
        actual_active = actual > 0
        evaluation_mask = lookback_active | actual_active
        evaluation_ids = revenue_panel.index.to_numpy(dtype=int)[evaluation_mask]
        actual_eval = actual[evaluation_mask]
        sequence_eval = sequence_forecast[evaluation_mask]
        independent_eval = independent_forecast[evaluation_mask]
        baseline_eval = baseline_forecast[evaluation_mask]

        sequence_metrics = self._evaluate(
            evaluation_ids, actual_eval, sequence_eval, top_k
        )
        independent_metrics = self._evaluate(
            evaluation_ids, actual_eval, independent_eval, top_k
        )
        baseline_metrics = self._evaluate(
            evaluation_ids, actual_eval, baseline_eval, top_k
        )
        aggregate_period_metrics = {
            "time_series_model": self._aggregate_period_metrics(
                actual_steps, sequence_steps, evaluation_mask
            ),
            "independent_direct_model": self._aggregate_period_metrics(
                actual_steps, independent_steps, evaluation_mask
            ),
            "recent_average_baseline": self._aggregate_period_metrics(
                actual_steps, baseline_steps, evaluation_mask
            ),
        }
        ranking_similarity = self._ranking_similarity(
            evaluation_ids, sequence_eval, independent_eval
        )
        top_rank_comparison = self._top_rank_comparison(
            evaluation_ids, actual_eval, sequence_eval, independent_eval
        )
        lower_is_better = ("mae", "rmse", "wmape", "smape")
        sequence_wins = sum(
            sequence_metrics[key] < independent_metrics[key]
            for key in lower_is_better
        )
        independent_wins = sum(
            independent_metrics[key] < sequence_metrics[key]
            for key in lower_is_better
        )
        if sequence_wins > independent_wins:
            assessment = "time_series_model"
        elif independent_wins > sequence_wins:
            assessment = "independent_direct_model"
        else:
            assessment = "mixed_no_clear_winner"
        best_error_method = (
            "time_series_model"
            if sequence_metrics["wmape"] <= independent_metrics["wmape"]
            else "independent_direct_model"
        )
        best_ranking_method = (
            "time_series_model"
            if sequence_metrics["ndcg_at_k"] >= independent_metrics["ndcg_at_k"]
            else "independent_direct_model"
        )

        test_start_day = data_profile["analysis_start_day"] + test_origin * PERIOD_DAYS
        test_end_day = test_start_day + horizon * PERIOD_DAYS - 1
        holdout_actual_vs_predicted = []
        for step in range(horizon):
            period_start = test_start_day + step * PERIOD_DAYS
            holdout_actual_vs_predicted.append({
                "period": int(test_origin + step + 1),
                "start_day": int(period_start),
                "end_day": int(period_start + PERIOD_DAYS - 1),
                "actual_revenue": round(float(actual_steps[evaluation_mask, step].sum()), 2),
                "time_series_revenue": round(float(sequence_steps[evaluation_mask, step].sum()), 2),
                "independent_revenue": round(float(independent_steps[evaluation_mask, step].sum()), 2),
                "baseline_revenue": round(float(baseline_steps[evaluation_mask, step].sum()), 2),
            })
        reconciliation_power = self._reconciliation_power(horizon)
        report = {
            "artifact_version": ARTIFACT_VERSION,
            "configuration_key": configuration_key,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_configuration": architecture_config,
            "forecast_method": "autoregressive_encoder_decoder_rnn_with_bptt",
            "model_architecture": {
                "time_series_model": "NumPy tanh encoder-decoder RNN",
                "independent_direct_model": (
                    "50/50 equal-product and revenue-weighted HistGradientBoosting ensemble"
                ),
                "recursive_feedback": True,
                "joint_multi_step_loss": True,
                "backpropagation_through_time": True,
                "hidden_units": int(validation_sequence.hidden_size),
                "feedback_rate": float(validation_sequence.feedback_rate),
            },
            "sequence_training": {
                "loss": "masked multi-step Huber loss on normalized log revenue",
                "epochs": int(validation_sequence.epochs),
                "loss_history": validation_sequence.training_history_,
                "supervised_forecast_steps": int(validation_sequence.supervised_steps_),
                "maximum_supervised_rollout": int(validation_sequence.max_supervised_horizon_),
                "supervised_steps_by_origin": supervised_by_origin,
            },
            "aggregate_reconciliation": {
                "power": reconciliation_power,
                "mode": (
                    "full" if reconciliation_power == 1.0
                    else "partial" if reconciliation_power > 0.0
                    else "disabled"
                ),
                "applied_equally_to_compared_models": True,
            },
            "target": f"monthly Product ID revenue for the next {horizon} complete 30-day period(s)",
            "horizon_months": horizon,
            "window_size_months": window_size,
            "sliding_step_months": sliding_step,
            "training_size": training_size,
            "validation": (
                "strict final-horizon chronological holdout; every training target ends before "
                "the holdout; overlapping windows stay on the training side; no random split or "
                "future-revenue features"
            ),
            "temporal_evidence_strength": "limited" if len(origins) < 6 else "moderate",
            "temporal_fit_origins": int(len(origins)),
            "fit_origin_periods": [origin + 1 for origin in origins],
            "available_pretest_origins": [origin + 1 for origin in all_candidates],
            "test_origin_period": test_origin + 1,
            "test_period": {
                "start_day": int(test_start_day),
                "end_day": int(test_end_day),
                "periods": list(range(test_origin + 1, test_origin + horizon + 1)),
            },
            "holdout_actual_vs_predicted": holdout_actual_vs_predicted,
            "samples": {
                "independent_train": int(len(direct_features)),
                "sequence_train": int(len(sequence_lags)),
                "independent_train_by_origin": direct_counts,
                "sequence_train_by_origin": sequence_counts,
                "products_total": int(len(revenue_panel)),
                "evaluation_products": int(evaluation_mask.sum()),
                "lookback_active_products": int(lookback_active.sum()),
                "cold_start_products_in_test": int((~lookback_active & actual_active).sum()),
            },
            "data_profile": data_profile,
            "time_series_model": sequence_metrics,
            "independent_direct_model": independent_metrics,
            "recent_average_baseline": baseline_metrics,
            "bias_diagnostics": {
                "time_series_bias_percent": sequence_metrics["bias_percent"],
                "independent_bias_percent": independent_metrics["bias_percent"],
                "recent_average_bias_percent": baseline_metrics["bias_percent"],
                "interpretation": (
                    "Negative values mean underprediction and positive values mean "
                    "overprediction. Sparse products, cold starts, and log-target "
                    "shrinkage can create negative aggregate bias; long recursive "
                    "rollouts can also accumulate feedback error in either direction."
                ),
            },
            "retransformation_correction": {
                "applied": True,
                "method": (
                    "Size-conditional log-space retransformation correction "
                    "(Duan-type smearing, solved per decile of predicted level)"
                ),
                "estimated_on": "training fold only; the holdout is never used to calibrate",
                "reason": (
                    "Both models are fitted on a log target and inverted with expm1, so "
                    "exponentiating a conditional mean of logs returns a geometric mean "
                    "and understates right-skewed revenue. Residual spread grows with "
                    "product size, so one global factor would inflate near-zero products; "
                    "a factor per predicted-level decile corrects the level without that "
                    "distortion."
                ),
                "sequence_factors": [
                    round(float(value), 6)
                    for value in np.asarray(
                        getattr(validation_sequence, "smearing_factors_", [1.0])
                    ).ravel()
                ],
                "direct_factors": [
                    round(float(value), 6)
                    for value in np.asarray(
                        getattr(
                            validation_direct["equal_product_weight"],
                            "smearing_factors_",
                            [1.0],
                        )
                    ).ravel()
                ],
            },
            "holdout_period_total_metrics": aggregate_period_metrics,
            "model_ranking_similarity": ranking_similarity,
            "top_product_rank_comparison": top_rank_comparison,
            "selection_guidance": {
                "best_for_revenue_error": best_error_method,
                "best_for_revenue_error_basis": "cumulative Product ID WMAPE",
                "overall_error_assessment": assessment,
                "error_metric_wins": {
                    "time_series_model": int(sequence_wins),
                    "independent_direct_model": int(independent_wins),
                    "metrics_compared": list(lower_is_better),
                },
                "best_for_top_k_ranking": best_ranking_method,
            },
            "required_caveats": [
                "The dataset provides 23 complete aligned periods, so the final holdout is one horizon block rather than many independent years.",
                "For long horizons, the validation-side RNN is supervised only for the future steps available before the holdout; the maximum supervised rollout is reported explicitly.",
                "Product rows increase cross-sectional samples but do not replace missing calendar history.",
                "Products with no revenue in the lookback are cold starts; revenue-only models predict zero for them and report their count.",
                "The legacy repurchase classifier has a different target and its classification Accuracy must not be compared with revenue errors.",
            ],
        }

        production_origins, production_candidates = self._production_origins(
            revenue_panel.shape[1], window_size, sliding_step, training_size
        )
        production_features, production_target, _, production_actual, production_counts = (
            self._build_training_samples(revenue_panel, production_origins, window_size)
        )
        production_direct = self._fit_estimators(
            production_features, production_target, production_actual
        )
        (
            production_lags,
            production_sequence_targets,
            production_sequence_mask,
            production_periods,
            production_sequence_counts,
            production_supervised_by_origin,
        ) = self._build_sequence_samples(
            revenue_panel,
            production_origins,
            window_size,
            horizon,
            revenue_panel.shape[1],
        )
        production_sequence = self._fit_sequence_model(
            production_lags,
            production_sequence_targets,
            production_sequence_mask,
            production_periods,
            architecture_config,
        )
        report["production_refit"] = {
            "fit_after_holdout_evaluation": True,
            "purpose": "future periods after the dataset as-of day only",
            "origin_periods": [origin + 1 for origin in production_origins],
            "available_origins": [origin + 1 for origin in production_candidates],
            "independent_samples": int(len(production_features)),
            "sequence_samples": int(len(production_lags)),
            "independent_samples_by_origin": production_counts,
            "sequence_samples_by_origin": production_sequence_counts,
            "sequence_supervised_steps_by_origin": production_supervised_by_origin,
            "sequence_maximum_supervised_rollout": int(
                production_sequence.max_supervised_horizon_
            ),
        }
        artifact = {
            "artifact_version": ARTIFACT_VERSION,
            "configuration_key": configuration_key,
            "training_configuration": architecture_config,
            "production_direct_estimators": production_direct,
            "production_sequence_model": production_sequence,
            "horizon": horizon,
            "window_size": window_size,
            "sliding_step": sliding_step,
            "data_profile": data_profile,
            "report": report,
        }
        path_args = (
            horizon, window_size, sliding_step, training_size,
            architecture_config["parameter_mode"], architecture_config["hidden_units"],
            architecture_config["feedback_rate"], architecture_config["epochs"],
        )
        artifact_path = self._artifact_path(*path_args)
        metrics_path = self._metrics_path(*path_args)
        with artifact_path.open("wb") as handle:
            pickle.dump(artifact, handle)
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        return report

    def get_report(
        self, horizon=3, window_size=6, sliding_step=1, training_size=0.8,
        parameter_mode="auto", hidden_units=None, feedback_rate=None, epochs=None,
    ):
        path = self._metrics_path(
            int(horizon), int(window_size), int(sliding_step), float(training_size),
            parameter_mode, hidden_units, feedback_rate, epochs,
        )
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as handle:
            return self._refresh_report_interpretation(json.load(handle))

    def forecast(
        self,
        horizon=3,
        window_size=6,
        sliding_step=1,
        top_n=20,
        revenue_threshold=None,
        forecast_method="recommended",
        training_size=0.8,
        parameter_mode="auto",
        hidden_units=None,
        feedback_rate=None,
        epochs=None,
    ):
        """Forecast with both compared models and return side-by-side product results."""
        horizon, window_size, sliding_step = int(horizon), int(window_size), int(sliding_step)
        training_size = float(training_size)
        self._validate_configuration(horizon, window_size, sliding_step, training_size)
        architecture_config = self._resolve_architecture(
            parameter_mode, hidden_units, feedback_rate, epochs
        )
        configuration_key = self._configuration_key(
            horizon, window_size, sliding_step, training_size, architecture_config
        )
        path = self._artifact_path(
            horizon, window_size, sliding_step, training_size,
            architecture_config["parameter_mode"], architecture_config["hidden_units"],
            architecture_config["feedback_rate"], architecture_config["epochs"],
        )
        if not path.exists():
            raise ValueError(
                "No saved model matches this exact horizon, window, step, and "
                "parameter configuration. Train and validate it first."
            )
        with path.open("rb") as handle:
            artifact = pickle.load(handle)
        if artifact.get("artifact_version") != ARTIFACT_VERSION:
            raise ValueError("The saved model is obsolete. Retrain it with the corrected validation pipeline.")
        if artifact.get("configuration_key") != configuration_key:
            raise ValueError("Saved-model configuration mismatch. Retrain this configuration.")
        artifact["report"] = self._refresh_report_interpretation(artifact["report"])

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
        latest_lags = observed[:, -window_size:]
        eligible = latest_lags.sum(axis=1) > 0
        raw_sequence_steps = np.zeros((len(observed), horizon), dtype=float)
        raw_sequence_steps[eligible] = artifact["production_sequence_model"].predict(
            latest_lags[eligible], observed.shape[1], horizon
        )
        raw_independent_steps = self._direct_predict(
            artifact["production_direct_estimators"],
            observed,
            horizon,
            window_size,
            observed.shape[1],
        )
        sequence_steps = self._reconcile_to_aggregate(
            raw_sequence_steps, observed, horizon
        )
        independent_steps = self._reconcile_to_aggregate(
            raw_independent_steps, observed, horizon
        )
        baseline_one_step = self._one_step_baseline(latest_lags)
        baseline_steps = np.repeat(baseline_one_step[:, None], horizon, axis=1)
        sequence_revenue = sequence_steps.sum(axis=1)
        independent_revenue = independent_steps.sum(axis=1)
        baseline_revenue = baseline_steps.sum(axis=1)
        requested_method = str(forecast_method or "recommended")
        valid_methods = {
            "recommended", "time_series_model", "independent_direct_model",
            "recent_average_baseline",
        }
        if requested_method not in valid_methods:
            raise ValueError(
                "forecast_method must be recommended, time_series_model, "
                "independent_direct_model, or recent_average_baseline."
            )
        if requested_method == "recommended":
            applied_method = artifact["report"]["selection_guidance"]["best_for_revenue_error"]
        else:
            applied_method = requested_method
        method_steps = {
            "time_series_model": sequence_steps,
            "independent_direct_model": independent_steps,
            "recent_average_baseline": baseline_steps,
        }
        selected_steps = method_steps[applied_method]
        predicted_revenue = selected_steps.sum(axis=1)

        recent_units = unit_panel.iloc[:, -window_size:].sum(axis=1).to_numpy(dtype=float)
        recent_revenue = revenue_panel.iloc[:, -window_size:].sum(axis=1).to_numpy(dtype=float)
        revenue_per_unit = np.divide(
            recent_revenue,
            recent_units,
            out=np.full_like(recent_revenue, np.nan),
            where=recent_units > 0,
        )
        # Quantity is not a consistent retail "unit" in this source (notably fuel and
        # weighted products).  Do not manufacture enormous unit forecasts when the
        # historical revenue/quantity rate is below one cent.
        unit_estimate_available = np.isfinite(revenue_per_unit) & (revenue_per_unit >= 0.01)
        predicted_units = np.divide(
            predicted_revenue,
            revenue_per_unit,
            out=np.full_like(predicted_revenue, np.nan),
            where=unit_estimate_available,
        )

        result = pd.DataFrame(
            {
                "product_id": revenue_panel.index.astype(int),
                "panel_position": np.arange(len(revenue_panel)),
                "predicted_revenue": predicted_revenue,
                "time_series_revenue": sequence_revenue,
                "independent_revenue": independent_revenue,
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
        department_forecast = result.groupby("department", dropna=False)[[
            "predicted_revenue", "time_series_revenue", "independent_revenue",
            "baseline_revenue",
        ]].sum().sort_values("predicted_revenue", ascending=False).reset_index()
        result = result.sort_values(
            ["predicted_revenue", "product_id"], ascending=[False, True]
        )
        top_n = max(1, min(int(top_n), 500))
        top_result = result.head(top_n).copy()
        top_result["forecast_rank"] = np.arange(1, len(top_result) + 1)
        top_result["revenue_change_time_series_vs_independent"] = (
            top_result["time_series_revenue"] - top_result["independent_revenue"]
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
                    "time_series_revenue": round(float(row.time_series_revenue), 2),
                    "independent_revenue": round(float(row.independent_revenue), 2),
                    "baseline_revenue": round(float(row.baseline_revenue), 2),
                    "revenue_change_time_series_vs_independent": round(
                        float(row.revenue_change_time_series_vs_independent), 2
                    ),
                    "revenue_per_unit": (
                        round(float(row.revenue_per_unit), 4)
                        if pd.notna(row.revenue_per_unit) else None
                    ),
                    "predicted_units": (
                        round(float(row.predicted_units), 2)
                        if pd.notna(row.predicted_units) else None
                    ),
                    "time_series_monthly_predictions": [
                        round(float(value), 2)
                        for value in sequence_steps[int(row.panel_position)]
                    ],
                    "independent_monthly_predictions": [
                        round(float(value), 2)
                        for value in independent_steps[int(row.panel_position)]
                    ],
                    "baseline_monthly_predictions": [
                        round(float(value), 2)
                        for value in baseline_steps[int(row.panel_position)]
                    ],
                }
            )

        forecast_start_day = data_profile["as_of_day"] + 1
        return {
            "horizon_months": horizon,
            "configuration_key": configuration_key,
            "saved_at_utc": artifact["report"].get("saved_at_utc"),
            "training_configuration": artifact.get("training_configuration", {}),
            "window_size_months": window_size,
            "sliding_step_months": sliding_step,
            "forecast_method_requested": requested_method,
            "forecast_method_applied": applied_method,
            "forecast_method": {
                "time_series_model": "autoregressive_encoder_decoder_rnn_with_bptt",
                "independent_direct_model": "independent_direct_gradient_boosting_ensemble",
                "recent_average_baseline": "recent_average_naive_benchmark",
            }[applied_method],
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
                    "time_series_revenue": round(float(row.time_series_revenue), 2),
                    "independent_revenue": round(float(row.independent_revenue), 2),
                    "baseline_revenue": round(float(row.baseline_revenue), 2),
                }
                for _, row in department_forecast.iterrows()
            ],
            "predictions": predictions,
        }


product_revenue_forecaster = ProductRevenueTimeSeriesForecaster()
