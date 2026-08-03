import numpy as np
import pandas as pd
from django.test import SimpleTestCase
from sklearn.ensemble import GradientBoostingClassifier

from .repurchase_classifier import MODEL_FEATURES, PredictiveMarketBasketAnalyzer
from .autoregressive_rnn import AutoregressiveRevenueRNN
from .time_series_forecasting import ProductRevenueTimeSeriesForecaster


class ZeroCorrectionModel:
    """Test double that leaves the recent-average baseline unchanged."""

    def predict(self, features):
        return np.zeros(len(features), dtype=float)


class ProductRevenueTimeSeriesTests(SimpleTestCase):
    def setUp(self):
        self.forecaster = ProductRevenueTimeSeriesForecaster()

    def test_final_horizon_is_purged_from_training_targets(self):
        origins, test_origin, _ = self.forecaster._training_origins(
            period_count=23,
            horizon=3,
            window_size=6,
            sliding_step=1,
            training_size=0.8,
        )
        self.assertEqual(test_origin, 20)
        self.assertTrue(origins)
        self.assertLess(max(origins), test_origin)
        self.assertTrue(set(origins).isdisjoint({20, 21, 22}))

    def test_auto_architecture_uses_professor_aligned_defaults(self):
        architecture = self.forecaster._resolve_architecture(
            "auto", hidden_units=64, feedback_rate=1.0, epochs=30
        )
        self.assertEqual(architecture["hidden_units"], 16)
        self.assertEqual(architecture["feedback_rate"], 0.5)
        self.assertEqual(architecture["epochs"], 10)
        self.assertTrue(architecture["recursive_feedback"])
        self.assertTrue(architecture["joint_multi_step_loss"])
        self.assertTrue(architecture["backpropagation_through_time"])

    def test_custom_architecture_has_a_distinct_saved_model_key(self):
        automatic = self.forecaster._resolve_architecture("auto")
        custom = self.forecaster._resolve_architecture("custom", 32, 0.75, 15)
        auto_key = self.forecaster._configuration_key(3, 6, 1, 0.8, automatic)
        custom_key = self.forecaster._configuration_key(3, 6, 1, 0.8, custom)
        self.assertNotEqual(auto_key, custom_key)
        self.assertIn("custom_hu32_fb075_e15", custom_key)

    def test_invalid_custom_architecture_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "hidden_units"):
            self.forecaster._resolve_architecture("custom", 12, 0.5, 10)

    def test_direct_forecast_does_not_compound_its_own_predictions(self):
        history = np.array([[10.0, 10.0, 10.0], [0.0, 0.0, 0.0]])
        forecast = self.forecaster._direct_predict(
            {"equal": ZeroCorrectionModel(), "weighted": ZeroCorrectionModel()},
            history,
            horizon=3,
            window_size=3,
            first_target_index=3,
        )
        np.testing.assert_allclose(forecast[0], [10.0, 10.0, 10.0])
        np.testing.assert_allclose(forecast[1], [0.0, 0.0, 0.0])

    def test_sequence_targets_are_masked_at_the_chronological_boundary(self):
        panel = pd.DataFrame(
            [np.arange(1, 11, dtype=float), np.arange(2, 12, dtype=float)],
            index=[101, 202],
        )
        _, targets, mask, periods, _, supervised = self.forecaster._build_sequence_samples(
            panel,
            origins=[3, 5, 7],
            window_size=3,
            horizon=4,
            training_end=8,
        )
        self.assertEqual(supervised, {"4": 4, "6": 3, "8": 1})
        self.assertTrue(np.all(periods < 8))
        self.assertTrue(np.all(targets[~mask] == 0.0))
        self.assertEqual(int(mask.sum()), 2 * (4 + 3 + 1))

    def test_final_step_loss_backpropagates_through_recurrent_model(self):
        model = AutoregressiveRevenueRNN(hidden_size=6, epochs=1, batch_size=4)
        model.parameters_ = model._initialize_parameters(np.random.default_rng(42))
        lags = np.array([[2.0, 3.0, 4.0], [5.0, 4.0, 3.0]])
        targets = np.array([[0.0, 0.0, 9.0], [0.0, 0.0, 8.0]])
        mask = np.array([[False, False, True], [False, False, True]])
        loss, gradients = model._loss_and_gradients(
            lags, targets, mask, np.array([3, 3]), np.ones(2)
        )
        self.assertGreater(loss, 0.0)
        self.assertGreater(np.linalg.norm(gradients["wh"]), 0.0)
        self.assertGreater(np.linalg.norm(gradients["wx"]), 0.0)

    def test_long_horizon_drift_guard_preserves_model_specific_path(self):
        observed = np.array([[50.0] * 6, [50.0] * 6])
        raw = np.array([
            np.linspace(40.0, 5.0, 12),
            np.linspace(40.0, 5.0, 12),
        ])
        reconciled = self.forecaster._reconcile_to_aggregate(raw, observed, 12)
        raw_ratio = raw[:, -1].sum() / raw[:, 0].sum()
        reconciled_ratio = reconciled[:, -1].sum() / reconciled[:, 0].sum()
        self.assertGreater(reconciled_ratio, raw_ratio)
        self.assertTrue(np.all(reconciled >= 0))
        expected = raw.sum(axis=0) * np.power(100.0 / raw.sum(axis=0), 0.25)
        np.testing.assert_allclose(reconciled.sum(axis=0), expected)
        self.assertFalse(np.allclose(reconciled.sum(axis=0), np.full(12, 100.0)))

    def test_production_refit_uses_latest_fully_observed_target(self):
        origins, candidates = self.forecaster._production_origins(
            period_count=23, window_size=6, sliding_step=2, training_size=0.8
        )
        self.assertEqual(candidates[-1], 22)
        self.assertEqual(origins[-1], 22)

    def test_perfect_period_total_forecast_has_zero_error(self):
        actual = np.array([[10.0, 20.0], [5.0, 7.0]])
        metrics = self.forecaster._aggregate_period_metrics(
            actual, actual, np.array([True, True])
        )
        self.assertEqual(metrics["wmape"], 0.0)
        self.assertEqual(metrics["bias_percent"], 0.0)

    def test_perfect_product_ranking_scores_one_at_all_cutoffs(self):
        ids = np.arange(1, 26)
        actual = np.arange(25, 0, -1, dtype=float)
        metrics = self.forecaster._ranking_metrics(ids, actual, actual)
        self.assertEqual(metrics["spearman"], 1.0)
        self.assertEqual(metrics["kendall_tau"], 1.0)
        for cutoff in ("5", "10", "20"):
            self.assertEqual(metrics["ranking_at_k"][cutoff]["precision_at_k"], 1.0)
            self.assertEqual(metrics["ranking_at_k"][cutoff]["jaccard_at_k"], 1.0)
            self.assertEqual(metrics["ranking_at_k"][cutoff]["ndcg_at_k"], 1.0)
            self.assertEqual(metrics["ranking_at_k"][cutoff]["rank_biased_overlap"], 1.0)

    def test_product_metrics_report_negative_bias_for_underprediction(self):
        metrics = self.forecaster._evaluate(
            np.array([1, 2]), np.array([100.0, 50.0]), np.array([80.0, 40.0]), 2
        )
        self.assertEqual(metrics["bias_percent"], -0.2)

    def test_product_metrics_report_positive_bias_for_overprediction(self):
        metrics = self.forecaster._evaluate(
            np.array([1, 2]), np.array([100.0, 50.0]), np.array([120.0, 60.0]), 2
        )
        self.assertEqual(metrics["bias_percent"], 0.2)

    def test_saved_report_explains_bias_in_both_directions(self):
        report = {"bias_diagnostics": {"interpretation": "stale text"}}
        refreshed = self.forecaster._refresh_report_interpretation(report)
        self.assertIn("underprediction", refreshed["bias_diagnostics"]["interpretation"])
        self.assertIn("overprediction", refreshed["bias_diagnostics"]["interpretation"])

    def test_impossible_window_and_holdout_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "Not enough complete history"):
            self.forecaster._training_origins(
                period_count=23,
                horizon=12,
                window_size=12,
                sliding_step=1,
                training_size=0.8,
            )


class RepurchaseClassifierTests(SimpleTestCase):
    def test_training_targets_end_before_chronological_test_snapshot(self):
        origins, test_origin, _, status = PredictiveMarketBasketAnalyzer._split_origins(
            max_day=711, horizon_days=90, training_size=0.8
        )
        self.assertEqual(status, "chronological_holdout")
        self.assertTrue(origins)
        self.assertTrue(all(origin + 90 <= test_origin for origin in origins))

    def test_twelve_month_holdout_is_not_fabricated(self):
        origins, test_origin, candidates, status = PredictiveMarketBasketAnalyzer._split_origins(
            max_day=711, horizon_days=360, training_size=0.8
        )
        self.assertEqual(origins, [])
        self.assertIsNone(test_origin)
        self.assertTrue(candidates)
        self.assertEqual(status, "unavailable_insufficient_independent_windows")

    def test_target_columns_are_never_model_features(self):
        self.assertNotIn("target", MODEL_FEATURES)
        self.assertFalse(any("repurchase" in feature for feature in MODEL_FEATURES))

    def test_gradient_boost_option_is_a_real_gradient_boosting_classifier(self):
        estimator = PredictiveMarketBasketAnalyzer._classifier("gradient_boost")
        self.assertIsInstance(estimator, GradientBoostingClassifier)
