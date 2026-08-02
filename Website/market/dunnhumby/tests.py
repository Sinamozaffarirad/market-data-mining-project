import numpy as np
from django.test import SimpleTestCase
from sklearn.ensemble import GradientBoostingClassifier

from .repurchase_classifier import MODEL_FEATURES, PredictiveMarketBasketAnalyzer
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

    def test_recursive_forecast_feeds_predictions_forward(self):
        history = np.array([[10.0, 10.0, 10.0], [0.0, 0.0, 0.0]])
        forecast = self.forecaster._recursive_predict(
            ZeroCorrectionModel(), history, horizon=3, window_size=3, first_target_index=3
        )
        np.testing.assert_allclose(forecast[0], [10.0, 10.0, 10.0])
        np.testing.assert_allclose(forecast[1], [0.0, 0.0, 0.0])

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
