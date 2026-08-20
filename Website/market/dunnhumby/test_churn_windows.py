from unittest import TestCase
import pandas as pd

from .churn_windows import (
    ChurnWindowConfig, WindowMethod, _metrics, _select_classification_threshold,
    build_customer_features, build_training_dataset, experiment_metadata, generate_time_windows,
)


class TimeWindowTests(TestCase):
    def test_sliding_windows_keep_labels_after_cutoff(self):
        config = ChurnWindowConfig(WindowMethod.SLIDING, 90, 30, 30)
        windows = list(generate_time_windows(1, 180, config))
        self.assertEqual(windows[0]['observation_end'], 90)
        self.assertEqual(windows[0]['label_start'], 91)
        self.assertEqual(windows[1]['observation_start'], 31)
        self.assertTrue(all(window['label_start'] > window['cutoff_day'] for window in windows))

    def test_non_overlapping_step_is_automatic(self):
        config = ChurnWindowConfig(WindowMethod.NON_OVERLAPPING, 90, 30)
        windows = list(generate_time_windows(1, 240, config))
        self.assertEqual(config.step_size(), 120)
        self.assertEqual(windows[1]['observation_start'], 121)

    def test_every_generated_window_has_safe_observation_and_label_boundaries(self):
        maximum_day = 12
        config = ChurnWindowConfig(WindowMethod.SLIDING, 3, 2, 3)
        windows = list(generate_time_windows(1, maximum_day, config))

        self.assertEqual([window["cutoff_day"] for window in windows], [3, 6, 9])
        for window in windows:
            self.assertEqual(window["observation_end"], window["cutoff_day"])
            self.assertEqual(window["label_start"], window["cutoff_day"] + 1)
            self.assertGreater(window["label_start"], window["cutoff_day"])
            self.assertLessEqual(window["label_end"], maximum_day)

    def test_incomplete_final_window_is_not_generated(self):
        config = ChurnWindowConfig(WindowMethod.SLIDING, 3, 2, 3)
        windows = list(generate_time_windows(1, 10, config))

        self.assertEqual([window["cutoff_day"] for window in windows], [3, 6])
        self.assertTrue(all(window["label_end"] <= 10 for window in windows))

    def test_purchase_gap_features_use_last_two_purchase_days(self):
        observation = _transactions([
            (1, 10, 101, 10.0),
            (1, 22, 102, 20.0),
            (1, 30, 103, 30.0),
            (2, 25, 201, 15.0),
        ])
        features = build_customer_features(observation, cutoff_day=30).set_index("household_key")
        self.assertEqual(features.loc[1, "days_since_previous_purchase"], 8)
        self.assertEqual(features.loc[1, "has_multiple_purchase_days"], 1)
        self.assertGreater(features.loc[1, "purchase_gap_std"], 0)
        self.assertEqual(features.loc[2, "has_multiple_purchase_days"], 0)
        self.assertEqual(features.loc[2, "days_since_previous_purchase"], 0)
        self.assertEqual(features.loc[2, "purchase_gap_std"], 0)

    def test_future_label_purchase_does_not_change_observation_features(self):
        transactions = _transactions([
            (1, 1, 101, 10.0),
            (1, 3, 102, 20.0),
            (1, 4, 103, 999.0),  # Label window purchase, not an observation feature.
            (1, 7, 104, 25.0),
        ])
        config = ChurnWindowConfig(WindowMethod.SLIDING, 3, 2, 3)
        dataset = build_training_dataset(transactions, config)
        first_window = dataset[(dataset.household_key == 1) & (dataset.cutoff_day == 3)].iloc[0]
        self.assertEqual(first_window.monetary, 30.0)
        self.assertEqual(first_window.frequency, 2)

    def test_changing_a_future_transaction_cannot_change_an_earlier_cutoff(self):
        base_transactions = _transactions([
            (1, 1, 101, 10.0),
            (1, 3, 102, 20.0),
            (1, 4, 103, 25.0),
            (1, 7, 104, 30.0),
        ])
        changed_future = pd.concat([base_transactions, _transactions([
            (1, 7, 105, 9999.0),
        ])], ignore_index=True)
        config = ChurnWindowConfig(WindowMethod.SLIDING, 3, 2, 3)

        original = build_training_dataset(base_transactions, config)
        changed = build_training_dataset(changed_future, config)
        original_row = original[(original.household_key == 1) & (original.cutoff_day == 3)].iloc[0]
        changed_row = changed[(changed.household_key == 1) & (changed.cutoff_day == 3)].iloc[0]

        for column in ["recency_days", "frequency", "monetary", "lifetime_frequency", "lifetime_monetary"]:
            self.assertEqual(original_row[column], changed_row[column], column)

    def test_lifetime_features_include_inactive_customer_without_future_data(self):
        lifetime = _transactions([
            (1, 2, 101, 10.0),
            (1, 8, 102, 20.0),
            (1, 12, 103, 999.0),  # Future of cutoff 10: must not be used.
            (2, 10, 201, 15.0),
        ])
        observation = lifetime[lifetime.day == 10]
        features = build_customer_features(
            observation,
            cutoff_day=10,
            lifetime_transactions=lifetime,
            observation_start=6,
        ).set_index("household_key")
        self.assertIn(1, features.index)
        self.assertEqual(features.loc[1, "frequency"], 0)
        self.assertEqual(features.loc[1, "customer_tenure_days"], 8)
        self.assertEqual(features.loc[1, "lifetime_frequency"], 2)
        self.assertEqual(features.loc[1, "lifetime_monetary"], 30.0)

    def test_validation_selects_the_best_f1_cutoff(self):
        threshold = _select_classification_threshold(
            [1, 1, 0, 0], [0.85, 0.45, 0.40, 0.10],
        )
        self.assertEqual(threshold, 0.45)

    def test_metrics_save_the_full_test_confusion_matrix(self):
        metrics = _metrics([1, 1, 0, 0], [0.90, 0.40, 0.60, 0.10], 0.50)
        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["true_negative"], 1)

    def test_experiment_metadata_records_the_training_definition(self):
        metadata = experiment_metadata()
        self.assertEqual(metadata["model_name"], "XGBoostClassifier")
        self.assertEqual(metadata["model_parameters"]["random_state"], 42)
        self.assertIn("prediction horizon", metadata["label_definition"])
        self.assertEqual(metadata["threshold_selection"]["primary_metric"], "F1")
        self.assertEqual(metadata["threshold_selection"]["tie_breaker"], "Higher recall")
        self.assertEqual(
            metadata["training_protocol_version"],
            "purged-temporal-v1",
        )
        self.assertIn("Purged chronological", metadata["time_split"])


def _transactions(rows):
    """Build the minimum transaction frame needed by churn-window feature tests."""
    return pd.DataFrame([
        {
            "household_key": household_key,
            "day": day,
            "basket_id": basket_id,
            "product_id": basket_id,
            "quantity": 1,
            "sales_value": sales_value,
            "retail_disc": 0.0,
            "coupon_disc": 0.0,
            "coupon_match_disc": 0.0,
            "department": "Test",
            "commodity_desc": "Test item",
        }
        for household_key, day, basket_id, sales_value in rows
    ])
