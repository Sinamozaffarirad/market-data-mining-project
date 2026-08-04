from unittest import TestCase

from .churn_windows import ChurnWindowConfig, WindowMethod, generate_time_windows


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
