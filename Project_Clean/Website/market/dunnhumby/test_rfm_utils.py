import unittest

import pandas as pd

from .rfm_utils import assign_rfm_segment, score_rfm_series


class SharedRFMTests(unittest.TestCase):
    def test_rfm_score_direction_is_correct(self):
        values = pd.Series([1, 2, 3, 4, 5])
        self.assertEqual(
            score_rfm_series(values, higher_is_better=True).tolist(), [1, 2, 3, 4, 5]
        )
        self.assertEqual(
            score_rfm_series(values, higher_is_better=False).tolist(), [5, 4, 3, 2, 1]
        )

    def test_specific_segments_precede_broad_segments(self):
        self.assertEqual(assign_rfm_segment(5, 5, 5), "Champions")
        self.assertEqual(assign_rfm_segment(1, 5, 5), "Need Attention")
        self.assertEqual(assign_rfm_segment(1, 3, 3), "Need Attention")
        self.assertEqual(assign_rfm_segment(1, 2, 2), "At Risk")

    def test_all_rfm_segments_are_reachable(self):
        examples = {
            "Hibernating": (1, 1, 1),
            "Potential Loyalists": (4, 3, 2),
            "New Customers": (4, 1, 1),
            "Loyal Customers": (3, 4, 3),
            "Big Spenders": (3, 2, 4),
            "Regular Customers": (3, 3, 2),
            "Lost": (3, 1, 1),
        }
        for expected, scores in examples.items():
            with self.subTest(segment=expected):
                self.assertEqual(assign_rfm_segment(*scores), expected)

    def test_rfm_produces_exactly_the_ten_supported_segments(self):
        expected_segments = {
            "Champions",
            "Need Attention",
            "At Risk",
            "Hibernating",
            "Potential Loyalists",
            "New Customers",
            "Loyal Customers",
            "Big Spenders",
            "Regular Customers",
            "Lost",
        }

        actual_segments = {
            assign_rfm_segment(r_score, f_score, m_score)
            for r_score in range(1, 6)
            for f_score in range(1, 6)
            for m_score in range(1, 6)
        }

        self.assertSetEqual(actual_segments, expected_segments)
