from django.test import SimpleTestCase

from .analytics import build_churn_feature_set
from .ml_models import ChurnPredictor


class LegacyChurnDeprecationTests(SimpleTestCase):
    def test_legacy_feature_builder_is_blocked(self):
        with self.assertRaisesMessage(RuntimeError, "Deprecated churn builder"):
            build_churn_feature_set()

    def test_legacy_churn_predictor_is_blocked(self):
        with self.assertRaisesMessage(RuntimeError, "ChurnPredictor is deprecated"):
            ChurnPredictor()
