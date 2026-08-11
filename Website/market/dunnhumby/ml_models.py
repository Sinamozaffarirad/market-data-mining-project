"""Machine-learning entry points used by the Dunnhumby views."""
from .repurchase_classifier import PredictiveMarketBasketAnalyzer


# The active predictive-basket implementation is the leakage-safe classifier in
# repurchase_classifier.py.  Keeping this import surface avoids changing views.
ml_analyzer = PredictiveMarketBasketAnalyzer()


class ChurnPredictor:
    """Deprecated compatibility placeholder for the obsolete churn pipeline."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "ChurnPredictor is deprecated. Use the time-window experiment system "
            "in Customer Segments to train and activate a churn model."
        )
