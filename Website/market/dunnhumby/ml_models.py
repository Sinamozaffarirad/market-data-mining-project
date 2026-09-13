
from .repurchase_classifier import PredictiveMarketBasketAnalyzer


ml_analyzer = PredictiveMarketBasketAnalyzer()


class ChurnPredictor:

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "ChurnPredictor is deprecated. Use the time-window experiment system "
            "in Customer Segments to train and activate a churn model."
        )
