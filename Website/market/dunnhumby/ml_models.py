"""Machine-learning entry points used by the Dunnhumby views."""
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from .analytics import build_churn_feature_set
from .repurchase_classifier import PredictiveMarketBasketAnalyzer


# The active predictive-basket implementation is the leakage-safe classifier in
# repurchase_classifier.py.  Keeping this import surface avoids changing views.
ml_analyzer = PredictiveMarketBasketAnalyzer()


class ChurnPredictor:
    """Train and serve the existing customer churn model."""

    def __init__(self, model=None):
        self.model = model or xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            n_estimators=100,
            random_state=42,
        )
        self.features = None
        self.target = None
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def prepare_data(self, churn_threshold_days=30):
        frame = build_churn_feature_set(prediction_point_offset=churn_threshold_days)
        if frame.empty:
            return False
        self.target = "is_churn"
        self.features = frame.drop(columns=[self.target, "household_key"])
        object_columns = self.features.select_dtypes(include=["object"]).columns
        self.features = pd.get_dummies(self.features, columns=object_columns, drop_first=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features,
            frame[self.target],
            test_size=0.2,
            random_state=42,
            stratify=frame[self.target],
        )
        return True

    def train_model(self):
        if self.X_train is None or self.y_train is None:
            return False
        self.model.fit(self.X_train, self.y_train)
        return True

    def evaluate_model(self):
        if self.X_test is None or self.y_test is None:
            return None
        predictions = self.model.predict(self.X_test)
        return {
            "accuracy": accuracy_score(self.y_test, predictions),
            "report": classification_report(self.y_test, predictions),
        }

    def get_feature_importance(self):
        if not hasattr(self.model, "feature_importances_") or self.features is None:
            return None
        return pd.DataFrame({
            "feature": self.features.columns,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)

    def run_prediction_pipeline(self, churn_threshold_days=14):
        if not self.prepare_data(churn_threshold_days):
            return False
        self.train_model()
        self.evaluate_model()
        return True

    def predict_probabilities(self, customers_df):
        if not hasattr(self.model, "predict_proba"):
            return None
        model_features = self.model.get_booster().feature_names
        aligned = customers_df.reindex(columns=model_features, fill_value=0)
        return self.model.predict_proba(aligned)[:, 1]
