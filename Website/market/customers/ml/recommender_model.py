# customers/ml/recommender_model.py
import pickle
import logging
from pathlib import Path

import pandas as pd
from django.utils import timezone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from .feature_engineering import FEATURE_COLUMNS

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "ml_models_cache" / "hybrid_recommender"
ARTIFACT_VERSION = 1
logger = logging.getLogger(__name__)


class HybridRecommenderModel:
    """
    Predicts P(household purchases this candidate product soon), trained
    on association/CF candidates labeled from held-out future transactions.
    """

    def __init__(self):
        self.pipeline = None
        self.metrics = {}
        self.popularity_map = {}
        self.cycle_map = {}
        self.trained_at = None

    @staticmethod
    def _artifact_path():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        return MODEL_DIR / f"hybrid_recommender_v{ARTIFACT_VERSION}.pkl"

    def train(self, train_df, test_df=None):
        X_train, y_train = train_df[FEATURE_COLUMNS], train_df["label"]
        self.pipeline = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.06, max_depth=6, random_state=42,
        )
        self.pipeline.fit(X_train, y_train)
        self.trained_at = timezone.now()

        if test_df is not None and len(test_df):
            X_test, y_test = test_df[FEATURE_COLUMNS], test_df["label"]
            proba = self.pipeline.predict_proba(X_test)[:, 1]
            self.metrics = {
                "roc_auc": round(float(roc_auc_score(y_test, proba)), 4) if y_test.nunique() > 1 else None,
                "pr_auc": round(float(average_precision_score(y_test, proba)), 4) if y_test.nunique() > 1 else None,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "positive_rate_train": round(float(y_train.mean()), 4),
                "positive_rate_test": round(float(y_test.mean()), 4),
                "trained_at": self.trained_at.isoformat(),
            }
        return self.metrics

    def predict_scores(self, features_df):
        if self.pipeline is None:
            raise ValueError("Model not trained/loaded.")
        proba = self.pipeline.predict_proba(features_df[FEATURE_COLUMNS])[:, 1]
        return pd.Series(proba, index=features_df.index)

    def save(self):
        artifact = {
            "artifact_version": ARTIFACT_VERSION,
            "pipeline": self.pipeline,
            "metrics": self.metrics,
            "popularity_map": self.popularity_map,
            "cycle_map": self.cycle_map,
            "trained_at": self.trained_at,
        }
        with self._artifact_path().open("wb") as f:
            pickle.dump(artifact, f)

    @classmethod
    def load(cls):
        path = cls._artifact_path()
        if not path.exists():
            return None
        try:
            with path.open("rb") as f:
                artifact = pickle.load(f)
        except (AttributeError, ImportError, ModuleNotFoundError, pickle.UnpicklingError) as exc:
            # scikit-learn estimators are not portable across every package
            # version. Keep recommendation pages available with their rule/CF
            # fallback until this machine retrains a compatible artifact.
            logger.warning(
                "Ignoring incompatible hybrid recommender artifact at %s: %s. "
                "Run `python manage.py train_hybrid_recommender` to rebuild it.",
                path,
                exc,
            )
            return None
        if artifact.get("artifact_version") != ARTIFACT_VERSION:
            return None
        model = cls()
        model.pipeline = artifact["pipeline"]
        model.metrics = artifact["metrics"]
        model.popularity_map = artifact["popularity_map"]
        model.cycle_map = artifact["cycle_map"]
        model.trained_at = artifact.get("trained_at")
        return model
