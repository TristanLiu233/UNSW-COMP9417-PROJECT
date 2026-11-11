"""Price-only interface for integration; not a full two-target submission."""
from pathlib import Path
import joblib
import pandas as pd
from .features import build_features_for_day
from .transforms import transform_features


class PriceClassifier:
    def __init__(self, artifact_dir):
        # Load only trusted artifacts produced by this project.
        bundle = joblib.load(Path(artifact_dir) / 'logistic.joblib')
        self.pipeline = bundle['pipeline']
        self.columns = bundle['columns']
        self.threshold = bundle['threshold']
        self.recipe = bundle.get('recipe', 'raw22')

    def predict(self, history, weather_next):
        x = transform_features(build_features_for_day(history, weather_next), self.recipe)[self.columns]
        probability = self.pipeline.predict_proba(x)[:, 1]
        return pd.DataFrame({'nsw_price_up': (probability >= self.threshold).astype(int)}, index=x.index)
