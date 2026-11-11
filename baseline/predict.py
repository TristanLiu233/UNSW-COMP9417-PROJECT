"""Predict with the four fitted Stage 1 models, in batch or for one market day."""

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ablation.models import SPECS
from src.common.features import build_features_for_day


def predict_models(bundle, inputs):
    """Apply each model to its ordered feature columns and return aligned forecasts."""
    forecasts = {}
    with threadpool_limits(limits=1):
        for name, spec in SPECS.items():
            fitted = bundle["models"][name]
            design = inputs.loc[:, fitted["columns"]]
            if not np.isfinite(design.to_numpy(dtype=float)).all():
                raise ValueError(f"{name} received a missing or non-finite feature")
            estimator = fitted["estimator"]
            if spec.task == "price":
                probabilities = estimator.predict_proba(design)[:, 1]
                if not np.isfinite(probabilities).all():
                    raise ValueError(f"{name} produced a non-finite probability")
                forecasts[name] = (probabilities >= fitted["threshold"]).astype(np.int8)
            else:
                predictions = estimator.predict(design)
                if not np.isfinite(predictions).all():
                    raise ValueError(f"{name} produced a non-finite demand forecast")
                forecasts[name] = predictions
    return pd.DataFrame(forecasts, index=inputs.index)


def predict_day(bundle, history, weather_next):
    """Forecast four columns for the 48 slots after the last complete history day."""
    return predict_models(bundle, build_features_for_day(history, weather_next))
