"""Equal-budget tuning on expanding training folds before the 2014 selection year."""

import numpy as np
import pandas as pd

from harness import dayahead
from .models import SPECS, predict_raw


def tune(model_name, columns, fold_data):
    """Return the chosen parameter, probability threshold (None for demand), and all scores.

    Each fold is (X_train, y_train, X_valid, y_valid); its feature data must be
    constructed with transformations fitted using only that fold's training data.
    For a given parameter, concatenate both held-out predictions before scoring.
    """
    spec = SPECS[model_name]
    folds = []
    previous_end = None
    for train_x, train_y, valid_x, valid_y in fold_data:
        if not isinstance(train_x.index, pd.DatetimeIndex) or not isinstance(valid_x.index, pd.DatetimeIndex):
            raise ValueError("Tuning requires timestamp-indexed folds")
        if train_x.empty or valid_x.empty or train_x.index.max() >= valid_x.index.min():
            raise ValueError("Each validation fold must follow its training fold")
        # Midnight Jan 1 belongs to Dec 31 in the kit's end-of-interval convention.
        if dayahead.market_day(valid_x.index).max() >= pd.Timestamp("2014-01-01"):
            raise ValueError("Tuning folds may not include a 2014 market day")
        if previous_end is not None and valid_x.index.min() <= previous_end:
            raise ValueError("Tuning validation folds must be ordered and disjoint")
        for frame, labels in ((train_x, train_y), (valid_x, valid_y)):
            if len(labels) != len(frame) or (isinstance(labels, pd.Series) and not labels.index.equals(frame.index)):
                raise ValueError("Tuning label and feature rows must match")
        previous_end = valid_x.index.max()
        folds.append((train_x.loc[:, columns], train_y, valid_x.loc[:, columns], np.asarray(valid_y)))
    if not folds:
        raise ValueError("At least one temporal fold is required")

    price = spec.task == "price"
    metric_name = "BalAcc" if price else "MAPE"
    thresholds = (0.40, 0.45, 0.50, 0.55, 0.60) if price else (None,)
    truth = np.concatenate([valid_y for _, _, _, valid_y in folds])
    rows = []
    for param in spec.grid:
        predictions = np.concatenate([predict_raw(model_name, param, train_x, train_y, valid_x)
                                      for train_x, train_y, valid_x, _ in folds])
        if predictions.shape != truth.shape or not np.isfinite(predictions).all():
            raise ValueError("Invalid tuning predictions")
        for threshold in thresholds:
            if price:
                score = dayahead.balanced_accuracy(truth, predictions >= threshold)
            else:
                score = dayahead.mape(truth, predictions)
            rows.append({"param": param, "threshold": threshold, "primary_metric": metric_name,
                         "score": score, "chosen": False})

    def rank(row):
        loss = -row["score"] if price else row["score"]
        threshold_distance = abs(round(100 * row["threshold"]) - 50) if price else 0
        return loss, threshold_distance, row["param"] != spec.default

    best = min(rows, key=rank)
    best["chosen"] = True
    return best["param"], best["threshold"], rows
