"""Sliding-window variant of P4's `logistic_price.final_training.fit_final` (design section 2.4).

Line-for-line the same as `fit_final` with its default plan (last calendar year as the single validation
fold, thresholds 0.25..0.60 step 0.01, pooled BalAcc, ties -> closest to 0.5 -> lower), except that the
training TARGETS are restricted to market days >= `start` (the tuning model: start <= day < validation start;
the final model: start <= day <= cutoff). Features are still built on the complete 2010-01-01 .. cutoff
prefix so lags and 28-day statistics have their full history.
With start = 2010-01-29 the result is bitwise identical to fit_final (tests/price_lr/test_fit_sliding.py).
"""
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from .env import ft, lu
from .env import lt


def fit_sliding(data, cutoff, start):
    prefix = ft.training_prefix(data, cutoff)
    cutoff = ft.market_date(cutoff)
    start = ft.market_date(start)
    config = json.loads(ft.SELECTION.read_text(encoding="utf-8"))["model"]
    base = ft.build_features(prefix)
    x = lt.transform_features(base, config["recipe"]).iloc[28 * 48:]
    y = (prefix.nsw_price > base.prev_price_mean).astype(int).reindex(x.index)
    if not np.isfinite(x.to_numpy()).all() or y.nunique() != 2:
        raise ValueError("Invalid training features or missing training class")
    days = (x.index - pd.Timedelta(minutes=30)).normalize()
    if not days[0] <= start <= cutoff:
        raise ValueError("start must lie within the eligible training days")
    window = days >= start
    vstart = cutoff - pd.DateOffset(years=1) + pd.Timedelta(days=1)
    thresholds = [i / 100 for i in range(25, 61)]
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("error", ConvergenceWarning)
        if not start < vstart:
            raise ValueError("window too short for a one-year validation fold")
        train, valid = window & (days < vstart), (days >= vstart) & (days <= cutoff)
        if y.loc[train].nunique() != 2 or not valid.any():
            raise ValueError("Insufficient fold data or missing training class")
        fold = {"train_start": str(days[train][0].date()), "train_end": str((vstart - pd.Timedelta(days=1)).date()),
                "validation_start": str(vstart.date()), "validation_end": str(cutoff.date()),
                "train_rows": int(train.sum()), "validation_rows": int(valid.sum())}
        fitted = lu.estimator(config["C"], config["class_weight"]).fit(x.loc[train], y.loc[train])
        truth = y.loc[valid].to_numpy()
        probability = fitted.predict_proba(x.loc[valid])[:, 1]
        search = [{"threshold": float(t), **lu.metrics(truth, probability >= t)} for t in sorted(set(thresholds))]
        best = min(search, key=lambda row: (-row["BalAcc"], round(abs(row["threshold"] - .5), 12), row["threshold"]))
        threshold = best["threshold"]
        pipeline = lu.estimator(config["C"], config["class_weight"]).fit(x.loc[window], y.loc[window])
    audit = {"training_policy": f"sliding_from_{start.date()}", "train_start": str(start.date()),
             "train_end": str(cutoff.date()), "train_rows": int(window.sum()), "recipe": config["recipe"],
             "C": config["C"], "class_weight": config["class_weight"], "threshold": threshold,
             "threshold_policy": "explicit_nested_temporal", "folds": [fold], "search": search, "random_state": 17,
             "selection_scores_are_not_outer_test_scores": True,
             "prefix_sha256": hashlib.sha256(pd.util.hash_pandas_object(prefix, index=True).values.tobytes()).hexdigest()}
    bundle = {"pipeline": pipeline, "columns": list(x.columns), "recipe": config["recipe"], "threshold": threshold,
              "train_end": audit["train_end"], "train_rows": int(window.sum()), "parameters": audit}
    return bundle, audit
