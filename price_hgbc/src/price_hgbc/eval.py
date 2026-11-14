"""Evaluation and decision-threshold search for the price-direction classifier.

* Metrics are the official ones: da.accuracy and da.balanced_accuracy (percent).
* Decision rule: predict 1 iff probability >= tau (the same rule the group's ablation uses).
* tau grid: 0.30, 0.31, ..., 0.70 (41 points), built from integers so every grid value is the
  correctly rounded double of k/100 (no accumulated float drift from np.arange(0.30, 0.71, 0.01)).
* Tie rule (fixed before any experiment): among taus whose BalAcc is within TIE_TOL_PP of the
  maximum, take the one closest to 0.50; if two are equally close, take the smaller one.
  Rationale: prefer the least deviation from the uninformed 0.5 cut-off; the tolerance only
  absorbs floating-point noise between identical confusion matrices (BalAcc values from
  different confusion matrices on >= 1000 rows differ by far more than 1e-9 pp).
"""
import numpy as np
import pandas as pd

from .kit import da

TAU_GRID = np.array([k / 100 for k in range(30, 71)])   # 41 points, 0.30 .. 0.70
TIE_TOL_PP = 1e-9                                        # percentage points
TAU_LO, TAU_HI = TAU_GRID[0], TAU_GRID[-1]


def predict_labels(proba, tau):
    """1 iff proba >= tau."""
    return (np.asarray(proba, dtype=float) >= tau).astype("int64")


def evaluate(y_true, y_pred):
    """Official Acc / BalAcc (percent) plus the counts needed to read them."""
    y, yhat = np.asarray(y_true, dtype="int64"), np.asarray(y_pred, dtype="int64")
    if y.shape != yhat.shape:
        raise ValueError("y_true and y_pred differ in length")
    return {"Acc": da.accuracy(y, yhat), "BalAcc": da.balanced_accuracy(y, yhat),
            "n": int(len(y)), "pos_rate_true": float(y.mean() * 100),
            "pos_rate_pred": float(yhat.mean() * 100),
            "tp": int(((y == 1) & (yhat == 1)).sum()), "tn": int(((y == 0) & (yhat == 0)).sum()),
            "fp": int(((y == 0) & (yhat == 1)).sum()), "fn": int(((y == 1) & (yhat == 0)).sum())}


def tau_curve(y_true, proba, grid=TAU_GRID):
    """Acc / BalAcc for every tau in the grid."""
    rows = []
    for tau in grid:
        m = evaluate(y_true, predict_labels(proba, tau))
        rows.append({"tau": float(tau), "BalAcc": m["BalAcc"], "Acc": m["Acc"],
                     "pos_rate_pred": m["pos_rate_pred"]})
    return pd.DataFrame(rows)


def select_tau(curve, metric="BalAcc", tol=TIE_TOL_PP):
    """Apply the fixed tie rule to a tau curve; returns (tau, tied_taus)."""
    vals = curve[metric].to_numpy(dtype=float)
    taus = curve["tau"].to_numpy(dtype=float)
    tied = taus[vals >= vals.max() - tol]
    # closest to 0.5 first, then the smaller tau; distances compared on the integer grid index
    order = sorted(tied, key=lambda t: (abs(round(t * 100) - 50), t))
    return float(order[0]), [float(t) for t in tied]


def tune_tau(y_true, proba, grid=TAU_GRID, metric="BalAcc"):
    """Best tau on (y_true, proba) by `metric` with the fixed tie rule.

    Returns a dict: tau, value of the metric at tau, the tied taus, whether tau is at a grid
    end point (to be reported as WARN), and the full curve (DataFrame, len(grid) rows).
    """
    curve = tau_curve(y_true, proba, grid)
    tau, tied = select_tau(curve, metric)
    at = curve.loc[np.isclose(curve["tau"], tau)].iloc[0]
    return {"tau": tau, metric: float(at[metric]), "Acc": float(at["Acc"]), "tied": tied,
            "at_endpoint": bool(np.isclose(tau, grid[0]) or np.isclose(tau, grid[-1])),
            "curve": curve}
