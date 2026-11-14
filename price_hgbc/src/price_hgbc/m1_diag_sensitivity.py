"""M1 diagnosis: is the group's HGB fit sensitive to last-bit floating-point differences?

Probe: refit with the SAME training rows in a different row order. The training set, labels,
params and the chronological train/valid split are unchanged; only the order in which float32
gradients are accumulated into histograms changes (ULP-level rounding). If this alone moves
hundreds of 2014 labels, then a different CPU/compiler/libm is *sufficient* to explain a 157-row gap.
It does not prove that a platform difference is the cause.

Diagnostic only: never used for model selection or reported scores.
Usage: python -m src.price_hgbc.m1_diag_sensitivity [order|ulp]
"""
import json
import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from .kit import DATA_CSV, KIT, RESULTS  # noqa: E402

if KIT not in sys.path:
    sys.path.insert(0, KIT)
from ablation import features as gf  # noqa: E402
from harness import dayahead as da  # noqa: E402

N_PERM = 10


def run():
    data = gf.load_study(DATA_CSV)
    X, truth = gf.build_features(data)
    days = da.market_day(data.index)
    tr = (days >= "2010-01-29") & (days < "2014-01-01")
    va = days.year == 2014
    scores = pd.read_csv(os.path.join(KIT, "ablation", "results", "scores.csv"))
    saved = pd.read_csv(os.path.join(KIT, "ablation", "results", "predictions", "boost_price.csv.gz"),
                        index_col="timestamp")
    y = truth.loc[va, "nsw_price_up"].to_numpy()
    rows = []
    for config in ("tuned_C", "tuned_CR"):
        r = scores[(scores.model == "boost_price") & (scores.config == config)].iloc[0]
        cols, param, thr = json.loads(r["columns"]), int(r["parameter"]), float(r["threshold"])
        Xtr, ytr, Xva = X.loc[tr, cols], truth.loc[tr, "nsw_price_up"], X.loc[va, cols]

        def fit(order):
            m = HistGradientBoostingClassifier(loss="log_loss", max_leaf_nodes=param, max_iter=100,
                                               learning_rate=0.1, early_stopping=False, random_state=0)
            m.fit(Xtr.iloc[order], ytr.iloc[order])
            return (m.predict_proba(Xva)[:, 1] >= thr).astype(int)

        base = fit(np.arange(len(Xtr)))
        s = saved[config].to_numpy()
        for k in range(N_PERM + 1):
            order = np.arange(len(Xtr)) if k == 0 else np.random.default_rng(1000 + k).permutation(len(Xtr))
            pred = base if k == 0 else fit(order)
            rows.append({"config": config, "perm": k, "BalAcc": da.balanced_accuracy(y, pred),
                         "mismatch_vs_time_order_fit": int((pred != base).sum()),
                         "mismatch_vs_saved": int((pred != s).sum())})
        print(config, "done", flush=True)
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "selfcheck", "m1_diag_sensitivity.csv")
    df.to_csv(out, index=False)
    summ = df[df.perm > 0].groupby("config").agg(
        BalAcc_min=("BalAcc", "min"), BalAcc_max=("BalAcc", "max"), BalAcc_sd=("BalAcc", "std"),
        mism_vs_base_median=("mismatch_vs_time_order_fit", "median"),
        mism_vs_base_max=("mismatch_vs_time_order_fit", "max"),
        any_exact_saved=("mismatch_vs_saved", lambda v: bool((v == 0).any())))
    print(df.to_string(index=False))
    print(summ.to_string())
    return df




# ---------------------------------------------------------------------------------------------
# Probe 2: nudge a seeded fraction of the float32 log-loss gradients by exactly one ULP each
# iteration (simulates a libm/compiler whose exp/expit disagrees in the last bit on some inputs).
# Result after the row-order probe showed 0 changes: float64 histogram sums absorb reordering.
# ---------------------------------------------------------------------------------------------
def run_ulp(fractions=(0.0001, 0.001, 0.01, 0.1), seeds=(1, 2, 3)):
    from sklearn._loss.loss import HalfBinomialLoss
    data = gf.load_study(DATA_CSV)
    X, truth = gf.build_features(data)
    days = da.market_day(data.index)
    tr = (days >= "2010-01-29") & (days < "2014-01-01")
    va = days.year == 2014
    scores = pd.read_csv(os.path.join(KIT, "ablation", "results", "scores.csv"))
    saved = pd.read_csv(os.path.join(KIT, "ablation", "results", "predictions", "boost_price.csv.gz"),
                        index_col="timestamp")
    y = truth.loc[va, "nsw_price_up"].to_numpy()
    original = HalfBinomialLoss.gradient_hessian
    rows = []
    try:
        for config in ("tuned_C", "tuned_CR"):
            r = scores[(scores.model == "boost_price") & (scores.config == config)].iloc[0]
            cols, param, thr = json.loads(r["columns"]), int(r["parameter"]), float(r["threshold"])
            Xtr, ytr, Xva = X.loc[tr, cols], truth.loc[tr, "nsw_price_up"], X.loc[va, cols]
            s = saved[config].to_numpy()

            def fit():
                m = HistGradientBoostingClassifier(loss="log_loss", max_leaf_nodes=param, max_iter=100,
                                                   learning_rate=0.1, early_stopping=False, random_state=0)
                m.fit(Xtr, ytr)
                return (m.predict_proba(Xva)[:, 1] >= thr).astype(int)

            HalfBinomialLoss.gradient_hessian = original
            base = fit()
            for frac in fractions:
                for seed in seeds:
                    rng = np.random.default_rng(seed)

                    def patched(self, *a, _rng=rng, _frac=frac, **k):
                        g, h = original(self, *a, **k)
                        idx = np.flatnonzero(_rng.random(g.shape[0]) < _frac)
                        g[idx] = np.nextafter(g[idx], np.where(_rng.random(idx.size) < 0.5,
                                                               np.inf, -np.inf).astype(g.dtype))
                        return g, h
                    HalfBinomialLoss.gradient_hessian = patched
                    pred = fit()
                    HalfBinomialLoss.gradient_hessian = original
                    rows.append({"config": config, "fraction_1ulp": frac, "seed": seed,
                                 "BalAcc": da.balanced_accuracy(y, pred),
                                 "BalAcc_minus_unperturbed": da.balanced_accuracy(y, pred) - da.balanced_accuracy(y, base),
                                 "mismatch_vs_unperturbed": int((pred != base).sum()),
                                 "mismatch_vs_saved": int((pred != s).sum())})
            print(config, "done", flush=True)
    finally:
        HalfBinomialLoss.gradient_hessian = original
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "selfcheck", "m1_diag_ulp.csv"), index=False)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "ulp":
    run_ulp()


if __name__ == "__main__" and (len(sys.argv) == 1 or sys.argv[1] == "order"):
    run()
