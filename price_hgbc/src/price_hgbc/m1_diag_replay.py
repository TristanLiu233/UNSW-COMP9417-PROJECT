"""M1 diagnosis: re-fit every saved ablation configuration with the group's OWN code and compare
with the group's saved predictions, model by model.

Why: if the saved files came from this code + data + library versions, the deterministic models
(Ridge, logistic) and the tree models should all match. The pattern of which ones match separates
"saved results do not come from this code" (everything drifts) from "only HistGradientBoosting
is not reproducible on this machine" (linear exact, HGB not).

Also replays the 2013 two-fold tuning table (tuning.csv) for boost_price and logistic.

The kit is only read: bytecode writing is disabled before importing ablation.*.
Usage: python -m src.price_hgbc.m1_diag_replay [--models boost_price logistic ridge boost_demand]
"""
import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .kit import DATA_CSV, KIT, RESULTS  # noqa: E402

if KIT not in sys.path:
    sys.path.insert(0, KIT)
from ablation import features as gf  # noqa: E402
from ablation import study as gs  # noqa: E402
from ablation.models import SPECS, predict_raw  # noqa: E402
from ablation.tuning import tune  # noqa: E402
from harness import dayahead as da  # noqa: E402

RES = os.path.join(KIT, "ablation", "results")
OUT = os.path.join(RESULTS, "selfcheck")


def replay_configs(models):
    data = gf.load_study(DATA_CSV)
    X, truth = gf.build_features(data)
    days = da.market_day(data.index)
    tr = (days >= "2010-01-29") & (days < "2014-01-01")
    va = days.year == 2014
    scores = pd.read_csv(os.path.join(RES, "scores.csv"))
    rows = []
    for model in models:
        task = SPECS[model].task
        target = "nsw_price_up" if task == "price" else "nsw_demand_mw"
        saved = pd.read_csv(os.path.join(RES, "predictions", f"{model}.csv.gz"),
                            parse_dates=["timestamp"], index_col="timestamp")
        assert saved.index.equals(X.index[va])
        cache = {}
        for _, r in scores[scores.model == model].iterrows():
            cols = tuple(json.loads(r["columns"]))
            param = r["parameter"]
            key = (cols, param)
            if key not in cache:
                t = time.perf_counter()
                cache[key] = predict_raw(model, param, X.loc[tr, list(cols)], truth.loc[tr, target],
                                         X.loc[va, list(cols)])
                fit_s = time.perf_counter() - t
            raw = cache[key]
            s = saved[r["config"]].to_numpy()
            row = {"model": model, "config": r["config"], "n_features": len(cols), "param": param,
                   "threshold": r["threshold"]}
            if task == "price":
                pred = (raw >= r["threshold"]).astype(int)
                y = truth.loc[va, target].to_numpy()
                row.update(n_label_mismatch=int((pred != s).sum()),
                           saved_BalAcc=float(r["BalAcc"]), replay_BalAcc=da.balanced_accuracy(y, pred))
                row["exact"] = row["n_label_mismatch"] == 0
            else:
                y = truth.loc[va, target].to_numpy()
                row.update(max_abs_diff_MW=float(np.max(np.abs(raw - s))),
                           saved_MAPE=float(r["MAPE"]), replay_MAPE=da.mape(y, raw))
                # saved CSV keeps full float repr; allow only representation-level noise
                row["exact"] = row["max_abs_diff_MW"] < 1e-6
            row["uses_day_type"] = any(c.startswith("prev_day_type") for c in cols)
            rows.append(row)
        print(f"{model}: {len(cache)} distinct fits", flush=True)
    return pd.DataFrame(rows)


def replay_tuning(models=("boost_price", "logistic")):
    data = gf.load_study(DATA_CSV)
    saved = pd.read_csv(os.path.join(RES, "tuning.csv"))
    scores = pd.read_csv(os.path.join(RES, "scores.csv"))
    folds = gs.temporal_folds(data, "price")
    rows = []
    for model in models:
        cols = json.loads(scores[(scores.model == model) & (scores.config == "C")]["columns"].iloc[0])
        _, _, got = tune(model, cols, folds)
        ref = saved[(saved.model == model) & (saved.source == "C")].reset_index(drop=True)
        for g, (_, s) in zip(got, ref.iterrows()):
            assert g["param"] == s["param"] and g["threshold"] == s["threshold"]
            rows.append({"model": model, "source": "C", "param": g["param"], "threshold": g["threshold"],
                         "saved_score": s["score"], "replay_score": g["score"],
                         "diff_pp": g["score"] - s["score"], "saved_chosen": bool(s["chosen"]),
                         "replay_chosen": bool(g["chosen"])})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["boost_price", "logistic", "ridge", "boost_demand"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cfg = replay_configs(args.models)
    cfg.to_csv(os.path.join(OUT, "m1_diag_replay_configs.csv"), index=False)
    tun = replay_tuning()
    tun.to_csv(os.path.join(OUT, "m1_diag_replay_tuning.csv"), index=False)
    summary = (cfg.groupby("model")
               .agg(configs=("config", "size"), exact=("exact", "sum"),
                    exact_without_day_type=("exact", lambda e: int(e[~cfg.loc[e.index, "uses_day_type"]].sum())),
                    configs_without_day_type=("uses_day_type", lambda u: int((~u).sum())))
               .reset_index())
    print(summary.to_string(index=False))
    print(tun.to_string(index=False))
    summary.to_csv(os.path.join(OUT, "m1_diag_replay_summary.csv"), index=False)


if __name__ == "__main__":
    main()
