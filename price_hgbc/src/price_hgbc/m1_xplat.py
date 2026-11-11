"""M1 cross-platform experiment: one run of the group's own HGB fits in the current environment.

Run once per (environment, repetition):
    python -m src.price_hgbc.m1_xplat --env macos-arm64 --run 1
Writes results/price_hgbc/selfcheck/xplat/xplat_<env>_run<k>.json and nothing else.

Each config is fitted with the group's own `ablation.models.predict_raw` (random_state=17 as the
group used; the seed is inert here, recorded once with 0 for tuned_C). The HGB estimator classes
imported into `ablation.models` are swapped, in this process only, for subclasses that remember
the fitted instance, so tree hashes come from the exact model that produced the predictions.
No kit file is touched (bytecode writing disabled; the kit can be mounted read-only).
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .kit import DATA_CSV, KIT, RESULTS  # noqa: E402

if KIT not in sys.path:
    sys.path.insert(0, KIT)
from ablation import features as gf  # noqa: E402
from ablation import models as gm  # noqa: E402
from harness import dayahead as da  # noqa: E402

OUT_DIR = os.path.join(RESULTS, "selfcheck", "xplat")
# representative configs: (model, config, reproduced exactly on macOS-arm64 earlier?)
CONFIGS = [
    ("boost_price", "tuned_C", False), ("boost_price", "tuned_CR", True),
    ("boost_price", "C", False), ("boost_price", "CR", True),
    ("boost_price", "CW", False), ("boost_price", "tuned_CW", True),
    ("boost_price", "F", False), ("boost_price", "tuned_F", True),
    ("boost_demand", "drop_F_qld_price", True), ("boost_demand", "C", False),
    ("boost_demand", "tuned_C", False),
]

_captured = []


class _CapC(gm.HistGradientBoostingClassifier):
    def fit(self, *a, **k):
        _captured.append(self)
        return super().fit(*a, **k)


class _CapR(gm.HistGradientBoostingRegressor):
    def fit(self, *a, **k):
        _captured.append(self)
        return super().fit(*a, **k)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def sha_frame(df):
    h = hashlib.sha256()
    h.update(json.dumps([[c, str(t)] for c, t in df.dtypes.items()]).encode())
    h.update(np.ascontiguousarray(df.to_numpy(dtype="float64")).tobytes())
    return h.hexdigest()


def tree_hashes(model):
    out = []
    for it in model._predictors:
        nodes = it[0].nodes
        out.append({"structure": sha(nodes[["feature_idx", "bin_threshold", "left", "right",
                                             "is_leaf"]].tobytes())[:16],
                    "values": sha(nodes["value"].tobytes())[:16]})
    return out


def env_block():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import env_fingerprint as ef  # same folder; self-contained helper
    env = ef.env_info()
    flags = None
    if sys.platform.startswith("linux"):
        try:
            for line in open("/proc/cpuinfo"):
                if line.lower().startswith(("flags", "features")):
                    flags = line.split(":", 1)[1].split()
                    break
        except OSError:
            pass
    env["cpu_flags_simd"] = sorted(f for f in (flags or []) if any(
        s in f for s in ("sse", "avx", "fma", "neon", "asimd", "sve", "f16c", "bmi")))
    env["uname"] = list(platform.uname())
    return env, ef.kernel_probes()


def run(env_label, rep):
    t0 = time.perf_counter()
    gm.HistGradientBoostingClassifier, gm.HistGradientBoostingRegressor = _CapC, _CapR
    env, kernels = env_block()
    data = gf.load_study(DATA_CSV)
    X, truth = gf.build_features(data)
    days = da.market_day(data.index)
    tr = (days >= "2010-01-29") & (days < "2014-01-01")
    va = days.year == 2014
    scores = pd.read_csv(os.path.join(KIT, "ablation", "results", "scores.csv"))
    used = sorted({c for m, cfg, _ in CONFIGS
                   for c in json.loads(scores[(scores.model == m) & (scores.config == cfg)]["columns"].iloc[0])})
    inputs = {"prefix_rows": len(data), "train_rows": int(tr.sum()), "valid_rows": int(va.sum()),
              "X_used_cols": used,
              "X_train_sha256": sha_frame(X.loc[tr, used]), "X_valid_sha256": sha_frame(X.loc[va, used]),
              "X_core18_train_sha256": sha_frame(X.loc[tr, gf.CORE]),
              "y_price_sha256": sha(truth["nsw_price_up"].to_numpy().astype("int64").tobytes()),
              "y_demand_sha256": sha(truth["nsw_demand_mw"].to_numpy().astype("float64").tobytes())}
    saved = {m: pd.read_csv(os.path.join(KIT, "ablation", "results", "predictions", f"{m}.csv.gz"),
                            index_col="timestamp") for m in ("boost_price", "boost_demand")}
    rows = []
    for model, config, mac_exact in CONFIGS:
        r = scores[(scores.model == model) & (scores.config == config)].iloc[0]
        cols, param = json.loads(r["columns"]), r["parameter"]
        task = gm.SPECS[model].task
        target = "nsw_price_up" if task == "price" else "nsw_demand_mw"
        _captured.clear()
        t = time.perf_counter()
        raw = gm.predict_raw(model, param, X.loc[tr, cols], truth.loc[tr, target], X.loc[va, cols])
        fit_s = time.perf_counter() - t
        assert len(_captured) == 1
        est = _captured[0]
        y = truth.loc[va, target].to_numpy()
        s = saved[model][config].to_numpy()
        row = {"model": model, "config": config, "mac_exact_before": mac_exact, "n_features": len(cols),
               "param": float(param), "fit_seconds": round(fit_s, 2),
               "params": {k: v for k, v in est.get_params().items()
                          if isinstance(v, (int, float, str, bool, type(None)))},
               "n_iter": int(est.n_iter_), "trees": tree_hashes(est),
               "bin_thresholds_sha256": sha(b"".join(np.asarray(b).tobytes()
                                                     for b in est._bin_mapper.bin_thresholds_)),
               "raw_sha256": sha(np.asarray(raw, dtype="float64").tobytes())}
        if task == "price":
            thr = float(r["threshold"])
            pred = (raw >= thr).astype(int)
            row.update(threshold=thr, BalAcc=da.balanced_accuracy(y, pred), Acc=da.accuracy(y, pred),
                       pred_sha256=sha(pred.astype("int64").tobytes()),
                       mismatch_vs_saved=int((pred != s).sum()),
                       saved_BalAcc=float(r["BalAcc"]), saved_Acc=float(r["Acc"]))
        else:
            diff = np.abs(raw - s)
            row.update(MAPE=da.mape(y, raw), MAE=da.mae(y, raw), pred_sha256=row["raw_sha256"],
                       mismatch_vs_saved=int((diff > 1e-6).sum()), max_abs_diff_vs_saved=float(diff.max()),
                       saved_MAPE=float(r["MAPE"]), saved_MAE=float(r["MAE"]))
        rows.append(row)
        print(f"{model:12s} {config:18s} fit {fit_s:6.1f}s  mismatch_vs_saved {row['mismatch_vs_saved']}", flush=True)
    # our rule: random_state=0 for tuned_C must give the same model as the group's 17
    r = scores[(scores.model == "boost_price") & (scores.config == "tuned_C")].iloc[0]
    cols = json.loads(r["columns"])
    m0 = gm.HistGradientBoostingClassifier.__mro__[1](
        loss="log_loss", max_leaf_nodes=15, max_iter=100, learning_rate=0.1,
        early_stopping=False, random_state=0).fit(X.loc[tr, cols], truth.loc[tr, "nsw_price_up"])
    p0 = m0.predict_proba(X.loc[va, cols])[:, 1]
    tuned_c = next(x for x in rows if x["config"] == "tuned_C" and x["model"] == "boost_price")
    rs0 = {"raw_sha256": sha(p0.astype("float64").tobytes()), "trees_equal_rs17": tree_hashes(m0) == tuned_c["trees"]}
    rs0["raw_equal_rs17"] = rs0["raw_sha256"] == tuned_c["raw_sha256"]
    out = {"env_label": env_label, "run": rep, "env": env, "kernels": kernels, "inputs": inputs,
           "configs": rows, "tuned_C_random_state_0": rs0,
           "seconds": round(time.perf_counter() - t0, 1)}
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"xplat_{env_label}_run{rep}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote", path, "in", out["seconds"], "s")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--run", type=int, required=True)
    a = ap.parse_args()
    run(a.env, a.run)
