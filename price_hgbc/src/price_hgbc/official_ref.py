"""M2.5: official price-direction reference = P1's `boost_price`, using P1's own code unchanged.

P1 code (imported, never copied or edited):
  comp9417_project_kit/ablation/features.py  -> load_study, build_features, CORE (= BOOST_PRICE_FEATURES, 18 cols)
  comp9417_project_kit/ablation/models.py    -> predict_raw("boost_price", 15, ...)
  branch `baseline`, commit 116e296 (recorded at run time from git).
There is no src/common/features.py, and `train_four.py` is not a file in any branch: guide §D
(ablation/消融说明.md) tells readers to assemble it from the doc. The importable package above is the code that
produced the group's results, so it is used directly.

Protocol (fixed, from P1): train market days 2010-01-29 .. 2013-12-31 (68,784 rows), validate 2014 (17,520),
predict 1 iff p >= 0.45. Labels/metrics/baselines checked with harness/dayahead.py.

Usage:
  python -m src.price_hgbc.official_ref                       # fit, write results/price_hgbc/official/*
  python -m src.price_hgbc.official_ref --fingerprint f.json  # refit only, write a fingerprint (stability runs)
  python -m src.price_hgbc.official_ref --reload-check        # load saved model, rebuild features, compare
"""
import argparse
import contextlib
import hashlib
import json
import os
import platform
import re
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from .kit import DATA_CSV, KIT, RESULTS, da

OFFICIAL = os.path.join(RESULTS, "official")
REF_JSON = os.path.join(OFFICIAL, "boost_price_ref.json")
PREDS = os.path.join(OFFICIAL, "preds_2014.parquet")
MODEL = os.path.join(OFFICIAL, "boost_price_model.joblib")
TRAIN = ("2010-01-29", "2013-12-31")
VALID_YEAR = 2014
DOC = {"Acc": 79.218, "BalAcc": 79.300, "recall_0": 76.325, "recall_1": 82.275}      # 消融说明.md (3 d.p.)
DOC_TOL = 0.0005                                                                       # half a unit in the 3rd decimal


# ---------------- P1 import ----------------
def p1():
    sys.dont_write_bytecode = True               # never write __pycache__ into the kit
    if KIT not in sys.path:
        sys.path.insert(0, KIT)
    from ablation import features as gf
    from ablation import models as gm
    return gf, gm


def _git(*args):
    return subprocess.run(["git", "-C", KIT, *args], capture_output=True, text=True, check=True).stdout.strip()


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    return sha_bytes(open(p, "rb").read())


def provenance():
    files = ["ablation/features.py", "ablation/models.py", "ablation/消融说明.md", "ablation/results/selection.csv"]
    clean = subprocess.run(["git", "-C", KIT, "diff", "--quiet", "HEAD", "--", "ablation"]).returncode == 0
    return {"repo": "comp9417_project_kit", "remote": _git("remote", "get-url", "origin"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"), "commit": _git("rev-parse", "HEAD"),
            "files_last_commit": {f: _git("log", "-1", "--format=%H", "--", f) for f in files},
            "files_sha256": {f: sha_file(os.path.join(KIT, f)) for f in files},
            "ablation_clean_vs_HEAD": clean, "imported": ["ablation.features", "ablation.models"]}


def doc_boost_price_features():
    """BOOST_PRICE_FEATURES as written in guide §B2 (parsed from the committed markdown)."""
    text = open(os.path.join(KIT, "ablation", "消融说明.md"), encoding="utf-8").read()
    block = re.search(r"BOOST_PRICE_FEATURES = \[(.*?)\]", text, re.S).group(1)
    return re.findall(r'"([^"]+)"', block)


def p1_selection():
    s = pd.read_csv(os.path.join(KIT, "ablation", "results", "selection.csv"))
    r = s[(s["model"] == "boost_price")].iloc[0]
    return {"config": r["config"], "columns": json.loads(r["columns"]), "parameter": int(r["parameter"]),
            "threshold": float(r["threshold"]), "train_rows": int(r["train_rows"]), "valid_rows": int(r["validation_rows"]),
            "saved_BalAcc": float(r["BalAcc"]), "saved_Acc": float(r["Acc"]),
            "saved_Recall0": float(r["Recall0"]), "saved_Recall1": float(r["Recall1"])}


@contextlib.contextmanager
def capture_p1_estimator(gm):
    """Record the estimator P1's predict_raw constructs (its parameters come from P1's code, not from here)."""
    built = []
    original = gm.HistGradientBoostingClassifier

    class Recorder(original):
        def fit(self, X, y, *a, **k):
            built.append(self.get_params())
            return super().fit(X, y, *a, **k)
    gm.HistGradientBoostingClassifier = Recorder
    try:
        yield built
    finally:
        gm.HistGradientBoostingClassifier = original


# ---------------- data ----------------
def design(data=None):
    gf, _ = p1()
    data = gf.load_study(DATA_CSV) if data is None else data
    X, truth = gf.build_features(data)                       # P1 default: day types fitted on <= 2013 only
    cols = list(gf.CORE)
    days = da.market_day(X.index)
    tr = (days >= pd.Timestamp(TRAIN[0])) & (days <= pd.Timestamp(TRAIN[1]))
    va = days.year == VALID_YEAR
    return data, X, truth, cols, tr, va


def official_labels(data, days):
    """nsw_price_up for each day straight from da.targets (one call per day on a two-day slice)."""
    out = []
    for d in days:
        block = data.loc[d - pd.Timedelta("1D") + pd.Timedelta("30min"): d + pd.Timedelta("1D")]
        out.append(da.targets(block, d)["nsw_price_up"])
    return pd.concat(out)


def sha_frame(X):
    h = hashlib.sha256()
    h.update(json.dumps([[c, str(t)] for c, t in X.dtypes.items()]).encode())
    h.update(np.asarray(X.index.asi8).tobytes())
    h.update(np.ascontiguousarray(X.to_numpy(dtype="float64")).tobytes())
    return h.hexdigest()


def sha_arr(a, dtype):
    return sha_bytes(np.ascontiguousarray(np.asarray(a, dtype=dtype)).tobytes())


def environment():
    from threadpoolctl import threadpool_info
    return {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                         "sklearn": sklearn.__version__, "joblib": joblib.__version__},
            "threadpools": [{k: t.get(k) for k in ("internal_api", "num_threads")} for t in threadpool_info()]}


def metrics(y, pred):
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    return {"Acc": da.accuracy(y, pred), "BalAcc": da.balanced_accuracy(y, pred),
            "recall_0": float(np.mean(pred[y == 0] == 0) * 100), "recall_1": float(np.mean(pred[y == 1] == 1) * 100),
            "tn": int(((y == 0) & (pred == 0)).sum()), "fp": int(((y == 0) & (pred == 1)).sum()),
            "fn": int(((y == 1) & (pred == 0)).sum()), "tp": int(((y == 1) & (pred == 1)).sum()), "n": int(len(y))}


def tree_fingerprint(model):
    trees = [sha_bytes(p.nodes.tobytes()) for it in model._predictors for p in it]
    return {"n_iter": int(model.n_iter_), "trees_sha256": sha_bytes("".join(trees).encode())}


# ---------------- run ----------------
def run():
    gf, gm = p1()
    sel = p1_selection()
    data, X, truth, cols, tr, va = design()
    Xtr, Xva = X.loc[tr, cols], X.loc[va, cols]
    ytr, yva = truth.loc[tr, "nsw_price_up"], truth.loc[va, "nsw_price_up"]
    assert (len(Xtr), len(Xva)) == (sel["train_rows"], sel["valid_rows"]) == (68784, 17520)
    assert np.isfinite(Xtr.to_numpy(float)).all() and np.isfinite(Xva.to_numpy(float)).all()
    tau = sel["threshold"]
    with capture_p1_estimator(gm) as built:
        proba = gm.predict_raw("boost_price", sel["parameter"], Xtr, ytr, Xva)          # P1's own fit
    assert len(built) == 1
    params = built[0]
    model = HistGradientBoostingClassifier(**params).fit(Xtr, ytr)                     # same params, plain class
    assert np.array_equal(model.predict_proba(Xva)[:, 1], proba), "saved model differs from P1 predict_raw"
    pred = (proba >= tau).astype("int64")
    m = metrics(yva, pred)
    info = {
        "model": "P1 boost_price (HistGradientBoostingClassifier)", "params": params,
        "features": cols, "n_features": len(cols), "tau": tau, "decision_rule": "1 iff P(up) >= tau",
        "train": {"first_day": str(da.market_day(Xtr.index).min().date()), "last_day": str(da.market_day(Xtr.index).max().date()),
                  "rows": int(len(Xtr))},
        "valid": {"first_day": str(da.market_day(Xva.index).min().date()), "last_day": str(da.market_day(Xva.index).max().date()),
                  "rows": int(len(Xva))},
        "metrics_2014": m,
        "doc_2014": DOC, "diff_vs_doc_pp": {k: m[k] - DOC[k] for k in DOC},
        "matches_doc_3dp": all(abs(m[k] - DOC[k]) <= DOC_TOL for k in DOC),
        "formal_basis": ("this environment's result is the formal baseline (user rule, M2.5 step 3): it differs from "
                         "the documented values, nothing was adjusted, and stability is checked by fresh-process re-runs"),
        "hashes": {"X_train": sha_frame(Xtr), "y_train": sha_arr(ytr, "int64"), "X_valid": sha_frame(Xva),
                   "y_valid": sha_arr(yva, "int64"), "proba_2014": sha_arr(proba, "float64"),
                   "pred_2014": sha_arr(pred, "int64")},
        "trees": tree_fingerprint(model),
        "p1": provenance(), "p1_selection_row": sel, "environment": environment(),
    }
    return info, {"model": model, "proba": proba, "pred": pred, "y": yva, "X": X, "truth": truth, "data": data,
                  "cols": cols, "Xva": Xva, "Xtr": Xtr, "ytr": ytr}


def write(info, arr):
    os.makedirs(OFFICIAL, exist_ok=True)
    joblib.dump(arr["model"], MODEL)
    pd.DataFrame({"proba": arr["proba"], "pred": arr["pred"], "y": arr["y"].to_numpy().astype("int64")},
                 index=arr["Xva"].index).to_parquet(PREDS)
    info["files"] = {"preds_2014.parquet": sha_file(PREDS), "boost_price_model.joblib": sha_file(MODEL)}
    b = official_baselines(arr["data"])
    info["baselines_2014_da"] = b
    json.dump(info, open(REF_JSON, "w"), indent=1, default=str)


def official_baselines(data):
    """b1 / b7 via da.make_inputs / da.baseline / da.targets for every 2014 target day."""
    days = [d for d in da.complete_days(data) if d.year == VALID_YEAR]
    truth, b1, b7 = [], [], []
    for d in days:
        h, w = da.make_inputs(data, d)
        truth.append(da.targets(data, d)["nsw_price_up"])
        b1.append(da.baseline(h, w, 1)["nsw_price_up"])
        b7.append(da.baseline(h, w, 7)["nsw_price_up"])
    y = pd.concat(truth)
    return {k: {"BalAcc": da.balanced_accuracy(y, pd.concat(v)), "Acc": da.accuracy(y, pd.concat(v))}
            for k, v in (("b1", b1), ("b7", b7))}


def reload_check():
    """Fresh process: load the saved model, rebuild P1 features, predict 2014, compare with the saved outputs."""
    ref = json.load(open(REF_JSON))
    model = joblib.load(MODEL)
    _, X, _, cols, _, va = design()
    proba = model.predict_proba(X.loc[va, cols])[:, 1]
    pred = (proba >= ref["tau"]).astype("int64")
    saved = pd.read_parquet(PREDS)
    ok = {"proba_equals_parquet": bool(np.array_equal(proba, saved["proba"].to_numpy())),
          "pred_equals_parquet": bool(np.array_equal(pred, saved["pred"].to_numpy())),
          "proba_hash_equals_json": sha_arr(proba, "float64") == ref["hashes"]["proba_2014"],
          "pred_hash_equals_json": sha_arr(pred, "int64") == ref["hashes"]["pred_2014"],
          "index_equal": bool(saved.index.equals(X.loc[va].index))}
    print(json.dumps(ok))
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fingerprint")
    ap.add_argument("--reload-check", action="store_true")
    a = ap.parse_args(argv)
    if a.reload_check:
        ok = reload_check()
        raise SystemExit(0 if all(ok.values()) else 1)
    info, arr = run()
    if a.fingerprint:
        keep = {k: info[k] for k in ("params", "features", "tau", "train", "valid", "metrics_2014", "hashes", "trees",
                                     "environment")}
        json.dump(keep, open(a.fingerprint, "w"), indent=1, default=str)
    else:
        write(info, arr)
    m = info["metrics_2014"]
    print(f"BalAcc {m['BalAcc']:.5f} Acc {m['Acc']:.5f} recall0 {m['recall_0']:.5f} recall1 {m['recall_1']:.5f} "
          f"| doc 79.300/79.218/76.325/82.275 | proba {info['hashes']['proba_2014'][:12]}")


if __name__ == "__main__":
    main()
