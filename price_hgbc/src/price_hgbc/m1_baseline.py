"""M1: current HGBC baseline for nsw_price_up with the M1 feature set and a 2013-tuned tau.

Protocol (no 2014 information in any choice):
  1. candidate model: fit on target days 2010-01-01 .. 2012-12-31 (first 7 days dropped: no lag-7);
  2. tau: tune_tau on target days 2013 with the candidate's probabilities (grid 0.30..0.70);
  3. final model: refit the same configuration on 2010-01-01 .. 2013-12-31;
  4. one evaluation on target days 2014 with the fixed tau; baselines b1 / b7 via da.baseline.
Steps 1-3 receive a frame that physically ends with market day 2013-12-31 (`before_2014`), so they
cannot read any 2014 price, demand or label.

There was no compliant 2013 tuning for this feature set to reuse: the group's 2013 tuning
(ablation/tuning.py) is for its own 18/22-column recipes, so tau is tuned here.

Model configuration: sklearn 1.9.1 HistGradientBoostingClassifier defaults, written out in full,
with the project rules early_stopping=False, random_state=0 and native categorical handling of
period / weekday. Nothing in it is tuned (no hyper-parameter search in M1).

Usage:
  python -m src.price_hgbc.m1_baseline                # full run, writes results/price_hgbc/m1*
  python -m src.price_hgbc.m1_baseline --out f.json   # fingerprint only (repeatability checks)
  python -m src.price_hgbc.m1_baseline --random-state 1 --out f.json   # sensitivity only
"""
import argparse
import hashlib
import json
import os
import platform
import sys

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from . import eval as ev
from . import features_simple as fs
from .kit import DATA_CSV, RESULTS, da

CUTOFF = pd.Timestamp("2014-01-01")
SPLITS = {"candidate_train": ("2010-01-01", "2012-12-31"), "tau_tune": ("2013-01-01", "2013-12-31"),
          "final_train": ("2010-01-01", "2013-12-31"), "test": ("2014-01-01", "2014-12-31")}


def hgbc_params(random_state=0):
    return dict(loss="log_loss", learning_rate=0.1, max_iter=100, max_leaf_nodes=31, max_depth=None,
                min_samples_leaf=20, l2_regularization=0.0, max_features=1.0, max_bins=255,
                categorical_features=list(fs.M1_CATEGORICAL), monotonic_cst=None,
                interaction_cst=None, warm_start=False, early_stopping=False, scoring="loss",
                validation_fraction=0.1, n_iter_no_change=10, tol=1e-7, verbose=0,
                random_state=random_state, class_weight=None)


# ---------- hashing ----------
def sha(b):
    return hashlib.sha256(b).hexdigest()


def sha_frame(X):
    h = hashlib.sha256()
    h.update(json.dumps([[c, str(t)] for c, t in X.dtypes.items()]).encode())
    h.update(np.asarray(X.index.asi8).tobytes())
    h.update(np.ascontiguousarray(X.to_numpy(dtype="float64")).tobytes())
    return h.hexdigest()


def sha_array(a, dtype):
    return sha(np.ascontiguousarray(np.asarray(a, dtype=dtype)).tobytes())


def model_fingerprint(model):
    trees = [sha(p.nodes.tobytes()) for it in model._predictors for p in it]
    bins = sha(b"".join(np.asarray(t, dtype="float64").tobytes() for t in model._bin_mapper.bin_thresholds_))
    return {"n_iter": int(model.n_iter_), "trees_sha256": sha("".join(trees).encode()),
            "bin_thresholds_sha256": bins,
            "is_categorical": [fs.M1_FEATURES[i] for i in np.flatnonzero(model.is_categorical_)]
            if model.is_categorical_ is not None else []}


def environment():
    from threadpoolctl import threadpool_info
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
            "executable": sys.executable,
            "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                         "sklearn": sklearn.__version__},
            "threadpools": [{k: t.get(k) for k in ("internal_api", "version", "num_threads")}
                            for t in threadpool_info()]}


# ---------- stages ----------
def before_2014(df):
    """Rows of market days < 2014-01-01 only (the stamp 2014-01-01 00:00 still belongs to 2013-12-31)."""
    return df[da.market_day(df.index) < CUTOFF]


def _fit(X, y, random_state):
    return HistGradientBoostingClassifier(**hgbc_params(random_state)).fit(X, y)


def select_tau(df_pre, random_state=0):
    """Steps 1-2 on a frame that ends before 2014."""
    assert da.market_day(df_pre.index).max() < CUTOFF
    X = fs.build_m1_features(df_pre)
    Xtr, ytr, dropped = fs.m1_design(df_pre, *SPLITS["candidate_train"], X)
    Xtu, ytu, dropped_tu = fs.m1_design(df_pre, *SPLITS["tau_tune"], X)
    cand = _fit(Xtr, ytr, random_state)
    p_tu = cand.predict_proba(Xtu)[:, 1]
    res = ev.tune_tau(ytu, p_tu)
    return {"tau": res, "candidate": cand, "Xtr": Xtr, "ytr": ytr, "Xtu": Xtu, "ytu": ytu,
            "p_tu": p_tu, "dropped_train": dropped, "dropped_tune": dropped_tu}


def fit_final(df_pre, random_state=0):
    """Step 3 on a frame that ends before 2014."""
    assert da.market_day(df_pre.index).max() < CUTOFF
    X, y, dropped = fs.m1_design(df_pre, *SPLITS["final_train"])
    return _fit(X, y, random_state), X, y, dropped


def test_frame(df):
    """Market days up to 2014-12-31 (2014 targets only need days <= 2014-12-31)."""
    return df[da.market_day(df.index) <= pd.Timestamp(SPLITS["test"][1])]


def run(df=None, random_state=0):
    df = da.load_dataset(DATA_CSV) if df is None else df
    pre = before_2014(df)
    sel = select_tau(pre, random_state)
    tau = sel["tau"]["tau"]
    final, Xf, yf, dropped_f = fit_final(pre, random_state)
    dte = test_frame(df)
    Xte, yte, dropped_te = fs.m1_design(dte, *SPLITS["test"])
    p_te = final.predict_proba(Xte)[:, 1]
    pred = ev.predict_labels(p_te, tau)
    m = ev.evaluate(yte, pred)
    m05 = ev.evaluate(yte, ev.predict_labels(p_te, 0.5))
    days = lambda X: da.market_day(X.index)  # noqa: E731
    info = {
        "model": "HistGradientBoostingClassifier", "params": hgbc_params(random_state),
        "fitted_params": {k: (v if not isinstance(v, (np.generic,)) else v.item())
                          for k, v in final.get_params().items()},
        "features": list(Xf.columns), "categorical_features": list(fs.M1_CATEGORICAL),
        "splits": SPLITS,
        "dates": {name: [str(days(X).min().date()), str(days(X).max().date()), int(days(X).nunique()), int(len(X))]
                  for name, X in (("candidate_train", sel["Xtr"]), ("tau_tune", sel["Xtu"]),
                                  ("final_train", Xf), ("test", Xte))},
        "dropped_days": {"candidate_train": [str(d.date()) for d in sel["dropped_train"]],
                         "tau_tune": [str(d.date()) for d in sel["dropped_tune"]],
                         "final_train": [str(d.date()) for d in dropped_f],
                         "test": [str(d.date()) for d in dropped_te]},
        "tau": tau, "tau_tied": sel["tau"]["tied"], "tau_at_endpoint": sel["tau"]["at_endpoint"],
        "tau_tune_BalAcc_2013": sel["tau"]["BalAcc"], "tau_tune_Acc_2013": sel["tau"]["Acc"],
        "test_2014": m, "test_2014_tau_0.50_reference": m05,
        "hashes": {"X_candidate_train": sha_frame(sel["Xtr"]), "y_candidate_train": sha_array(sel["ytr"], "int64"),
                   "X_tau_tune": sha_frame(sel["Xtu"]), "y_tau_tune": sha_array(sel["ytu"], "int64"),
                   "proba_tau_tune": sha_array(sel["p_tu"], "float64"),
                   "X_final_train": sha_frame(Xf), "y_final_train": sha_array(yf, "int64"),
                   "X_test": sha_frame(Xte), "y_test": sha_array(yte, "int64"),
                   "proba_test": sha_array(p_te, "float64"), "pred_test": sha_array(pred, "int64")},
        "candidate_model": model_fingerprint(sel["candidate"]), "final_model": model_fingerprint(final),
        "environment": environment(),
    }
    arrays = {"curve": sel["tau"]["curve"], "Xte": Xte, "yte": yte, "p_te": p_te, "pred": pred,
              "final": final}
    return info, arrays


def baselines_2014(df, index):
    """b1 / b7 on the same 2014 rows, straight from da.baseline / da.targets."""
    from .m0_baselines import baseline_predictions
    _, truth, b1, b7 = baseline_predictions(df, 2014)
    assert truth.index.equals(index)
    return truth, b1, b7


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="write only the run fingerprint JSON to this path")
    ap.add_argument("--random-state", type=int, default=0)
    a = ap.parse_args(argv)
    df = da.load_dataset(DATA_CSV)
    info, arr = run(df, a.random_state)
    if a.out:
        json.dump(info, open(a.out, "w"), indent=1, default=str)
        print(f"tau {info['tau']:.2f}  2014 BalAcc {info['test_2014']['BalAcc']:.5f}  "
              f"Acc {info['test_2014']['Acc']:.5f}  proba {info['hashes']['proba_test'][:12]}")
        return info
    if a.random_state != 0:
        raise SystemExit("formal results use random_state=0; use --out for sensitivity runs")
    truth, b1, b7 = baselines_2014(df, arr["Xte"].index)
    assert np.array_equal(truth["nsw_price_up"].to_numpy(), arr["yte"].to_numpy())
    out_dir = os.path.join(RESULTS, "m1")
    os.makedirs(out_dir, exist_ok=True)
    te = info["dates"]["test"]
    rows = [{"model": "M1_HGBC", "tau": info["tau"], **{k: info["test_2014"][k] for k in ("Acc", "BalAcc", "n")}}]
    rows.append({"model": "M1_HGBC_tau0.50_reference", "tau": 0.5,
                 **{k: info["test_2014_tau_0.50_reference"][k] for k in ("Acc", "BalAcc", "n")}})
    for name, b in (("b1", b1), ("b7", b7)):
        mb = ev.evaluate(truth["nsw_price_up"], b["nsw_price_up"])
        rows.append({"model": name, "tau": np.nan, **{k: mb[k] for k in ("Acc", "BalAcc", "n")}})
    tab = pd.DataFrame(rows)
    tab["test_first_day"], tab["test_last_day"], tab["test_days"] = te[0], te[1], te[2]
    tab["train_first_day"], tab["train_last_day"] = info["dates"]["final_train"][:2]
    tab["tau_tuned_on"] = "2013 (candidate fit 2010-2012)"
    tab.to_csv(os.path.join(RESULTS, "m1.csv"), index=False, float_format="%.6f")
    arr["curve"].to_csv(os.path.join(out_dir, "m1_tau_curve_2013.csv"), index=False, float_format="%.6f")
    pd.DataFrame({"y": arr["yte"].to_numpy(), "proba": arr["p_te"], "pred": arr["pred"],
                  "b1": b1["nsw_price_up"].to_numpy(), "b7": b7["nsw_price_up"].to_numpy()},
                 index=arr["Xte"].index).to_csv(os.path.join(out_dir, "m1_pred_2014.csv.gz"),
                                                float_format="%.17g")
    json.dump(info, open(os.path.join(out_dir, "m1_run.json"), "w"), indent=1, default=str)
    print(tab.to_string(index=False))
    print("saved", os.path.join(RESULTS, "m1.csv"), "and", out_dir)
    return info


if __name__ == "__main__":
    main()
