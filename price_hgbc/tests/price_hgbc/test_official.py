"""M2.5 self-checks: the official reference is P1's boost_price, run with P1's own (imported) code.

Tolerances fixed in advance: hashes / features / predictions bitwise; documented values compared at their printed
precision (3 d.p., ±0.0005); b1 BalAcc compared with the stated 73.986 at 3 d.p.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import KIT, ROOT
from src.price_hgbc import official_ref as ofr
from src.price_hgbc.kit import da

ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
DOC_ORDER = ["slot_sin", "slot_cos", "season_sin", "season_cos", "weekend",
             "weekday_0", "weekday_1", "weekday_2", "weekday_3", "weekday_4", "weekday_5", "weekday_6",
             "nsw_demand_lag1", "nsw_price_lag1", "nsw_demand_lag7", "nsw_price_lag7",
             "prev_price_mean", "lag1_price_gap"]                  # guide §B2, transcribed for an independent check


@pytest.fixture(scope="module")
def ref():
    return json.load(open(ofr.REF_JSON))


@pytest.fixture(scope="module")
def built():
    return ofr.design()


@pytest.fixture(scope="module")
def preds():
    return pd.read_parquet(ofr.PREDS)


# ---------------- step 2: P1 code, imported ----------------
def test_p1_code_is_imported_unmodified_and_recorded(ref):
    gf, gm = ofr.p1()
    assert os.path.realpath(gf.__file__) == os.path.realpath(os.path.join(KIT, "ablation", "features.py"))
    assert os.path.realpath(gm.__file__) == os.path.realpath(os.path.join(KIT, "ablation", "models.py"))
    head = subprocess.run(["git", "-C", KIT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert ref["p1"]["commit"] == head == "116e2966310232894c71d9f7b73bc468a09eeee2"
    assert ref["p1"]["branch"] == "baseline" and ref["p1"]["ablation_clean_vs_HEAD"]
    assert subprocess.run(["git", "-C", KIT, "diff", "--quiet", "HEAD", "--", "ablation"]).returncode == 0
    for f, sha in ref["p1"]["files_sha256"].items():
        assert ofr.sha_file(os.path.join(KIT, f)) == sha, f
    st = subprocess.run(["git", "-C", KIT, "status", "--porcelain"], capture_output=True, text=True).stdout
    assert st.strip().splitlines() == ["?? data/"]                       # no __pycache__ or edits in the kit


def test_no_copy_of_p1_feature_code_in_src():
    """The official path imports P1; no local re-implementation of the 18-column recipe is used."""
    src = open(os.path.join(ROOT, "src", "price_hgbc", "official_ref.py"), encoding="utf-8").read()
    assert "from ablation import features as gf" in src and "gm.predict_raw(" in src
    assert "np.sin(" not in src and "shift(48" not in src


def test_boost_price_features_exactly_18_in_documented_order(ref):
    gf, _ = ofr.p1()
    assert list(gf.CORE) == ofr.doc_boost_price_features() == DOC_ORDER == ref["features"]
    assert len(ref["features"]) == ref["n_features"] == 18
    assert ref["p1_selection_row"]["columns"] == DOC_ORDER


# ---------------- step 3: reproduction ----------------
def test_rows_and_dates(ref, built):
    _, X, _, cols, tr, va = built
    assert (int(tr.sum()), int(va.sum())) == (68784, 17520)
    assert ref["train"] == {"first_day": "2010-01-29", "last_day": "2013-12-31", "rows": 68784}
    assert ref["valid"] == {"first_day": "2014-01-01", "last_day": "2014-12-31", "rows": 17520}
    assert X.index[va][0] == pd.Timestamp("2014-01-01 00:30") and X.index[va][-1] == pd.Timestamp("2015-01-01 00:00")


def test_params_and_tau_are_p1s(ref):
    p = ref["params"]
    assert (p["loss"], p["max_leaf_nodes"], p["max_iter"], p["learning_rate"]) == ("log_loss", 15, 100, 0.1)
    assert p["early_stopping"] is False and p["random_state"] == 17
    assert ref["tau"] == ref["p1_selection_row"]["threshold"] == 0.45


def test_p1_labels_equal_official_targets(built):
    data, X, truth, _, tr, va = built
    days = sorted(set(da.market_day(X.index[tr | va])))
    off = ofr.official_labels(data, days)
    assert off.index.equals(truth.index[tr | va])
    assert np.array_equal(off.to_numpy(), truth.loc[tr | va, "nsw_price_up"].to_numpy())


def test_metrics_recorded_and_doc_difference_reported(ref, preds):
    m = ofr.metrics(preds["y"], preds["pred"])
    for k in ("Acc", "BalAcc", "recall_0", "recall_1", "tp", "tn", "fp", "fn"):
        assert m[k] == ref["metrics_2014"][k], k
    assert abs(np.mean([m["recall_0"], m["recall_1"]]) - m["BalAcc"]) < 1e-9
    for k, doc in ofr.DOC.items():
        assert abs(ref["diff_vs_doc_pp"][k] - (m[k] - doc)) < 1e-12
    assert ref["matches_doc_3dp"] == all(abs(m[k] - ofr.DOC[k]) <= ofr.DOC_TOL for k in ofr.DOC)
    s = ref["p1_selection_row"]                                         # doc numbers = P1's saved selection row
    assert abs(s["saved_BalAcc"] - 79.300) < 5e-4 and abs(s["saved_Recall1"] - 82.275) < 5e-4


def test_two_fresh_process_refits_identical_and_equal_recorded(tmp_path, ref):
    outs = []
    for k in (1, 2):
        out = tmp_path / f"run{k}.json"
        subprocess.run([sys.executable, "-m", "src.price_hgbc.official_ref", "--fingerprint", str(out)], cwd=ROOT,
                       env=ENV, check=True, capture_output=True, timeout=900)
        outs.append(json.load(open(out)))
    for key in outs[0]:
        assert outs[0][key] == outs[1][key] == json.loads(json.dumps(ref[key], default=str)), key


# ---------------- step 4: official files ----------------
def test_official_files(ref, preds, built):
    _, X, truth, _, _, va = built
    assert list(preds.columns) == ["proba", "pred", "y"]
    assert preds.index.equals(X.index[va])
    assert np.array_equal(preds["y"].to_numpy(), truth.loc[va, "nsw_price_up"].to_numpy())
    assert np.array_equal(preds["pred"].to_numpy(), (preds["proba"] >= ref["tau"]).astype("int64").to_numpy())
    assert ofr.sha_arr(preds["proba"], "float64") == ref["hashes"]["proba_2014"]
    assert ofr.sha_arr(preds["pred"], "int64") == ref["hashes"]["pred_2014"]
    assert ofr.sha_file(ofr.PREDS) == ref["files"]["preds_2014.parquet"]
    assert ofr.sha_file(ofr.MODEL) == ref["files"]["boost_price_model.joblib"]
    for k in ("params", "features", "tau", "train", "valid", "metrics_2014", "hashes", "environment", "p1"):
        assert k in ref
    assert ref["environment"]["python"].startswith("3.11.") and ref["environment"]["versions"]["sklearn"] == "1.9.1"


def test_fresh_process_reload_predicts_bitwise():
    r = subprocess.run([sys.executable, "-m", "src.price_hgbc.official_ref", "--reload-check"], cwd=ROOT, env=ENV,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr[-2000:]
    assert all(json.loads(r.stdout.strip().splitlines()[-1]).values())


def test_saved_model_equals_p1_predict_raw(ref, preds, built):
    """The joblib model is the same fit as P1's predict_raw (its proba is what the parquet holds)."""
    import joblib
    _, X, _, cols, _, va = built
    assert np.array_equal(joblib.load(ofr.MODEL).predict_proba(X.loc[va, cols])[:, 1], preds["proba"].to_numpy())


def test_b1_balacc_via_official_baseline(built, ref):
    b = ofr.official_baselines(built[0])
    assert round(b["b1"]["BalAcc"], 3) == 73.986 and round(b["b7"]["BalAcc"], 3) == 74.282
    assert b == ref["baselines_2014_da"]


# ---------------- step 5: leakage ----------------
def test_shuffling_market_data_from_2014_06_15_leaves_that_days_features_unchanged(built, ref):
    import joblib
    data, X, _, cols, _, _ = built
    day = pd.Timestamp("2014-06-15")
    late = da.market_day(data.index) >= day
    alt = data.copy()
    rng = np.random.default_rng(615)
    for c in da.MARKET_COLS:
        alt.loc[late, c] = rng.permutation(alt.loc[late, c].to_numpy())
    rows = data.index[da.market_day(data.index) == day]
    assert len(rows) == 48 and not np.array_equal(alt.loc[rows, "nsw_price"], data.loc[rows, "nsw_price"])
    gf, _ = ofr.p1()
    X2, _ = gf.build_features(alt)
    a, b = X.loc[rows, cols], X2.loc[rows, cols]
    assert list(a.columns) == list(b.columns) == ref["features"] and a.dtypes.equals(b.dtypes)
    assert np.ascontiguousarray(a.to_numpy()).tobytes() == np.ascontiguousarray(b.to_numpy()).tobytes()
    model = joblib.load(ofr.MODEL)
    assert np.array_equal(model.predict_proba(a)[:, 1], model.predict_proba(b)[:, 1])
    assert not np.array_equal(X.loc[rows + pd.Timedelta("1D"), cols].to_numpy(),
                              X2.loc[rows + pd.Timedelta("1D"), cols].to_numpy())   # the next day does change
