"""M1 reproduction diagnosis: automated evidence (does NOT assert the 79.300 gate; see test_m1_gate.py)."""
import json
import os
import subprocess
import sys
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from conftest import KIT, ROOT

SELF = os.path.join(ROOT, "results", "price_hgbc", "selfcheck")
FP_SCRIPT = os.path.join(ROOT, "src", "price_hgbc", "env_fingerprint.py")
# SHA-256 of the group's saved benchmark files at commit 116e296 (must never change)
SAVED_SHA = {
    "ablation/results/predictions/boost_price.csv.gz": "d1804b3231ed26e9237b985ce0915e17f099e5b6480509e14011039e919aef93",
    "ablation/results/selection.csv": "27bf7a5d8b80b19541e3b05404d880ee2e2287d573156cf5d29c2833969326a9",
    "ablation/results/scores.csv": "b7faa24416cd838a7e8602940bcdb8320c17d3c90d355a0dfa453d616b1ee263",
    "ablation/results/tuning.csv": "261c070ac75cf71a67a0dce4709b21b79d93fd2e1f2b07fa33a4588e5cf62d30",
    "ablation/results/run.json": "e6b1ca3ab826f916c0ceea9f9c6fd3ce2228fef6473e7ed384f60034a670924c",
}
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")


def _fp_module():
    sys.dont_write_bytecode = True
    import importlib.util
    spec = importlib.util.spec_from_file_location("env_fingerprint", FP_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fp():
    return _fp_module().model_fingerprint(KIT, threads=1, random_state=17)


# ---- benchmark and kit integrity ----
def test_saved_group_results_unchanged():
    from importlib import import_module  # noqa: F401
    mod = _fp_module()
    for rel, want in SAVED_SHA.items():
        assert mod.sha_file(os.path.join(KIT, rel)) == want, rel


def test_kit_matches_git_head():
    diff = subprocess.run(["git", "-C", KIT, "diff", "--quiet", "HEAD"], capture_output=True)
    status = subprocess.run(["git", "-C", KIT, "status", "--porcelain"], capture_output=True, text=True)
    assert diff.returncode == 0
    assert status.stdout.strip().splitlines() == ["?? data/"]


# ---- inputs identical to the group's run ----
def test_data_prefix_matches_run_json(fp):
    assert fp["data"]["prefix_matches_run_json"]


def test_our_features_bitwise_equal_group_features(fp):
    from src.price_hgbc import features_simple as fs
    from src.price_hgbc.kit import DATA_CSV, da
    mod = _fp_module()
    df = da.load_dataset(DATA_CSV)
    X = fs.build_features(df)
    Xtr, ytr = fs.design(df, "2010-01-29", "2013-12-31", X)
    Xva, yva = fs.design(df, "2014-01-01", "2014-12-31", X)
    assert mod.sha_frame(Xtr) == fp["features"]["X_train_sha256"]
    assert mod.sha_frame(Xva) == fp["features"]["X_valid_sha256"]
    assert mod.sha(ytr.to_numpy().astype("int64").tobytes()) == fp["features"]["y_train_sha256"]
    assert mod.sha(yva.to_numpy().astype("int64").tobytes()) == fp["features"]["y_valid_sha256"]


def test_group_saved_2014_truth_and_baseline_equal_ours():
    from src.price_hgbc import features_simple as fs
    from src.price_hgbc.kit import DATA_CSV, da
    v = pd.read_csv(os.path.join(KIT, "ablation", "results", "validation.csv.gz"), index_col="timestamp")
    df = da.load_dataset(DATA_CSV)
    X = fs.build_features(df)
    _, y = fs.design(df, "2014-01-01", "2014-12-31", X)
    b1 = (X.loc[y.index, "nsw_price_lag1"] > X.loc[y.index, "prev_price_mean"]).astype(int)
    assert np.array_equal(v["nsw_price_up"].to_numpy(), y.to_numpy())
    assert np.array_equal(v["baseline1_price"].to_numpy(), b1.to_numpy())


def test_no_exact_price_ties_2010_2014():
    """Exact decimal arithmetic: no price equals the previous day's mean, so no summation order
    can flip a label (min nonzero |price - mean| = 1/4800)."""
    from src.price_hgbc.kit import DATA_CSV
    prices = pd.read_csv(DATA_CSV, usecols=["timestamp", "nsw_price"], dtype={"nsw_price": str},
                         nrows=len(pd.date_range("2010-01-01 00:30", "2015-01-01 00:00", freq="30min")))
    P = [Fraction(p) for p in prices["nsw_price"]]
    days = [P[i:i + 48] for i in range(0, len(P), 48)]
    ties = sum(1 for d in range(1, len(days)) for p in days[d] if p == sum(days[d - 1], Fraction(0)) / 48)
    assert ties == 0


# ---- model config equals the group's ----
def test_model_params_match_group_spec(fp):
    p = fp["model"]["params"]
    assert (p["loss"], p["max_leaf_nodes"], p["max_iter"], p["learning_rate"]) == ("log_loss", 15, 100, 0.1)
    assert p["early_stopping"] is False and p["max_features"] == 1.0 and p["max_bins"] == 255
    assert p["categorical_features"] == "from_dtype" and fp["model"]["is_categorical_"] is None
    assert fp["features"]["train_rows"] == 68784 and fp["features"]["train_index_monotonic"]


# ---- within-machine determinism ----
def test_random_state_has_no_effect():
    mod = _fp_module()
    a = mod.model_fingerprint(KIT, 1, 0)["model"]
    b = mod.model_fingerprint(KIT, 1, 17)["model"]
    assert a["proba_sha256"] == b["proba_sha256"] and a["trees"] == b["trees"]


def test_fresh_processes_and_thread_counts_identical(tmp_path):
    outs = []
    for i, threads in enumerate((1, 1, 8)):
        out = tmp_path / f"fp{i}.json"
        subprocess.run([sys.executable, FP_SCRIPT, "--kit", KIT, "--threads", str(threads), "--out", str(out)],
                       check=True, capture_output=True, env=ENV, timeout=600)
        outs.append(json.load(open(out))["model"])
    for o in outs[1:]:
        assert o["proba_sha256"] == outs[0]["proba_sha256"]
        assert o["trees"] == outs[0]["trees"]
        assert o["bin_thresholds_sha256"] == outs[0]["bin_thresholds_sha256"]


def test_fingerprint_compare_detects_tree_divergence(tmp_path):
    mod = _fp_module()
    a = {"model": {"trees": [{"structure": "x", "values": "1"}, {"structure": "y", "values": "2"}]}}
    b = {"model": {"trees": [{"structure": "x", "values": "1"}, {"structure": "z", "values": "2"}]}}
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a)); pb.write_text(json.dumps(b))
    d = mod.compare(str(pa), str(pb))
    assert d == ["model.trees: first differing tree = iteration 1 (fields ['structure'])"]


# ---- controls: deterministic models reproduce exactly here ----
@pytest.mark.parametrize("model", ["logistic", "ridge"])
def test_linear_tuned_c_reproduces_exactly(model):
    sys.dont_write_bytecode = True
    if KIT not in sys.path:
        sys.path.insert(0, KIT)
    from ablation import features as gf
    from ablation.models import predict_raw
    from harness import dayahead as da
    from src.price_hgbc.kit import DATA_CSV
    data = gf.load_study(DATA_CSV)
    X, truth = gf.build_features(data)
    days = da.market_day(data.index)
    tr, va = (days >= "2010-01-29") & (days < "2014-01-01"), days.year == 2014
    r = pd.read_csv(os.path.join(KIT, "ablation", "results", "scores.csv"))
    r = r[(r.model == model) & (r.config == "tuned_C")].iloc[0]
    cols = json.loads(r["columns"])
    target = "nsw_price_up" if model == "logistic" else "nsw_demand_mw"
    raw = predict_raw(model, r["parameter"], X.loc[tr, cols], truth.loc[tr, target], X.loc[va, cols])
    saved = pd.read_csv(os.path.join(KIT, "ablation", "results", "predictions", f"{model}.csv.gz"),
                        index_col="timestamp")["tuned_C"].to_numpy()
    if model == "logistic":
        assert np.array_equal((raw >= r["threshold"]).astype(int), saved)
    else:
        assert np.max(np.abs(raw - saved)) < 1e-6


# ---- slow: full replay and perturbation probes reproduce the recorded evidence ----
@pytest.mark.slow
def test_full_replay_matches_recorded_evidence():
    r = subprocess.run([sys.executable, "-m", "src.price_hgbc.m1_diag_replay"], cwd=ROOT, env=ENV,
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-2000:]
    s = pd.read_csv(os.path.join(SELF, "m1_diag_replay_summary.csv")).set_index("model")
    assert s.loc["logistic", "exact"] == s.loc["logistic", "configs"] == 85
    assert s.loc["ridge", "exact"] == s.loc["ridge", "configs"] == 83
    assert (s.loc["boost_price", "exact"], s.loc["boost_demand", "exact"]) == (28, 1)


@pytest.mark.slow
def test_hgb_invariant_to_row_order_and_1ulp_gradients():
    for arg, name in (("order", "m1_diag_sensitivity.csv"), ("ulp", "m1_diag_ulp.csv")):
        r = subprocess.run([sys.executable, "-m", "src.price_hgbc.m1_diag_sensitivity", arg], cwd=ROOT,
                           env=ENV, capture_output=True, text=True, timeout=1800)
        assert r.returncode == 0, r.stderr[-2000:]
        df = pd.read_csv(os.path.join(SELF, name))
        col = "mismatch_vs_time_order_fit" if arg == "order" else "mismatch_vs_unperturbed"
        assert (df[col] == 0).all()
