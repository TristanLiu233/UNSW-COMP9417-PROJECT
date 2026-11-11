"""[ARCHIVED in M2.5: M1 is a non-formal sensitivity analysis; run with -m archive]
M1 self-checks (tasks 1-5): M1 features, eval/tune_tau, 2013-tuned baseline, repeatability,
historical comparison. Tolerances are fixed here, before looking at any M1 result:
  * features, labels, hashes, probabilities: bitwise / exact;
  * metric cross-checks against sklearn: 1e-9 percentage points (float re-association only).
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from conftest import KIT, ROOT
from src.price_hgbc import eval as ev
from src.price_hgbc import features_simple as fs
from src.price_hgbc import m1_baseline as mb
from src.price_hgbc.common import add_day_period
from src.price_hgbc.kit import ARCHIVE, DATA_CSV, RESULTS, da

pytestmark = pytest.mark.archive
SELF = os.path.join(RESULTS, "selfcheck")
M1_DIR = os.path.join(ARCHIVE, "m1")            # M2.5: moved to the archive (bytes unchanged)
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
D1 = pd.Timedelta("1D")
HALF = pd.Timedelta("30min")
MARKET = ["nsw_price", "nsw_demand"]


@pytest.fixture(scope="module")
def df():
    return da.load_dataset(DATA_CSV)


@pytest.fixture(scope="module")
def X(df):
    return fs.build_m1_features(df)


@pytest.fixture(scope="module")
def m1(df):
    return mb.run(df, random_state=0)


def _rows(index, day):
    return index[(index > day) & (index <= day + D1)]


def _same(a, b):
    """Bitwise equality of two frames: columns, dtypes, index and value bytes."""
    return (list(a.columns) == list(b.columns) and a.dtypes.equals(b.dtypes) and a.index.equals(b.index)
            and np.ascontiguousarray(a.to_numpy()).tobytes() == np.ascontiguousarray(b.to_numpy()).tobytes())


SAMPLE_DAYS = [pd.Timestamp(d) for d in ("2010-01-08", "2011-06-15", "2013-03-03", "2013-12-31",
                                         "2014-01-01", "2014-03-04", "2014-12-31", "2024-05-09", "2026-10-31")]


# =========================== task 1: features ===========================
def test_m1_columns_and_order():
    assert fs.M1_FEATURES == (["up_lag1", "up_lag7"] + list(da.WEATHER_COLS)
                              + ["period", "weekday", "price_gap_lag1", "price_gap_lag7"])
    assert len(fs.M1_FEATURES) == 18 and len(set(fs.M1_FEATURES)) == 18


def test_group_recipe_untouched():
    """The historical 18-col recipe keeps its own column list and has no categorical features."""
    assert fs.BOOST_PRICE_FEATURES[0] == "slot_sin" and len(fs.BOOST_PRICE_FEATURES) == 18
    assert fs.CATEGORICAL_FEATURES == []


@pytest.mark.parametrize("day", SAMPLE_DAYS)
@pytest.mark.parametrize("mode", ["permute", "random"])
def test_leakage_next_day_price_demand_do_not_change_features(df, X, day, mode):
    """Change the TRUE D+1 nsw_price and nsw_demand; D+1 feature rows must be bitwise unchanged
    (labels are allowed to change). Checked for the bulk builder and the prediction-time path."""
    rng = np.random.default_rng(0)
    rows = _rows(df.index, day)
    assert len(rows) == 48
    alt = df.copy()
    for c in MARKET:
        v = alt.loc[rows, c].to_numpy()
        alt.loc[rows, c] = rng.permutation(v) if mode == "permute" else rng.normal(1e3, 1e3, len(v))
    assert not np.array_equal(alt.loc[rows, "nsw_price"].to_numpy(), df.loc[rows, "nsw_price"].to_numpy())
    Xalt = fs.build_m1_features(alt)
    assert _same(Xalt.loc[rows], X.loc[rows])
    h0, w0 = da.make_inputs(df, day)
    h1, w1 = da.make_inputs(alt, day)
    assert _same(fs.m1_features_for_day(h1, w1), X.loc[rows])
    assert _same(fs.m1_features_for_day(h0, w0), X.loc[rows])


@pytest.mark.parametrize("day", SAMPLE_DAYS)
def test_leakage_all_next_day_market_columns(df, X, day):
    """Stronger variant: all six D+1 market columns (NSW/QLD/VIC demand and price) set to NaN."""
    rows = _rows(df.index, day)
    alt = df.copy()
    alt.loc[rows, da.MARKET_COLS] = np.nan
    assert _same(fs.build_m1_features(alt).loc[rows], X.loc[rows])


def test_prediction_path_equals_bulk_features(df, X):
    rng = np.random.default_rng(1)
    days = [d for d in da.complete_days(df) if d >= pd.Timestamp("2010-01-08")]
    for day in [days[i] for i in rng.choice(len(days), 25, replace=False)]:
        h, w = da.make_inputs(df, day)
        assert _same(fs.m1_features_for_day(h, w), X.loc[w.index]), day


def test_up_lag1_consistent_with_price_above_prev_mean(df, X):
    """P(D,t) > P̄(D)  <=>  up_lag1 == 1, for every target row 2010-01-08 .. 2026-10-31.

    P(D,t) by timestamp arithmetic (ts - 1 day), P̄(D) by Series.mean over the day's 48 rows
    (the reduction dayahead.py uses), both independent of build_m1_features' shift/groupby."""
    p = df["nsw_price"]
    days = da.complete_days(df)
    mean = pd.Series({d: p.loc[d + HALF: d + D1].mean() for d in days})
    Xs = X.loc[da.market_day(X.index) >= pd.Timestamp("2010-01-08")]
    target_day = da.market_day(Xs.index)
    p_d = p.reindex(Xs.index - D1).to_numpy()
    p_d6 = p.reindex(Xs.index - 7 * D1).to_numpy()
    pbar = mean.reindex(target_day - D1).to_numpy()
    assert np.isfinite(p_d).all() and np.isfinite(pbar).all()
    assert int(((p_d > pbar).astype(int) != Xs["up_lag1"].to_numpy()).sum()) == 0
    assert int(((p_d6 > pbar).astype(int) != Xs["up_lag7"].to_numpy()).sum()) == 0
    assert np.array_equal(Xs["price_gap_lag1"].to_numpy(), p_d - pbar)
    assert np.array_equal(Xs["price_gap_lag7"].to_numpy(), p_d6 - pbar)
    assert np.array_equal((Xs["price_gap_lag1"] > 0).astype(int), Xs["up_lag1"])


@pytest.mark.parametrize("day", SAMPLE_DAYS)
def test_up_lags_equal_official_baselines(df, X, day):
    h, w = da.make_inputs(df, day)
    assert np.array_equal(X.loc[w.index, "up_lag1"].to_numpy(), da.baseline(h, w, 1)["nsw_price_up"].to_numpy())
    assert np.array_equal(X.loc[w.index, "up_lag7"].to_numpy(), da.baseline(h, w, 7)["nsw_price_up"].to_numpy())


def test_labels_match_da_targets_on_5_random_days(df, X):
    rng = np.random.default_rng(9417)
    days = fs.day_range(df, "2010-01-08", "2026-10-31")
    picked = [days[i] for i in rng.choice(len(days), 5, replace=False)]
    for d in picked:
        _, y, _ = fs.m1_design(df, d, d, X)
        t = da.targets(df, d)["nsw_price_up"]
        assert y.index.equals(t.index) and np.array_equal(y.to_numpy(), t.to_numpy()), d
        assert y.index[0] == d + HALF and y.index[-1] == d + D1      # day boundary: 00:30 .. next 00:00


def test_calendar_and_weather_columns(df, X):
    ref = add_day_period(df)
    assert np.array_equal(X["period"].to_numpy(), ref["period"].to_numpy().astype("int64"))
    assert np.array_equal(X["weekday"].to_numpy(), pd.DatetimeIndex(ref["day"]).day_of_week.to_numpy())
    assert X["period"].between(0, 47).all() and X["weekday"].between(0, 6).all()
    assert _same(X[da.WEATHER_COLS], df[da.WEATHER_COLS].astype("float64"))   # weather of the target row


def test_categorical_features_only_expected(m1):
    info, arr = m1
    assert fs.M1_CATEGORICAL == ["period", "weekday"]
    assert info["params"]["categorical_features"] == ["period", "weekday"]
    assert info["final_model"]["is_categorical"] == ["period", "weekday"]
    assert info["candidate_model"]["is_categorical"] == ["period", "weekday"]
    for c in ("period", "weekday"):
        assert str(arr["Xte"][c].dtype) == "int64"


def test_no_nan_inf_and_dropped_days_recorded(df, X):
    Xs, y, dropped = fs.m1_design(df, "2010-01-01", "2026-10-31", X)
    assert np.isfinite(Xs.to_numpy(dtype=float)).all()
    assert dropped == list(pd.date_range("2010-01-01", "2010-01-07"))      # no D-6 lag yet
    assert len(Xs) == (len(da.complete_days(df)) - 7) * 48 == len(y)
    assert set(np.unique(y)) == {0, 1}


# =========================== task 2: eval / tune_tau ===========================
def test_tau_grid_has_41_points():
    assert len(ev.TAU_GRID) == 41
    assert ev.TAU_GRID[0] == 0.30 and ev.TAU_GRID[-1] == 0.70
    assert all(ev.TAU_GRID[k] == float(f"0.{30 + k:02d}") for k in range(41))
    assert len(ev.tune_tau(np.array([0, 1]), np.array([0.2, 0.8]))["curve"]) == 41


def test_evaluate_matches_official_and_sklearn():
    rng = np.random.default_rng(3)
    for p1 in (0.1, 0.5, 0.9):
        y = (rng.random(5000) < p1).astype(int)
        yhat = (rng.random(5000) < 0.5).astype(int)
        m = ev.evaluate(y, yhat)
        assert m["BalAcc"] == da.balanced_accuracy(y, yhat) and m["Acc"] == da.accuracy(y, yhat)
        assert abs(m["BalAcc"] - balanced_accuracy_score(y, yhat) * 100) < 1e-9
        assert abs(m["Acc"] - accuracy_score(y, yhat) * 100) < 1e-9
        assert m["tp"] + m["tn"] + m["fp"] + m["fn"] == 5000


def test_evaluate_on_2014_baseline_equals_m0(df):
    from src.price_hgbc.m0_baselines import baseline_predictions
    _, truth, b1, _ = baseline_predictions(df, 2014)
    m = ev.evaluate(truth["nsw_price_up"], b1["nsw_price_up"])
    m0 = pd.read_csv(os.path.join(RESULTS, "m0_baselines_2014.csv"), index_col=0)
    assert abs(m["BalAcc"] - m0.loc["b1", "BalAcc"]) < 1e-5 and abs(m["Acc"] - m0.loc["b1", "Acc"]) < 1e-5


def test_tau_05_equals_direct_threshold_prediction(m1):
    """eval at tau=0.5 == the fitted model's own predict() (argmax, i.e. p > 0.5)."""
    _, arr = m1
    p = arr["p_te"]
    assert int((p == 0.5).sum()) == 0                      # no exact ties at the cut-off
    direct = arr["final"].predict(arr["Xte"])
    assert np.array_equal(ev.predict_labels(p, 0.5), direct)
    assert ev.evaluate(arr["yte"], ev.predict_labels(p, 0.5)) == ev.evaluate(arr["yte"], direct)


def test_tune_tau_finds_040_on_constructed_data():
    # positives at 0.405, negatives at 0.395 (+ easy points): only tau = 0.40 separates them
    y = np.r_[np.ones(60), np.zeros(60), np.ones(20), np.zeros(20)].astype(int)
    p = np.r_[np.full(60, 0.405), np.full(60, 0.395), np.full(20, 0.9), np.full(20, 0.1)]
    r = ev.tune_tau(y, p)
    assert r["tau"] == 0.40 and r["tied"] == [0.40] and r["BalAcc"] == 100.0
    c = r["curve"].set_index("tau")["BalAcc"]
    assert c.loc[0.39] < 100 and c.loc[0.41] < 100


def test_tie_rule_closest_to_half_then_smaller():
    def curve(vals):
        return pd.DataFrame({"tau": ev.TAU_GRID, "BalAcc": vals, "Acc": vals})
    v = np.full(41, 70.0)
    v[[2, 5, 9]] = 80.0                                    # 0.32, 0.35, 0.39 tied
    assert ev.select_tau(curve(v)) == (0.39, [0.32, 0.35, 0.39])
    v = np.full(41, 70.0)
    v[[15, 25]] = 80.0                                     # 0.45 and 0.55: equally close -> smaller
    assert ev.select_tau(curve(v))[0] == 0.45
    v = np.full(41, 70.0)
    v[[0, 40]] = 80.0 + np.array([0, 5e-10])               # within TIE_TOL -> tie -> 0.30
    assert ev.select_tau(curve(v))[0] == 0.30
    # plateau on real predictions: identical confusion matrices for 0.45..0.55 -> 0.50
    y = np.array([0] * 50 + [1] * 50)
    p = np.r_[np.full(50, 0.2), np.full(50, 0.8)]
    assert ev.tune_tau(y, p)["tau"] == 0.50
    assert [ev.tune_tau(y, p)["tau"] for _ in range(3)] == [0.50] * 3


def test_endpoint_flag():
    y = np.array([0] * 50 + [1] * 50)
    p = np.r_[np.full(50, 0.295), np.full(50, 0.305)]      # only 0.30 separates them
    r = ev.tune_tau(y, p)
    assert r["tau"] == 0.30 and r["at_endpoint"]


# =========================== task 3: M1 baseline protocol ===========================
def test_splits_are_chronological(m1):
    info, _ = m1
    d = {k: (pd.Timestamp(v[0]), pd.Timestamp(v[1])) for k, v in info["dates"].items()}
    assert d["candidate_train"][1] < pd.Timestamp("2013-01-01") <= d["tau_tune"][0]
    assert d["tau_tune"] == (pd.Timestamp("2013-01-01"), pd.Timestamp("2013-12-31"))
    assert d["final_train"][1] < pd.Timestamp("2014-01-01")                       # final train max day < 2014
    assert d["test"] == (pd.Timestamp("2014-01-01"), pd.Timestamp("2014-12-31"))
    assert info["dates"]["test"][2:] == [365, 17520]
    assert info["dates"]["tau_tune"][2:] == [365, 17520]
    assert info["dates"]["final_train"][2:] == [1454, 1454 * 48]
    assert info["dropped_days"]["test"] == [] and info["dropped_days"]["tau_tune"] == []


def test_selection_stages_cannot_see_2014(df, m1):
    """Replace every market value of market days >= 2014-01-01 (prices, demands -> labels) with noise:
    tau, the tuning probabilities and the final model must be bitwise identical."""
    info, _ = m1
    alt = df.copy()
    late = da.market_day(alt.index) >= mb.CUTOFF
    rng = np.random.default_rng(5)
    for c in da.MARKET_COLS:
        alt.loc[late, c] = rng.normal(100, 300, int(late.sum()))
    pre = mb.before_2014(alt)
    assert da.market_day(pre.index).max() == pd.Timestamp("2013-12-31")
    assert pre.index.max() == pd.Timestamp("2014-01-01 00:00")
    sel = mb.select_tau(pre)
    final, Xf, yf, _ = mb.fit_final(pre)
    assert sel["tau"]["tau"] == info["tau"]
    assert mb.sha_array(sel["p_tu"], "float64") == info["hashes"]["proba_tau_tune"]
    assert mb.model_fingerprint(final) == info["final_model"]
    assert mb.sha_frame(Xf) == info["hashes"]["X_final_train"]


def test_m1_outputs_saved_and_consistent(m1):
    info, arr = m1
    tab = pd.read_csv(os.path.join(ARCHIVE, "m1.csv")).set_index("model")
    assert {"M1_HGBC", "b1", "b7"} <= set(tab.index)
    for c in ("Acc", "BalAcc", "tau"):
        assert c in tab.columns
    assert tab.loc["M1_HGBC", "tau"] == info["tau"]
    assert abs(tab.loc["M1_HGBC", "BalAcc"] - info["test_2014"]["BalAcc"]) < 1e-5
    assert abs(tab.loc["M1_HGBC", "Acc"] - info["test_2014"]["Acc"]) < 1e-5
    m0 = pd.read_csv(os.path.join(RESULTS, "m0_baselines_2014.csv"), index_col=0)
    for b in ("b1", "b7"):
        assert abs(tab.loc[b, "BalAcc"] - m0.loc[b, "BalAcc"]) < 1e-5
    saved = json.load(open(os.path.join(M1_DIR, "m1_run.json")))
    assert saved["hashes"] == info["hashes"] and saved["final_model"] == info["final_model"]
    pred = pd.read_csv(os.path.join(M1_DIR, "m1_pred_2014.csv.gz"), index_col=0, parse_dates=True)
    assert np.array_equal(pred["pred"].to_numpy(), ev.predict_labels(pred["proba"], info["tau"]))
    assert np.array_equal(pred["proba"].to_numpy(), arr["p_te"])          # %.17g round-trips float64
    assert np.array_equal(pred["b1"].to_numpy(), arr["Xte"]["up_lag1"].to_numpy())
    assert np.array_equal(pred["b7"].to_numpy(), arr["Xte"]["up_lag7"].to_numpy())
    curve = pd.read_csv(os.path.join(M1_DIR, "m1_tau_curve_2013.csv"))
    assert len(curve) == 41


def test_fixed_model_config(m1):
    info, _ = m1
    p = info["params"]
    assert p["early_stopping"] is False and p["random_state"] == 0
    assert info["fitted_params"]["early_stopping"] is False and info["fitted_params"]["random_state"] == 0
    assert info["final_model"]["n_iter"] == p["max_iter"] == 100
    assert info["environment"]["python"].startswith("3.11.")
    assert info["environment"]["versions"]["sklearn"] == "1.9.1"


# =========================== task 4: repeatability ===========================
def test_in_process_rerun_bitwise_identical(df, m1):
    a, _ = m1
    b, _ = mb.run(df, random_state=0)
    assert a["hashes"] == b["hashes"]
    assert a["final_model"] == b["final_model"] and a["candidate_model"] == b["candidate_model"]
    assert a["params"] == b["params"] and a["tau"] == b["tau"]


@pytest.mark.slow
def test_two_fresh_processes_bitwise_identical_and_equal_recorded(tmp_path):
    outs = []
    for k in (1, 2):
        out = tmp_path / f"run{k}.json"
        subprocess.run([sys.executable, "-m", "src.price_hgbc.m1_baseline", "--out", str(out)],
                       cwd=ROOT, env=ENV, check=True, capture_output=True, timeout=900)
        outs.append(json.load(open(out)))
    rec = json.load(open(os.path.join(M1_DIR, "m1_run.json")))
    for key in ("hashes", "final_model", "candidate_model", "params", "tau", "dates", "features"):
        assert outs[0][key] == outs[1][key] == rec[key], key


# =========================== task 5: historical comparison ===========================
def test_historical_evidence_kept():
    gate = open(os.path.join(ROOT, "tests", "price_hgbc", "test_m1_gate.py")).read()
    assert "assert abs(bal - GROUP_BALACC) <= 0.01" in gate and "@pytest.mark.historical_repro" in gate
    blocked = open(os.path.join(SELF, "M1_repro_blocked.md")).read()
    assert "157 / 17,520" in blocked
    from src.price_hgbc.m1_repro_check import GROUP_BALACC
    assert GROUP_BALACC == 79.29988384512427


def test_m1_predictions_are_not_the_historical_ones(m1):
    _, arr = m1
    saved = pd.read_csv(os.path.join(KIT, "ablation", "results", "predictions", "boost_price.csv.gz"),
                        parse_dates=["timestamp"], index_col="timestamp")["tuned_C"]
    assert saved.index.equals(arr["Xte"].index)
    assert not np.array_equal(saved.to_numpy(), arr["pred"])
