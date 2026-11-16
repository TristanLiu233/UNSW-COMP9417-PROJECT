"""M5 self-checks: Stage 1 freeze (P1 boost_price refit to 2014), causal day-by-day interface, 2015-2026 batch.
Batch vs day-by-day must be bitwise (FAIL otherwise, never WARN). Metric re-computations exact or 1e-9 pp.
"""
import json
import os
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
import pytest

from conftest import ROOT
from src.price_hgbc import features_simple as fs
from src.price_hgbc import m5
from src.price_hgbc import official_ref as ofr
from src.price_hgbc.kit import DATA_CSV, da

ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
OFFICIAL = {"boost_price_ref.json": "06b022774a10dad545dacb9a73d910a73eccd37d1ebc14d3f605d6e9ff210725",
            "preds_2014.parquet": "b4c9890d0149d69c902dcc10685db82c7b2d19d253a57a72ce0483a1f93c0c29",
            "boost_price_model.joblib": "9206389cfbebe6436550875273c172d5557ddb7e781252c06899281a38b128bf"}
D1 = pd.Timedelta("1D")


@pytest.fixture(scope="module")
def df():
    return da.load_dataset(DATA_CSV)


@pytest.fixture(scope="module")
def model():
    return joblib.load(m5.P["model"])


@pytest.fixture(scope="module")
def ref():
    return json.load(open(ofr.REF_JSON))


@pytest.fixture(scope="module")
def preds():
    return pd.read_parquet(m5.P["preds"])


@pytest.fixture(scope="module")
def trained():
    return m5.train()


def _rows(index, day):
    day = pd.Timestamp(day)
    return index[(index > day) & (index <= day + D1)]


# ---------------- guards ----------------
def test_ref_and_m4_untouched():
    for f, want in OFFICIAL.items():
        assert ofr.sha_file(os.path.join(ofr.OFFICIAL, f)) == want, f
    cus = json.load(open(m5.CUSUM_CONFIG))
    assert cus["ref_update"] == "A"
    st = subprocess.run(["git", "-C", ofr.KIT, "status", "--porcelain"], capture_output=True, text=True).stdout
    assert st.strip().splitlines() == ["?? data/"]


# ---------------- task 1: model ----------------
def test_model_params_equal_ref(model, ref):
    cfg = json.load(open(m5.P["config"]))
    assert model.get_params() == ref["params"] == cfg["params"]
    assert ref["params"]["early_stopping"] is False and ref["params"]["random_state"] == 17
    assert list(model.feature_names_in_) == ref["features"] == cfg["features"] and len(ref["features"]) == 18
    assert cfg["tau"] == ref["tau"] == 0.45
    src = open(os.path.join(ROOT, "src", "price_hgbc", "m5.py")).read()
    assert "random_state=" not in src and "max_leaf_nodes=" not in src            # nothing restated in code


def test_training_window(trained, df):
    _, Xtr, ytr, data, _ = trained
    d = da.market_day(Xtr.index)
    assert (d.min(), d.max()) == (pd.Timestamp("2010-01-29"), pd.Timestamp("2014-12-31"))
    assert (d.nunique(), len(Xtr)) == (1798, 86304)
    assert data.index.max() == pd.Timestamp("2015-01-01 00:00")                   # nothing after 2014 read
    assert Xtr.index.is_unique and Xtr.index.is_monotonic_increasing
    # first valid training day satisfies the lag requirement (lag-7 needs 7 earlier days; P1 warm-up is 28)
    assert (d.min() - pd.Timestamp("2010-01-01")).days >= 7 and np.isfinite(Xtr.to_numpy(float)).all()
    lab = ofr.official_labels(data, sorted(set(d)))
    assert np.array_equal(lab.to_numpy(), ytr.to_numpy())                         # labels = da.targets
    bx = m5.batch_features(df).loc[Xtr.index]
    assert np.array_equal(bx.to_numpy(), Xtr.to_numpy())                          # same features as the batch builder


def test_reload_bitwise_and_manifest(model, trained, preds):
    m_new = trained[0]
    X = m5.batch_features(da.load_dataset(DATA_CSV)).loc[preds.index[:4800]]
    assert np.array_equal(model.predict_proba(X)[:, 1], m_new.predict_proba(X)[:, 1])
    assert np.array_equal(model.predict_proba(X)[:, 1], preds["p_up"].to_numpy()[:4800])
    man = json.load(open(m5.P["manifest"]))
    for k, v in man["files"].items():
        assert m5.sha_file(os.path.join(ROOT, v["path"])) == v["sha256"], k
    assert man["versions"]["sklearn"] == "1.9.1" and man["train_seconds"] > 0 and man["trained_at"]
    assert m5.tree_fp(model) == man["fingerprint"]["trees"] == m5.tree_fp(m_new)


@pytest.mark.slow
def test_fresh_process_refit_identical(tmp_path):
    man = json.load(open(m5.P["manifest"]))["fingerprint"]
    outs = []
    for k in (1, 2):
        f = tmp_path / f"fp{k}.json"
        subprocess.run([sys.executable, "-m", "src.price_hgbc.m5", "--fingerprint", str(f)], cwd=ROOT, env=ENV,
                       check=True, capture_output=True, timeout=900)
        outs.append(json.load(open(f)))
    for key in ("trees", "X_train_sha256", "y_train_sha256", "p_up_sha256", "pred_sha256"):
        assert outs[0][key] == outs[1][key] == man[key], key


# ---------------- task 2: day-by-day interface ----------------
@pytest.mark.parametrize("day", m5.CHECK_DAYS)
def test_daily_interface(df, model, ref, preds, day):
    h, w = da.make_inputs(df, day)
    m5._DTM.clear()                                                               # include the one-off fit in the timing
    t0 = time.perf_counter()
    out = m5.predict_price(model, ref["tau"], h, w)
    assert time.perf_counter() - t0 < 5
    assert out.index.equals(w.index) and list(out.columns) == ["p_up", "nsw_price_up"]
    start = out.index - pd.Timedelta("30min")
    assert list(start.hour * 2 + start.minute // 30) == list(range(48))          # 48 periods, in order
    assert np.isfinite(out["p_up"]).all() and out["p_up"].between(0, 1).all()
    assert out["nsw_price_up"].dtype == np.int64 and np.array_equal(out["nsw_price_up"], (out["p_up"] >= 0.45).astype(int))
    again = m5.predict_price(model, ref["tau"], h, w)
    assert np.array_equal(out["p_up"].to_numpy(), again["p_up"].to_numpy())     # repeat identical
    b = preds.loc[w.index]
    assert np.array_equal(out["p_up"].to_numpy(), b["p_up"].to_numpy())         # batch = day-by-day, bitwise
    assert np.array_equal(out["nsw_price_up"].to_numpy(), b["pred"].to_numpy())


def test_check_days_cover_years_boundaries_and_leap_days():
    d = pd.DatetimeIndex(m5.CHECK_DAYS)
    assert len(d) >= 10 and set(d.year) == set(range(2015, 2027))
    assert {"2016-02-29", "2020-02-29", "2024-02-29"} <= set(m5.CHECK_DAYS)
    assert {"2016-01-01", "2018-01-01", "2021-01-01", "2025-01-01"} <= set(m5.CHECK_DAYS)


@pytest.mark.parametrize("day", ["2016-02-29", "2021-01-01", "2026-10-31"])
def test_daily_ignores_next_day_truth_and_future(df, model, ref, day):
    h, w = da.make_inputs(df, day)
    base = m5.predict_price(model, ref["tau"], h, w)
    w2 = w.copy()
    for c in da.MARKET_COLS:                                                      # smuggled D+1 truth is ignored
        w2[c] = np.random.default_rng(1).normal(500, 900, len(w2))
    assert np.array_equal(m5.predict_price(model, ref["tau"], h, w2)["p_up"], base["p_up"])
    alt = df.copy()                                                               # change D+1 and later truth
    late = da.market_day(alt.index) >= pd.Timestamp(day)
    for c in da.MARKET_COLS:
        alt.loc[late, c] = np.random.default_rng(2).permutation(alt.loc[late, c].to_numpy())
    h3, w3 = da.make_inputs(alt, day)
    assert np.array_equal(m5.predict_price(model, ref["tau"], h3, w3)["p_up"], base["p_up"])


@pytest.mark.parametrize("day", ["2015-06-15", "2019-01-01", "2024-02-29", "2026-05-20"])
def test_batch_causal_shuffle_from_next_day(df, model, preds, day):
    """User rule C(1): shuffle the market data of D+1 onwards in the whole file; D+1 features / predictions unchanged."""
    alt = df.copy()
    late = da.market_day(alt.index) >= pd.Timestamp(day)
    rng = np.random.default_rng(3)
    for c in da.MARKET_COLS:
        alt.loc[late, c] = rng.permutation(alt.loc[late, c].to_numpy())
    rows = _rows(df.index, day)
    a, b = m5.batch_features(df).loc[rows], m5.batch_features(alt).loc[rows]
    assert np.ascontiguousarray(a.to_numpy()).tobytes() == np.ascontiguousarray(b.to_numpy()).tobytes()
    assert np.array_equal(model.predict_proba(b)[:, 1], preds.loc[rows, "p_up"].to_numpy())


@pytest.mark.parametrize("day", ["2015-03-02", "2018-12-31", "2022-06-15", "2026-10-31"])
def test_features_match_p1_definition_column_by_column(df, day):
    """Adapter columns vs an independent bit-exact transcription of P1's recipe (features_simple.build_features)."""
    h, w = da.make_inputs(df, day)
    got = m5.features_for_day(h, w, list(fs.BOOST_PRICE_FEATURES))
    want = fs.build_features(df).loc[w.index]
    for c in fs.BOOST_PRICE_FEATURES:
        assert np.array_equal(got[c].to_numpy(dtype=float), want[c].to_numpy(dtype=float)), c


# ---------------- task 3: prediction file ----------------
def test_prediction_file_integrity(preds, ref):
    assert list(preds.columns) == ["day", "period", "p_up", "pred"]                # no labels stored
    assert preds.index.is_unique and preds.index.is_monotonic_increasing
    days = pd.date_range("2015-01-01", "2026-10-31")
    assert (preds.groupby("day").size() == 48).all() and list(pd.DatetimeIndex(sorted(preds["day"].unique()))) == list(days)
    assert len(preds) == 48 * len(days) == 207456
    assert (preds["day"] == da.market_day(preds.index)).all()
    start = preds.index - pd.Timedelta("30min")
    assert (preds["period"] == start.hour * 2 + start.minute // 30).all()
    assert np.isfinite(preds["p_up"]).all() and preds["p_up"].between(0, 1).all()
    assert (preds["pred"] == (preds["p_up"] >= ref["tau"]).astype(int)).all()


def test_daily_evidence_file():
    d = pd.read_csv(m5.P["daily"])
    assert len(d) >= 10 and d["p_up_bitwise_equal"].all() and d["pred_equal"].all() and (d["max_abs_diff"] == 0).all()
    assert (d["seconds"] < 5).all() and (d["periods"] == 48).all()


# ---------------- task 4: development scores ----------------
def test_dev_scores_2015_2024_only_and_recomputable(df, preds):
    y = pd.read_csv(m5.P["yearly"])
    mth = pd.read_csv(m5.P["monthly"])
    assert list(y["year"]) == list(range(2015, 2025)) and mth["month"].min() == "2015-01" and mth["month"].max() == "2024-12"
    days = [d for d in da.complete_days(df) if d.year == 2015]
    truth = pd.concat([da.targets(df.loc[d - D1 + pd.Timedelta("30min"): d + D1], d)["nsw_price_up"] for d in days])
    p15 = preds.loc[truth.index, "pred"]
    r15 = y.set_index("year").loc[2015]
    assert abs(da.balanced_accuracy(truth, p15) - r15["model_BalAcc"]) < 1e-9 and abs(da.accuracy(truth, p15) - r15["model_Acc"]) < 1e-9
    gap = abs(r15["model_BalAcc"] - json.load(open(ofr.REF_JSON))["metrics_2014"]["BalAcc"])
    assert gap < 10                                                                # else WARN + feature check
    src = open(os.path.join(ROOT, "src", "price_hgbc", "m5.py")).read()
    assert 'DEV_END = pd.Timestamp("2024-12-31")' in src and 'preds[preds["day"] <= DEV_END]' in src


# ---------------- task 5: frozen files ----------------
def test_readme_frozen():
    from src.price_hgbc import m5_readme
    md = open(m5.P["readme"], encoding="utf-8").read()
    assert md == m5_readme.build()
    man = json.load(open(m5.P["manifest"]))
    for v in man["files"].values():
        assert v["sha256"] in md
    for must in ("τ = 0.45 was chosen on 2014 for models trained on 2010–2013", "2025–2026 scores must not be used for any choice",
                 "price_hgbc workstream", "μ₀ and σ₀ come from an earlier model", "budget of at most 6 retrains",
                 "ref_update = **A**"):
        assert must in md, must
