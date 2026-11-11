"""Checks of the pre-registered S3 sensitivity run (m6/s3_sensitivity_preregistration.md section 4)."""
import json
import os

import joblib
import numpy as np
import pandas as pd
import pytest

from conftest import ROOT
from src.price_hgbc import cusum as cs
from src.price_hgbc import m5, m6
from src.price_hgbc import m6_sensitivity as sv
from src.price_hgbc.kit import da

PREREG_SHA = "0b71324010f56adfdf4088318b084312646c013b35fd3cdb2415589172fa13cc"          # frozen at approval
OFFICIAL_M6 = {"s3_predictions.parquet": "2fb504c2164efa220a5fb0d65f1a57651efd4c6bc1184d929843589589f5a717",
               "s3_events.csv": "ca13d412850144feb5a469ad61d8aec1c6d0f45eeecd702a411705e3f29ebb9b",
               "models_log.csv": "74e89f1df8271feb82fe93c28585d5a5d5c2604fde7f4045fe9bec81909c6ced",
               "strategy_comparison_dev.csv": "1fb584b572ac25368b1261ddb19b03a9984d63d225cac8ade4ceaa6c009d1c02",
               "s3_daily_cusum.csv": "3f0fa69f0e78be8f8a6ea6d193cb146b68a8b989bab45ae2ff8fecbe5761682d"}
FROZEN = {"results/price_hgbc/official/boost_price_ref.json": "06b022774a10dad545dacb9a73d910a73eccd37d1ebc14d3f605d6e9ff210725",
          "results/price_hgbc/m4/cusum_config.json": "b0c5ebd2be66cb9cdf910ee42ef4660ad6b1d6986aec542005f8856df187d156",
          "results/price_hgbc/m5/stage1_model.joblib": "bca5291ba0e143db518f220f112b5bcfad127a42d3de7bd3a167edbea246a1d6",
          "results/price_hgbc/m5/predictions.parquet": "9014a23e12df76f1885dd00176c480a8d6a1ab645cc23b00791a81e4226ba8d9"}
D1 = pd.Timedelta("1D")


def _tag(k):
    return k.lower().replace("-", "")


@pytest.fixture(scope="module")
def ctx():
    return m6.Context()


@pytest.fixture(scope="module")
def out():
    r = {}
    for k in sv.NAMES:
        r[k] = {"p": pd.read_parquet(os.path.join(sv.OUT, f"{_tag(k)}_predictions.parquet")),
                "e": pd.read_csv(os.path.join(sv.OUT, f"{_tag(k)}_events.csv"), parse_dates=["alarm_day"]),
                "d": pd.read_csv(os.path.join(sv.OUT, f"{_tag(k)}_daily_cusum.csv"), parse_dates=["day"])}
    r["log"] = pd.read_csv(sv.F["log"])
    return r


def test_prereg_frozen_and_official_outputs_unchanged():
    assert m5.sha_file(sv.PREREG) == PREREG_SHA
    for f, want in OFFICIAL_M6.items():
        assert m5.sha_file(os.path.join(m6.OUT, f)) == want, f
    for rel, want in FROZEN.items():
        assert m5.sha_file(os.path.join(ROOT, rel)) == want, rel
    assert json.load(open(sv.F["meta"]))["prereg_sha256"] == PREREG_SHA


def test_1_control_reproduces_official_s3(out):
    a, off = out["S3-A"], pd.read_parquet(os.path.join(m6.OUT, "s3_predictions.parquet"))
    assert a["p"].index.equals(off.index)
    assert np.array_equal(a["p"]["p_up"].to_numpy(), off["p_up"].to_numpy())
    assert np.array_equal(a["p"]["pred"].to_numpy(), off["pred"].to_numpy())
    oe = pd.read_csv(os.path.join(m6.OUT, "s3_events.csv"), parse_dates=["alarm_day"])
    assert list(a["e"]["alarm_day"]) == list(oe["alarm_day"]) and a["e"]["retrained"].tolist() == oe["retrained"].tolist()
    assert np.array_equal(a["e"]["S_at_alarm"].to_numpy(), oe["S_at_alarm"].to_numpy())
    od = pd.read_csv(os.path.join(m6.OUT, "s3_daily_cusum.csv"))
    assert np.array_equal(a["d"]["S"].to_numpy(), od["S"].to_numpy())


@pytest.mark.parametrize("k", sv.NAMES)
def test_2_budget_cooldown_effective_and_s0_prefix(out, k):
    lg = out["log"][(out["log"]["strategy"] == k) & (out["log"]["model_id"] != "M5_frozen")]
    v = sv.variants(json.load(open(m6.CUSUM_CONFIG)), json.load(open(sv.F["meta"]))["h_for_S3K"])[k]
    assert 1 <= len(lg) <= 6
    al = pd.to_datetime(lg["alarm_day"]).reset_index(drop=True)
    assert all((al[i + 1] - al[i]).days >= v["cooldown"] for i in range(len(al) - 1))
    assert list(pd.to_datetime(lg["effective_from"])) == list(al + D1)
    s0 = pd.read_parquet(os.path.join(m6.OUT, "s0_predictions.parquet"))
    m = out[k]["p"]["day"] < al.min() + D1
    assert np.array_equal(out[k]["p"].loc[m, "p_up"].to_numpy(), s0.loc[m, "p_up"].to_numpy())
    ev = out[k]["e"]
    assert int(ev["retrained"].sum()) == len(lg)
    assert set(ev.loc[~ev["retrained"], "suppressed_reason"]) <= {"cooldown", "budget"}


@pytest.mark.slow
def test_3_s3b_future_shuffle_does_not_change_past(ctx, out):
    T = pd.Timestamp("2017-06-30")
    alt = ctx.df.copy()
    late = da.market_day(alt.index) > T
    rng = np.random.default_rng(8)
    for col in da.MARKET_COLS:
        alt.loc[late, col] = rng.permutation(alt.loc[late, col].to_numpy())
    v = sv.variants(json.load(open(m6.CUSUM_CONFIG)), 5)["S3-B"]
    p, _, ev, _ = m6.run_s3(m6.Context(alt), end=T, save=False, variant=v)
    want = out["S3-B"]["e"]
    want = want[want["alarm_day"] <= T]
    assert list(pd.to_datetime(ev["alarm_day"])) == list(want["alarm_day"])
    assert np.array_equal(p["p_up"].to_numpy(), out["S3-B"]["p"].loc[p.index, "p_up"].to_numpy())


def test_4_s3b_reference_updates_reproducible(ctx, out):
    rb = pd.read_csv(sv.F["refB"])
    for _, r in rb.iterrows():
        assert pd.Timestamp(r["val_end"]) <= pd.Timestamp(r["alarm_day"]) and r["ref_from_days"] >= 300
    r = rb.iloc[0]
    gate = m6.LabelGate(ctx.labels)
    gate.advance(r["alarm_day"])
    _, _, lg = m6.retrain(ctx, gate, r["alarm_day"], "expanding")
    s = cs.reference_stats(lg["_val_daily_logloss"].to_numpy())
    assert (s["q99"], s["mu0"], s["sigma0"]) == (r["ref_q99_after"], r["ref_mu0_after"], r["ref_sigma0_after"])
    d = out["S3-B"]["d"]
    for _, x in rb.iterrows():                                                    # reference in force from the next day
        nxt = d[d["day"] == pd.Timestamp(x["alarm_day"]) + D1].iloc[0]
        assert (nxt["mu0"], nxt["sigma0"], nxt["q99"]) == (x["ref_mu0_after"], x["ref_sigma0_after"], x["ref_q99_after"])


def test_5_s3k_calibration_before_replay():
    cal = pd.read_csv(sv.F["calib"])
    meta = json.load(open(sv.F["meta"]))
    assert m5.sha_file(sv.F["calib"]) == meta["calibration_sha256_before_replay"]
    assert os.path.getmtime(sv.F["calib"]) < os.path.getmtime(os.path.join(sv.OUT, "s3k_predictions.parquet"))
    assert (cal["k"] == 0.75).all() and list(cal["h"]) == list(cs.H_CANDIDATES)
    v = cal["arl0_median"].to_numpy()
    assert all(v[i + 1] >= v[i] for i in range(len(v) - 1))
    ok = cal[cal["arl0_median"] >= 730]
    assert cal.loc[cal["selected"], "h"].tolist() == [ok["h"].min()] == [meta["h_for_S3K"]]
    d = pd.read_csv(os.path.join(sv.OUT, "s3k_daily_cusum.csv"))
    assert (d["k"] == 0.75).all() and (d["h"] == meta["h_for_S3K"]).all()


@pytest.mark.parametrize("k", ["S3-B", "S3-K", "S3-C"])
def test_6_batch_equals_daily(ctx, out, k):
    lg = out["log"][(out["log"]["strategy"] == k) & (out["log"]["model_id"] != "M5_frozen")].reset_index(drop=True)
    for i in (0, len(lg) // 2, len(lg) - 1):
        x = lg.iloc[i]
        model = joblib.load(os.path.join(ROOT, x["model_path"]))
        first, last = pd.Timestamp(x["effective_from"]), pd.Timestamp(x["effective_to"])
        for d in (first, (first + (last - first) / 2).normalize(), last):
            h, w = da.make_inputs(ctx.df, d)
            o = m5.predict_price(model, x["tau"], h, w)
            b = out[k]["p"].loc[w.index]
            assert (b["model_id"] == x["model_id"]).all()
            assert np.array_equal(o["p_up"].to_numpy(), b["p_up"].to_numpy()), (k, x["model_id"], d)


def test_7_8_no_holdout_scores_and_tables_consistent():
    y = pd.read_csv(sv.F["yearly"])
    assert list(y["year"]) == list(range(2015, 2025))
    for k in sv.NAMES:
        d = pd.read_csv(os.path.join(sv.OUT, f"{_tag(k)}_daily_cusum.csv"))
        assert d.loc[d["day"] > "2024-12-31", "logloss"].isna().all()
    b = pd.read_csv(sv.F["boot"])
    prim = b[b["role"] == "primary"]
    assert set(prim["a"]) == {"S3-B", "S3-K", "S3-C"} and (prim["b"] == "S3-A").all()
    assert np.allclose(prim["p_holm"].to_numpy(), sv.holm(prim["p_boot"].to_numpy()))
    assert json.load(open(sv.F["meta"]))["boot_index_sha256"] == json.load(open(os.path.join(m6.OUT, "eval_meta.json")))["boot_index_sha256"]
    cmp_ = pd.read_csv(sv.F["cmp"]).set_index("variant")
    for k in sv.NAMES:
        r = b[(b["a"] == k) & (b["b"] == "S1")].iloc[0]
        rule = k != "S3-A" and r["diff_pp"] >= 0.5 and r["diff_lo"] > 0 and cmp_.loc[k, "retrains_total"] <= 6
        assert bool(cmp_.loc[k, "meets_adoption_rule"]) == bool(rule)


def test_holm_and_boot_p():
    assert list(sv.holm(np.array([0.01, 0.04, 0.03]))) == [0.03, 0.06, 0.06]
    assert sv.boot_p(np.array([1.0, 2.0, 3.0])) == 0.0 and sv.boot_p(np.array([-1.0, 1.0])) == 1.0
