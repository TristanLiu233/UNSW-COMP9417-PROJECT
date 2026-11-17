"""M7 checks (results/price_lr/strategy_design.md section 6). Run: pytest tests/price_lr -q"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import pytest

from src.price_lr import data as D
from src.price_lr.env import REPLAY_CSV, RESULTS, ROOT, check_inputs, ft
from src.price_lr.fit_sliding import fit_sliding
from src.price_lr.replay import BUDGET, COOLDOWN_DAYS, CUSUM_CONFIG, SCHEDULE
from src.price_hgbc import cusum as cs
from src.price_hgbc.m6 import LabelGate

R = lambda *p: os.path.join(RESULTS, *p)
DEV_END = pd.Timestamp("2024-12-31")


@pytest.fixture(scope="module")
def raw():
    return D.load_raw()


@pytest.fixture(scope="module")
def log():
    return pd.read_csv(R("models_log.csv"))


def test_group_inputs_unchanged():
    check_inputs(full=True)


def test_s0_bitwise_equal_to_p4_replay():
    s0 = pd.read_parquet(R("s0_predictions.parquet"))
    r = pd.read_csv(REPLAY_CSV, index_col="timestamp", parse_dates=True, float_precision="round_trip")
    assert len(r) == 207456 and r.index.equals(s0.index)
    assert np.array_equal(s0["p_up"].to_numpy(), r["log_interactions31_probability"].to_numpy())
    assert np.array_equal(s0["pred"].to_numpy(), r["log_interactions31"].to_numpy())
    assert (s0["model_id"] == "LR_frozen").all()


def test_budget_and_effective_dates(log):
    for s, g in log.groupby("strategy"):
        re = g[g["model_id"] != "LR_frozen"]
        assert len(re) <= BUDGET
        for _, l in re.iterrows():
            assert pd.Timestamp(l["effective_from"]) == pd.Timestamp(l["cutoff"]) + pd.Timedelta("1D")
        eff = pd.to_datetime(g["effective_from"]).tolist()
        to = pd.to_datetime(g["effective_to"]).tolist()
        assert eff[0] == pd.Timestamp("2015-01-01") and to[-1] == pd.Timestamp("2026-10-31")
        assert all(to[i] + pd.Timedelta("1D") == eff[i + 1] for i in range(len(eff) - 1))
    for s in ("S1", "S2"):
        cut = pd.to_datetime(log[(log["strategy"] == s) & (log["model_id"] != "LR_frozen")]["cutoff"]).tolist()
        assert cut == SCHEDULE
    assert len(log[log["strategy"] == "S0"]) == 1


def test_s3_rules(log):
    ev = pd.read_csv(R("s3_events.csv"), parse_dates=["alarm_day"])
    rt = ev[ev["retrained"]]
    s3 = log[(log["strategy"] == "S3") & (log["model_id"] != "LR_frozen")]
    assert pd.to_datetime(s3["cutoff"]).tolist() == rt["alarm_day"].tolist()
    gaps = rt["alarm_day"].diff().dt.days.dropna()
    assert (gaps >= COOLDOWN_DAYS).all() and len(rt) <= BUDGET
    sup = ev[~ev["retrained"]]
    assert set(sup["suppressed_reason"]) <= {"cooldown", "budget"}
    last = None
    for _, e in ev.iterrows():                      # each suppression reason follows the rule order
        if e["retrained"]:
            last, n = e["alarm_day"], int(e["retrain_index"])
            continue
        cool_ok = last is None or (e["alarm_day"] - last).days >= COOLDOWN_DAYS
        assert e["suppressed_reason"] == ("cooldown" if not cool_ok else "budget")
    assert set(s3["trainer"]) == {"fit_final"}


def test_cusum_constants_and_path():
    cfg = json.load(open(CUSUM_CONFIG))
    cs.validate_config(cfg)
    m = cfg["schemes"]["main"]
    d = pd.read_csv(R("cusum", "daily_err_2014.csv"))
    s = cs.reference_stats(d["logloss"], truncate=True)
    assert (s["q99"], s["mu0"], s["sigma0"]) == (m["q99"], m["mu0"], m["sigma0"])
    assert m["k"] == 0.5 and m["h"] in cs.H_CANDIDATES and m["meets_target"] and cfg["ref_update"] == "A"
    arl = pd.read_csv(R("cusum", "arl0_by_h.csv"))
    main = arl[arl["scheme"] == "main"]
    assert m["h"] == main[main["arl0_median"] >= 730]["h"].min()
    dly = pd.read_csv(R("s3_daily_cusum.csv"), parse_dates=["day"])
    dev = dly[dly["day"] <= DEV_END]
    det = cs.Cusum.from_config(cfg, "main")
    out = cs.run_batch(pd.Series(dev["logloss"].to_numpy(), index=dev["day"]), det)
    assert np.array_equal(out["S"].to_numpy(), dev["S"].to_numpy())
    assert np.array_equal(out["alarm"].to_numpy(), dev["alarm"].to_numpy())


def test_fit_sliding_equals_fit_final_on_full_window(raw):
    cut = "2016-12-31"
    a, aa = ft.fit_final(D.prefix(raw, cut), cut)
    b, ab = fit_sliding(D.prefix(raw, cut), cut, "2010-01-29")
    pa, pb = a["pipeline"], b["pipeline"]
    for x, y in ((pa[0].mean_, pb[0].mean_), (pa[0].var_, pb[0].var_), (pa[-1].coef_, pb[-1].coef_),
                 (pa[-1].intercept_, pb[-1].intercept_)):
        assert np.array_equal(x, y)
    assert aa["threshold"] == ab["threshold"] and aa["search"] == ab["search"] and aa["train_rows"] == ab["train_rows"]
    f = D.Frame(raw)
    m = f.rows("2017-01-01", "2017-01-03")
    assert np.array_equal(pa.predict_proba(f.X.loc[m]), pb.predict_proba(f.X.loc[m]))


def test_sliding_window_rows(raw, log):
    b, a = fit_sliding(D.prefix(raw, "2016-12-31"), "2016-12-31", "2014-01-01")
    assert a["train_rows"] == 1096 * 48
    assert a["folds"][0]["train_start"] == "2014-01-01" and a["folds"][0]["train_end"] == "2015-12-31"
    s2 = log[log["strategy"] == "S2"].set_index("model_id")
    assert s2.loc["S2_r1_2016-12-31", "train_rows"] == 1096 * 48 and s2.loc["S2_r1_2016-12-31", "tau"] == a["threshold"]


def test_label_gate_and_future_isolation(raw):
    cut = pd.Timestamp("2018-12-31")
    fut = raw.copy()
    late = fut.index > cut + pd.Timedelta("1D")
    fut.loc[late, "nsw_price"] = -fut.loc[late, "nsw_price"] * 7 + 3           # corrupt every future label/price
    for fn, args in ((ft.fit_final, ()), (fit_sliding, ("2016-01-01",))):
        a, aa = fn(raw, cut, *args)
        b, ab = fn(fut, cut, *args)
        assert np.array_equal(a["pipeline"][-1].coef_, b["pipeline"][-1].coef_) and aa["threshold"] == ab["threshold"]
    y = pd.Series(1, index=pd.date_range("2015-01-01 00:30", periods=96, freq="30min"))
    g = LabelGate(y)
    with pytest.raises(RuntimeError):
        g.until("2015-01-01")
    g.advance("2015-01-01")
    assert len(g.until("2015-01-01")) == 48
    with pytest.raises(RuntimeError):
        g.day("2015-01-02")


def test_saved_models_match_log(log):
    for _, l in log[log["model_id"] != "LR_frozen"].iterrows():
        b = joblib.load(os.path.join(ROOT, l["model_path"]))
        assert b["threshold"] == l["tau"] and b["train_end"] == l["cutoff"]


def test_no_holdout_scores_in_summaries():
    for f in ("yearly_dev.csv",):
        assert pd.read_csv(R(f))["year"].max() == 2024
    assert pd.read_csv(R("monthly_dev.csv"))["month"].max() == "2024-12"
    assert (pd.read_csv(R("strategy_comparison_dev.csv"))["days"] == 3653).all()
    dly = pd.read_csv(R("s3_daily_cusum.csv"), parse_dates=["day"])
    late = dly[dly["day"] > DEV_END]
    assert late["logloss"].isna().all() and late["S"].isna().all()
    assert pd.to_datetime(pd.read_csv(R("s0_monitor_cusum.csv"))["day"]).max() <= DEV_END
    g = pd.read_csv(R("g_analog", "g_analog_weeks.csv"))
    assert pd.to_datetime(g["week_end"]).max() <= DEV_END and len(g) == 20
    assert pd.to_datetime(pd.read_csv(R("g_analog", "g_age37_43_deployed.csv"))["window_end"]).max() <= DEV_END
    st = json.load(open(R("stage3_provisional.json")))
    assert st["uses_2025_2026"] is False and "2025" not in json.dumps(st["BalAcc_2015_2024"])
    assert json.load(open(R("eval_meta.json")))["latest_timestamp_read"] == "2025-01-01 00:00:00"


def test_consistency_record():
    c = json.load(open(R("selfcheck", "consistency.json")))
    assert c["scheduled_all_ok"] and len(c["scheduled"]) == 18
    assert all(v for k, v in c["s3_daily_vs_batch"].items() if k != "rows")


def test_stage3_provisional_follows_rule():
    st = json.load(open(R("stage3_provisional.json")))
    boot = pd.read_csv(R("paired_bootstrap_dev.csv"))
    cmp_ = pd.read_csv(R("strategy_comparison_dev.csv")).set_index("strategy")
    F = st["F"]
    assert F == cmp_.loc[["S0", "S1", "S2", "S3"], "BalAcc"].idxmax()
    for r in st["F_minus_others"]:
        b = boot[(boot["a"] == F) & (boot["b"] == r["vs"])].iloc[0]
        assert r["ci_includes_0"] == bool(b["diff_lo"] <= 0 <= b["diff_hi"])
    assert st["P_provisional"] in st["candidate_set"] and st["R_backup"] != st["P_provisional"]
