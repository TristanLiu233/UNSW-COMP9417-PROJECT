"""M6 self-checks: S0-S3 causal replay (P1 boost_price, REF params, M4 CUSUM option A), development evaluation.
Hard gates: S0 = M5 bitwise; S1/S2 exactly 5 retrains; S3 <= 6; batch = day-by-day; nothing read from the future.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import pytest

from conftest import ROOT
from src.price_hgbc import cusum as cs
from src.price_hgbc import m5, m6, m6_eval
from src.price_hgbc import official_ref as ofr
from src.price_hgbc.kit import da

FROZEN = {"results/price_hgbc/official/boost_price_ref.json": "06b022774a10dad545dacb9a73d910a73eccd37d1ebc14d3f605d6e9ff210725",
          "results/price_hgbc/official/preds_2014.parquet": "b4c9890d0149d69c902dcc10685db82c7b2d19d253a57a72ce0483a1f93c0c29",
          "results/price_hgbc/m4/cusum_design.md": "78fe92dbd3f1cdbe657ab8faacd289126a7a690ca4ddd34f71f15590c884f654",
          "results/price_hgbc/m5/stage1_model.joblib": "bca5291ba0e143db518f220f112b5bcfad127a42d3de7bd3a167edbea246a1d6",
          "results/price_hgbc/m5/predictions.parquet": "9014a23e12df76f1885dd00176c480a8d6a1ab645cc23b00791a81e4226ba8d9",
          "results/price_hgbc/m5/stage1_config.json": "f7cea1483d46159402593745ad0a26f7c70700d3ab74c20a022daa5243706504"}
D1 = pd.Timedelta("1D")


@pytest.fixture(scope="module")
def ctx():
    return m6.Context()


@pytest.fixture(scope="module")
def preds():
    return {s: pd.read_parquet(os.path.join(m6.OUT, f"{s.lower()}_predictions.parquet")) for s in ("S0", "S1", "S2", "S3")}


@pytest.fixture(scope="module")
def log():
    return pd.read_csv(os.path.join(m6.OUT, "models_log.csv"))


@pytest.fixture(scope="module")
def events():
    return pd.read_csv(os.path.join(m6.OUT, "s3_events.csv"), parse_dates=["alarm_day"])


def _seg(log, s):
    return log[(log["strategy"] == s) & (log["model_id"] != "M5_frozen")].reset_index(drop=True)


# ---------------- guards ----------------
def test_frozen_files_unchanged():
    for rel, want in FROZEN.items():
        assert m5.sha_file(os.path.join(ROOT, rel)) == want, rel
    assert json.load(open(m6.CUSUM_CONFIG))["ref_update"] == "A"


# ---------------- task 1: interface ----------------
def test_prediction_tables(preds, log):
    m5p = pd.read_parquet(m5.P["preds"])
    for s, p in preds.items():
        assert list(p.columns) == ["day", "period", "p_up", "pred", "strategy", "model_id"], s      # no labels
        assert p.index.equals(m5p.index) and (p["day"] == m5p["day"]).all() and (p["period"] == m5p["period"]).all()
        assert (p.groupby("day").size() == 48).all() and p["day"].nunique() == 4322
        assert np.isfinite(p["p_up"]).all() and p["p_up"].between(0, 1).all() and (p["strategy"] == s).all()
        segs = log[log["strategy"] == s]
        for _, r in segs.iterrows():                                          # each day predicted by the model in force
            m = (p["day"] >= r["effective_from"]) & (p["day"] <= r["effective_to"])
            assert (p.loc[m, "model_id"] == r["model_id"]).all() and m.any(), (s, r["model_id"])
            assert (p.loc[m, "pred"] == (p.loc[m, "p_up"] >= r["tau"]).astype(int)).all()


def test_label_gate_blocks_future(ctx):
    g = m6.LabelGate(ctx.labels)
    with pytest.raises(RuntimeError):
        g.until("2015-01-01")
    g.advance("2016-12-31")
    assert da.market_day(g.until("2016-12-31").index).max() == pd.Timestamp("2016-12-31")
    with pytest.raises(RuntimeError):
        g.until("2017-01-01")
    with pytest.raises(RuntimeError):
        g.day("2017-01-01")


def test_logs_traceable(log):
    for c in ("model_id", "trigger", "cutoff", "train_start", "train_end", "train_rows", "rows_gt_200k", "tau",
              "effective_from", "effective_to", "total_seconds", "trees_sha256", "model_path", "model_sha256"):
        assert c in log.columns, c
    r = log[log["model_id"] != "M5_frozen"]
    assert (r["rows_gt_200k"] == (r["train_rows"] > 200_000)).all()
    for _, x in r.iterrows():
        assert m5.sha_file(os.path.join(ROOT, x["model_path"])) == x["model_sha256"]
        assert pd.Timestamp(x["train_end"]) < pd.Timestamp(x["effective_from"])


# ---------------- task 2: S0 ----------------
def test_s0_equals_m5_bitwise(preds, log):
    m5p = pd.read_parquet(m5.P["preds"])
    s0 = preds["S0"]
    assert len(s0) == len(m5p) == 207456
    assert np.array_equal(s0["p_up"].to_numpy(), m5p["p_up"].to_numpy())
    assert np.array_equal(s0["pred"].to_numpy(), m5p["pred"].to_numpy())
    assert len(_seg(log, "S0")) == 0
    y5 = pd.read_csv(m5.P["yearly"]).set_index("year")
    y6 = pd.read_csv(m6_eval.P["yearly"]).set_index("year")
    assert np.array_equal(y5["model_BalAcc"].to_numpy(), y6["S0_BalAcc"].to_numpy())
    assert np.array_equal(y5["model_Acc"].to_numpy(), y6["S0_Acc"].to_numpy())


# ---------------- tasks 3-4: S1 / S2 ----------------
@pytest.mark.parametrize("s,window", [("S1", "expanding"), ("S2", "sliding3y")])
def test_scheduled_retrains(log, s, window):
    r = _seg(log, s)
    assert len(r) == 5
    assert list(r["cutoff"]) == [f"{y}-12-31" for y in (2016, 2018, 2020, 2022, 2024)]
    assert list(r["effective_from"]) == [f"{y + 1}-01-01" for y in (2016, 2018, 2020, 2022, 2024)]
    for _, x in r.iterrows():
        y = int(x["cutoff"][:4])
        assert x["val_start"] == f"{y}-01-01" and x["val_end"] == f"{y}-12-31" and x["val_days"] == (366 if y % 4 == 0 else 365)
        assert pd.Timestamp(x["tune_fit_end"]) < pd.Timestamp(x["val_start"])     # tau's validation days not fitted
        if s == "S1":
            assert x["train_start"] == "2010-01-29"
        else:
            assert x["train_start"] == f"{y - 2}-01-01"
            days = (pd.Timestamp(x["train_end"]) - pd.Timestamp(x["train_start"])).days + 1
            assert x["train_days"] == days == 1096 and x["train_rows"] == days * 48
        assert set([x["tau"]]) <= {0.40, 0.45, 0.50, 0.55, 0.60}


@pytest.mark.parametrize("s", ["S1", "S2", "S3"])
def test_batch_equals_daily_three_segments(ctx, preds, log, s):
    r = _seg(log, s)
    picks = [0, len(r) // 2, len(r) - 1]
    for i in picks:
        x = r.iloc[i]
        model = joblib.load(os.path.join(ROOT, x["model_path"]))
        first = pd.Timestamp(x["effective_from"])
        last = min(pd.Timestamp(x["effective_to"]), pd.Timestamp("2026-10-31"))
        for d in (first, first + (last - first) / 2, last):
            d = d.normalize()
            h, w = da.make_inputs(ctx.df, d)
            out = m5.predict_price(model, x["tau"], h, w)
            b = preds[s].loc[w.index]
            assert (b["model_id"] == x["model_id"]).all()
            assert np.array_equal(out["p_up"].to_numpy(), b["p_up"].to_numpy()), (s, x["model_id"], d)
            assert np.array_equal(out["nsw_price_up"].to_numpy(), b["pred"].to_numpy())


@pytest.mark.parametrize("model_id,window", [("S1_r1_2016-12-31", "expanding"), ("S2_r3_2020-12-31", "sliding3y")])
def test_retrained_models_reproducible(ctx, log, model_id, window):
    x = log.set_index("model_id").loc[model_id]
    gate = m6.LabelGate(ctx.labels)
    gate.advance(x["cutoff"])
    model, tau, lg = m6.retrain(ctx, gate, x["cutoff"], window)
    assert lg["trees_sha256"] == x["trees_sha256"] and tau == x["tau"]
    saved = joblib.load(os.path.join(ROOT, x["model_path"]))
    X = ctx.X.loc[ctx.rows(x["effective_from"], x["effective_from"])]
    assert np.array_equal(model.predict_proba(X)[:, 1], saved.predict_proba(X)[:, 1])


# ---------------- task 5: S3 ----------------
def test_s3_rules(preds, log, events):
    r = _seg(log, "S3")
    assert 1 <= len(r) <= 6
    al = pd.to_datetime(r["alarm_day"])
    assert all((al.iloc[i + 1] - al.iloc[i]).days >= 180 for i in range(len(al) - 1))
    assert list(pd.to_datetime(r["effective_from"])) == list(al + D1)
    first = pd.to_datetime(r["effective_from"]).min()
    m = preds["S3"]["day"] < first
    assert np.array_equal(preds["S3"].loc[m, "p_up"].to_numpy(), preds["S0"].loc[m, "p_up"].to_numpy())
    assert (r["train_start"] == "2010-01-29").all()                              # S3 window = S1 window
    ev = events
    assert int(ev["retrained"].sum()) == len(r)
    last_rt = None
    for _, e in ev.iterrows():                                                    # suppression reasons consistent
        if e["retrained"]:
            last_rt = e["alarm_day"]
            continue
        n_before = int(ev[(ev["alarm_day"] < e["alarm_day"]) & ev["retrained"]].shape[0])
        in_cool = last_rt is not None and (e["alarm_day"] - last_rt).days < 180
        assert e["suppressed_reason"] == ("cooldown" if in_cool else "budget"), e["alarm_day"]
        if not in_cool:
            assert n_before == 6
    after_budget = ev[ev["alarm_day"] > al.max()]
    assert (~after_budget["retrained"]).all() and (after_budget["suppressed_reason"] == "budget").all() or len(r) < 6


def test_s3_cusum_option_a_and_resets(ctx, preds):
    cfg = json.load(open(m6.CUSUM_CONFIG))["schemes"]["main"]
    dly = pd.read_csv(os.path.join(m6.OUT, "s3_daily_cusum.csv"), parse_dates=["day"])
    dev = dly[dly["day"] <= m6.DEV_END]
    assert dly.loc[dly["day"] > m6.DEV_END, "logloss"].isna().all()               # hold-out daily scores not published
    det = cs.Cusum(cfg["mu0"], cfg["sigma0"], cfg["q99"], cfg["k"], cfg["h"])
    rb = cs.run_batch(pd.Series(dev["logloss"].to_numpy(), index=dev["day"]), det)  # batch recomputation, option A fixed
    assert np.array_equal(rb["S"].to_numpy(), dev["S"].to_numpy())
    assert rb["alarm"].tolist() == dev["alarm"].tolist()
    p = preds["S3"]
    m = p["day"] <= m6.DEV_END
    y = ctx.labels.loc[p.index[m]]
    e = cs.daily_logloss(y.to_numpy(), p.loc[m, "p_up"].to_numpy(), p.index[m])
    assert np.array_equal(e.to_numpy(), dev["logloss"].to_numpy())               # errors use the stored predictions
    for d in dev.loc[dev["alarm"], "day"]:                                        # reset after every alarm
        nxt = dev[dev["day"] == d + D1]
        z = (min(nxt["logloss"].iloc[0], cfg["q99"]) - cfg["mu0"]) / cfg["sigma0"]
        assert nxt["S"].iloc[0] == pytest.approx(max(0.0, z - cfg["k"]), abs=1e-12)


def test_s3_daily_mode_equals_batch():
    a = pd.read_parquet(os.path.join(m6.OUT, "s3_predictions.parquet"))
    b = pd.read_parquet(os.path.join(m6.OUT, "s3_daily_mode_predictions.parquet"))
    assert a.index.equals(b.index) and np.array_equal(a["p_up"].to_numpy(), b["p_up"].to_numpy())
    assert (a["model_id"] == b["model_id"]).all()
    x = pd.read_csv(os.path.join(m6.OUT, "s3_daily_cusum.csv"))
    y = pd.read_csv(os.path.join(m6.OUT, "s3_daily_mode_cusum.csv"))
    assert x["alarm"].tolist() == y["alarm"].tolist() and x["retrained"].tolist() == y["retrained"].tolist()
    assert np.array_equal(x["S"].to_numpy(), y["S"].to_numpy())


def test_s3_repeat_identical(ctx, preds, events):
    p, l, ev, _ = m6.run_s3(ctx, end=pd.Timestamp("2016-12-31"), save=False)
    assert np.array_equal(p["p_up"].to_numpy(), preds["S3"].loc[p.index, "p_up"].to_numpy())
    want = events[events["alarm_day"] <= "2016-12-31"]
    assert list(pd.to_datetime(ev["alarm_day"])) == list(want["alarm_day"])
    assert ev["retrained"].tolist() == want["retrained"].tolist()


@pytest.mark.slow
def test_s3_future_changes_do_not_affect_past(ctx, events, preds):
    T = pd.Timestamp("2017-06-30")
    alt = ctx.df.copy()
    late = da.market_day(alt.index) > T
    rng = np.random.default_rng(6)
    for col in da.MARKET_COLS:
        alt.loc[late, col] = rng.permutation(alt.loc[late, col].to_numpy())
    c2 = m6.Context(alt)
    p, _, ev, _ = m6.run_s3(c2, end=T, save=False)
    want = events[events["alarm_day"] <= T]
    assert list(pd.to_datetime(ev["alarm_day"])) == list(want["alarm_day"])
    assert np.array_equal(p["p_up"].to_numpy(), preds["S3"].loc[p.index, "p_up"].to_numpy())


@pytest.mark.slow
def test_s3_daily_mode_rerun_short(ctx, preds, events):
    end = pd.Timestamp("2017-06-30")                                              # covers 4 retrains
    p, _, ev, _ = m6.run_s3(ctx, mode="daily", end=end, save=False)
    assert np.array_equal(p["p_up"].to_numpy(), preds["S3"].loc[p.index, "p_up"].to_numpy())
    assert list(pd.to_datetime(ev["alarm_day"])) == list(events[events["alarm_day"] <= end]["alarm_day"])


# ---------------- task 6: evaluation ----------------
def test_evaluation_tables(preds, log, events):
    cmp_ = pd.read_csv(m6_eval.P["cmp"]).set_index("strategy")
    assert (cmp_["days"] == 3653).all() and (cmp_["rows"] == 175344).all()
    for s in ("S0", "S1", "S2", "S3"):
        lg = _seg(log, s)
        assert cmp_.loc[s, "retrains_total"] == len(lg)
        assert cmp_.loc[s, "train_seconds_total"] == pytest.approx(lg["total_seconds"].sum() if len(lg) else 0.0)
    assert cmp_.loc["S3", "s3_actual_retrains_dev"] == int(events[events["alarm_day"] <= "2024-12-31"]["retrained"].sum())
    y = pd.read_csv(m6_eval.P["yearly"])
    mo = pd.read_csv(m6_eval.P["monthly"])
    assert list(y["year"]) == list(range(2015, 2025)) and mo["month"].max() == "2024-12"
    b = pd.read_csv(m6_eval.P["boot"])
    assert set(map(tuple, b[["a", "b"]].to_numpy().tolist())) == {("S1", "S0"), ("S2", "S0"), ("S3", "S0"), ("S3", "S1"),
                                                                  ("S0", "b1"), ("S1", "b1"), ("S2", "b1"), ("S3", "b1")}
    assert (b["n_boot"] == 1000).all() and (b["seed"] == 0).all() and (b["n_days"] == 3653).all()
    for _, r in b.iterrows():
        assert abs(r["diff_pp"] - (cmp_.loc[r["a"], "BalAcc"] - cmp_.loc[r["b"], "BalAcc"])) < 1e-9


def test_evaluation_recomputable(ctx, preds):
    cmp_ = pd.read_csv(m6_eval.P["cmp"]).set_index("strategy")
    p = preds["S2"]
    m = p["day"] <= m6.DEV_END
    y = ctx.labels.loc[p.index[m]]
    assert abs(da.balanced_accuracy(y, p.loc[m, "pred"]) - cmp_.loc["S2", "BalAcc"]) < 1e-9
    src = open(os.path.join(ROOT, "src", "price_hgbc", "m6_eval.py")).read()
    assert "DEV = (pd.Timestamp(\"2015-01-01\"), m6.DEV_END)" in src and "assert max(days) <= DEV[1]" in src


def test_figures_complete():
    cmp_ = pd.read_csv(m6_eval.P["trade"])
    dly = pd.read_csv(os.path.join(m6.OUT, "s3_daily_cusum.csv"), parse_dates=["day"])
    dly = dly[dly["day"] <= m6.DEV_END]
    for fig, ax in (m6_eval.fig_yearly(pd.read_csv(m6_eval.P["yearly"])), m6_eval.fig_cusum(dly, None), m6_eval.fig_trade(cmp_)):
        assert ax.get_title(loc="left").strip() and ax.get_xlabel().strip() and ax.get_ylabel().strip()
    for k in ("fig_year", "fig_cusum", "fig_trade"):
        assert os.path.getsize(m6_eval.P[k]) > 0
