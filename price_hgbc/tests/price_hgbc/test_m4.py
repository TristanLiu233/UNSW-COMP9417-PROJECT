"""M4 self-checks: error-based CUSUM (synthetic behaviour, 2014 calibration, h choice, data boundary, config).
Tolerances fixed in advance: recursion results exact (same code path); stored CSV numbers round-trip exactly (%.17g).
"""
import json
import math
import os
import re

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT
from src.price_hgbc import cusum as cs
from src.price_hgbc import m4
from src.price_hgbc import official_ref as ofr

DESIGN_SHA = "78fe92dbd3f1cdbe657ab8faacd289126a7a690ca4ddd34f71f15590c884f654"   # frozen before any M4 number
OFFICIAL = {"boost_price_ref.json": "06b022774a10dad545dacb9a73d910a73eccd37d1ebc14d3f605d6e9ff210725",
            "preds_2014.parquet": "b4c9890d0149d69c902dcc10685db82c7b2d19d253a57a72ce0483a1f93c0c29",
            "boost_price_model.joblib": "9206389cfbebe6436550875273c172d5557ddb7e781252c06899281a38b128bf"}
MU, SD = 0.44, 0.2
NO_CLIP = MU + 50 * SD


@pytest.fixture(scope="module")
def cfg():
    return json.load(open(m4.P["config"]))


@pytest.fixture(scope="module")
def R():
    return m4.run()


# =========================== guards ===========================
def test_design_frozen_and_ref_unchanged():
    assert ofr.sha_file(m4.P["design"]) == DESIGN_SHA
    for f, want in OFFICIAL.items():
        assert ofr.sha_file(os.path.join(ofr.OFFICIAL, f)) == want, f


# =========================== task 2: synthetic behaviour ===========================
def test_1_constant_error_never_alarms():
    det = cs.Cusum(MU, SD, NO_CLIP, 0.5, 3)                                   # smallest candidate h
    out = [det.update(MU) for _ in range(10000)]
    assert not any(a for _, a in out) and max(s for s, _ in out) == 0


@pytest.mark.parametrize("h", cs.H_CANDIDATES)
def test_2_shift_detection_delay(h, capsys):
    rng = np.random.default_rng(1)
    delays = {}
    for shift in (1.0, 3.0):
        d = []
        for _ in range(200):
            e = MU + SD * rng.standard_normal(800)
            e[200:] += shift * SD
            det = cs.Cusum(MU, SD, NO_CLIP, 0.5, h)
            al = [det.update(x)[1] for x in e]
            d.append(next(t for t in range(200, 800) if al[t]) - 199)          # day 200 is delay 1
        delays[shift] = float(np.mean(d))
    with capsys.disabled():
        print(f"\n[h={h}] mean delay: shift 1σ0 {delays[1.0]:.2f} d, shift 3σ0 {delays[3.0]:.2f} d "
              f"(bound {h / 2.5 + 3:.2f})")
    assert delays[3.0] < h / (3 - 0.5) + 3
    det = cs.Cusum(MU, SD, NO_CLIP, 0.5, h)                                    # noise-free 3σ0 step
    first = next(t for t in range(1, 100) if det.update(MU + 3 * SD)[1])
    assert first < h / (3 - 0.5) + 3


def test_3_reset_on_alarm_day():
    det = cs.Cusum(MU, SD, NO_CLIP, 0.5, 3)
    seq = [MU + 1.9 * SD] * 3                                                  # +1.4/day -> 1.4, 2.8, 4.2 (alarm)
    out = [det.update(x) for x in seq]
    assert [a for _, a in out] == [False, False, True] and out[-1][0] == pytest.approx(4.2)
    assert det.state == 0.0
    S, a = det.update(MU + 1.9 * SD)
    assert (S, a) == (pytest.approx(1.4), False)                               # normal accumulation after reset


def test_4_online_equals_batch_and_vectorised():
    rng = np.random.default_rng(2)
    e = MU + SD * (rng.standard_normal(3000) + np.r_[np.zeros(1500), np.full(1500, 0.7)])
    det = cs.Cusum(MU, SD, MU + 3 * SD, 0.5, 4)
    online = [det.update(x)[1] for x in e]
    batch = cs.run_batch(pd.Series(e), cs.Cusum(MU, SD, MU + 3 * SD, 0.5, 4))
    assert list(np.flatnonzero(online)) == list(np.flatnonzero(batch["alarm"].to_numpy()))
    assert cs.simulate_alarm_counts(e[None, :], MU, SD, MU + 3 * SD, 0.5, 4)[0] == sum(online)


def test_5_causality():
    rng = np.random.default_rng(3)
    e = MU + SD * rng.standard_normal(1000)
    D = 600
    alt = e.copy()
    alt[D + 1:] = rng.permutation(alt[D + 1:])
    a = cs.run_batch(pd.Series(e), cs.Cusum(MU, SD, MU + 3 * SD, 0.5, 4))
    b = cs.run_batch(pd.Series(alt), cs.Cusum(MU, SD, MU + 3 * SD, 0.5, 4))
    assert np.array_equal(a["S"].to_numpy()[:D + 1], b["S"].to_numpy()[:D + 1])


def test_6_truncation_caps_the_daily_increment(cfg):
    s = cfg["schemes"]["main"]
    det = cs.Cusum(s["mu0"], s["sigma0"], s["q99"], s["k"], 1e9)
    S, _ = det.update(s["mu0"] + 100 * s["sigma0"])
    cap = (s["q99"] - s["mu0"]) / s["sigma0"] - s["k"]
    assert S <= cap + 1e-12 and S == pytest.approx(cap)


# =========================== task 1: daily errors ===========================
def test_daily_errors_2014(R):
    d = pd.read_csv(m4.P["daily"])
    assert len(d) == 365 and d["day"].is_unique
    p = pd.read_parquet(ofr.PREDS)
    assert (p.groupby((p.index - pd.Timedelta("30min")).normalize()).size() == 48).all()
    r = json.load(open(ofr.REF_JSON))
    assert ofr.sha_arr(p["proba"], "float64") == r["hashes"]["proba_2014"]       # REF model's predictions
    assert np.isfinite(d["logloss"]).all() and (d["logloss"] > 0).all()
    st = pd.read_csv(m4.P["ref_stats"]).set_index("scheme")
    assert st.loc["main", "n_truncated"] == int(d["truncated"].sum()) and 3 <= st.loc["main", "n_truncated"] <= 4
    assert st.loc["main", "q99"] == np.quantile(d["logloss"], 0.99)
    et = np.minimum(d["logloss"], st.loc["main", "q99"])
    assert st.loc["main", "mu0"] == et.mean() and st.loc["main", "sigma0"] == et.std(ddof=1)
    diag = pd.read_csv(m4.P["diag"]).set_index("stat")["value"]
    assert all(f"acf_lag{l}_logloss_trunc" in diag for l in range(1, 8))
    assert all(f"mean_logloss_trunc_weekday{w}" in diag for w in range(7))
    assert diag["corr_logloss_misclass"] == pytest.approx(np.corrcoef(d["logloss"], d["misclass"])[0, 1])
    assert pd.DataFrame(R["daily"])["logloss"].to_numpy() == pytest.approx(d["logloss"].to_numpy(), abs=0)


def test_logloss_definition_by_hand():
    y = np.array([1] * 24 + [0] * 24)
    p = np.r_[np.full(24, 0.8), np.full(24, 0.0)]                              # 0 -> clipped to 1e-6
    idx = pd.date_range("2014-03-04 00:30", periods=48, freq="30min")
    got = cs.daily_logloss(y, p, idx).iloc[0]
    assert got == pytest.approx((24 * -np.log(0.8) + 24 * -np.log(1 - 1e-6)) / 48)


# =========================== task 3: ARL0 and h ===========================
def test_bootstrap_same_seed_identical(R):
    d = R["daily"]
    blocks = m4.week_blocks(d["day"])
    a = m4.arl_table(d, R["stats"], blocks, m4.boot_matrix(len(blocks)))
    b = m4.arl_table(d, R["stats"], blocks, m4.boot_matrix(len(blocks)))
    pd.testing.assert_frame_equal(a, b)
    assert len(blocks) == 51 and m4.boot_matrix(len(blocks)).shape == (1000, 522)


def test_arl0_monotone_and_selection_rule():
    t = pd.read_csv(m4.P["arl"])
    for sch, g in t.groupby("scheme"):
        g = g.sort_values("h")
        v = g["arl0_median"].to_numpy()
        assert all(v[i + 1] >= v[i] for i in range(len(v) - 1)), sch            # pairwise (inf >= inf is True)
        ok = g[g["arl0_median"] >= 730]
        assert g.loc[g["selected"], "h"].tolist() == [ok["h"].min()]
    main = t[(t["scheme"] == "main") & t["selected"]].iloc[0]
    assert main["h"] == 6 and main["arl0_median"] >= 730 and main["meets_target"]


def test_selected_h_on_real_2014_reported_not_reselected(cfg):
    path = pd.read_csv(m4.P["path"])
    main = path[path["scheme"] == "main"]
    assert len(main) == 365
    t = pd.read_csv(m4.P["arl"])
    assert cfg["schemes"]["main"]["h"] == t[(t["scheme"] == "main") & (t["arl0_median"] >= 730)]["h"].min()
    alarms = main.loc[main["alarm"], "day"].tolist()
    assert alarms == ["2014-09-26"]                                             # WARN: reported as is


def test_no_data_after_2014_read():
    log = json.load(open(m4.P["log"]))
    assert pd.Timestamp(log["latest_timestamp_read"]) <= pd.Timestamp("2015-01-01 00:00")
    allowed = {"results/price_hgbc/official/boost_price_ref.json", "results/price_hgbc/official/preds_2014.parquet",
               "comp9417_project_kit/data/nem_weather_halfhourly_2010_2026.csv"}
    assert {r["file"] for r in log["reads"]} <= allowed
    for f in ("src/price_hgbc/m4.py", "src/price_hgbc/cusum.py"):
        src = open(os.path.join(ROOT, f)).read()
        after_2014 = r"20(1[6-9]|2\d)-\d\d-\d\d|2015-(0[2-9]|1[0-2])-\d\d|2015-01-(0[2-9]|[12]\d|3[01])"
        assert not re.search(r"load_dataset\(|read_csv\(\s*(DATA_CSV|ofr\.DATA_CSV)|" + after_2014, src), f
    assert "nrows=len(expected)" in open(os.path.join(ofr.KIT, "ablation", "features.py")).read()   # P1 prefix read


# =========================== task 5: config ===========================
def test_config_valid_and_reproduces_2014(cfg):
    assert cs.validate_config(cfg)
    bad = json.loads(json.dumps(cfg))
    del bad["schemes"]["main"]["sigma0"]
    with pytest.raises(ValueError):
        cs.validate_config(bad)
    bad = dict(cfg, ref_update="C")
    with pytest.raises(ValueError):
        cs.validate_config(bad)
    det = cs.Cusum.from_config(m4.P["config"])
    d = pd.read_csv(m4.P["daily"])
    out = cs.run_batch(pd.Series(d["logloss"].to_numpy(), index=d["day"]), det)
    ref = pd.read_csv(m4.P["path"]).query("scheme == 'main'")
    assert np.array_equal(out["S"].to_numpy(), ref["S"].to_numpy())
    assert out["alarm"].tolist() == ref["alarm"].tolist()


def _numbers(x):
    if isinstance(x, bool) or x is None:
        return []
    if isinstance(x, (int, float)):
        return [float(x)]
    if isinstance(x, dict):
        return [n for v in x.values() for n in _numbers(v)]
    if isinstance(x, list):
        return [n for v in x for n in _numbers(v)]
    return []


def test_every_config_number_has_a_csv_source(cfg):
    pool = set()
    for f in os.listdir(m4.OUT):
        if f.endswith(".csv"):
            t = pd.read_csv(os.path.join(m4.OUT, f))
            for col in t.select_dtypes("number").columns:
                pool |= set(t[col].dropna().astype(float).tolist())
    nums = _numbers({k: v for k, v in cfg.items() if k not in ("code_sha256", "ref")}) + [cfg["ref"]["tau"]]
    missing = [n for n in nums if n not in pool]
    assert not missing, missing


def test_config_hashes(cfg):
    assert cfg["ref"]["sha256"] == OFFICIAL["boost_price_ref.json"] and cfg["ref"]["preds_2014_sha256"] == OFFICIAL["preds_2014.parquet"]
    for f, sha in cfg["code_sha256"].items():
        assert m4.sha_file(f) == sha, f"{f} changed after the config was written - re-run python -m src.price_hgbc.m4"
    assert cfg["design_sha256"] == DESIGN_SHA and cfg["code_commit"] is None
    assert cfg["ref_update"] == "A" and cfg["ref_update_decision"]["decided_by"] == "user"   # user decision 2026-11-11


# =========================== ref_update options ===========================
def test_ref_update_options(cfg):
    cur = cfg["schemes"]["main"]
    assert cs.reference_for_retrain("A", cur) == {k: cur[k] for k in ("mu0", "sigma0", "q99")}
    b = pd.read_csv(m4.P["optB"]).set_index("option")
    rng = np.random.default_rng(4)
    e = 0.45 + 0.15 * rng.standard_normal(365)
    got = cs.reference_for_retrain("B", cur, e)
    s = cs.reference_stats(e)
    assert got == {k: s[k] for k in ("mu0", "sigma0", "q99")}
    with pytest.raises(ValueError):
        cs.reference_for_retrain("B", cur, e[:100])
    with pytest.raises(ValueError):
        cs.reference_for_retrain("C", cur)
    assert b.loc["B (demo)", "n_days"] == 365 and b.loc["A", "q99"] == cur["q99"]


# =========================== figures ===========================
def test_figures_complete():
    d = pd.read_csv(m4.P["daily"])
    st = pd.read_csv(m4.P["ref_stats"]).set_index("scheme").loc["main"]
    path = pd.read_csv(m4.P["path"]).query("scheme == 'main'")
    delay = pd.read_csv(m4.P["delay"])
    for fig, ax in (m4.fig_err(d, st), m4.fig_path(path, 6), m4.fig_arl(pd.read_csv(m4.P["arl"])), m4.fig_delay(delay)):
        assert ax.get_title(loc="left").strip() and ax.get_xlabel().strip() and ax.get_ylabel().strip()
    _, ax = m4.fig_arl(pd.read_csv(m4.P["arl"]))
    assert ax.get_yscale() == "log"
    _, ax = m4.fig_delay(delay)
    assert sorted(t.get_text() for t in ax.texts) == sorted(f"{v:.1f}" for v in delay["mean_delay_days"])
    for k in ("fig_err", "fig_S", "fig_arl", "fig_delay"):
        assert os.path.getsize(m4.P[k]) > 0
