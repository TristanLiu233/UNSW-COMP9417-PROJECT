"""M3 self-checks (explanation of the official boost_price; nothing here may change REF or tau).
Tolerances fixed in advance: predictions / hashes / permutation scores bitwise; CSV round-trips 1e-6 pp;
metric re-computations 1e-9 pp.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.inspection import permutation_importance

from conftest import ROOT
from src.price_hgbc import ablate, m3_handoff, m3_permutation as mp, m3_spike as ms
from src.price_hgbc import m3_common as c
from src.price_hgbc import official_ref as ofr
from src.price_hgbc.kit import da

ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
# REF files at the start of M3 (M3 must not change them)
REF_FILES = {
    "boost_price_ref.json": "06b022774a10dad545dacb9a73d910a73eccd37d1ebc14d3f605d6e9ff210725",
    "preds_2014.parquet": "b4c9890d0149d69c902dcc10685db82c7b2d19d253a57a72ce0483a1f93c0c29",
    "boost_price_model.joblib": "9206389cfbebe6436550875273c172d5557ddb7e781252c06899281a38b128bf",
}


@pytest.fixture(scope="module")
def ref():
    return c.ref()


@pytest.fixture(scope="module")
def loaded():
    return c.load()


@pytest.fixture(scope="module")
def abl():
    return pd.read_csv(ablate.CSV)


def test_ref_untouched_and_no_post_2014_data(loaded):
    for f, want in REF_FILES.items():
        assert ofr.sha_file(os.path.join(ofr.OFFICIAL, f)) == want, f
    assert loaded[0].index.max() == pd.Timestamp("2015-01-01 00:00")           # P1 load_study prefix only
    assert da.market_day(loaded[0].index).max() == pd.Timestamp("2014-12-31")


# ---------------- 1. feature groups ----------------
def test_feature_groups_exact(ref):
    g = pd.read_csv(os.path.join(c.OUT, "m3_feature_groups.csv"))
    assert len(g) == 18 and g["feature"].is_unique
    assert list(g["feature"]) == ref["features"]                                # same names, REF order
    flat = [x for _, cols in c.GROUPS.values() for x in cols]
    assert sorted(flat) == sorted(ref["features"]) and len(flat) == len(set(flat))   # exclusive and complete
    assert set(g["group"]) == set(c.GROUPS) and not set(g["feature"]) & set(da.WEATHER_COLS)
    assert g.groupby("group").size().to_dict() == {k: len(v[1]) for k, v in c.GROUPS.items()}


# ---------------- 2. ablation ----------------
def test_full_features_equals_official_bitwise(abl, ref):
    full = abl.set_index("row").loc["full_features"]
    assert full["proba_sha256"] == ref["hashes"]["proba_2014"] and full["pred_sha256"] == ref["hashes"]["pred_2014"]
    p = pd.read_parquet(ablate.PREDS)
    off = pd.read_parquet(ofr.PREDS)
    assert p.index.equals(off.index) and np.array_equal(p["full_features"].to_numpy(), off["proba"].to_numpy())
    assert abs(full["BalAcc"] - ref["metrics_2014"]["BalAcc"]) < 1e-6 and full["n_features"] == 18


def test_each_group_ablated_exactly_once(abl):
    assert list(abl["row"]) == ["full_features"] + list(c.GROUPS)
    for _, r in abl.iloc[1:].iterrows():
        dropped = json.loads(r["dropped"])
        assert dropped == c.GROUPS[r["row"]][1]
        assert r["n_dropped"] == len(dropped) and r["n_features"] == 18 - len(dropped)
    g = pd.read_csv(os.path.join(c.OUT, "m3_feature_groups.csv")).groupby("group").size()
    assert all(abl.set_index("row").loc[k, "n_dropped"] == g[k] for k in c.GROUPS)


def test_ablation_settings_and_intervals(abl, ref):
    meta = json.load(open(ablate.JSON))
    assert meta["params"] == ref["params"] and meta["tau"] == ref["tau"] == 0.45
    assert (meta["n_boot"], meta["seed"], meta["n_days"]) == (1000, 0, 365)
    assert (abl["tau_fixed"] == 0.45).all()
    assert set(abl["tau_2014_optimistic"]) <= set(ablate.P1_THRESHOLDS)
    full = abl.iloc[0]
    assert (full["delta_BalAcc_pp"], full["delta_lo"], full["delta_hi"]) == (0, 0, 0)
    for _, r in abl.iloc[1:].iterrows():
        assert abs(r["delta_BalAcc_pp"] - (r["BalAcc"] - full["BalAcc"])) < 1e-6
        assert r["delta_lo"] <= r["delta_hi"]
        if r["delta_lo"] <= 0 <= r["delta_hi"]:
            assert r["verdict"].startswith("未检测到显著贡献")
        else:
            assert "未检测" not in r["verdict"]


def test_ablation_bootstrap_paired_and_recomputable(abl, loaded):
    p = pd.read_parquet(ablate.PREDS)
    off = pd.read_parquet(ofr.PREDS)
    y = off["y"].to_numpy()
    days = sorted(set(da.market_day(p.index)))
    B = c.boot_indices(len(days))
    assert c.sha(np.ascontiguousarray(B).tobytes()) == json.load(open(ablate.JSON))["boot_index_sha256"]
    s_full = c.boot_balacc(y, (p["full_features"] >= 0.45).astype(int), p.index, B)
    for g in ("threshold_gap", "season"):
        lo, hi = c.ci(c.boot_balacc(y, (p[g] >= 0.45).astype(int), p.index, B) - s_full)
        r = abl.set_index("row").loc[g]
        assert abs(lo - r["delta_lo"]) < 1e-6 and abs(hi - r["delta_hi"]) < 1e-6
    # count-based BalAcc == official metric on resampled rows
    pred = (p["slot"] >= 0.45).astype(int).to_numpy()
    cnt = c.day_counts(y, pred, p.index)
    dd = np.asarray(da.market_day(p.index))
    for b in B[:3]:
        rows = np.concatenate([np.flatnonzero(dd == cnt.index[j]) for j in b])
        s = cnt.to_numpy()[b].sum(axis=0)
        got = c.balacc_from_counts(*[s[cnt.columns.get_loc(k)] for k in ("tp", "fn", "tn", "fp")])[0]
        assert abs(got - da.balanced_accuracy(y[rows], pred[rows])) < 1e-9


def test_ablation_rerun_identical(abl):
    tab, meta, _ = ablate.run()
    assert list(tab["proba_sha256"]) == list(abl["proba_sha256"])
    for k in ("BalAcc", "Acc", "delta_lo", "delta_hi", "BalAcc_2014_optimistic"):
        assert np.allclose(tab[k], abl[k], rtol=0, atol=1e-6), k
    assert meta["boot_index_sha256"] == json.load(open(ablate.JSON))["boot_index_sha256"]


@pytest.mark.slow
def test_ablation_two_fresh_processes_identical(tmp_path):
    outs = []
    for k in (1, 2):
        f = tmp_path / f"a{k}.json"
        subprocess.run([sys.executable, "-m", "src.price_hgbc.ablate", "--fingerprint", str(f)], cwd=ROOT, env=ENV,
                       check=True, capture_output=True, timeout=900)
        outs.append(json.load(open(f)))
    assert outs[0] == outs[1]


def test_p1_retune_rule():
    y = np.array([0] * 50 + [1] * 50)
    p = np.r_[np.full(50, 0.2), np.full(50, 0.8)]                              # all thresholds tie -> 0.50
    assert ablate.p1_retune(y, p)[0] == 0.50
    p = np.r_[np.full(50, 0.42), np.full(50, 0.43)]                            # no P1 threshold separates: all 50 -> 0.50
    assert ablate.p1_retune(y, p)[:2] == (0.50, 50.0)
    p = np.r_[np.full(50, 0.38), np.full(50, 0.42)]                            # only 0.40 separates -> 0.40, BalAcc 100
    assert ablate.p1_retune(y, p)[:2] == (0.40, 100.0)


# ---------------- 3. permutation importance ----------------
def test_scorer_uses_tau_045_and_equals_ref(loaded, ref):
    import joblib
    _, X, truth, cols, _, va = loaded
    model = joblib.load(ofr.MODEL)
    Xv, yv = X.loc[va, cols], truth.loc[va, "nsw_price_up"].to_numpy()
    s = mp.make_scorer(ref["tau"])
    assert s(model, Xv, yv) == ref["metrics_2014"]["BalAcc"]
    at_half = da.balanced_accuracy(yv, (model.predict_proba(Xv)[:, 1] >= 0.5).astype(int))
    assert at_half != ref["metrics_2014"]["BalAcc"]                             # the 0.5 cut would differ
    assert json.load(open(mp.JSON))["baseline_BalAcc"] == ref["metrics_2014"]["BalAcc"]


def test_permutation_tables(ref):
    col, grp = pd.read_csv(mp.COL_CSV), pd.read_csv(mp.GRP_CSV)
    reps = [f"rep{k}" for k in range(10)]
    assert set(col["feature"]) == set(ref["features"]) and len(col) == 18
    assert set(grp["group"]) == set(c.GROUPS) and len(grp) == 6
    for t in (col, grp):
        assert all(r in t.columns for r in reps) and "rep10" not in t.columns
        assert np.allclose(t[reps].mean(axis=1), t["importance_mean_pp"], atol=1e-6)
    m = json.load(open(mp.JSON))
    assert (m["n_repeats"], m["random_state"], m["refit"]) == (10, 0, False)
    flagged = set(col.loc[~col["interpret_alone"], "feature"])
    must = {f"weekday_{i}" for i in range(7)} | {"weekend", "slot_sin", "slot_cos", "season_sin", "season_cos",
                                                "lag1_price_gap", "nsw_price_lag1", "prev_price_mean"}
    assert flagged == must


def test_per_column_equals_sklearn_and_group_scheme_matches(loaded, ref):
    import joblib
    _, X, truth, cols, _, va = loaded
    model = joblib.load(ofr.MODEL)
    Xv, yv = X.loc[va, cols], truth.loc[va, "nsw_price_up"].to_numpy()
    scorer = mp.make_scorer(ref["tau"])
    pi = permutation_importance(model, Xv, yv, scoring=scorer, n_repeats=10, random_state=0)
    col = pd.read_csv(mp.COL_CSV).set_index("feature").loc[cols]
    assert np.allclose(col[[f"rep{k}" for k in range(10)]].to_numpy(), pi.importances, atol=1e-6)
    some = ["lag1_price_gap", "nsw_demand_lag7", "season_sin"]
    gp = mp.group_permutation(model, Xv, yv, scorer, {x: [x] for x in some})
    base = scorer(model, Xv, yv)
    for x in some:                                                              # singleton groups == sklearn, bitwise
        assert np.array_equal(base - gp[x], pi.importances[cols.index(x)])
    g2 = mp.group_permutation(model, Xv, yv, scorer, {"slot": c.GROUPS["slot"][1]})
    grp = pd.read_csv(mp.GRP_CSV).set_index("group")
    assert np.allclose(base - g2["slot"], grp.loc["slot", [f"rep{k}" for k in range(10)]].to_numpy(dtype=float), atol=1e-6)


def _texts(ax):
    return [t.get_text() for t in ax.texts]


def test_permutation_figures_match_csv():
    col, grp = pd.read_csv(mp.COL_CSV), pd.read_csv(mp.GRP_CSV)
    for fig_fn, df in ((mp.plot_columns, col), (mp.plot_groups, grp)):
        fig, ax = fig_fn(df)
        assert ax.get_title(loc="left").strip() and ax.get_xlabel().strip()
        assert sorted(_texts(ax)) == sorted(f"{v:.2f}" for v in df["importance_mean_pp"])
    assert os.path.getsize(mp.FIG_COL) > 0 and os.path.getsize(mp.FIG_GRP) > 0


# ---------------- 4. price spikes ----------------
def test_spike_2014_partition_and_metrics(loaded):
    t = pd.read_csv(ms.CSV_2014).set_index("group")
    parts = t.loc[ms.LABELS]
    assert parts["days"].sum() == t.loc["all", "days"] == 365
    assert parts["rows"].sum() == t.loc["all", "rows"] == 17520
    assert ms.BINS == [-np.inf, 300.0, 1000.0, np.inf] and ms.LABELS == ["<=300", "(300,1000]", ">1000"]
    for g, r in parts.iterrows():
        assert r["warn_lt_30_days"] == (r["days"] < 30)
        assert r["HGBC_tn"] + r["HGBC_fp"] + r["HGBC_fn"] + r["HGBC_tp"] == r["rows"]
        if r["classes_present"] == 1:                                             # kit convention: recall of that class
            assert abs(r["HGBC_BalAcc"] - (r["HGBC_recall_0"] if np.isnan(r["HGBC_recall_1"]) else r["HGBC_recall_1"])) < 1e-9
    assert abs(t.loc["all", "b1_BalAcc"] - 73.98617484807123) < 1e-6
    assert abs(t.loc["all", "HGBC_BalAcc"] - 79.1867969333506) < 1e-6


def test_spike_each_day_in_one_group(loaded):
    data = loaded[0]
    days = sorted(set(da.market_day(pd.read_parquet(ofr.PREDS).index)))
    grp, dmax = ms.spike_group_of_target_days(data, days)
    assert grp.index.is_unique and set(grp.unique()) <= set(ms.LABELS) and len(grp) == 365
    assert ((grp == "<=300") == (dmax <= 300)).all() and ((grp == ">1000") == (dmax > 1000)).all()
    d = pd.Timestamp("2014-03-04")                                               # D = 2014-03-03, recomputed by hand
    assert dmax[d] == data.loc[d - pd.Timedelta("1D") + pd.Timedelta("30min"): d, "nsw_price"].max()


def test_spike_b1_from_official_baseline(loaded):
    data = loaded[0]
    t = pd.read_csv(ms.CSV_2014).set_index("group")
    for d in (pd.Timestamp("2014-01-01"), pd.Timestamp("2014-07-15")):
        h, w = da.make_inputs(data, d)
        assert np.array_equal(ms.b1_2014(data, [d]).to_numpy(), da.baseline(h, w, 1)["nsw_price_up"].to_numpy())
    assert t.loc["all", "b1_tn"] + t.loc["all", "b1_tp"] == round(t.loc["all", "b1_Acc"] / 100 * 17520)


def test_spike_label_only(loaded):
    t = pd.read_csv(ms.CSV_LABEL).set_index("group")
    parts = t.loc[ms.LABELS]
    assert parts["days"].sum() == t.loc["all", "days"] == 1825                 # 2010-01-02 .. 2014-12-31
    assert parts["rows"].sum() == t.loc["all", "rows"] == 1825 * 48
    assert (parts["warn_lt_30_days"] == (parts["days"] < 30)).all()
    m = json.load(open(ms.JSON))["claim_9_7pct"]
    assert m["high_group_days"] == t.loc[">1000", "days"]
    assert abs(m["high_group_class1_share"] - t.loc[">1000", "true_class1_share"]) < 1e-6
    assert m["matches_9_7_at_0_1pp"] == (abs(m["high_group_class1_share"] - 9.7) < 0.05)


def test_spike_figures_match_csv():
    t14, tl = pd.read_csv(ms.CSV_2014), pd.read_csv(ms.CSV_LABEL)
    fig, ax = ms.plot_balacc(t14)
    a = t14[t14["group"] != "all"]
    assert ax.get_title(loc="left").strip() and ax.get_xlabel().strip() and ax.get_ylabel().strip()
    assert sorted(_texts(ax)) == sorted([f"{v:.1f}" for v in a["HGBC_BalAcc"]] + [f"{v:.1f}" for v in a["b1_BalAcc"]])
    fig, ax = ms.plot_share(t14, tl)
    b = tl[tl["group"] != "all"]
    assert ax.get_title(loc="left").strip() and ax.get_ylabel().strip()
    assert sorted(_texts(ax)) == sorted([f"{v:.1f}" for v in a["true_class1_share"]] + [f"{v:.1f}" for v in b["true_class1_share"]])
    for f in (ms.FIG_BA, ms.FIG_SHARE, ablate.FIG):
        assert os.path.getsize(f) > 0


def test_ablation_figure_matches_csv(abl):
    fig, ax = ablate.plot(abl)
    assert ax.get_title(loc="left").strip() and ax.get_xlabel().strip()
    assert sorted(_texts(ax)) == sorted(f"{v:+.2f}" for v in abl.iloc[1:]["delta_BalAcc_pp"])


# ---------------- 5. handoff ----------------
def test_handoff_content():
    md = open(m3_handoff.OUT_MD, encoding="utf-8").read()
    assert "这些是解释性结果，不用于重新选型" in md
    assert "未发送给 P1" in md and "已发送" not in md
    assert md == m3_handoff.build()                                             # regenerated from the saved outputs
    abl = pd.read_csv(ablate.CSV).set_index("row")
    assert f"{abl.loc['threshold_gap', 'delta_BalAcc_pp']:+.3f}" in md
    col = pd.read_csv(mp.COL_CSV)
    for feat in col.head(10)["feature"]:
        assert f"`{feat}`" in md
    for contrast in m3_handoff.P1_CONTRASTS:
        assert f"`{contrast}`" in md
