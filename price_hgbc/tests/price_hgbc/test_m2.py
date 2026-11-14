"""[ARCHIVED in M2.5: M2 is a non-formal sensitivity analysis; run with -m archive]
M2 self-checks: extended features, 2a feature-set decision, 48-config grid + M1_ref, selection,
paired day bootstrap, 2014 confirmation, outputs. Tolerances fixed in advance:
features / hashes / labels bitwise; metric re-computations 1e-9 pp; CSV round-trips 1e-6 pp.
"""
import json
import os
import sys

import holidays
import numpy as np
import pandas as pd
import pytest

from conftest import KIT, ROOT
from src.price_hgbc import eval as ev
from src.price_hgbc import extra_features as xf
from src.price_hgbc import features_simple as fs
from src.price_hgbc import m1_baseline as m1b
from src.price_hgbc import m2
from src.price_hgbc.kit import ARCHIVE, DATA_CSV, RESULTS, da  # noqa: F401

pytestmark = pytest.mark.archive
D1, HALF = pd.Timedelta("1D"), pd.Timedelta("30min")
P = m2.paths()
SAMPLE_DAYS = [pd.Timestamp(d) for d in ("2010-01-29", "2012-07-04", "2013-06-30", "2013-07-01",
                                         "2014-01-27", "2014-12-31", "2022-06-15", "2026-10-31")]
# SHA-256 of the M1 artefacts at the end of M1 (must never change; paths moved to the archive in M2.5)
M1_SHA = {
    "results/price_hgbc/_archive_m1m2/m1.csv": "64e8f3875fbe265a89b7f8ca888317e971c9ec23f4e2105c6685a72808087774",
    "results/price_hgbc/_archive_m1m2/m1/m1_pred_2014.csv.gz": "fd2e4b46ce935cfd674dad60181b39bb4f51d9d69cb1a0bcfa524b5139e293ac",
    "results/price_hgbc/_archive_m1m2/m1/m1_run.json": "408ce7e401fb693b93f43eebb10c626285c8b3698dda6d50545ea9982d8b8d8c",
    "results/price_hgbc/_archive_m1m2/m1/m1_tau_curve_2013.csv": "cdf34ad29ef63ff6a6672671147fbfc6bbb988f01c4bd7e4296db109624866ee",
    "src/price_hgbc/features_simple.py": "d8a8ebd5fe963f2822861019057fed886952fcb20271bfc1dad66443196cc7ff",
    "src/price_hgbc/eval.py": "9a32a2b706430c87cac61d0e9d4fc026bcad9a9376d22fc126fcf6dae4ea4b43",
    "src/price_hgbc/m1_baseline.py": "27d031296399510d0ba125329e340ecf0e5c5813abd04fa871614c0125ab6135",
}


@pytest.fixture(scope="module")
def df():
    return da.load_dataset(DATA_CSV)


@pytest.fixture(scope="module")
def X(df):
    return xf.build_ext_features(df)


@pytest.fixture(scope="module")
def data(df):
    return m2.Data(df)


@pytest.fixture(scope="module")
def grid():
    return pd.read_csv(P["grid"])


@pytest.fixture(scope="module")
def best():
    return json.load(open(P["best"]))


def _rows(index, day):
    return index[(index > day) & (index <= day + D1)]


def _same(a, b):
    return (list(a.columns) == list(b.columns) and a.dtypes.equals(b.dtypes) and a.index.equals(b.index)
            and np.ascontiguousarray(a.to_numpy()).tobytes() == np.ascontiguousarray(b.to_numpy()).tobytes())


# =========================== guards ===========================
def test_m1_formal_artefacts_unchanged():
    for rel, want in M1_SHA.items():
        assert m1b.sha(open(os.path.join(ROOT, rel), "rb").read()) == want, rel


# =========================== task 1: extended features ===========================
@pytest.mark.parametrize("day", SAMPLE_DAYS)
@pytest.mark.parametrize("col", ["nsw_price", "nsw_demand"])
def test_leakage_shuffle_next_day_price_or_demand(df, X, day, col):
    rows = _rows(df.index, day)
    alt = df.copy()
    alt.loc[rows, col] = np.random.default_rng(0).permutation(alt.loc[rows, col].to_numpy())
    assert not np.array_equal(alt.loc[rows, col].to_numpy(), df.loc[rows, col].to_numpy())
    assert _same(xf.build_ext_features(alt).loc[rows], X.loc[rows])
    h, w = da.make_inputs(alt, day)
    assert _same(xf.ext_features_for_day(h, w), X.loc[rows])


@pytest.mark.parametrize("day", SAMPLE_DAYS)
def test_up_lag1_official_and_extended_matches_m1(df, X, day):
    h, w = da.make_inputs(df, day)
    assert int((X.loc[w.index, "up_lag1"].to_numpy() != da.baseline(h, w, 1)["nsw_price_up"].to_numpy()).sum()) == 0
    assert _same(X.loc[w.index, fs.M1_FEATURES], fs.build_m1_features(df).loc[w.index])


def test_up_lag1_zero_mismatch_all_rows(df, X):
    p = df["nsw_price"]
    mean = pd.Series({d: p.loc[d + HALF: d + D1].mean() for d in da.complete_days(df)})
    Xs = X.loc[da.market_day(X.index) >= pd.Timestamp("2010-01-08")]
    pbar = mean.reindex(da.market_day(Xs.index) - D1).to_numpy()
    assert int(((p.reindex(Xs.index - D1).to_numpy() > pbar).astype(int) != Xs["up_lag1"].to_numpy()).sum()) == 0


@pytest.mark.parametrize("day", SAMPLE_DAYS)
def test_extra_columns_use_day_D_and_D_plus_1(df, X, day):
    """Independent re-computation from the 48 rows of D (and D+1 weather) for target day D+1 = day."""
    D = day - D1
    pD = df["nsw_price"].loc[D + HALF: D + D1]
    assert len(pD) == 48
    r = X.loc[_rows(df.index, day)]
    assert (r["prev_price_max"] == pD.max()).all()
    assert (r["prev_price_std"] == pD.std(ddof=0)).all()
    assert (r["prev_neg_share"] == (pD < 0).sum() / 48).all()
    assert (r["prev_n_gt300"] == (pD > 300).sum()).all() and (r["prev_n_gt1000"] == (pD > 1000).sum()).all()
    assert (r["prev_spike_ratio"] == pD.max() / max(pD.mean(), xf.SPIKE_FLOOR)).all()
    tD = df["syd_temp"].loc[D + HALF: D + D1].to_numpy()
    tD1 = df["syd_temp"].loc[day + HALF: day + D1].to_numpy()
    assert np.array_equal(r["syd_temp_diff"].to_numpy(), tD1 - tD)
    hol = holidays.country_holidays("AU", subdiv="NSW", years=[day.year])
    assert (r["is_holiday"] == int(day.date() in hol)).all()


def test_spike_ratio_finite_and_floor_rule(df, X):
    Xs, _, _ = fs.m1_design(df, "2010-01-08", "2026-10-31", X)
    assert np.isfinite(Xs["prev_spike_ratio"]).all()
    days_mean = df["nsw_price"].groupby(da.market_day(df.index)).mean()
    assert int((days_mean < xf.SPIKE_FLOOR).sum()) == 0      # recorded: floor never triggers on the released data
    # synthetic: zero and negative daily means -> denominator floored at 1 $/MWh
    sub = df.loc["2014-01-01 00:30":"2014-01-11 00:00"].copy()
    for k, level in ((3, 0.0), (5, -50.0)):
        d = pd.Timestamp("2014-01-01") + k * D1
        rows = _rows(sub.index, d)
        sub.loc[rows, "nsw_price"] = level + np.linspace(-10, 10, 48)
        ex = xf.build_extra(sub).loc[_rows(sub.index, d + D1), "prev_spike_ratio"]
        assert np.isfinite(ex).all() and (ex == sub.loc[rows, "nsw_price"].max() / 1.0).all()


def test_count_and_share_ranges(df, X):
    Xs, _, _ = fs.m1_design(df, "2010-01-08", "2026-10-31", X)
    for c in ("prev_n_gt300", "prev_n_gt1000"):
        assert Xs[c].between(0, 48).all() and (Xs[c] == Xs[c].round()).all()
    assert Xs["prev_neg_share"].between(0, 1).all()
    assert (Xs["prev_n_gt1000"] <= Xs["prev_n_gt300"]).all()
    assert set(Xs["is_holiday"].unique()) == {0, 1}


def test_holidays_around_australia_day_2014_and_observed_days(X):
    def flag(d):
        return set(X.loc[_rows(X.index, pd.Timestamp(d)), "is_holiday"])
    assert flag("2014-01-25") == {0} and flag("2014-01-26") == {0}          # Sunday 26 Jan is not the NSW holiday
    assert flag("2014-01-27") == {1}                                         # Monday 27 Jan (Australia Day, NSW)
    assert flag("2014-01-28") == {0}
    assert flag("2015-12-28") == {1}                                         # Boxing Day (observed)
    assert flag("2022-01-03") == {1}                                         # New Year's Day (observed)
    hol = xf.nsw_holidays(range(2010, 2027))
    per_day = X["is_holiday"].groupby(da.market_day(X.index)).agg(["min", "max"])
    assert (per_day["min"] == per_day["max"]).all()                         # constant within a market day
    assert np.array_equal(per_day["max"].to_numpy(), np.array([int(d.date() in hol) for d in per_day.index]))
    assert int(per_day["max"].sum()) == len([d for d in hol if d <= pd.Timestamp("2026-10-31").date()])


def test_no_unexpected_nan_inf(df, X):
    Xs, _, dropped = fs.m1_design(df, "2010-01-01", "2026-10-31", X)
    assert np.isfinite(Xs.to_numpy(dtype=float)).all()
    assert dropped == list(pd.date_range("2010-01-01", "2010-01-07"))


def test_column_order_and_determinism(df, X):
    assert list(X.columns) == xf.EXT_FEATURES and len(xf.EXT_FEATURES) == 26
    assert xf.EXT_FEATURES[:18] == fs.M1_FEATURES
    assert _same(xf.build_ext_features(df), X)


def test_bulk_equals_day_by_day(df, X):
    rng = np.random.default_rng(2)
    days = fs.day_range(df, "2010-01-08", "2026-10-31")
    for d in [days[i] for i in rng.choice(len(days), 25, replace=False)] + SAMPLE_DAYS:
        h, w = da.make_inputs(df, d)
        assert _same(xf.ext_features_for_day(h, w), X.loc[w.index]), d


def test_reuses_p1_definitions(df, X):
    """prev_price_max / negative share / price gap equal the group's ablation (P1) columns 2010-01-29..2014."""
    sys.dont_write_bytecode = True
    if KIT not in sys.path:
        sys.path.insert(0, KIT)
    from ablation import features as gf
    gX, _ = gf.build_features(gf.load_study(DATA_CSV))
    rows = gX.index[da.market_day(gX.index) >= pd.Timestamp("2010-01-29")]
    assert np.array_equal(X.loc[rows, "prev_price_max"].to_numpy(), gX.loc[rows, "prev_price_max"].to_numpy())
    assert np.array_equal(X.loc[rows, "prev_neg_share"].to_numpy(), gX.loc[rows, "prev_negative_slots"].to_numpy() / 48)
    assert np.array_equal(X.loc[rows, "price_gap_lag1"].to_numpy(), gX.loc[rows, "lag1_price_gap"].to_numpy())


def test_feature_groups_file():
    g = pd.read_csv(P["groups"])
    assert list(g["feature"]) == xf.EXT_FEATURES and list(g["order"]) == list(range(26))
    assert set(g["group"]) == {"Price lag", "Price distribution", "Price spike", "Weather", "Calendar"}
    assert set(g.loc[g["categorical"], "feature"]) == {"period", "weekday"}
    assert g["allowed_at_prediction"].all() and int(g["in_M1"].sum()) == 18
    for c in ("feature", "group", "source", "date", "categorical", "allowed_at_prediction"):
        assert g[c].notna().all()


# =========================== 2a ===========================
def test_2a_decision_follows_rule():
    s = json.load(open(P["2a_json"]))
    t = pd.read_csv(P["2a"]).set_index("feature_set")
    gain = t.loc["M1_ext", "val_BalAcc"] - t.loc["M1", "val_BalAcc"]
    assert abs(gain - s["gain_pp"]) < 1e-6
    lo, hi = s["diff_ci95"]
    expect = "M1_ext" if gain >= 0.5 and (lo > 0 or hi < 0) else "M1"
    assert s["winner"] == expect
    assert s["n_boot"] == 1000 and s["seed"] == 0 and s["n_days"] == 365
    assert json.loads(t.loc["M1", "params_json"]) == json.loads(t.loc["M1_ext", "params_json"]) | {
        "categorical_features": ["period", "weekday"]}


def test_2a_m1_arm_equals_grid_m1_ref(grid):
    t = pd.read_csv(P["2a"]).set_index("feature_set")
    ref = grid.set_index("config_id").loc["M1_ref"]
    assert t.loc["M1", "proba_sha256"] == ref["proba_sha256"] and t.loc["M1", "tau"] == ref["tau"]


# =========================== task 2: grid ===========================
def test_grid_has_48_unique_configs_plus_reference(grid):
    assert grid["config_id"].is_unique and (grid["status"] == "done").all()
    g = grid[grid["config_id"] != "M1_ref"]
    assert len(g) == 48 and (grid["config_id"] == "M1_ref").sum() == 1
    got = set(map(tuple, g[["max_leaf_nodes", "learning_rate", "max_iter", "l2_regularization"]].to_numpy().tolist()))
    want = {(a, b, c, d) for a in (15, 31, 63) for b in (0.05, 0.1) for c in (100, 200) for d in (0, 0.1, 1, 10)}
    assert got == want
    assert list(g["config_id"]) == [c["config_id"] for c in m2.grid_configs()]


def test_grid_params_fixed_and_match_definition(grid):
    winner = json.load(open(P["2a_json"]))["winner"]
    for _, r in grid.iterrows():
        p = json.loads(r["params_json"])
        fset = r["feature_set"]
        cfg = m2.m1_ref_config() if r["config_id"] == "M1_ref" else m2.cfg_from_row(r)
        assert p == json.loads(json.dumps(m2.params_for(cfg, fset), sort_keys=True))
        assert p["early_stopping"] is False and p["class_weight"] is None
        assert p["random_state"] == (0 if r["config_id"] == "M1_ref" else 17)
        assert fset == ("M1" if r["config_id"] == "M1_ref" else winner)
    ref = json.loads(grid.set_index("config_id").loc["M1_ref", "params_json"])
    m1 = json.load(open(os.path.join(ARCHIVE, "m1", "m1_run.json")))["params"]
    assert ref == json.loads(json.dumps(m1, sort_keys=True))                # M1_ref = M1 parameters verbatim


def test_each_config_has_41_tau_points(grid):
    assert (grid["n_tau_points"] == 41).all()
    c = pd.read_csv(P["curves"])
    assert (c.groupby("config_id").size() == 41).all() and set(c["config_id"]) == set(grid["config_id"])
    for cid, cc in c.groupby("config_id"):
        r = grid.set_index("config_id").loc[cid]
        assert np.allclose(cc["tau"], ev.TAU_GRID)
        assert ev.select_tau(cc.reset_index(drop=True))[0] == r["tau"]


def test_fold_dates_chronological(grid, data):
    assert m2.TRAIN_START == "2010-01-29"
    assert (grid["train_range"] == "2010-01-29..2012-12-31 | 2010-01-29..2013-06-30").all()
    assert (grid["valid_range"] == "2013-01-01..2013-06-30 | 2013-07-01..2013-12-31").all()
    for end, vs, ve in m2.FOLDS:
        Xtr, _ = data.design("M1", m2.TRAIN_START, end)
        Xva, _ = data.design("M1", vs, ve)
        dtr, dva = da.market_day(Xtr.index), da.market_day(Xva.index)
        assert dtr.min() == pd.Timestamp("2010-01-29") and dtr.max() < dva.min() and dva.max() < pd.Timestamp("2014-01-01")
    assert (grid["val_rows"] == 17520).all()


def test_2014_not_used_in_selection(df, grid):
    """Noise in every market value of days >= 2014-01-01 leaves validation results bitwise identical."""
    alt = df.copy()
    late = da.market_day(alt.index) >= m1b.CUTOFF
    rng = np.random.default_rng(7)
    for c in da.MARKET_COLS:
        alt.loc[late, c] = rng.normal(100, 300, int(late.sum()))
    d = m2.Data(alt)
    g = grid.set_index("config_id")
    for cid in ("M1_ref", g.index[0]):
        r = g.loc[cid]
        cfg = m2.m1_ref_config() if cid == "M1_ref" else m2.cfg_from_row({"config_id": cid, **r.to_dict()})
        row, _ = m2.validation_run(d, r["feature_set"], cfg)
        assert row["proba_sha256"] == r["proba_sha256"] and row["tau"] == r["tau"]


def test_random_state_unused_for_m1_ref(data, grid):
    cfg = m2.m1_ref_config()
    p = m2.params_for(cfg, "M1")
    p17 = dict(p, random_state=17)
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xtr, ytr = data.design("M1", m2.TRAIN_START, "2012-12-31")
    Xva, _ = data.design("M1", "2013-01-01", "2013-06-30")
    a = HistGradientBoostingClassifier(**p).fit(Xtr, ytr).predict_proba(Xva)[:, 1]
    b = HistGradientBoostingClassifier(**p17).fit(Xtr, ytr).predict_proba(Xva)[:, 1]
    assert np.array_equal(a, b)


def test_tau_endpoint_flag_matches_tau(grid):
    assert (grid["tau_at_endpoint"] == grid["tau"].isin([0.30, 0.70])).all()


def test_resume_equals_one_shot(tmp_path, data, grid):
    cfgs = [c for c in m2.grid_configs() if c["max_leaf_nodes"] == 15 and c["max_iter"] == 100
            and c["learning_rate"] == 0.1][:3]
    fset = grid.set_index("config_id").loc[cfgs[0]["config_id"], "feature_set"]
    quiet = lambda *a: None  # noqa: E731
    one = m2.run_grid(data, fset, cfgs, str(tmp_path / "one.csv"), str(tmp_path / "one_c.csv"), log=quiet)
    g2, c2 = str(tmp_path / "two.csv"), str(tmp_path / "two_c.csv")
    m2.run_grid(data, fset, cfgs, g2, c2, max_new=1, log=quiet)                # interrupted after 1
    partial = pd.read_csv(g2)
    crashed = pd.DataFrame([{"config_id": cfgs[1]["config_id"], "feature_set": fset, "status": "running"}])
    pd.concat([partial, crashed]).to_csv(g2, index=False)                     # simulated crash mid-config
    two = m2.run_grid(data, fset, cfgs, g2, c2, log=quiet)
    one, two = pd.read_csv(tmp_path / "one.csv"), pd.read_csv(g2)              # compare what is on disk
    keep = [c for c in one.columns if c not in ("fit_seconds", "env")]
    assert len(two) == 3 and two["config_id"].is_unique and (two["status"] == "done").all()
    pd.testing.assert_frame_equal(one[keep].reset_index(drop=True), two[keep].reset_index(drop=True))
    assert pd.read_csv(c2).groupby("config_id").size().tolist() == [41, 41, 41]
    # resuming a finished grid runs nothing and leaves the file unchanged
    before = open(g2, "rb").read()
    calls = []
    m2.run_grid(data, fset, cfgs, g2, c2, log=calls.append)
    assert calls == [] and open(g2, "rb").read() == before
    # agrees with the full grid run
    full = grid.set_index("config_id")
    for _, r in two.iterrows():
        assert r["proba_sha256"] == full.loc[r["config_id"], "proba_sha256"]
    # a row whose definition changed is refused, not mixed in
    bad = pd.read_csv(g2)
    bad.loc[0, "params_json"] = "{}"
    bad.to_csv(g2, index=False)
    with pytest.raises(ValueError):
        m2.run_grid(data, fset, cfgs, g2, c2, log=quiet)


def test_best_config_rerun_identical(data, grid, best):
    r = grid.set_index("config_id").loc[best["best_config_id"]]
    cfg = m2.cfg_from_row({"config_id": best["best_config_id"], **r.to_dict()})
    row, _ = m2.validation_run(data, r["feature_set"], cfg)
    for k in ("proba_sha256", "pred_sha256", "tau", "X_valid_sha256", "y_valid_sha256"):
        assert row[k] == r[k], k
    for k in ("val_BalAcc", "val_Acc"):                                      # CSV stores 10 significant digits
        assert abs(row[k] - r[k]) < 1e-6, k


# =========================== task 3: selection / bootstrap ===========================
def test_rank_rule_on_constructed_grid():
    rows = []
    for cid, L, it, l2, bal in (("a", 63, 200, 0.0, 80.00), ("b", 31, 200, 0.0, 79.97), ("c", 31, 100, 0.0, 79.96),
                                ("d", 31, 100, 10.0, 79.955), ("e", 15, 100, 0.0, 79.90)):
        rows.append({"config_id": cid, "max_leaf_nodes": L, "max_iter": it, "l2_regularization": l2,
                     "learning_rate": 0.1, "val_BalAcc": bal, "status": "done"})
    ranked, tied = m2.rank(pd.DataFrame(rows))
    assert set(tied) == {"a", "b", "c", "d"}                                 # e is 0.10 pp below the top
    assert ranked.iloc[0]["config_id"] == "d"                               # 31 leaves, 100 iter, larger l2
    assert list(ranked["config_id"]) == ["d", "a", "b", "c", "e"]


def test_selection_reproduces_from_grid(grid, best):
    ranked, tied = m2.rank(grid)
    assert ranked.iloc[0]["config_id"] == best["best_config_id"]
    top5 = pd.read_csv(P["top5"])
    assert list(top5["config_id"]) == list(ranked["config_id"][:5])
    g = grid.set_index("config_id")
    for _, r in top5.iterrows():
        assert abs(r["val_BalAcc"] - g.loc[r["config_id"], "val_BalAcc"]) < 1e-6
    assert set(best["validation"]["tied_with_top"]) == set(tied)


def test_decision_uses_only_2013(grid, best):
    v = best["validation"]
    ref = grid.set_index("config_id").loc["M1_ref"]
    bst = grid.set_index("config_id").loc[best["best_config_id"]]
    assert abs(v["gain_pp"] - (bst["val_BalAcc"] - ref["val_BalAcc"])) < 1e-6
    b = pd.read_csv(P["boot"])
    row = b[(b["part"] == "2b") & (b["a"] == best["best_config_id"]) & (b["b"] == "M1_ref")].iloc[0]
    assert abs(row["diff_lo"] - v["gain_ci95"][0]) < 1e-6 and abs(row["diff_hi"] - v["gain_ci95"][1]) < 1e-6
    adopt = v["gain_pp"] >= 0.5 and (v["gain_ci95"][0] > 0 or v["gain_ci95"][1] < 0)
    assert best["decision"] == ("adopt_M2" if adopt else "retain_M1")
    assert best["class_weight_from_M1"] is None and best["params"]["class_weight"] is None
    assert best["params"]["random_state"] == 17 and best["params"]["early_stopping"] is False


def test_bootstrap_balacc_equals_official_on_resamples(df):
    pred = pd.read_csv(P["pred"], index_col=0, parse_dates=True)
    y, yhat = pred["y"].to_numpy(), pred["pred_M2_best"].to_numpy()
    c = m2.day_counts(y, yhat, pred.index)
    B = m2.boot_indices(len(c), 3, 11)
    days = np.asarray(da.market_day(pred.index))
    for idx in B:
        rows = np.concatenate([np.flatnonzero(days == c.index[j]) for j in idx])
        s = c.to_numpy()[idx].sum(axis=0)
        got = m2.balacc_from_counts(*[s[c.columns.get_loc(k)] for k in ("tp", "fn", "tn", "fp")])[0]
        assert abs(got - da.balanced_accuracy(y[rows], yhat[rows])) < 1e-9
    assert m2.balacc_from_counts([5], [0], [0], [0])[0] == 100.0            # one class present -> its recall


def test_bootstrap_deterministic_and_paired():
    pred = pd.read_csv(P["pred"], index_col=0, parse_dates=True)
    y = pred["y"].to_numpy()
    preds = {"a": pred["pred_M2_best"].to_numpy(), "b": pred["pred_M1_ref"].to_numpy()}
    s1, B1, _ = m2.paired_bootstrap(y, preds, pred.index)
    s2, B2, _ = m2.paired_bootstrap(y, preds, pred.index)
    assert np.array_equal(B1, B2) and all(np.array_equal(s1[k], s2[k]) for k in s1)
    b = pd.read_csv(P["boot"])
    for _, part in b.groupby("part"):
        assert part["boot_index_sha256"].nunique() == 1                     # one index matrix per comparison set
    r = b[(b["part"] == "2014 confirmation") & (b["b"] == "M1_ref")].iloc[0]
    lo, hi = m2.ci(s1["a"] - s1["b"])
    assert abs(lo - r["diff_lo"]) < 1e-6 and abs(hi - r["diff_hi"]) < 1e-6
    assert (b["n_boot"] == 1000).all() and (b["seed"] == 0).all()


def test_2014_confirmation_numbers(best):
    pred = pd.read_csv(P["pred"], index_col=0, parse_dates=True)
    assert len(pred) == 17520 and da.market_day(pred.index).min() == pd.Timestamp("2014-01-01")
    t = best["test_2014_confirmation"]
    for cid, col in ((best["best_config_id"], "pred_M2_best"), ("M1_ref", "pred_M1_ref")):
        assert abs(da.balanced_accuracy(pred["y"], pred[col]) - t[cid]["BalAcc"]) < 1e-9
        assert abs(da.accuracy(pred["y"], pred[col]) - t[cid]["Acc"]) < 1e-9
        assert t[cid]["final_train"][:2] == ["2010-01-29", "2013-12-31"]
    m1pred = pd.read_csv(os.path.join(ARCHIVE, "m1", "m1_pred_2014.csv.gz"), index_col=0, parse_dates=True)
    assert np.array_equal(m1pred["y"].to_numpy(), pred["y"].to_numpy())
    assert np.array_equal(m1pred["b1"].to_numpy(), pred["b1"].to_numpy())
    assert abs(best["reference_numbers"]["b7_2014_BalAcc"] - 74.28175) < 1e-4


# =========================== task 4: outputs ===========================
def test_all_outputs_readable_and_linked(grid, best):
    for k in ("grid", "top5", "groups", "boot", "curves", "2a", "ranked"):
        assert len(pd.read_csv(P[k])) > 0, k
    assert best["best_config_id"] in set(grid["config_id"])
    b = pd.read_csv(P["boot"])
    ids = set(grid["config_id"]) | {"b1", "b7", "M1 features", "M1_ext features"}
    assert set(b["a"]) <= ids and set(b["b"]) <= ids
    assert set(b["part"]) == {"2a", "2b", "2014 confirmation"}


# =========================== current-environment repeatability (fresh processes) ===========================
@pytest.mark.slow
def test_fresh_process_rerun_reproduces_all_m2_outputs():
    """`m2 grid` on a finished grid is a no-op; `m2 select` rewrites byte-identical outputs."""
    import subprocess
    files = [P[k] for k in ("grid", "curves", "top5", "best", "boot", "pred", "ranked", "groups")]
    before = {f: m1b.sha(open(f, "rb").read()) for f in files}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for stage in ("grid", "select"):
        subprocess.run([sys.executable, "-m", "src.price_hgbc.m2", stage], cwd=ROOT, env=env, check=True,
                       capture_output=True, timeout=1800)
    after = {f: m1b.sha(open(f, "rb").read()) for f in files}
    assert after == before
