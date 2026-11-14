"""M2: feature-set check (2a), 48-config HGBC grid + M1_ref (2b), selection, paired day bootstrap,
one fixed 2014 confirmation. M1 artefacts are read-only and never written here.

Pre-declared rules (fixed before any M2 number was computed; user rules A-E):
  Validation  : the group's 2013 expanding folds (ablation/study.py `temporal_folds`):
                fold 1 train 2010-01-29 .. 2012-12-31, valid 2013-01-01 .. 2013-06-30;
                fold 2 train 2010-01-29 .. 2013-06-30, valid 2013-07-01 .. 2013-12-31.
                Held-out probabilities of both folds are pooled (17,520 rows); tau is searched on
                the pooled set with eval.tune_tau (0.30..0.70, 41 points, ties -> closest to 0.5,
                then smaller). Training start 2010-01-29 = P1 (28-day warm-up).
  2a          : M1 parameters, M1 features vs M1 + extended features. Extended wins iff
                BalAcc gain >= 0.5 pp AND the paired whole-day bootstrap 95 % interval of the
                difference excludes 0 (n_boot 1000, seed 0). Otherwise keep M1 features.
  2b          : 48 configs (max_leaf_nodes {15,31,63} x learning_rate {0.05,0.1} x max_iter {100,200}
                x l2 {0,0.1,1,10}) on the 2a winner, random_state 17, early_stopping False,
                class_weight as in M1 (None). Plus reference row M1_ref = M1 parameters + M1
                features, same folds and tau search.
  Ranking     : 2013 pooled BalAcc. Configs within < 0.05 pp of the top are tied; among them pick
                smaller max_leaf_nodes, then smaller max_iter, then larger l2_regularization;
                residual ties: higher BalAcc, then config_id order.
  Adoption    : M2 is adopted iff its 2013 BalAcc >= M1_ref + 0.5 pp AND the paired bootstrap
                95 % interval (M2 - M1_ref) excludes 0. Otherwise the conclusion is "retain M1".
  2014        : evaluated once after the decision, for confirmation only; it cannot change it.

Usage: python -m src.price_hgbc.m2 {2a|grid|select|all} [--max-new K]
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import eval as ev
from . import extra_features as xf
from . import features_simple as fs
from . import m1_baseline as m1b
from .kit import ARCHIVE, DATA_CSV, RESULTS, da  # noqa: F401

OUT = ARCHIVE                          # M2.5: M2 outputs live in the archive
OUT_DIR = os.path.join(ARCHIVE, "m2")
TRAIN_START = "2010-01-29"
FOLDS = [("2012-12-31", "2013-01-01", "2013-06-30"), ("2013-06-30", "2013-07-01", "2013-12-31")]
FINAL_TRAIN = (TRAIN_START, "2013-12-31")
TEST = ("2014-01-01", "2014-12-31")
N_BOOT, BOOT_SEED = 1000, 0
GAIN_PP = 0.5            # rules A and C
RANK_TIE_PP = 0.05       # rule C
TOP5_NOISE_PP = 0.3      # task 3.4 wording
M2_SEED = 17
M1_REF = "M1_ref"

GRID_VALUES = {"max_leaf_nodes": (15, 31, 63), "learning_rate": (0.05, 0.1), "max_iter": (100, 200),
               "l2_regularization": (0.0, 0.1, 1.0, 10.0)}


def grid_configs():
    out = []
    for leaves in GRID_VALUES["max_leaf_nodes"]:
        for lr in GRID_VALUES["learning_rate"]:
            for it in GRID_VALUES["max_iter"]:
                for l2 in GRID_VALUES["l2_regularization"]:
                    cid = f"g{len(out):02d}_L{leaves}_lr{lr:g}_it{it}_l2{l2:g}"
                    out.append({"config_id": cid, "max_leaf_nodes": leaves, "learning_rate": lr,
                                "max_iter": it, "l2_regularization": l2})
    return out


def m1_ref_config():
    p = m1b.hgbc_params(0)
    return {"config_id": M1_REF, **{k: p[k] for k in GRID_VALUES}}


def params_for(cfg, feature_set):
    """Full HGBC parameters: M1's documented configuration with the four grid values replaced.
    M1_ref keeps M1's parameters verbatim (random_state 0); grid configs use random_state 17."""
    seed = 0 if cfg["config_id"] == M1_REF else M2_SEED
    p = m1b.hgbc_params(seed)
    p.update({k: cfg[k] for k in GRID_VALUES})
    p["categorical_features"] = list(xf.EXT_CATEGORICAL if feature_set == "M1_ext" else fs.M1_CATEGORICAL)
    assert p["early_stopping"] is False and p["class_weight"] == m1b.hgbc_params(0)["class_weight"]
    return p


# ---------------- data / folds ----------------
class Data:
    """Feature matrices built once per feature set on a frame that ends before 2014."""

    def __init__(self, df):
        self.df = df
        self.pre = m1b.before_2014(df)
        assert da.market_day(self.pre.index).max() < m1b.CUTOFF
        self._X = {}

    def X(self, feature_set):
        if feature_set not in self._X:
            self._X[feature_set] = xf.build(self.pre, feature_set)
        return self._X[feature_set]

    def design(self, feature_set, first, last):
        X, y, dropped = fs.m1_design(self.pre, first, last, self.X(feature_set))
        assert dropped == [], dropped            # 2010-01-29 start: every day has its lags
        return X, y


def fold_info():
    return [{"fold": i + 1, "train": [TRAIN_START, end], "valid": [vs, ve]} for i, (end, vs, ve) in enumerate(FOLDS)]


def validation_run(data, feature_set, cfg):
    """Pooled 2013 held-out probabilities for one config + its tau search."""
    params = params_for(cfg, feature_set)
    ys, ps, idx, secs = [], [], [], 0.0
    for end, vs, ve in FOLDS:
        Xtr, ytr = data.design(feature_set, TRAIN_START, end)
        Xva, yva = data.design(feature_set, vs, ve)
        assert da.market_day(Xtr.index).max() < da.market_day(Xva.index).min()
        t = time.perf_counter()
        model = HistGradientBoostingClassifier(**params).fit(Xtr, ytr)
        secs += time.perf_counter() - t
        ps.append(model.predict_proba(Xva)[:, 1])
        ys.append(yva.to_numpy())
        idx.append(Xva.index)
    y, p, index = np.concatenate(ys), np.concatenate(ps), idx[0].append(idx[1])
    res = ev.tune_tau(y, p)
    pred = ev.predict_labels(p, res["tau"])
    Xv = pd.concat([data.design(feature_set, vs, ve)[0] for _, vs, ve in FOLDS])
    row = {"config_id": cfg["config_id"], "feature_set": feature_set,
           "features_sha256": m1b.sha(json.dumps(xf.FEATURE_SETS[feature_set]).encode()),
           "X_valid_sha256": m1b.sha_frame(Xv), "y_valid_sha256": m1b.sha_array(y, "int64"),
           **{k: cfg[k] for k in GRID_VALUES}, "random_state": params["random_state"],
           "class_weight": str(params["class_weight"]), "params_json": json.dumps(params, sort_keys=True),
           "train_range": " | ".join(f"{TRAIN_START}..{e}" for e, _, _ in FOLDS),
           "valid_range": " | ".join(f"{a}..{b}" for _, a, b in FOLDS),
           "tau": res["tau"], "tau_at_endpoint": res["at_endpoint"], "tau_tied": json.dumps(res["tied"]),
           "n_tau_points": len(res["curve"]), "val_Acc": res["Acc"], "val_BalAcc": res["BalAcc"],
           "val_rows": int(len(y)), "fit_seconds": round(secs, 3),
           "proba_sha256": m1b.sha_array(p, "float64"), "pred_sha256": m1b.sha_array(pred, "int64")}
    return row, {"y": y, "proba": p, "pred": pred, "index": index, "curve": res["curve"]}


# ---------------- bootstrap ----------------
def day_counts(y, pred, index):
    """Per market day: tp, fn, tn, fp (rows are whole days)."""
    days = da.market_day(pd.DatetimeIndex(index))
    f = pd.DataFrame({"tp": (y == 1) & (pred == 1), "fn": (y == 1) & (pred == 0),
                      "tn": (y == 0) & (pred == 0), "fp": (y == 0) & (pred == 1)}).astype("int64")
    c = f.groupby(np.asarray(days)).sum()
    assert (c.sum(axis=1) == 48).all()
    return c


def balacc_from_counts(tp, fn, tn, fp):
    """Mean recall over the classes present (identical to da.balanced_accuracy)."""
    tp, fn, tn, fp = (np.asarray(a, dtype="float64") for a in (tp, fn, tn, fp))
    pos, neg = tp + fn, tn + fp
    r1 = np.where(pos > 0, tp / np.where(pos > 0, pos, 1), np.nan)
    r0 = np.where(neg > 0, tn / np.where(neg > 0, neg, 1), np.nan)
    return np.nanmean(np.vstack([r1, r0]), axis=0) * 100


def boot_indices(n_days, n_boot=N_BOOT, seed=BOOT_SEED):
    return np.random.default_rng(seed).integers(0, n_days, size=(n_boot, n_days))


def paired_bootstrap(y, preds, index, n_boot=N_BOOT, seed=BOOT_SEED):
    """Whole-day paired bootstrap: one index matrix shared by every model in `preds`."""
    counts = {k: day_counts(y, v, index) for k, v in preds.items()}
    days = next(iter(counts.values())).index
    assert all(c.index.equals(days) for c in counts.values())
    B = boot_indices(len(days), n_boot, seed)
    samples = {}
    for k, c in counts.items():
        arr = c[["tp", "fn", "tn", "fp"]].to_numpy()
        s = arr[B].sum(axis=1)                       # (n_boot, 4)
        samples[k] = balacc_from_counts(s[:, 0], s[:, 1], s[:, 2], s[:, 3])
    return samples, B, days


def ci(a):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------- grid (resumable) ----------------
def _atomic_csv(frame, path):
    tmp = path + ".tmp"
    frame.to_csv(tmp, index=False, float_format="%.10g")
    os.replace(tmp, path)


def _read(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def run_grid(data, feature_set, configs, out_csv, curves_csv, max_new=None, log=print):
    """Idempotent: configs with status 'done' are kept; 'running' (interrupted) or missing ones run.
    Existing rows whose definition no longer matches (params / feature set) raise instead of mixing."""
    grid = _read(out_csv)
    curves = _read(curves_csv)
    grid = grid if grid is not None else pd.DataFrame()
    curves = curves if curves is not None else pd.DataFrame(columns=["config_id", "tau", "BalAcc", "Acc", "pos_rate_pred"])
    if len(grid):
        assert grid["config_id"].is_unique, "duplicate config_id in existing grid"
    env = m1b.environment()
    env_s = f"py{env['python']} sklearn{env['versions']['sklearn']} {env['platform']}"
    new = 0
    for cfg in configs:
        cid = cfg["config_id"]
        if len(grid) and cid in set(grid["config_id"]):
            old = grid.set_index("config_id").loc[cid]
            want = json.dumps(params_for(cfg, feature_set), sort_keys=True)
            if old["status"] == "done":
                if old["params_json"] != want or old["feature_set"] != feature_set:
                    raise ValueError(f"{cid}: existing row does not match the current definition")
                continue
            grid = grid[grid["config_id"] != cid]
            curves = curves[curves["config_id"] != cid]
        if max_new is not None and new >= max_new:
            break
        running = {"config_id": cid, "feature_set": feature_set, "status": "running"}
        _atomic_csv(pd.concat([grid, pd.DataFrame([running])], ignore_index=True), out_csv)
        row, arr = validation_run(data, feature_set, cfg)
        row.update({"env": env_s, "status": "done"})
        grid = pd.concat([grid, pd.DataFrame([row])], ignore_index=True)
        c = arr["curve"].copy()
        c.insert(0, "config_id", cid)
        curves = pd.concat([curves, c], ignore_index=True)
        _atomic_csv(curves, curves_csv)
        _atomic_csv(grid, out_csv)
        new += 1
        log(f"{cid:28s} tau {row['tau']:.2f} BalAcc {row['val_BalAcc']:.4f} ({row['fit_seconds']:.1f}s)")
    return grid


# ---------------- selection ----------------
def rank(grid):
    """Apply the pre-declared ranking rule to the 48 grid rows (M1_ref excluded)."""
    g = grid[(grid["config_id"] != M1_REF) & (grid["status"] == "done")].copy()
    top = g["val_BalAcc"].max()
    tied = g[g["val_BalAcc"] > top - RANK_TIE_PP]
    best = tied.sort_values(["max_leaf_nodes", "max_iter", "l2_regularization", "val_BalAcc", "config_id"],
                            ascending=[True, True, False, False, True]).iloc[0]["config_id"]
    rest = g[g["config_id"] != best].sort_values(["val_BalAcc", "config_id"], ascending=[False, True])
    order = [best] + list(rest["config_id"])
    g = g.set_index("config_id").loc[order].reset_index()
    g.insert(0, "rank", range(1, len(g) + 1))
    g["tied_with_top"] = g["config_id"].isin(set(tied["config_id"]))
    return g, list(tied["config_id"])


def cfg_from_row(r):
    return {"config_id": r["config_id"], **{k: (int(r[k]) if k in ("max_leaf_nodes", "max_iter") else float(r[k]))
                                            for k in GRID_VALUES}}


def fit_final(data, feature_set, cfg):
    X, y = data.design(feature_set, *FINAL_TRAIN)
    return HistGradientBoostingClassifier(**params_for(cfg, feature_set)).fit(X, y), X, y


def test_rows(df, feature_set):
    dte = m1b.test_frame(df)
    X, y, dropped = fs.m1_design(dte, *TEST, xf.build(dte, feature_set))
    assert dropped == []
    return X, y


# ---------------- stages ----------------
def stage_2a(data, log=print):
    cfg = m1_ref_config()
    rows, arrs = {}, {}
    for fset in ("M1", "M1_ext"):
        rows[fset], arrs[fset] = validation_run(data, fset, cfg)
    assert np.array_equal(arrs["M1"]["y"], arrs["M1_ext"]["y"])
    assert arrs["M1"]["index"].equals(arrs["M1_ext"]["index"])
    s, B, days = paired_bootstrap(arrs["M1"]["y"], {k: v["pred"] for k, v in arrs.items()}, arrs["M1"]["index"])
    diff = s["M1_ext"] - s["M1"]
    gain = rows["M1_ext"]["val_BalAcc"] - rows["M1"]["val_BalAcc"]
    lo, hi = ci(diff)
    winner = "M1_ext" if (gain >= GAIN_PP and (lo > 0 or hi < 0)) else "M1"
    tab = pd.DataFrame([{**rows[k], "status": "done"} for k in ("M1", "M1_ext")])
    tab["config_id"] = ["2a_M1_features", "2a_M1_ext_features"]
    summary = {"gain_pp": gain, "diff_ci95": [lo, hi], "rule": f"gain >= {GAIN_PP} pp and CI excludes 0",
               "winner": winner, "n_days": int(len(days)), "n_boot": N_BOOT, "seed": BOOT_SEED,
               "boot_index_sha256": m1b.sha(np.ascontiguousarray(B).tobytes())}
    boot = [{"part": "2a", "period": "2013 pooled folds", "a": "M1_ext features", "b": "M1 features",
             "BalAcc_a": rows["M1_ext"]["val_BalAcc"], "BalAcc_b": rows["M1"]["val_BalAcc"],
             "diff_pp": gain, "diff_lo": lo, "diff_hi": hi,
             "a_lo": ci(s["M1_ext"])[0], "a_hi": ci(s["M1_ext"])[1], "b_lo": ci(s["M1"])[0], "b_hi": ci(s["M1"])[1],
             "n_days": len(days), "n_boot": N_BOOT, "seed": BOOT_SEED, "boot_index_sha256": summary["boot_index_sha256"]}]
    log(f"2a: M1 {rows['M1']['val_BalAcc']:.4f}  M1_ext {rows['M1_ext']['val_BalAcc']:.4f}  "
        f"gain {gain:+.4f} pp  CI [{lo:+.4f}, {hi:+.4f}] -> {winner}")
    return tab, summary, boot


def paths():
    return {"grid": os.path.join(OUT, "m2_grid.csv"), "curves": os.path.join(OUT_DIR, "m2_tau_curves_2013.csv"),
            "2a": os.path.join(OUT_DIR, "m2_featureset_2a.csv"), "2a_json": os.path.join(OUT_DIR, "m2_featureset_2a.json"),
            "top5": os.path.join(OUT, "m2_top5.csv"), "best": os.path.join(OUT, "m2_best_config.json"),
            "groups": os.path.join(OUT, "m2_feature_groups.csv"), "boot": os.path.join(OUT, "m2_bootstrap.csv"),
            "pred": os.path.join(OUT_DIR, "m2_pred_2014.csv.gz"), "ranked": os.path.join(OUT_DIR, "m2_grid_ranked.csv")}


def stage_select(data, log=print):
    P = paths()
    grid = pd.read_csv(P["grid"])
    assert grid["config_id"].is_unique and (grid["status"] == "done").all()
    winner = json.load(open(P["2a_json"]))["winner"]
    ranked, tied = rank(grid)
    ranked.to_csv(P["ranked"], index=False, float_format="%.10g")
    ref = grid.set_index("config_id").loc[M1_REF]
    top5 = ranked.head(5).copy()
    best_id = top5.iloc[0]["config_id"]
    # independent re-runs (same pipeline, fresh fits) must reproduce the grid rows bitwise
    runs = {}
    for cid in list(top5["config_id"]) + [M1_REF]:
        r = grid.set_index("config_id").loc[cid]
        fset = r["feature_set"]
        cfg = m1_ref_config() if cid == M1_REF else cfg_from_row({"config_id": cid, **r.to_dict()})
        row, arr = validation_run(data, fset, cfg)
        assert row["proba_sha256"] == r["proba_sha256"] and row["tau"] == r["tau"], cid
        runs[cid] = (row, arr)
    y, index = runs[M1_REF][1]["y"], runs[M1_REF][1]["index"]
    s, B, days = paired_bootstrap(y, {k: v[1]["pred"] for k, v in runs.items()}, index)
    bidx = m1b.sha(np.ascontiguousarray(B).tobytes())
    boot = []
    for cid in top5["config_id"]:
        for other in (M1_REF, best_id):
            if other == cid:
                continue
            d = s[cid] - s[other]
            boot.append({"part": "2b", "period": "2013 pooled folds", "a": cid, "b": other,
                         "BalAcc_a": runs[cid][0]["val_BalAcc"], "BalAcc_b": runs[other][0]["val_BalAcc"],
                         "diff_pp": runs[cid][0]["val_BalAcc"] - runs[other][0]["val_BalAcc"],
                         "diff_lo": ci(d)[0], "diff_hi": ci(d)[1], "a_lo": ci(s[cid])[0], "a_hi": ci(s[cid])[1],
                         "b_lo": ci(s[other])[0], "b_hi": ci(s[other])[1], "n_days": len(days),
                         "n_boot": N_BOOT, "seed": BOOT_SEED, "boot_index_sha256": bidx})
    top5["BalAcc_lo"] = [ci(s[c])[0] for c in top5["config_id"]]
    top5["BalAcc_hi"] = [ci(s[c])[1] for c in top5["config_id"]]
    top5["gap_to_best_pp"] = top5["val_BalAcc"] - top5.iloc[0]["val_BalAcc"]
    top5["gap_to_M1_ref_pp"] = top5["val_BalAcc"] - ref["val_BalAcc"]
    top5["differs_from_best"] = [", ".join(f"{k}={top5.iloc[i][k]:g}" for k in GRID_VALUES
                                           if top5.iloc[i][k] != top5.iloc[0][k]) or "-" for i in range(len(top5))]
    top5.to_csv(P["top5"], index=False, float_format="%.10g")
    d_best = s[best_id] - s[M1_REF]
    gain = runs[best_id][0]["val_BalAcc"] - ref["val_BalAcc"]
    lo, hi = ci(d_best)
    adopt = bool(gain >= GAIN_PP and (lo > 0 or hi < 0))
    decision = "adopt_M2" if adopt else "retain_M1"
    spread = float(top5["val_BalAcc"].max() - top5["val_BalAcc"].min())
    log(f"best {best_id}: 2013 {runs[best_id][0]['val_BalAcc']:.4f} vs M1_ref {ref['val_BalAcc']:.4f} "
        f"gain {gain:+.4f} CI [{lo:+.4f},{hi:+.4f}] -> {decision}; top5 spread {spread:.4f} pp")

    # ---- 2014: one fixed confirmation (after the decision) ----
    best_row = grid.set_index("config_id").loc[best_id]
    fset_best = best_row["feature_set"]
    preds14, info14 = {}, {}
    for cid, fset, cfg, tau in ((best_id, fset_best, cfg_from_row({"config_id": best_id, **best_row.to_dict()}),
                                 float(best_row["tau"])),
                                (M1_REF, "M1", m1_ref_config(), float(ref["tau"]))):
        model, Xf, yf = fit_final(data, fset, cfg)
        Xte, yte = test_rows(data.df, fset)
        p = model.predict_proba(Xte)[:, 1]
        pred = ev.predict_labels(p, tau)
        preds14[cid] = (p, pred, Xte, yte)
        info14[cid] = {"tau": tau, **ev.evaluate(yte, pred), "feature_set": fset,
                       "final_train": [str(da.market_day(Xf.index).min().date()), str(da.market_day(Xf.index).max().date()), int(len(Xf))],
                       "proba_sha256": m1b.sha_array(p, "float64"), "pred_sha256": m1b.sha_array(pred, "int64"),
                       "final_model": m1b.model_fingerprint(model) if fset == "M1" else _fp_ext(model, fset)}
    yte = preds14[best_id][3]
    assert np.array_equal(yte.to_numpy(), preds14[M1_REF][3].to_numpy())
    Xm1 = preds14[M1_REF][2]
    b1, b7 = Xm1["up_lag1"].to_numpy(), Xm1["up_lag7"].to_numpy()
    s14, B14, days14 = paired_bootstrap(yte.to_numpy(), {best_id: preds14[best_id][1], M1_REF: preds14[M1_REF][1],
                                                          "b1": b1, "b7": b7}, yte.index)
    b14 = m1b.sha(np.ascontiguousarray(B14).tobytes())
    for a, b in ((best_id, M1_REF), (best_id, "b7"), (M1_REF, "b7")):
        ba = info14[a]["BalAcc"] if a in info14 else da.balanced_accuracy(yte, {"b1": b1, "b7": b7}[a])
        bb = info14[b]["BalAcc"] if b in info14 else da.balanced_accuracy(yte, {"b1": b1, "b7": b7}[b])
        boot.append({"part": "2014 confirmation", "period": "2014", "a": a, "b": b, "BalAcc_a": ba, "BalAcc_b": bb,
                     "diff_pp": ba - bb, "diff_lo": ci(s14[a] - s14[b])[0], "diff_hi": ci(s14[a] - s14[b])[1],
                     "a_lo": ci(s14[a])[0], "a_hi": ci(s14[a])[1], "b_lo": ci(s14[b])[0], "b_hi": ci(s14[b])[1],
                     "n_days": len(days14), "n_boot": N_BOOT, "seed": BOOT_SEED, "boot_index_sha256": b14})
    pd.DataFrame({"y": yte.to_numpy(), "proba_M2_best": preds14[best_id][0], "pred_M2_best": preds14[best_id][1],
                  "proba_M1_ref": preds14[M1_REF][0], "pred_M1_ref": preds14[M1_REF][1], "b1": b1, "b7": b7},
                 index=yte.index).to_csv(P["pred"], float_format="%.17g",
                                         compression={"method": "gzip", "mtime": 0})  # byte-reproducible .gz
    m1_official = json.load(open(os.path.join(ARCHIVE, "m1", "m1_run.json")))
    best_cfg = {
        "decision": decision, "decision_rule": f"2013 BalAcc gain >= {GAIN_PP} pp vs M1_ref and paired day-bootstrap 95% CI excludes 0",
        "feature_set_2a_winner": winner, "best_config_id": best_id, "feature_set": fset_best,
        "features": xf.FEATURE_SETS[fset_best], "params": json.loads(best_row["params_json"]),
        "class_weight_from_M1": m1_official["params"]["class_weight"], "tau": float(best_row["tau"]),
        "tau_at_endpoint": bool(best_row["tau_at_endpoint"]),
        "validation": {"folds": fold_info(), "BalAcc_2013": runs[best_id][0]["val_BalAcc"],
                       "Acc_2013": runs[best_id][0]["val_Acc"], "M1_ref_BalAcc_2013": float(ref["val_BalAcc"]),
                       "M1_ref_Acc_2013": float(ref["val_Acc"]), "M1_ref_tau": float(ref["tau"]),
                       "gain_pp": gain, "gain_ci95": [lo, hi], "tied_with_top": tied, "top5_spread_pp": spread,
                       "top5_within_noise_warning": spread < TOP5_NOISE_PP},
        "test_2014_confirmation": info14,
        "reference_numbers": {"M1_official_2014_BalAcc": m1_official["test_2014"]["BalAcc"],
                              "b7_2014_BalAcc": da.balanced_accuracy(yte, b7), "b1_2014_BalAcc": da.balanced_accuracy(yte, b1),
                              "target_BalAcc": 82.0},
        "environment": m1b.environment(),
    }
    json.dump(best_cfg, open(P["best"], "w"), indent=1, default=str)
    _write_boot(boot, P["boot"], parts=("2b", "2014 confirmation"))
    log(f"2014: {best_id} BalAcc {info14[best_id]['BalAcc']:.4f} Acc {info14[best_id]['Acc']:.4f} | "
        f"M1_ref {info14[M1_REF]['BalAcc']:.4f}")
    return best_cfg


def _fp_ext(model, fset):
    import hashlib
    trees = [hashlib.sha256(p.nodes.tobytes()).hexdigest() for it in model._predictors for p in it]
    cols = xf.FEATURE_SETS[fset]
    return {"n_iter": int(model.n_iter_), "trees_sha256": hashlib.sha256("".join(trees).encode()).hexdigest(),
            "is_categorical": [cols[i] for i in np.flatnonzero(model.is_categorical_)]}


def _write_boot(rows, path, parts):
    """Replace only the given parts in m2_bootstrap.csv (this file is M2-owned)."""
    old = _read(path)
    new = pd.DataFrame(rows)
    if old is not None:
        old = old[~old["part"].isin(parts)]
        new = pd.concat([old, new], ignore_index=True)
    order = {"2a": 0, "2b": 1, "2014 confirmation": 2}
    new = new.sort_values("part", key=lambda s: s.map(order), kind="stable")
    _atomic_csv(new, path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["2a", "grid", "select", "all"])
    ap.add_argument("--max-new", type=int, default=None)
    a = ap.parse_args(argv)
    os.makedirs(OUT_DIR, exist_ok=True)
    P = paths()
    data = Data(da.load_dataset(DATA_CSV))
    xf.feature_groups().to_csv(P["groups"], index=False)
    if a.stage in ("2a", "all"):
        tab, summary, boot = stage_2a(data)
        _atomic_csv(tab, P["2a"])
        json.dump(summary, open(P["2a_json"], "w"), indent=1)
        _write_boot(boot, P["boot"], parts=("2a",))
    if a.stage in ("grid", "all"):
        winner = json.load(open(P["2a_json"]))["winner"]
        configs = grid_configs()
        grid = run_grid(data, winner, configs, P["grid"], P["curves"], max_new=a.max_new)
        if set(c["config_id"] for c in configs) <= set(grid.loc[grid["status"] == "done", "config_id"]):
            run_grid(data, "M1", [m1_ref_config()], P["grid"], P["curves"])   # reference row last
    if a.stage in ("select", "all"):
        stage_select(data)


if __name__ == "__main__":
    main()
