"""M4: calibrate the error-based CUSUM on 2014 (design: results/price_hgbc/m4/cusum_design.md, pre-declared).

Data read (and logged to m4/data_access_log.json):
  official/preds_2014.parquet (2014 out-of-sample predictions of REF), official/boost_price_ref.json,
  and - only for the option-B demonstration - P1's load_study (rows up to 2015-01-01 00:00 = end of 2014).
Nothing from 2015 onwards is read. REF and tau are not modified.
Usage: python -m src.price_hgbc.m4
"""
import hashlib
import json
import math
import os

import numpy as np
import pandas as pd

from . import cusum as cs
from . import m3_common as c
from . import official_ref as ofr
from .kit import ROOT, da

OUT = os.path.join(os.path.dirname(c.OUT), "m4")
P = {k: os.path.join(OUT, v) for k, v in {
    "daily": "daily_err_2014.csv", "ref_stats": "reference_stats_2014.csv", "diag": "err_diagnostics_2014.csv",
    "arl": "arl0_by_h.csv", "path": "cusum_path_2014.csv", "delay": "delay_by_shift.csv",
    "optB": "ref_update_B_demo_2013.csv", "consts": "design_constants.csv", "config": "cusum_config.json", "log": "data_access_log.json",
    "design": "cusum_design.md", "fig_err": "fig_m4_daily_err_2014.png", "fig_S": "fig_m4_cusum_path_2014.png",
    "fig_arl": "fig_m4_arl0_by_h.png", "fig_delay": "fig_m4_delay_by_shift.png"}.items()}
N_DAYS_SIM, N_REP, SEED, BLOCK = 3650, 1000, 0, 7
SHIFTS = (0.5, 1.0, 2.0, 3.0)
CODE_FILES = ["src/price_hgbc/cusum.py", "src/price_hgbc/m4.py"]
ACCESS = []
# User decision after the M4 review: keep the 2014 reference after every retrain (date and context in
# results/price_hgbc/selfcheck/M4.md section 8; no date literal here, the data-boundary grep forbids later dates).
REF_UPDATE_DECISION = {"choice": "A", "decided_by": "user", "recorded_in": "results/price_hgbc/selfcheck/M4.md section 8",
                       "note": "after a retrain keep mu0, sigma0, q99 from 2014 (h = 6 calibration stays as validated)"}


def _log(path, kind, max_ts):
    ACCESS.append({"file": os.path.relpath(path, ROOT), "kind": kind, "max_timestamp": str(max_ts)})


# ---------------- task 1 ----------------
def daily_errors():
    r = c.ref()
    _log(ofr.REF_JSON, "json", "n/a")
    p = pd.read_parquet(ofr.PREDS)
    _log(ofr.PREDS, "parquet", p.index.max())
    assert p.index.max() == pd.Timestamp("2015-01-01 00:00")
    assert ofr.sha_arr(p["proba"], "float64") == r["hashes"]["proba_2014"]
    assert ofr.sha_arr(p["pred"], "int64") == r["hashes"]["pred_2014"]
    ll = cs.daily_logloss(p["y"], p["proba"], p.index)
    mc = cs.daily_misclass(p["y"], (p["proba"] >= r["tau"]).astype(int), p.index)
    d = pd.DataFrame({"day": ll.index, "logloss": ll.to_numpy(), "misclass": mc.to_numpy()})
    d["weekday"] = pd.DatetimeIndex(d["day"]).day_of_week
    return d, r


def ref_stats_table(d):
    rows = []
    for scheme, col, trunc in (("main", "logloss", True), ("misclass", "misclass", True), ("no_trunc", "logloss", False)):
        s = cs.reference_stats(d[col], truncate=trunc)
        rows.append({"scheme": scheme, "error": col, "truncate": trunc, **s, "k": cs.K_DEFAULT})
    return pd.DataFrame(rows)


def diagnostics(d, q99):
    e = d["logloss"].to_numpy()
    et = np.minimum(e, q99)
    s = pd.Series(et)
    rows = [{"stat": f"acf_lag{l}_logloss_trunc", "value": float(s.autocorr(l))} for l in range(1, 8)]
    rows += [{"stat": f"acf_lag{l}_logloss_raw", "value": float(pd.Series(e).autocorr(l))} for l in range(1, 8)]
    for wd, g in d.assign(et=et).groupby("weekday"):
        rows.append({"stat": f"mean_logloss_trunc_weekday{wd}", "value": float(g["et"].mean())})
    rows.append({"stat": "corr_logloss_misclass", "value": float(np.corrcoef(d["logloss"], d["misclass"])[0, 1])})
    return pd.DataFrame(rows)


# ---------------- task 3 ----------------
def week_blocks(days):
    """Indices of full Monday-Sunday weeks inside the 365 days (design B5)."""
    days = pd.DatetimeIndex(days)
    starts = [i for i in range(len(days) - 6) if days[i].day_of_week == 0]
    blocks = np.array([np.arange(i, i + BLOCK) for i in starts])
    assert all(days[b[-1]].day_of_week == 6 for b in blocks)
    return blocks


def boot_matrix(n_blocks, seed=SEED):
    n_draw = math.ceil(N_DAYS_SIM / BLOCK)
    return np.random.default_rng(seed).integers(0, n_blocks, size=(N_REP, n_draw))


def bootstrap_series(values, blocks, W):
    """(N_REP, 3650) series of daily values built from week blocks (weekday phase kept)."""
    return np.asarray(values, dtype=float)[blocks[W].reshape(W.shape[0], -1)][:, :N_DAYS_SIM]


def arl_table(d, stats, blocks, W):
    rows = []
    for _, s in stats.iterrows():
        x = np.minimum(d[s["error"]].to_numpy(), s["q99"])          # bootstrap material: truncated errors
        E = bootstrap_series(x, blocks, W)
        for h in cs.H_CANDIDATES:
            arl = cs.arl_from_counts(cs.simulate_alarm_counts(E, s["mu0"], s["sigma0"], s["q99"], s["k"], h), N_DAYS_SIM)
            rows.append({"scheme": s["scheme"], "error": s["error"], "truncate": s["truncate"], "k": s["k"], "h": h,
                         "arl0_median": cs.quantile_with_inf(arl, 0.5), "arl0_q05": cs.quantile_with_inf(arl, 0.05),
                         "arl0_q95": cs.quantile_with_inf(arl, 0.95), "share_reps_no_alarm": float(np.isinf(arl).mean()),
                         "mean_alarms_per_3650d": float(np.mean(N_DAYS_SIM / np.where(np.isinf(arl), np.inf, arl)))})
    t = pd.DataFrame(rows)
    t["meets_target"] = t["arl0_median"] >= cs.ARL0_TARGET
    t["selected"] = False
    for sch, g in t.groupby("scheme"):
        ok = g[g["meets_target"]]
        pick = ok["h"].min() if len(ok) else max(cs.H_CANDIDATES)              # design B8
        t.loc[(t["scheme"] == sch) & (t["h"] == pick), "selected"] = True
    return t


def delay_table(stats, d, blocks, W, h):
    s = stats.set_index("scheme").loc["main"]
    rng = np.random.default_rng(SEED)
    Z = rng.standard_normal((N_REP, N_DAYS_SIM))
    real = bootstrap_series(np.minimum(d["logloss"].to_numpy(), s["q99"]), blocks, W)
    rows = []
    for delta in SHIFTS:
        for noise, base in (("iid_normal", s["mu0"] + s["sigma0"] * Z), ("block_bootstrap_2014", real)):
            first = cs.simulate_first_alarm(base + delta * s["sigma0"], s["mu0"], s["sigma0"], s["q99"], s["k"], h)
            first = np.where(np.isinf(first), N_DAYS_SIM, first)
            rows.append({"shift_sigma0": delta, "noise": noise, "h": h, "mean_delay_days": float(first.mean()),
                         "median_delay_days": float(np.median(first)), "q95_delay_days": float(np.quantile(first, 0.95))})
    return pd.DataFrame(rows)


# ---------------- option B demonstration ----------------
def option_b_demo(r):
    from sklearn.ensemble import HistGradientBoostingClassifier
    data, X, truth, cols, _, _ = ofr.design()
    _log(ofr.DATA_CSV if hasattr(ofr, "DATA_CSV") else "data", "csv via P1 load_study", data.index.max())
    assert data.index.max() <= pd.Timestamp("2015-01-01 00:00")
    days = da.market_day(X.index)
    tr = (days >= pd.Timestamp("2010-01-29")) & (days <= pd.Timestamp("2012-12-31"))
    te = (days >= pd.Timestamp("2013-01-01")) & (days <= pd.Timestamp("2013-12-31"))
    m = HistGradientBoostingClassifier(**r["params"]).fit(X.loc[tr, cols], truth.loc[tr, "nsw_price_up"])
    p = m.predict_proba(X.loc[te, cols])[:, 1]
    e = cs.daily_logloss(truth.loc[te, "nsw_price_up"], p, X.index[te])
    s = cs.reference_stats(e, truncate=True)
    return pd.DataFrame([{"option": "B (demo)", "window": "train 2010-01-29..2012-12-31, OOS 2013", **s},
                         {"option": "A", "window": "2014 OOS of REF", **{}}]), e


# ---------------- figures ----------------
def fig_err(d, s):
    plt = c.mpl()
    fig, ax = plt.subplots(figsize=(9, 3.8), facecolor=c.SURFACE)
    c.style(ax)
    ax.grid(axis="y", color=c.GRID, linewidth=0.8)
    x = pd.DatetimeIndex(d["day"])
    ax.plot(x, d["logloss"], color=c.GRAY, linewidth=1, label="e_D raw")
    ax.plot(x, np.minimum(d["logloss"], s["q99"]), color=c.BLUE, linewidth=1.6, label="e_D truncated at q99")
    ax.axhline(s["mu0"], color=c.INK2, linewidth=1, label=f"μ₀ = {s['mu0']:.4f}")
    ax.axhline(s["q99"], color=c.ORANGE, linewidth=1.2, linestyle=(0, (4, 3)), label=f"q99 = {s['q99']:.4f}")
    ax.set_xlabel("Target day D (2014)", color=c.INK2)
    ax.set_ylabel("Daily mean log loss (nats per half-hour)", color=c.INK2)
    ax.set_title("2014 daily error of boost_price (log loss), raw and truncated", color=c.INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=c.INK2, ncol=4, loc="upper left")
    fig.tight_layout()
    return fig, ax


def fig_path(path, h):
    plt = c.mpl()
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=c.SURFACE)
    c.style(ax)
    ax.grid(axis="y", color=c.GRID, linewidth=0.8)
    x = pd.DatetimeIndex(path["day"])
    ax.plot(x, path["S"], color=c.BLUE, linewidth=1.6, label="S_D (before reset)")
    ax.axhline(h, color=c.ORANGE, linewidth=1.2, linestyle=(0, (4, 3)), label=f"h = {h:g}")
    al = path[path["alarm"]]
    if len(al):
        ax.scatter(pd.DatetimeIndex(al["day"]), al["S"], color=c.ORANGE, s=40, zorder=3, label="alarm")
    ax.set_xlabel("Target day D (2014)", color=c.INK2)
    ax.set_ylabel("CUSUM statistic S_D (σ₀ units)", color=c.INK2)
    ax.set_title(f"Upper CUSUM on 2014 (k = 0.5, h = {h:g}): {len(al)} alarm(s)", color=c.INK, fontsize=11, loc="left")
    ax.set_ylim(top=max(h, float(path["S"].max())) * 1.25)
    ax.legend(frameon=False, fontsize=8, labelcolor=c.INK2, loc="upper center", ncol=3)
    fig.tight_layout()
    return fig, ax


def fig_arl(arl):
    plt = c.mpl()
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=c.SURFACE)
    c.style(ax)
    ax.grid(axis="both", color=c.GRID, linewidth=0.8)
    cap = N_DAYS_SIM                                   # +inf (no alarm in the replicate) is drawn at the 3650-day cap
    for sch, col, lab in (("main", c.BLUE, "log loss (main)"), ("misclass", c.ORANGE, "misclassification (comparison)")):
        g = arl[arl["scheme"] == sch].sort_values("h")
        med = np.minimum(g["arl0_median"], cap)
        ax.fill_between(g["h"], np.minimum(g["arl0_q05"], cap), np.minimum(g["arl0_q95"], cap), color=col, alpha=0.15, linewidth=0)
        ax.plot(g["h"], med, color=col, linewidth=2, marker="o", markersize=6, label=lab)
        capped = g[np.isinf(g["arl0_median"])]
        ax.scatter(capped["h"], np.full(len(capped), cap), marker="^", s=70, color=col, zorder=4)
        sel = g[g["selected"]]
        ax.scatter(sel["h"], sel["arl0_median"], s=120, facecolors="none", edgecolors=col, linewidths=2, zorder=4)
    ax.axhline(cap, color=c.GRID, linewidth=1)
    ax.text(10, cap * 1.08, "▲ > 3650 (no alarm in most replicates)", color=c.INK2, fontsize=8, ha="right")
    ax.axhline(cs.ARL0_TARGET, color=c.INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(3, cs.ARL0_TARGET * 1.08, "target ARL₀ = 730 days", color=c.INK2, fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("Threshold h (σ₀ units)", color=c.INK2)
    ax.set_ylabel("In-control ARL₀ (days, log scale)", color=c.INK2)
    ax.set_title("ARL₀ by h: median, 5–95 % band (2014 weekly block bootstrap); ○ = selected", color=c.INK, fontsize=10, loc="left")
    ax.set_ylim(top=cap * 1.6)
    ax.legend(frameon=False, fontsize=9, labelcolor=c.INK2, loc="lower right")
    fig.tight_layout()
    return fig, ax


def fig_delay(delay):
    plt = c.mpl()
    fig, ax = plt.subplots(figsize=(6.5, 4), facecolor=c.SURFACE)
    c.style(ax)
    ax.grid(axis="both", color=c.GRID, linewidth=0.8)
    for noise, col, lab in (("iid_normal", c.BLUE, "iid N(0,1) noise"), ("block_bootstrap_2014", c.ORANGE, "2014 weekly block bootstrap")):
        g = delay[delay["noise"] == noise].sort_values("shift_sigma0")
        ax.plot(g["shift_sigma0"], g["mean_delay_days"], color=col, linewidth=2, marker="o", label=lab)
        for x, y in zip(g["shift_sigma0"], g["mean_delay_days"]):
            ax.annotate(f"{y:.1f}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8, color=c.INK2)
    ax.set_xlabel("Mean shift of the daily error (σ₀ units)", color=c.INK2)
    ax.set_ylabel("Mean detection delay (days)", color=c.INK2)
    ax.set_title(f"Detection delay at the selected h = {int(delay['h'].iloc[0])}", color=c.INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=c.INK2)
    fig.tight_layout()
    return fig, ax


# ---------------- config ----------------
def sha_file(p):
    return hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest()


def _num(x):
    return cs.NO_ALARM if (isinstance(x, float) and math.isinf(x)) else x


def build_config(stats, arl, r):
    schemes = {}
    for sch, role in (("main", "selected detector (log loss)"), ("misclass", "comparison only (not used)")):
        s = stats.set_index("scheme").loc[sch]
        a = arl[(arl["scheme"] == sch) & arl["selected"]].iloc[0]
        schemes[sch] = {"error": s["error"], "truncate": bool(s["truncate"]), "q99": float(s["q99"]), "mu0": float(s["mu0"]),
                        "sigma0": float(s["sigma0"]), "k": float(s["k"]), "h": int(a["h"]),
                        "arl0_median": _num(float(a["arl0_median"])), "arl0_q05": _num(float(a["arl0_q05"])),
                        "arl0_q95": _num(float(a["arl0_q95"])), "meets_target": bool(a["meets_target"]), "role": role}
    return {
        "error_definitions": {"logloss": "mean over the 48 half-hours of day D of -[y log p + (1-y) log(1-p)], p clipped",
                              "misclass": "share of day D's 48 half-hours with 1[p >= tau] != y, tau = REF tau"},
        "logloss_clip": list(cs.LOGLOSS_CLIP), "k": cs.K_DEFAULT, "h_candidates": list(cs.H_CANDIDATES),
        "arl0_target_days": cs.ARL0_TARGET, "schemes": schemes,
        "timing": "update S_D at the end of day D; after an alarm the retrained model forecasts from D+1",
        "ref_update_options": {"A": "keep the 2014 mu0, sigma0, q99 after every retrain",
                               "B": "re-estimate q99, mu0, sigma0 from the new model's out-of-sample daily errors on the last "
                                    "12 months of its training window (inner fit inside the same retraining)"},
        "ref_update": REF_UPDATE_DECISION["choice"], "ref_update_decision": REF_UPDATE_DECISION,
        "bootstrap": {"method": "weekly block bootstrap (full Mon-Sun weeks of 2014)", "block_days": BLOCK,
                      "series_days": N_DAYS_SIM, "replicates": N_REP, "seed": SEED, "arl_per_replicate": "3650 / n_alarms"},
        "data_period": {"calibration": "2014 (out-of-sample predictions of REF)", "latest_timestamp_read": "2015-01-01 00:00"},
        "ref": {"path": "results/price_hgbc/official/boost_price_ref.json", "sha256": ofr.sha_file(ofr.REF_JSON),
                "preds_2014_sha256": ofr.sha_file(ofr.PREDS), "tau": r["tau"]},
        "code_commit": None, "code_commit_note": "project root is not a git repository; code identified by SHA-256 below",
        "code_sha256": {f: sha_file(f) for f in CODE_FILES}, "design_sha256": sha_file(os.path.relpath(P["design"], ROOT)),
    }


def run():
    ACCESS.clear()
    d, r = daily_errors()
    stats = ref_stats_table(d)
    main = stats.set_index("scheme").loc["main"]
    d["logloss_trunc"] = np.minimum(d["logloss"], main["q99"])
    d["misclass_trunc"] = np.minimum(d["misclass"], stats.set_index("scheme").loc["misclass", "q99"])
    d["truncated"] = d["logloss"] > main["q99"]
    diag = diagnostics(d, main["q99"])
    blocks = week_blocks(d["day"])
    W = boot_matrix(len(blocks))
    arl = arl_table(d, stats, blocks, W)
    h = int(arl[(arl["scheme"] == "main") & arl["selected"]]["h"].iloc[0])
    det = cs.Cusum(main["mu0"], main["sigma0"], main["q99"], main["k"], h)
    path = cs.run_batch(pd.Series(d["logloss"].to_numpy(), index=d["day"]), det)
    hm = int(arl[(arl["scheme"] == "misclass") & arl["selected"]]["h"].iloc[0])
    sm = stats.set_index("scheme").loc["misclass"]
    path_m = cs.run_batch(pd.Series(d["misclass"].to_numpy(), index=d["day"]), cs.Cusum(sm["mu0"], sm["sigma0"], sm["q99"], sm["k"], hm))
    path["scheme"], path_m["scheme"] = "main", "misclass"
    delay = delay_table(stats, d, blocks, W, h)
    optb, e13 = option_b_demo(r)
    optb.loc[optb["option"] == "A", ["q99", "mu0", "sigma0", "n_days", "n_truncated"]] = \
        main[["q99", "mu0", "sigma0", "n_days", "n_truncated"]].to_numpy()
    meta = {"blocks": len(blocks), "block_first_day": str(pd.Timestamp(d["day"].iloc[blocks[0][0]]).date()),
            "block_last_day": str(pd.Timestamp(d["day"].iloc[blocks[-1][-1]]).date()),
            "boot_matrix_sha256": hashlib.sha256(np.ascontiguousarray(W).tobytes()).hexdigest()}
    return {"daily": d, "stats": stats, "diag": diag, "arl": arl, "path": pd.concat([path, path_m], ignore_index=True),
            "delay": delay, "optB": optb, "meta": meta, "ref": r, "h": h}


def main():
    os.makedirs(OUT, exist_ok=True)
    R = run()
    R["daily"].to_csv(P["daily"], index=False, float_format="%.17g")
    R["stats"].to_csv(P["ref_stats"], index=False, float_format="%.17g")
    R["diag"].to_csv(P["diag"], index=False, float_format="%.17g")
    R["arl"].to_csv(P["arl"], index=False, float_format="%.17g")
    R["path"].to_csv(P["path"], index=False, float_format="%.17g")
    R["delay"].to_csv(P["delay"], index=False, float_format="%.17g")
    R["optB"].to_csv(P["optB"], index=False, float_format="%.17g")
    consts = [("k", cs.K_DEFAULT), ("arl0_target_days", cs.ARL0_TARGET), ("logloss_clip_low", cs.LOGLOSS_CLIP[0]),
              ("logloss_clip_high", cs.LOGLOSS_CLIP[1]), ("bootstrap_block_days", BLOCK), ("bootstrap_series_days", N_DAYS_SIM),
              ("bootstrap_replicates", N_REP), ("bootstrap_seed", SEED), ("bootstrap_weeks", R["meta"]["blocks"]),
              ("ref_tau", R["ref"]["tau"])] + [(f"h_candidate_{i}", h) for i, h in enumerate(cs.H_CANDIDATES)]
    pd.DataFrame(consts, columns=["constant", "value"]).to_csv(P["consts"], index=False, float_format="%.17g")
    cfg = build_config(R["stats"], R["arl"], R["ref"])
    cfg["bootstrap"].update(R["meta"])
    cs.validate_config(cfg)
    json.dump(cfg, open(P["config"], "w"), indent=1, allow_nan=False)
    json.dump({"reads": ACCESS, "latest_timestamp_read": max(a["max_timestamp"] for a in ACCESS if a["max_timestamp"] != "n/a")},
              open(P["log"], "w"), indent=1)
    s = R["stats"].set_index("scheme").loc["main"]
    for fn, args, key in ((fig_err, (pd.read_csv(P["daily"]), s), "fig_err"),
                          (fig_path, (pd.read_csv(P["path"]).query("scheme == 'main'"), R["h"]), "fig_S"),
                          (fig_arl, (pd.read_csv(P["arl"]),), "fig_arl"), (fig_delay, (pd.read_csv(P["delay"]),), "fig_delay")):
        fig, _ = fn(*args)
        fig.savefig(P[key], dpi=150, facecolor=c.SURFACE)
    pd.set_option("display.width", 250)
    print(R["stats"].to_string(index=False))
    print(R["diag"].to_string(index=False))
    print(R["arl"].to_string(index=False))
    print("selected h:", R["h"], "| 2014 alarms (main):", int(R["path"].query("scheme=='main'")["alarm"].sum()),
          "| misclass alarms:", int(R["path"].query("scheme=='misclass'")["alarm"].sum()))
    print(R["delay"].to_string(index=False))
    print(R["optB"].to_string(index=False))
    print(json.dumps(R["meta"]))


if __name__ == "__main__":
    main()
