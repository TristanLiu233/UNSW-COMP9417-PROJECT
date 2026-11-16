"""Pre-registered S3 sensitivity run (results/price_hgbc/m6/s3_sensitivity_preregistration.md, approved and frozen).

Variants (one factor each vs the official S3): S3-A control; S3-B option B reference; S3-K k = 0.75 with h re-selected
by the M4 rule on 2014; S3-C 730-day cooldown. Evaluation 2015-2024 only; 2025-2026 simulated, never scored.
Exploratory: the official M6 S3 result is unchanged.
Usage: python -m src.price_hgbc.m6_sensitivity
"""
import json
import os
import time

import numpy as np
import pandas as pd

from . import cusum as cs
from . import m3_common as c
from . import m4, m6, m6_eval
from .kit import da

OUT = os.path.join(m6.OUT, "sensitivity")
MODELS = os.path.join(OUT, "models")
PREREG = os.path.join(m6.OUT, "s3_sensitivity_preregistration.md")
K_VARIANT, COOLDOWN_C = 0.75, 730
NAMES = ["S3-A", "S3-B", "S3-K", "S3-C"]
F = {k: os.path.join(OUT, v) for k, v in {
    "calib": "s3k_calibration.csv", "cmp": "sensitivity_comparison_dev.csv", "yearly": "sensitivity_yearly_dev.csv",
    "boot": "sensitivity_bootstrap_dev.csv", "refB": "s3b_reference_updates.csv", "log": "models_log_sensitivity.csv",
    "zyear": "sensitivity_z_by_year.csv", "meta": "sensitivity_meta.json", "fig_paths": "fig_sensitivity_cusum_paths.png",
    "fig_year": "fig_sensitivity_yearly.png"}.items()}


def calibrate_k():
    """S3-K: k = 0.75; h = smallest candidate with median ARL0 >= 730 on the 2014 weekly block bootstrap (M4 rule)."""
    d, _ = m4.daily_errors()
    stats = m4.ref_stats_table(d)
    main = stats[stats["scheme"] == "main"].copy()
    main["k"] = K_VARIANT
    blocks = m4.week_blocks(d["day"])
    t = m4.arl_table(d, main, blocks, m4.boot_matrix(len(blocks)))
    t["written_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return t


def variants(cfg, h_k):
    s = cfg["schemes"]["main"]
    return {"S3-A": {"name": "S3-A", "ref_update": "A", "k": s["k"], "h": s["h"], "cooldown": m6.COOLDOWN_DAYS},
            "S3-B": {"name": "S3-B", "ref_update": "B", "k": s["k"], "h": s["h"], "cooldown": m6.COOLDOWN_DAYS},
            "S3-K": {"name": "S3-K", "ref_update": "A", "k": K_VARIANT, "h": h_k, "cooldown": m6.COOLDOWN_DAYS},
            "S3-C": {"name": "S3-C", "ref_update": "A", "k": s["k"], "h": s["h"], "cooldown": COOLDOWN_C}}


def boot_p(d):
    """Two-sided bootstrap p-value: twice the smaller tail share around 0 (capped at 1)."""
    return float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))


def holm(ps):
    order = np.argsort(ps)
    adj, running = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def evaluate(df, preds, logs, events, dailies):
    dev = {k: v[(v["day"] >= m6_eval.DEV[0]) & (v["day"] <= m6_eval.DEV[1])] for k, v in preds.items()}
    idx = dev["S0"].index
    assert all(v.index.equals(idx) for v in dev.values())
    t = m6_eval.truth_and_baselines(df, idx)
    for k, v in dev.items():
        t[k] = v["pred"].to_numpy()
    t["day"] = da.market_day(idx)
    days = sorted(t["day"].unique())
    B = c.boot_indices(len(days))
    sha_B = c.sha(np.ascontiguousarray(B).tobytes())
    assert sha_B == json.load(open(m6_eval.P["meta"]))["boot_index_sha256"]          # same resampled days as M6
    cols = list(dev) + ["b1"]
    samp = {k: c.boot_balacc(t["y"].to_numpy(), t[k].to_numpy(), idx, B) for k in cols}
    bal = {k: da.balanced_accuracy(t["y"], t[k]) for k in cols}
    acc = {k: da.accuracy(t["y"], t[k]) for k in cols}
    rows = []
    for role, pairs in (("primary", [(v, "S3-A") for v in ("S3-B", "S3-K", "S3-C")]),
                        ("secondary", [(v, "S1") for v in NAMES]),
                        ("descriptive", [(v, "S0") for v in NAMES] + [(v, "b1") for v in NAMES])):
        for a, b in pairs:
            dd = samp[a] - samp[b]
            lo, hi = c.ci(dd)
            rows.append({"role": role, "a": a, "b": b, "BalAcc_a": bal[a], "BalAcc_b": bal[b], "diff_pp": bal[a] - bal[b],
                         "diff_lo": lo, "diff_hi": hi, "ci_excludes_0": bool(lo > 0 or hi < 0), "p_boot": boot_p(dd)})
    boot = pd.DataFrame(rows)
    prim = boot["role"] == "primary"
    boot["p_holm"] = np.nan
    boot.loc[prim, "p_holm"] = holm(boot.loc[prim, "p_boot"].to_numpy())
    cmp_ = []
    for k in NAMES:
        lg = logs[k][logs[k]["model_id"] != "M5_frozen"]
        ev = events[k].copy()
        ev["alarm_day"] = pd.to_datetime(ev["alarm_day"])
        evd = ev[ev["alarm_day"] <= m6_eval.DEV[1]]
        vs1 = boot[(boot["a"] == k) & (boot["b"] == "S1")].iloc[0]
        cmp_.append({"variant": k, "Acc": acc[k], "BalAcc": bal[k], "BalAcc_minus_S3A": bal[k] - bal["S3-A"],
                     "BalAcc_minus_S1": bal[k] - bal["S1"], "BalAcc_minus_S0": bal[k] - bal["S0"], "BalAcc_minus_b1": bal[k] - bal["b1"],
                     "retrains_total": len(lg), "retrain_days": ";".join(str(x) for x in lg["alarm_day"]),
                     "budget_exhausted_on": str(lg["alarm_day"].iloc[-1]) if len(lg) >= m6.BUDGET else "",
                     "alarms_dev": len(evd), "suppressed_cooldown_dev": int((evd["suppressed_reason"] == "cooldown").sum()),
                     "suppressed_budget_dev": int((evd["suppressed_reason"] == "budget").sum()),
                     "train_seconds_total": float(lg["total_seconds"].sum()),
                     # pre-registered adoption rule (section 6, default): >= 0.5 pp over S1, CI excludes 0, within budget
                     "meets_adoption_rule": bool(k != "S3-A" and vs1["diff_pp"] >= 0.5 and vs1["diff_lo"] > 0 and len(lg) <= m6.BUDGET)})
    for k in ("S0", "S1", "b1"):
        cmp_.append({"variant": k, "Acc": acc[k], "BalAcc": bal[k], "BalAcc_minus_S3A": bal[k] - bal["S3-A"],
                     "BalAcc_minus_S1": bal[k] - bal["S1"], "BalAcc_minus_S0": bal[k] - bal["S0"], "BalAcc_minus_b1": bal[k] - bal["b1"]})
    yr = []
    for y, g in t.groupby(t["day"].dt.year):
        yr.append({"year": y, **{f"{k}_BalAcc": da.balanced_accuracy(g["y"], g[k]) for k in cols}})
    z = []
    for k in NAMES:
        dly = dailies[k]
        dly = dly[pd.to_datetime(dly["day"]) <= m6_eval.DEV[1]]
        zz = (np.minimum(dly["logloss"], dly["q99"]) - dly["mu0"]) / dly["sigma0"]
        for y, g in zz.groupby(pd.to_datetime(dly["day"]).dt.year):
            z.append({"variant": k, "year": y, "mean_z": float(g.mean())})
    return pd.DataFrame(cmp_), pd.DataFrame(yr), boot, pd.DataFrame(z), sha_B


def fig_paths(dailies, logs):
    plt = c.mpl()
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True, facecolor=c.SURFACE)
    for ax, k in zip(axes, NAMES):
        c.style(ax)
        dly = dailies[k]
        dly = dly[pd.to_datetime(dly["day"]) <= m6_eval.DEV[1]]
        x = pd.to_datetime(dly["day"])
        ax.plot(x, dly["S"], color=c.BLUE, linewidth=0.7)
        ax.plot(x, dly["h"], color=c.INK2, linewidth=1, linestyle=(0, (4, 3)))
        rt = dly[dly["retrained"]]
        ax.scatter(pd.to_datetime(rt["day"]), rt["S"], s=40, color=c.ORANGE, zorder=3)
        ax.set_ylabel("S_D", color=c.INK2)
        ax.set_title(f"{k}: {int(dly['retrained'].sum())} retrains in 2015–2024 (orange), h dashed",
                     color=c.INK, fontsize=10, loc="left")
    axes[-1].set_xlabel("Target day (2015–2024)", color=c.INK2)
    fig.tight_layout()
    return fig, axes


def fig_yearly(yearly):
    plt = c.mpl()
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=c.SURFACE)
    c.style(ax)
    ax.grid(axis="y", color=c.GRID, linewidth=0.8)
    style = {"S1": (c.BLUE, "-"), "S3-A": (c.ORANGE, "-"), "S3-B": ("#1baf7a", "-"), "S3-K": ("#6250d6", "-"),
             "S3-C": ("#e87ba4", "-"), "S0": (c.GRAY, "-")}
    for k, (col, ls) in style.items():
        ax.plot(yearly["year"], yearly[f"{k}_BalAcc"], color=col, linestyle=ls, linewidth=2 if k in ("S1", "S3-A") else 1.6,
                marker="o", markersize=4, label=k)
    ax.set_xlabel("Year (target days)", color=c.INK2)
    ax.set_ylabel("BalAcc (%)", color=c.INK2)
    ax.set_title("S3 sensitivity variants vs S1 and S0, yearly BalAcc 2015–2024", color=c.INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=c.INK2, ncol=6, loc="lower left")
    fig.tight_layout()
    return fig, ax


def main():
    os.makedirs(OUT, exist_ok=True)
    prereg_sha = m6.m5.sha_file(PREREG)
    calib = calibrate_k()
    calib.to_csv(F["calib"], index=False, float_format="%.17g")                     # written before any S3-K replay
    calib_sha = m6.m5.sha_file(F["calib"])
    h_k = int(calib[calib["selected"]]["h"].iloc[0])
    ctx = m6.Context()
    cfg = json.load(open(m6.CUSUM_CONFIG))
    preds, logs, events, dailies = {}, {}, {}, {}
    for k, v in variants(cfg, h_k).items():
        t0 = time.perf_counter()
        p, l, e, d = m6.run_s3(ctx, variant=v, save=(k != "S3-A"), models_dir=MODELS)
        preds[k], logs[k], events[k], dailies[k] = p, l, e, d
        tag = k.lower().replace("-", "")
        p.to_parquet(os.path.join(OUT, f"{tag}_predictions.parquet"))
        e.to_csv(os.path.join(OUT, f"{tag}_events.csv"), index=False, float_format="%.17g")
        m6.mask_holdout(d).to_csv(os.path.join(OUT, f"{tag}_daily_cusum.csv"), index=False, float_format="%.17g")
        print(f"{k}: {len(l) - 1} retrains, {len(e)} alarms, {time.perf_counter() - t0:.0f}s")
    pd.concat(logs.values(), ignore_index=True).to_csv(F["log"], index=False)
    lb = logs["S3-B"]
    lb[lb["model_id"] != "M5_frozen"][["model_id", "alarm_day", "val_start", "val_end", "ref_from_days", "ref_q99_after",
                                        "ref_mu0_after", "ref_sigma0_after"]].to_csv(F["refB"], index=False, float_format="%.17g")
    for s in ("S0", "S1"):
        preds[s] = pd.read_parquet(os.path.join(m6.OUT, f"{s.lower()}_predictions.parquet"))
    cmp_, yearly, boot, z, sha_B = evaluate(ctx.df, preds, logs, events, dailies)
    cmp_.to_csv(F["cmp"], index=False, float_format="%.17g")
    yearly.to_csv(F["yearly"], index=False, float_format="%.17g")
    boot.to_csv(F["boot"], index=False, float_format="%.17g")
    z.to_csv(F["zyear"], index=False, float_format="%.17g")
    json.dump({"prereg_sha256": prereg_sha, "calibration_sha256_before_replay": calib_sha, "h_for_S3K": h_k,
               "boot_index_sha256": sha_B, "dev_period": ["2015-01-01", "2024-12-31"],
               "note": "exploratory sensitivity; official M6 S3 unchanged; 2025-2026 not scored"},
              open(F["meta"], "w"), indent=1)
    for fn, args, key in ((fig_paths, (dailies, logs), "fig_paths"), (fig_yearly, (pd.read_csv(F["yearly"]),), "fig_year")):
        fig, _ = fn(*args)
        fig.savefig(F[key], dpi=150, facecolor=c.SURFACE)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(calib[["h", "k", "arl0_median", "arl0_q05", "arl0_q95", "meets_target", "selected"]].to_string(index=False))
    print(cmp_.round(3).to_string(index=False))
    print(boot.round(4).to_string(index=False))
    print(yearly.round(2).to_string(index=False))
    print(pd.read_csv(F["refB"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
