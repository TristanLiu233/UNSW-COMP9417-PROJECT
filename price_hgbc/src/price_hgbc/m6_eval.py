"""M6 evaluation on the development period 2015-01-01 .. 2024-12-31 only (2025-2026 are never scored here).

Outputs (results/price_hgbc/m6/):
  strategy_comparison_dev.csv, yearly_dev.csv, monthly_dev.csv, paired_bootstrap_dev.csv, tradeoff_dev.csv,
  option_a_yearly.csv, option_a_retrain_effect.csv, fig_m6_yearly_balacc.png, fig_m6_s3_cusum.png, fig_m6_tradeoff.png
Usage: python -m src.price_hgbc.m6_eval
"""
import json
import os

import numpy as np
import pandas as pd

from . import cusum as cs
from . import m3_common as c
from . import m6
from .kit import DATA_CSV, da

STRATS = ["S0", "S1", "S2", "S3"]
DEV = (pd.Timestamp("2015-01-01"), m6.DEV_END)
P = {k: os.path.join(m6.OUT, v) for k, v in {
    "cmp": "strategy_comparison_dev.csv", "yearly": "yearly_dev.csv", "monthly": "monthly_dev.csv",
    "boot": "paired_bootstrap_dev.csv", "trade": "tradeoff_dev.csv", "a_year": "option_a_yearly.csv",
    "a_effect": "option_a_retrain_effect.csv", "fig_year": "fig_m6_yearly_balacc.png", "fig_cusum": "fig_m6_s3_cusum.png",
    "fig_trade": "fig_m6_tradeoff.png", "meta": "eval_meta.json"}.items()}


def load_preds():
    out = {}
    for s in STRATS:
        p = pd.read_parquet(os.path.join(m6.OUT, f"{s.lower()}_predictions.parquet"))
        out[s] = p[(p["day"] >= DEV[0]) & (p["day"] <= DEV[1])]
    idx = out["S0"].index
    assert all(v.index.equals(idx) for v in out.values())                 # identical dates and periods
    return out


def truth_and_baselines(df, idx):
    days = sorted(set(da.market_day(idx)))
    assert max(days) <= DEV[1]
    y, b1, b7 = [], [], []
    for d in days:
        block = df.loc[d - pd.Timedelta("7D") + pd.Timedelta("30min"): d + pd.Timedelta("1D")]
        h, w = da.make_inputs(block, d)
        y.append(da.targets(block, d)["nsw_price_up"])
        b1.append(da.baseline(h, w, 1)["nsw_price_up"])
        b7.append(da.baseline(h, w, 7)["nsw_price_up"])
    t = pd.DataFrame({"y": pd.concat(y), "b1": pd.concat(b1), "b7": pd.concat(b7)})
    assert t.index.equals(idx)
    return t


def scores(y, pred):
    return {"Acc": da.accuracy(y, pred), "BalAcc": da.balanced_accuracy(y, pred)}


def run():
    df = da.load_dataset(DATA_CSV)
    preds = load_preds()
    idx = preds["S0"].index
    t = truth_and_baselines(df, idx)
    for s in STRATS:
        t[s] = preds[s]["pred"].to_numpy()
    t["day"] = da.market_day(idx)
    log = pd.read_csv(os.path.join(m6.OUT, "models_log.csv"))
    ev = pd.read_csv(os.path.join(m6.OUT, "s3_events.csv"))
    ev["alarm_day"] = pd.to_datetime(ev["alarm_day"])
    evd = ev[ev["alarm_day"] <= DEV[1]]
    s0 = scores(t["y"], t["S0"])
    b1 = scores(t["y"], t["b1"])
    rows = []
    for s in STRATS + ["b1", "b7"]:
        sc = scores(t["y"], t[s])
        lg = log[(log["strategy"] == s) & (log["model_id"] != "M5_frozen")] if s in STRATS else log.iloc[0:0]
        lg_dev = lg[pd.to_datetime(lg["effective_from"]) <= DEV[1]]
        rows.append({"strategy": s, "days": t["day"].nunique(), "rows": len(t), **sc,
                     "BalAcc_minus_b1": sc["BalAcc"] - b1["BalAcc"], "BalAcc_minus_S0": sc["BalAcc"] - s0["BalAcc"],
                     "Acc_minus_b1": sc["Acc"] - b1["Acc"], "Acc_minus_S0": sc["Acc"] - s0["Acc"],
                     "retrains_total": len(lg), "retrains_effective_in_dev": len(lg_dev),
                     "train_seconds_total": float(lg["total_seconds"].sum()) if len(lg) else 0.0,
                     "train_seconds_mean": float(lg["total_seconds"].mean()) if len(lg) else 0.0,
                     "cusum_alarms_dev": len(evd) if s == "S3" else np.nan,
                     "cusum_suppressed_cooldown_dev": int((evd["suppressed_reason"] == "cooldown").sum()) if s == "S3" else np.nan,
                     "cusum_suppressed_budget_dev": int((evd["suppressed_reason"] == "budget").sum()) if s == "S3" else np.nan,
                     "s3_actual_retrains_dev": int(evd["retrained"].sum()) if s == "S3" else np.nan})
    cmp_ = pd.DataFrame(rows)
    # yearly / monthly
    yr, mo = [], []
    for key, out in (("year", yr), ("month", mo)):
        k = t["day"].dt.year if key == "year" else t["day"].dt.to_period("M").astype(str)
        for kv, g in t.groupby(k):
            row = {key: kv, "days": g["day"].nunique(), "class1_share": float(g["y"].mean() * 100)}
            for s in STRATS + ["b1", "b7"]:
                sc = scores(g["y"], g[s])
                row[f"{s}_Acc"], row[f"{s}_BalAcc"] = sc["Acc"], sc["BalAcc"]
            out.append(row)
    yearly, monthly = pd.DataFrame(yr), pd.DataFrame(mo)
    # paired whole-day bootstrap
    days = sorted(t["day"].unique())
    B = c.boot_indices(len(days))
    samp = {s: c.boot_balacc(t["y"].to_numpy(), t[s].to_numpy(), idx, B) for s in STRATS + ["b1"]}
    pairs = [(s, "S0") for s in ("S1", "S2", "S3")] + [("S3", "S1")] + [(s, "b1") for s in STRATS]
    brow = []
    for a, b in pairs:
        d = samp[a] - samp[b]
        lo, hi = c.ci(d)
        brow.append({"a": a, "b": b, "BalAcc_a": scores(t["y"], t[a])["BalAcc"], "BalAcc_b": scores(t["y"], t[b])["BalAcc"],
                     "diff_pp": scores(t["y"], t[a])["BalAcc"] - scores(t["y"], t[b])["BalAcc"], "diff_lo": lo, "diff_hi": hi,
                     "ci_excludes_0": bool(lo > 0 or hi < 0), "n_days": len(days), "n_boot": c.N_BOOT, "seed": c.BOOT_SEED})
    boot = pd.DataFrame(brow)
    trade = cmp_[cmp_["strategy"].isin(STRATS)][["strategy", "retrains_total", "retrains_effective_in_dev", "BalAcc", "BalAcc_minus_S0"]].copy()
    bs = boot[boot["b"] == "S0"].set_index("a")
    trade["gain_vs_S0_lo"] = [bs.loc[s, "diff_lo"] if s in bs.index else 0.0 for s in trade["strategy"]]
    trade["gain_vs_S0_hi"] = [bs.loc[s, "diff_hi"] if s in bs.index else 0.0 for s in trade["strategy"]]
    trade["train_seconds_total"] = cmp_.set_index("strategy").loc[trade["strategy"], "train_seconds_total"].to_numpy()
    # option A analysis (S3 detector state, dev period only)
    cfg = json.load(open(m6.CUSUM_CONFIG))["schemes"]["main"]
    dly = pd.read_csv(os.path.join(m6.OUT, "s3_daily_cusum.csv"), parse_dates=["day"])
    dly = dly[dly["day"] <= DEV[1]]
    ll = {s: cs.daily_logloss(t["y"], preds[s]["p_up"], idx) for s in STRATS}
    assert np.allclose(ll["S3"].to_numpy(), dly["logloss"].to_numpy(), rtol=0, atol=1e-12)
    arow = []
    for y in range(2015, 2025):
        m = pd.DatetimeIndex(ll["S0"].index).year == y
        e3 = ll["S3"][m]
        g = dly[dly["day"].dt.year == y]
        gy = evd[evd["alarm_day"].dt.year == y]
        arow.append({"year": y, **{f"{s}_mean_logloss": float(ll[s][m].mean()) for s in STRATS},
                     "S3_logloss_trunc_mean": float(np.minimum(e3, cfg["q99"]).mean()),
                     "S3_logloss_trunc_sd": float(np.minimum(e3, cfg["q99"]).std(ddof=1)),
                     "mu0": cfg["mu0"], "sigma0": cfg["sigma0"],
                     "S3_mean_z": float(((np.minimum(e3, cfg["q99"]) - cfg["mu0"]) / cfg["sigma0"]).mean()),
                     "share_days_truncated": float((e3 > cfg["q99"]).mean()),
                     "S3_mean_S": float(g["S"].mean()), "S3_max_S": float(g["S"].max()),
                     "alarms": len(gy), "retrains": int(gy["retrained"].sum()),
                     "suppressed_cooldown": int((gy["suppressed_reason"] == "cooldown").sum()),
                     "suppressed_budget": int((gy["suppressed_reason"] == "budget").sum())})
    a_year = pd.DataFrame(arow)
    erow = []
    for _, r in ev[ev["retrained"]].iterrows():
        a = r["alarm_day"]
        eff = a + pd.Timedelta("1D")
        before = ll["S3"][(ll["S3"].index > a - pd.Timedelta("60D")) & (ll["S3"].index <= a)]
        after_s3 = ll["S3"][(ll["S3"].index >= eff) & (ll["S3"].index < eff + pd.Timedelta("60D"))]
        after_s0 = ll["S0"][(ll["S0"].index >= eff) & (ll["S0"].index < eff + pd.Timedelta("60D"))]
        erow.append({"alarm_day": a.date(), "retrain_index": int(r["retrain_index"]), "S3_logloss_60d_before": float(before.mean()),
                     "S3_logloss_60d_after": float(after_s3.mean()), "S0_logloss_60d_after_same_days": float(after_s0.mean()),
                     "after_minus_S0_same_days": float(after_s3.mean() - after_s0.mean())})
    effect = pd.DataFrame(erow)
    meta = {"dev_period": [str(DEV[0].date()), str(DEV[1].date())], "boot_index_sha256": c.sha(np.ascontiguousarray(B).tobytes()),
            "budget_exhausted_on": str(ev[ev["retrained"]]["alarm_day"].max().date()) if int(ev["retrained"].sum()) >= m6.BUDGET else None,
            "note": "2025-2026 are not scored; S3 alarms there are simulation state only"}
    return cmp_, yearly, monthly, boot, trade, a_year, effect, meta, dly


COL = {"S0": "#a3a29c", "S1": c.BLUE, "S2": "#1baf7a", "S3": c.ORANGE, "b1": c.INK2}


def fig_yearly(yearly):
    plt = c.mpl()
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=c.SURFACE)
    c.style(ax)
    ax.grid(axis="y", color=c.GRID, linewidth=0.8)
    for s in STRATS + ["b1"]:
        ax.plot(yearly["year"], yearly[f"{s}_BalAcc"], color=COL[s], linewidth=2 if s != "b1" else 1.4,
                linestyle="-" if s != "b1" else (0, (4, 3)), marker="o", markersize=4, label=s)
    ax.set_xlabel("Year (target days)", color=c.INK2)
    ax.set_ylabel("BalAcc (%)", color=c.INK2)
    ax.set_title("Yearly BalAcc 2015–2024: S0 frozen, S1 expanding, S2 3-year, S3 CUSUM; b1 dashed",
                 color=c.INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=c.INK2, ncol=5, loc="lower left")
    fig.tight_layout()
    return fig, ax


def fig_cusum(dly, log):
    plt = c.mpl()
    cfg = json.load(open(m6.CUSUM_CONFIG))["schemes"]["main"]
    fig, ax = plt.subplots(figsize=(10, 3.8), facecolor=c.SURFACE)
    c.style(ax)
    ax.grid(axis="y", color=c.GRID, linewidth=0.8)
    ax.plot(dly["day"], dly["S"], color=c.BLUE, linewidth=0.8, label="S_D (before reset)")
    ax.axhline(cfg["h"], color=c.INK2, linewidth=1, linestyle=(0, (4, 3)), label=f"h = {cfg['h']}")
    al = dly[dly["alarm"] & ~dly["retrained"]]
    rt = dly[dly["retrained"]]
    ax.scatter(al["day"], al["S"], s=14, color=c.GRAY, zorder=3, label="alarm, suppressed")
    ax.scatter(rt["day"], rt["S"], s=50, color=c.ORANGE, zorder=4, label="alarm → retrain")
    ax.set_xlabel("Target day (2015–2024)", color=c.INK2)
    ax.set_ylabel("CUSUM S_D (σ₀ units)", color=c.INK2)
    ax.set_title("S3 CUSUM path with option A reference (2014): alarms and retrains", color=c.INK, fontsize=11, loc="left")
    ax.set_ylim(0, max(cfg["h"], dly["S"].max()) * 1.25)
    ax.legend(frameon=False, fontsize=8, labelcolor=c.INK2, ncol=4, loc="upper left")
    fig.tight_layout()
    return fig, ax


def fig_trade(trade):
    plt = c.mpl()
    fig, ax = plt.subplots(figsize=(6.5, 4), facecolor=c.SURFACE)
    c.style(ax)
    ax.grid(axis="both", color=c.GRID, linewidth=0.8)
    for _, r in trade.iterrows():
        s = r["strategy"]
        ax.errorbar(r["retrains_effective_in_dev"], r["BalAcc_minus_S0"],
                    yerr=[[r["BalAcc_minus_S0"] - r["gain_vs_S0_lo"]], [r["gain_vs_S0_hi"] - r["BalAcc_minus_S0"]]],
                    fmt="o", color=COL[s], ecolor=COL[s], capsize=4, markersize=8)
        ax.annotate(f"{s} {r['BalAcc_minus_S0']:+.2f}", (r["retrains_effective_in_dev"], r["BalAcc_minus_S0"]),
                    xytext=(8, 4), textcoords="offset points", fontsize=9, color=c.INK2)
    ax.axhline(0, color=c.INK2, linewidth=1)
    ax.set_xlabel("Retrains in force during 2015–2024", color=c.INK2)
    ax.set_ylabel("BalAcc gain over S0 (pp), 95 % paired CI", color=c.INK2)
    ax.set_title("Retrains vs gain over the frozen model (2015–2024)", color=c.INK, fontsize=11, loc="left")
    fig.tight_layout()
    return fig, ax


def main():
    cmp_, yearly, monthly, boot, trade, a_year, effect, meta, dly = run()
    for k, v in (("cmp", cmp_), ("yearly", yearly), ("monthly", monthly), ("boot", boot), ("trade", trade),
                 ("a_year", a_year), ("a_effect", effect)):
        v.to_csv(P[k], index=False, float_format="%.17g")
    json.dump(meta, open(P["meta"], "w"), indent=1)
    log = pd.read_csv(os.path.join(m6.OUT, "models_log.csv"))
    for fn, args, key in ((fig_yearly, (pd.read_csv(P["yearly"]),), "fig_year"), (fig_cusum, (dly, log), "fig_cusum"),
                          (fig_trade, (pd.read_csv(P["trade"]),), "fig_trade")):
        fig, _ = fn(*args)
        fig.savefig(P[key], dpi=150, facecolor=c.SURFACE)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(cmp_.round(3).to_string(index=False))
    print(yearly[["year"] + [f"{s}_BalAcc" for s in STRATS + ["b1"]]].round(2).to_string(index=False))
    print(boot.round(3).to_string(index=False))
    print(a_year.round(3).to_string(index=False))
    print(effect.round(3).to_string(index=False))
    print(meta)


if __name__ == "__main__":
    main()
