"""M7 step 3: evaluation on 2015-01-01 .. 2024-12-31 only, plus the provisional Stage 3 choice (design 5, 11.1-11.2).

Nothing from 2025-2026 is scored, summarised or plotted here.
Usage: python -m src.price_lr.evaluate
"""
import json
import os

import numpy as np
import pandas as pd

from src.price_hgbc import m3_common as c
from src.price_hgbc import m6_eval as m6e

from . import data as D
from .env import RESULTS, da
from .replay import CUSUM_CONFIG, DEV_END, FROZEN_ID

STRATS = ["S0", "S1", "S2", "S3"]
DEV = (pd.Timestamp("2015-01-01"), DEV_END)
WINDOW = {"S0": "expanding", "S1": "expanding", "S2": "sliding", "S3": "expanding"}
P = {k: os.path.join(RESULTS, v) for k, v in {
    "cmp": "strategy_comparison_dev.csv", "yearly": "yearly_dev.csv", "monthly": "monthly_dev.csv",
    "boot": "paired_bootstrap_dev.csv", "trade": "tradeoff_dev.csv", "meta": "eval_meta.json",
    "stage3": "stage3_provisional.json", "fig_year": "fig_m7_yearly_balacc.png", "fig_cusum": "fig_m7_s3_cusum.png",
    "fig_trade": "fig_m7_tradeoff.png"}.items()}


def scores(y, pred):
    return {"Acc": da.accuracy(y, pred), "BalAcc": da.balanced_accuracy(y, pred)}


def load():
    preds = {}
    for s in STRATS:
        p = pd.read_parquet(os.path.join(RESULTS, f"{s.lower()}_predictions.parquet"))
        preds[s] = p[(p["day"] >= DEV[0]) & (p["day"] <= DEV[1])]
    idx = preds["S0"].index
    assert all(v.index.equals(idx) for v in preds.values())
    assert preds["S0"]["day"].max() == DEV[1]
    return preds, idx


def provisional(cmp_, boot):
    """Design 11.1-11.2 (no 2025-2026 number is used)."""
    c4 = cmp_[cmp_["strategy"].isin(STRATS)].set_index("strategy")
    rank = c4["BalAcc"].sort_values(ascending=False)
    F = rank.index[0]
    cand = [F]
    steps = []
    for x in rank.index[1:]:
        r = boot[(boot["a"] == F) & (boot["b"] == x)].iloc[0]
        inc = bool(r["diff_lo"] <= 0 <= r["diff_hi"])
        steps.append({"vs": x, "diff_pp": r["diff_pp"], "lo": r["diff_lo"], "hi": r["diff_hi"], "ci_includes_0": inc})
        if inc:
            cand.append(x)
    key = lambda s: (int(c4.loc[s, "retrains_effective_in_dev"]), 0 if WINDOW[s] == "expanding" else 1, -c4.loc[s, "BalAcc"])
    P_ = min(cand, key=key)
    R = [s for s in rank.index if s != P_][0]
    return {"rule": "strategy_design.md section 11.1-11.2", "uses_2025_2026": False, "ranking_2015_2024": list(rank.index),
            "BalAcc_2015_2024": {s: float(c4.loc[s, "BalAcc"]) for s in STRATS}, "F": F,
            "F_minus_others": steps, "candidate_set": cand,
            "tie_keys": {s: {"retrains_effective_in_dev": int(c4.loc[s, "retrains_effective_in_dev"]), "window": WINDOW[s],
                             "BalAcc": float(c4.loc[s, "BalAcc"])} for s in cand},
            "P_provisional": P_, "R_backup": R,
            "pending": "11-18: veto check on 2025-01-01..2026-10-31 (section 11.4), then final training (11.5)"}


def run():
    df = pd.read_csv(D.DATA_CSV, index_col="timestamp", parse_dates=True)
    df = df.loc[df.index <= DEV[1] + pd.Timedelta("1D")]                  # nothing after 2025-01-01 00:00 is read
    preds, idx = load()
    t = D.truth_and_baselines(df, *DEV)
    assert t.index.equals(idx)
    for s in STRATS:
        t[s] = preds[s]["pred"].to_numpy()
    t["day"] = preds["S0"]["day"].to_numpy()
    log = pd.read_csv(os.path.join(RESULTS, "models_log.csv"))
    ev = pd.read_csv(os.path.join(RESULTS, "s3_events.csv"), parse_dates=["alarm_day"])
    evd = ev[ev["alarm_day"] <= DEV[1]]
    s0, b1 = scores(t["y"], t["S0"]), scores(t["y"], t["b1"])
    rows = []
    for s in STRATS + ["b1", "b7"]:
        sc = scores(t["y"], t[s])
        lg = log[(log["strategy"] == s) & (log["model_id"] != FROZEN_ID)] if s in STRATS else log.iloc[0:0]
        lg_dev = lg[pd.to_datetime(lg["effective_from"]) <= DEV[1]]
        rows.append({"strategy": s, "days": t["day"].nunique(), "rows": len(t), **sc,
                     "BalAcc_minus_b1": sc["BalAcc"] - b1["BalAcc"], "BalAcc_minus_S0": sc["BalAcc"] - s0["BalAcc"],
                     "Acc_minus_b1": sc["Acc"] - b1["Acc"], "Acc_minus_S0": sc["Acc"] - s0["Acc"],
                     "retrains_total": len(lg), "retrains_effective_in_dev": len(lg_dev),
                     "train_seconds_total": float(lg["total_seconds"].sum()) if len(lg) else 0.0,
                     "cusum_alarms_dev": len(evd) if s == "S3" else np.nan,
                     "cusum_suppressed_cooldown_dev": int((evd["suppressed_reason"] == "cooldown").sum()) if s == "S3" else np.nan,
                     "cusum_suppressed_budget_dev": int((evd["suppressed_reason"] == "budget").sum()) if s == "S3" else np.nan,
                     "s3_actual_retrains_dev": int(evd["retrained"].sum()) if s == "S3" else np.nan})
    cmp_ = pd.DataFrame(rows)
    yr, mo = [], []
    for key, out in (("year", yr), ("month", mo)):
        k = pd.DatetimeIndex(t["day"]).year if key == "year" else pd.DatetimeIndex(t["day"]).to_period("M").astype(str)
        for kv, g in t.groupby(k):
            row = {key: kv, "days": g["day"].nunique(), "class1_share": float(g["y"].mean() * 100)}
            for s in STRATS + ["b1", "b7"]:
                sc = scores(g["y"], g[s])
                row[f"{s}_Acc"], row[f"{s}_BalAcc"] = sc["Acc"], sc["BalAcc"]
            out.append(row)
    yearly, monthly = pd.DataFrame(yr), pd.DataFrame(mo)
    days = sorted(t["day"].unique())
    assert len(days) == 3653
    B = c.boot_indices(len(days))
    samp = {s: c.boot_balacc(t["y"].to_numpy(), t[s].to_numpy(), idx, B) for s in STRATS + ["b1"]}
    F = max(STRATS, key=lambda s: scores(t["y"], t[s])["BalAcc"])
    pairs = [(s, "S0") for s in ("S1", "S2", "S3")] + [("S3", "S1")] + [(s, "b1") for s in STRATS]
    pairs += [(F, s) for s in STRATS if s != F and (F, s) not in pairs]
    brow = []
    for a, b in pairs:
        d = samp[a] - samp[b]
        lo, hi = c.ci(d)
        ba, bb = scores(t["y"], t[a])["BalAcc"], scores(t["y"], t[b])["BalAcc"]
        brow.append({"a": a, "b": b, "BalAcc_a": ba, "BalAcc_b": bb, "diff_pp": ba - bb, "diff_lo": lo, "diff_hi": hi,
                     "ci_excludes_0": bool(lo > 0 or hi < 0), "n_days": len(days), "n_boot": c.N_BOOT, "seed": c.BOOT_SEED})
    boot = pd.DataFrame(brow)
    trade = cmp_[cmp_["strategy"].isin(STRATS)][["strategy", "retrains_total", "retrains_effective_in_dev", "BalAcc", "BalAcc_minus_S0"]].copy()
    bs = boot[boot["b"] == "S0"].drop_duplicates("a").set_index("a")
    trade["gain_vs_S0_lo"] = [bs.loc[s, "diff_lo"] if s in bs.index else 0.0 for s in trade["strategy"]]
    trade["gain_vs_S0_hi"] = [bs.loc[s, "diff_hi"] if s in bs.index else 0.0 for s in trade["strategy"]]
    trade["train_seconds_total"] = cmp_.set_index("strategy").loc[trade["strategy"], "train_seconds_total"].to_numpy()
    meta = {"dev_period": [str(DEV[0].date()), str(DEV[1].date())], "boot_index_sha256": c.sha(np.ascontiguousarray(B).tobytes()),
            "latest_timestamp_read": str(df.index.max()), "labels_equal_da": True,
            "note": "2025-2026 are not scored; S3 alarms there are simulation state only"}
    return cmp_, yearly, monthly, boot, trade, meta


def fig_cusum(dly):
    plt = c.mpl()
    cfg = json.load(open(CUSUM_CONFIG))["schemes"]["main"]
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
    ax.set_title("LR S3: CUSUM path (2014 reference, option A), alarms and retrains", color=c.INK, fontsize=11, loc="left")
    ax.set_ylim(0, max(cfg["h"], dly["S"].max()) * 1.25)
    ax.legend(frameon=False, fontsize=8, labelcolor=c.INK2, ncol=4, loc="upper left")
    fig.tight_layout()
    return fig, ax


def main():
    cmp_, yearly, monthly, boot, trade, meta = run()
    for k, v in (("cmp", cmp_), ("yearly", yearly), ("monthly", monthly), ("boot", boot), ("trade", trade)):
        v.to_csv(P[k], index=False, float_format="%.17g")
    json.dump(meta, open(P["meta"], "w"), indent=1)
    st3 = provisional(cmp_, boot)
    json.dump(st3, open(P["stage3"], "w"), indent=1, default=float)
    dly = pd.read_csv(os.path.join(RESULTS, "s3_daily_cusum.csv"), parse_dates=["day"])
    dly = dly[dly["day"] <= DEV[1]]
    fig, ax = m6e.fig_yearly(pd.read_csv(P["yearly"]))
    ax.set_title("LR yearly BalAcc 2015–2024: S0 frozen, S1 expanding, S2 3-year, S3 CUSUM; b1 dashed",
                 color=c.INK, fontsize=10, loc="left")
    fig.savefig(P["fig_year"], dpi=150, facecolor=c.SURFACE)
    fig, _ = fig_cusum(dly)
    fig.savefig(P["fig_cusum"], dpi=150, facecolor=c.SURFACE)
    fig, ax = m6e.fig_trade(pd.read_csv(P["trade"]))
    ax.set_title("LR: retrains vs gain over the frozen model (2015–2024)", color=c.INK, fontsize=11, loc="left")
    fig.savefig(P["fig_trade"], dpi=150, facecolor=c.SURFACE)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(cmp_[["strategy", "Acc", "BalAcc", "BalAcc_minus_b1", "BalAcc_minus_S0", "retrains_total", "retrains_effective_in_dev"]].round(3).to_string(index=False))
    print(yearly[["year", "class1_share"] + [f"{s}_BalAcc" for s in STRATS + ["b1"]]].round(2).to_string(index=False))
    print(boot.round(3).to_string(index=False))
    print(json.dumps(st3, indent=1, default=float))


if __name__ == "__main__":
    main()
