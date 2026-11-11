"""Generate labeled plots from saved evaluation outputs, without retraining."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .stage2 import STRATEGIES

COLORS = {"frozen": "#124e79", "periodic_expanding": "#a34310",
          "periodic_recent4y": "#17796b", "cusum_expanding": "#743a8e",
          "b1": "#5b6874", "b7": "#a2a9af"}
LABELS = {"frozen": "HGBR frozen", "periodic_expanding": "Periodic expanding",
          "periodic_recent4y": "Periodic trailing 4 years", "cusum_expanding": "CUSUM expanding",
          "b1": "Yesterday same slot", "b7": "Last week same slot"}


def save(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=170, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style(ax, date_years=False):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.24)
    if date_years:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def render(stage1, stage2, out):
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titleweight": "semibold", "svg.fonttype": "none"})
    monthly = pd.read_csv(stage2 / "monthly.csv")
    monthly["month"] = pd.to_datetime(monthly["month"], format="%Y-%m")
    frozen = monthly.loc[monthly["model"] == "frozen"]
    fig, axs = plt.subplots(2, 1, figsize=(10, 5.8), sharex=True, sharey=True, layout="constrained")
    for ax, stem, title in zip(axs, ["normal", "day_bootstrap"],
                              ["A. Independent half-hours", "B. Resampling whole market days"]):
        ax.plot(frozen["month"], frozen["mape_pct"], color=COLORS["frozen"], lw=1.5, label="Frozen HGBR")
        ax.fill_between(frozen["month"], frozen[f"{stem}_ci_low"], frozen[f"{stem}_ci_high"],
                        color=COLORS["frozen"], alpha=.25, label="95% confidence interval")
        ax.set(title=title, ylabel="Monthly MAPE (%)")
        ax.legend(loc="upper left", frameon=False)
        style(ax, date_years=True)
    axs[-1].set_xlabel("Market month (fixed AEST)")
    fig.suptitle("Frozen HGBR: monthly demand error and uncertainty")
    save(fig, out / "frozen_mape_intervals")

    fig, ax = plt.subplots(figsize=(10.3, 4.5), layout="constrained")
    for name in (*STRATEGIES, "b1", "b7"):
        g = monthly.loc[monthly["model"] == name]
        ax.plot(g["month"], g["mape_pct"], color=COLORS[name], lw=1.2, label=LABELS[name],
                ls="--" if name in {"b1","b7"} else "-")
    ax.axvline(pd.Timestamp("2025-01-01"), color="#353e47", lw=1, ls=":")
    ax.set(title="Demand adaptation and official baselines",
           ylabel="Monthly MAPE (%)", xlabel="Market month (fixed AEST)")
    ax.legend(ncol=3, loc="upper left", frameon=False)
    style(ax, date_years=True)
    save(fig, out / "adaptation_monthly")

    d = pd.read_csv(stage2 / "daily_monitors.csv", parse_dates=["market_day"])
    events = pd.read_csv(stage2 / "drift_alarms.csv", parse_dates=["observed_day"])
    fig, axs = plt.subplots(3, 1, figsize=(10.2, 7.1), sharex=True, layout="constrained")
    for ax, col, title, threshold, detector, color in [
        (axs[0], "input_score_28d", "A. Input-only detector", True, "input_only", "#17796b"),
        (axs[1], "frozen_cusum", "B. Frozen HGBR error CUSUM", True, "frozen_error_cusum", "#743a8e"),
        (axs[2], "frozen_mape_pct", "C. Frozen HGBR daily MAPE (28-day mean)", False, "", "#124e79"),
    ]:
        score = d[col] if threshold else d[col].rolling(28).mean()
        ax.plot(d["market_day"], score, color=color, lw=1)
        ax.set_title(title, loc="left")
        if threshold:
            a = events.loc[events["detector"] == detector]
            if not a.empty:
                ax.axhline(a["threshold"].iloc[0], color="#626b76", lw=1, ls="--", label="Threshold")
                ax.scatter(a["observed_day"], a["score"], s=8, color="#a34310", label="Alert", zorder=3)
                ax.legend(loc="upper left", frameon=False, ncol=2)
        style(ax, date_years=True)
    axs[0].set_ylabel("Standardized score")
    axs[1].set_ylabel("CUSUM (pp)")
    axs[2].set(ylabel="MAPE (%)", xlabel="Market day (fixed AEST)")
    save(fig, out / "drift_monitors")

    g = pd.read_csv(stage2 / "half_hour_predictions.csv.gz", parse_dates=["timestamp","market_day"])
    fig, axs = plt.subplots(2, 1, figsize=(10.1, 5.8), layout="constrained")
    for ax, start, stop in zip(axs, ["2015-01-01", "2026-10-25"], ["2015-01-08", "2026-11-01"]):
        w = g.loc[g["market_day"].ge(start) & g["market_day"].lt(stop)]
        for name, label, col, ls in [
            ("actual_demand_mw", "Observed", "#1c232a", "-"), ("frozen", "Frozen HGBR", COLORS["frozen"], "-"),
            ("b1", LABELS["b1"], COLORS["b1"], "--"), ("b7", LABELS["b7"], COLORS["b7"], ":"),
        ]:
            ax.plot(w["timestamp"], w[name], color=col, lw=1.1, ls=ls, label=label)
        ax.set(title=f"{start} through {w['market_day'].iloc[-1].date()}",
               ylabel="NSW demand (MW)", xlabel="Half-hour interval end (fixed AEST)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.15 * (hi-lo))
        ax.legend(loc="upper right", frameon=False, ncol=2)
        style(ax)
    save(fig, out / "observed_vs_predicted")

    v = pd.read_csv(stage1 / "validation_weekday_holiday.csv")
    chosen = pd.read_csv(stage1 / "p3_shared_row.csv").iloc[0]["target"]
    fig, axs = plt.subplots(1, 2, figsize=(10.1, 3.5), layout="constrained")
    for ax, kind, order, names in [
        (axs[0], "weekday", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
         ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]),
        (axs[1], "holiday_status", ["Non-holiday weekday","Non-holiday weekend","Public holiday"],
         ["Weekday","Weekend","Public holiday"]),
    ]:
        a = v.loc[v["grouping"] == kind].pivot(index="group", columns="model", values="mape_pct").loc[order]
        x = np.arange(len(order))
        ax.bar(x-.18, a[chosen], width=.36, color=COLORS["frozen"], label="Selected HGBR")
        ax.bar(x+.18, a["b1"], width=.36, color="#a4adb7", label="Yesterday same slot")
        ax.set_xticks(x, names)
        ax.set(title="A. Weekday" if kind=="weekday" else "B. NSW public holidays", ylabel="MAPE (%)")
        ax.legend(frameon=False, fontsize=9)
        style(ax)
    fig.suptitle("2014 validation: demand error by calendar group")
    save(fig, out / "validation_calendar")
    print(f"Rendered 5 figure pairs in {out}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage1", required=True, type=Path)
    p.add_argument("--stage2", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    render(args.stage1, args.stage2, args.out)


if __name__ == "__main__":
    main()
