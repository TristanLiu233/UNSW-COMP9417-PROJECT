from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUTS = Path(__file__).resolve().parent / "outputs"
COMPARISON = OUTPUTS / "comparison"

MODELS = {
    "base": "Base Ridge",
    "by_slot": "By-slot Ridge",
    "hdd_cdd": "Ridge + HDD/CDD",
    "by_slot_hdd_cdd": "By-slot Ridge + HDD/CDD",
}


def main():
    # 原始时间戳是区间结束时间，最后一档属于 2026-10-31。
    expected_timestamps = pd.date_range(
        "2015-01-01 00:30", "2026-11-01 00:00", freq="30min"
    )
    market_days = expected_timestamps - pd.Timedelta(minutes=30)
    expected_years = pd.Series(market_days.year).value_counts().sort_index()
    expected_months = (
        pd.Series(market_days.to_period("M").astype(str))
        .value_counts()
        .sort_index()
    )

    overall_rows = []
    yearly_tables = []
    monthly_curves = {}

    for variant, label in MODELS.items():
        folder = OUTPUTS / variant
        predictions = pd.read_csv(
            folder / f"ridge_predictions_{variant}.csv",
            usecols=["timestamp", "predicted_demand_mw"],
            parse_dates=["timestamp"],
        )
        timestamps = pd.DatetimeIndex(predictions["timestamp"])
        if not timestamps.equals(expected_timestamps):
            raise ValueError(f"{variant} 的预测时间戳与完整回放期不一致。")
        if not np.isfinite(predictions["predicted_demand_mw"]).all():
            raise ValueError(f"{variant} 的预测中存在缺失值或无穷值。")

        yearly = pd.read_csv(
            folder / f"ridge_yearly_scores_{variant}.csv"
        ).set_index("year").sort_index()
        monthly = pd.read_csv(
            folder / f"ridge_monthly_scores_{variant}.csv"
        ).set_index("month").sort_index()

        for scores, expected_counts in (
            (yearly, expected_years),
            (monthly, expected_months),
        ):
            if (
                not scores.index.equals(expected_counts.index)
                or not np.array_equal(scores["n_slots"], expected_counts)
            ):
                raise ValueError(f"{variant} 的评分表覆盖范围不一致。")
            if not np.isfinite(scores[["mape_pct", "mae_mw"]].to_numpy()).all():
                raise ValueError(f"{variant} 的评分表存在缺失值或无穷值。")

        weights = yearly["n_slots"]
        overall_rows.append({
            "model": variant,
            "n_slots": int(weights.sum()),
            "mape_pct": np.average(yearly["mape_pct"], weights=weights),
            "mae_mw": np.average(yearly["mae_mw"], weights=weights),
        })
        yearly_tables.append(
            yearly[["mape_pct", "mae_mw"]].add_prefix(f"{variant}_")
        )
        monthly_curves[label] = monthly["mape_pct"]

    overall = pd.DataFrame(overall_rows).set_index("model")
    overall["mape_delta_vs_base_pp"] = (
        overall["mape_pct"] - overall.loc["base", "mape_pct"]
    )
    overall["mae_delta_vs_base_mw"] = (
        overall["mae_mw"] - overall.loc["base", "mae_mw"]
    )
    overall = overall.sort_values("mape_pct")

    yearly_comparison = pd.concat(yearly_tables, axis=1)
    yearly_comparison.insert(0, "last_day", [
        f"{year}-10-31" if year == 2026 else f"{year}-12-31"
        for year in yearly_comparison.index
    ])

    COMPARISON.mkdir(parents=True, exist_ok=True)
    overall.to_csv(COMPARISON / "ridge_comparison_overall.csv")
    yearly_comparison.to_csv(COMPARISON / "ridge_comparison_yearly.csv")

    monthly = pd.DataFrame(monthly_curves)
    dates = pd.to_datetime(monthly.index, format="%Y-%m")
    fig, ax = plt.subplots(figsize=(13, 5))
    for label, scores in monthly.items():
        ax.plot(dates, scores, linewidth=1.3, label=label)

    ax.set_title("Monthly NSW demand MAPE across four frozen Ridge models")
    ax.set_xlabel("Market month (Jan 2015 – Oct 2026)")
    ax.set_ylabel("MAPE (%)")
    ax.set_xlim(dates[0], dates[-1])
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(COMPARISON / "ridge_comparison_monthly_mape.png", dpi=180)
    plt.close(fig)

    print("回放期：2015-01-01 至 2026-10-31")
    print(overall.round({
        "mape_pct": 3,
        "mae_mw": 2,
        "mape_delta_vs_base_pp": 3,
        "mae_delta_vs_base_mw": 2,
    }).to_string())
    print(f"\n对比结果目录：{COMPARISON}")


if __name__ == "__main__":
    main()