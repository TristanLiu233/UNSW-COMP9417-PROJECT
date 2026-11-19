"""Descriptive demand and price analysis for report Section 2.1."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_DATA = ROOT / "data" / "nem_weather_halfhourly_2010_2026.csv"
DEFAULT_OUTPUT = HERE / "outputs" / "data_patterns"

VARIABLES = {
    "nsw_demand": ("NSW demand", "MW"),
    "nsw_price": ("NSW price", "AUD/MWh"),
}
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def load_data(path):
    data = pd.read_csv(
        path,
        usecols=["timestamp", *VARIABLES],
        parse_dates=["timestamp"],
    )

    if data.empty or data["timestamp"].isna().any():
        raise ValueError("Empty data or invalid timestamps.")

    timestamps = pd.DatetimeIndex(data["timestamp"])

    if timestamps.tz is not None:
        raise ValueError(
            "Expected course timestamps in fixed AEST without a timezone."
        )
    if timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
        raise ValueError("Timestamps must be unique and ordered.")

    expected = pd.date_range(
        timestamps.min(), timestamps.max(), freq="30min"
    )
    if not timestamps.equals(expected):
        raise ValueError("Missing or irregular half-hour timestamps.")

    for column in VARIABLES:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if np.isinf(data[column].to_numpy(dtype=float)).any():
            raise ValueError(f"Infinite values in {column}.")

    # Course timestamps mark interval endings.
    starts = data["timestamp"] - pd.Timedelta(minutes=30)
    data["market_day"] = starts.dt.normalize()
    data["year"] = starts.dt.year
    data["month"] = starts.dt.month
    data["weekday"] = starts.dt.day_of_week
    data["slot"] = starts.dt.hour * 2 + starts.dt.minute // 30
    data["day_group"] = np.where(
        data["weekday"] < 5, "Weekday", "Weekend"
    )

    slots_per_day = data.groupby("market_day").size()
    if not slots_per_day.eq(48).all():
        raise ValueError("Every market day must contain exactly 48 slots.")

    demand = data["nsw_demand"].dropna()
    if demand.le(0).any():
        raise ValueError("Observed demand must be positive.")

    return data


def summary_table(data):
    rows = []
    for column, (name, unit) in VARIABLES.items():
        values = data[column].dropna()
        rows.append({
            "variable": name,
            "unit": unit,
            "observed": len(values),
            "missing": int(data[column].isna().sum()),
            "mean": values.mean(),
            "std": values.std(ddof=1),
            "median": values.median(),
            "min": values.min(),
            "p01": values.quantile(0.01),
            "p95": values.quantile(0.95),
            "p99": values.quantile(0.99),
            "max": values.max(),
            "negative_pct": (
                100 * values.lt(0).mean() if len(values) else np.nan
            ),
        })
    return pd.DataFrame(rows)


def grouped_statistics(data, keys):
    """Raw observation summaries; missing values are excluded."""
    return data.groupby(keys, observed=True).agg(
        slots=("timestamp", "size"),
        demand_observed=("nsw_demand", "count"),
        demand_mean_mw=("nsw_demand", "mean"),
        demand_median_mw=("nsw_demand", "median"),
        price_observed=("nsw_price", "count"),
        price_mean_aud_mwh=("nsw_price", "mean"),
        price_median_aud_mwh=("nsw_price", "median"),
        price_p99_aud_mwh=(
            "nsw_price", lambda x: x.quantile(0.99)
        ),
        negative_price_pct=(
            "nsw_price",
            lambda x: (
                100 * x.dropna().lt(0).mean()
                if x.notna().any() else np.nan
            ),
        ),
    ).reset_index()


def make_demand_figure(intraday, weekly, monthly, output):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))

    for group in ["Weekday", "Weekend"]:
        part = intraday[intraday["day_group"] == group]
        axes[0].plot(
            part["slot"] / 2,
            part["demand_mean_mw"],
            label=group,
        )
    axes[0].set(
        xlabel="Interval start (AEST)",
        ylabel="Mean NSW demand (MW)",
        title="Intraday",
        xlim=(0, 23.5),
    )
    axes[0].set_xticks([0, 6, 12, 18])
    axes[0].legend(frameon=False)

    axes[1].plot(
        weekly["weekday"],
        weekly["demand_mean_mw"],
        marker="o",
    )
    axes[1].set(
        xlabel="Market-day weekday",
        ylabel="Mean NSW demand (MW)",
        title="Weekly",
    )
    axes[1].set_xticks(range(7), WEEKDAYS)

    axes[2].plot(
        monthly["month"],
        monthly["demand_mean_mw"],
        marker="o",
    )
    axes[2].set(
        xlabel="Month",
        ylabel="Mean NSW demand (MW)",
        title="Monthly: full years 2010–2025",
    )
    axes[2].set_xticks(range(1, 13))

    for ax in axes:
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output / "demand_temporal_patterns.png", dpi=220)
    plt.close(fig)


def make_price_figure(data, yearly, output):
    prices = data["nsw_price"].dropna()
    if prices.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # Show the centre clearly without silently discarding tails.
    lower, upper = prices.quantile([0.01, 0.99])
    if lower == upper:
        lower, upper = lower - 1, upper + 1

    central = prices[prices.between(lower, upper)]
    axes[0].hist(central, bins=70, color="tab:blue", alpha=0.8)
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set(
        xlabel="NSW price (AUD/MWh)",
        ylabel="Half-hour observations",
        title="Price distribution: P1–P99 range",
    )
    excluded = len(prices) - len(central)
    axes[0].text(
        0.97, 0.95,
        f"{excluded:,} observations outside displayed range",
        transform=axes[0].transAxes,
        ha="right", va="top", fontsize=8,
    )

    axes[1].plot(
        yearly["year"],
        yearly["negative_price_pct"],
        marker="o",
        label="Observed period",
    )
    # Flag the incomplete final year explicitly.
    last = yearly[yearly["year"] == 2026]
    if not last.empty:
        axes[1].scatter(
            last["year"], last["negative_price_pct"],
            s=100, facecolors="none", edgecolors="black",
            label="2026: Jan–Oct",
        )
    axes[1].set(
        xlabel="Market year",
        ylabel="Negative-price observations (%)",
        title="Negative-price frequency",
    )
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output / "price_distribution_patterns.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data = load_data(args.data)
    args.output.mkdir(parents=True, exist_ok=True)

    summary = summary_table(data)
    intraday = grouped_statistics(data, ["day_group", "slot"])
    weekly = grouped_statistics(data, ["weekday"])
    yearly = grouped_statistics(data, ["year"])

    # Exclude partial 2026 from pooled calendar-month comparisons.
    full_year_data = data[data["year"].between(2010, 2025)]
    monthly = grouped_statistics(full_year_data, ["month"])

    tables = {
        "summary_statistics.csv": summary,
        "intraday_weekday_weekend.csv": intraday,
        "weekday_statistics.csv": weekly,
        "month_statistics_full_years.csv": monthly,
        "yearly_statistics.csv": yearly,
    }

    for filename, table in tables.items():
        table.to_csv(args.output / filename, index=False)

    coverage = pd.DataFrame([{
        "first_market_day": data["market_day"].min().date(),
        "last_market_day": data["market_day"].max().date(),
        "market_days": data["market_day"].nunique(),
        "half_hours": len(data),
        "missing_demand": data["nsw_demand"].isna().sum(),
        "missing_price": data["nsw_price"].isna().sum(),
    }])
    coverage.to_csv(args.output / "data_coverage.csv", index=False)

    make_demand_figure(intraday, weekly, monthly, args.output)
    make_price_figure(data, yearly, args.output)

    print("\nData coverage:")
    print(coverage.to_string(index=False))
    print("\nSummary statistics:")
    print(summary.round(3).to_string(index=False))
    print(f"\nOutputs: {args.output.resolve()}")
    print(
        "\nNotes:\n"
        "- Statistics exclude missing values; counts are reported.\n"
        "- Intraday and weekly summaries pool the observed period.\n"
        "- Calendar-month summaries use full years 2010–2025.\n"
        "- Weekdays include public holidays falling Monday–Friday.\n"
        "- Price histogram displays P1–P99 only; tables retain all values.\n"
        "- Negative-price frequency is not the classification positive rate.\n"
        "- These are descriptive summaries, not causal comparisons."
    )


if __name__ == "__main__":
    main()