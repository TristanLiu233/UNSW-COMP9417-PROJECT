from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / "data" / "nem_weather_halfhourly_2010_2026.csv"
OUTPUT_DIR = ROOT / "outputs" / "day_type_changes"

MODEL_TAG = "loyo_k2_pca95"
LABELS_PATH = (
    ROOT / "outputs" / "day_type_clustering"
    / f"day_type_assignments_{MODEL_TAG}.csv"
)

TRAIN_START, TRAIN_END = "2010-01-01", "2013-12-31"
RECENT_START, RECENT_END = "2024-01-01", "2025-12-31"
SLOTS = pd.RangeIndex(48, name="slot")


def load_labels():
    labels = pd.read_csv(
        LABELS_PATH,
        usecols=["market_day", "day_type", "distance_to_centroid"],
        parse_dates=["market_day"],
    ).set_index("market_day").sort_index()

    if labels.empty or labels.index.hasnans or not labels.index.is_unique:
        raise ValueError("日型表应包含有效日期，且每个 market_day 只能出现一次。")

    if set(labels["day_type"]) != {1, 2}:
        raise ValueError("PCA 95% 的日型表应包含 Type 1 和 Type 2。")

    distances = pd.to_numeric(labels["distance_to_centroid"], errors="raise")
    if not np.isfinite(distances).all() or (distances < 0).any():
        raise ValueError("distance_to_centroid 必须是有限非负数。")

    labels["distance_to_centroid"] = distances
    return labels


def load_profiles(market_days):
    data = pd.read_csv(DATA_PATH, usecols=["timestamp", "nsw_demand"])
    timestamps = pd.to_datetime(data["timestamp"], errors="raise")
    if timestamps.isna().any():
        raise ValueError("需求数据存在缺失时间戳。")
    if timestamps.dt.tz is not None:
        timestamps = timestamps.dt.tz_convert("Etc/GMT-10").dt.tz_localize(None)

    # 原始时间戳是区间结束时间，午夜 00:00 的记录属于前一天最后一档。
    starts = timestamps - pd.Timedelta(minutes=30)
    days = starts.dt.normalize()
    used = days.isin(market_days)

    if (starts[used] != starts[used].dt.floor("30min")).any():
        raise ValueError("需求时间戳没有对齐到半小时。")

    demand = pd.DataFrame({
        "market_day": days[used],
        "slot": starts[used].dt.hour * 2 + starts[used].dt.minute // 30,
        "demand": pd.to_numeric(data.loc[used, "nsw_demand"], errors="raise"),
    })
    daily = (
        demand.pivot(index="market_day", columns="slot", values="demand")
        .reindex(index=market_days, columns=SLOTS)
    )
    if not np.isfinite(daily.to_numpy()).all():
        raise ValueError("至少有一个已标注日期缺少有效的 48 个半小时需求值。")

    daily_means = daily.mean(axis=1)
    if (daily_means <= 0).any():
        raise ValueError("归一化要求每日平均需求大于零。")

    return daily.div(daily_means, axis=0)


def summarize_distances(labels, day_type, coverage):
    distances = labels.loc[
        labels["day_type"].eq(day_type), "distance_to_centroid"
    ]
    training = distances.loc[TRAIN_START:TRAIN_END]
    if training.empty:
        raise ValueError(f"训练期没有 Type {day_type}，无法计算参考距离。")

    training_p90 = training.quantile(0.90)
    years = distances.index.year
    grouped = distances.groupby(years)

    annual = coverage.copy()
    annual["day_type"] = day_type
    annual["type_days"] = grouped.size().reindex(annual.index, fill_value=0)
    annual["type_share_pct"] = 100 * annual["type_days"] / annual["observed_days"]
    annual["median_distance"] = grouped.median()
    annual["p90_distance"] = grouped.quantile(0.90)
    annual["pct_above_training_p90"] = (
        100 * distances.gt(training_p90).groupby(years).mean()
    )
    annual["training_p90"] = training_p90
    return annual


def plot_distances(annual, day_type, path):
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(annual.index, annual["median_distance"], "o-", label="Yearly median")
    ax.plot(annual.index, annual["p90_distance"], "o-", label="Yearly P90")
    ax.axhline(
        annual["training_p90"].iloc[0],
        color="0.45",
        linestyle="--",
        label=f"2010–2013 Type {day_type} P90",
    )

    ticks = [
        str(year) if last == pd.Timestamp(year, 12, 31)
        else f"{year}\n(to {last:%d %b})"
        for year, last in annual["last_observed_day"].items()
    ]
    ax.set_xticks(annual.index, ticks, rotation=45, ha="right")
    ax.set(
        title=f"Type {day_type} distance to the fixed training centroid",
        xlabel="Market year",
        ylabel="Euclidean distance in PCA space",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def compare_profiles(profiles, labels, day_type, stem):
    type_days = labels.index[labels["day_type"].eq(day_type)]
    type_profiles = profiles.loc[type_days]
    periods = (
        ("2010–2013", type_profiles.loc[TRAIN_START:TRAIN_END], "tab:blue"),
        ("2024–2025", type_profiles.loc[RECENT_START:RECENT_END], "tab:orange"),
    )

    hours = SLOTS.to_numpy() / 2
    tables = []
    fig, ax = plt.subplots(figsize=(11, 4.8))

    for period, daily, color in periods:
        mean = daily.mean().to_numpy()
        p10 = daily.quantile(0.10).to_numpy()
        p90 = daily.quantile(0.90).to_numpy()
        tables.append(pd.DataFrame({
            "day_type": day_type,
            "period": period,
            "n_days": len(daily),
            "slot": SLOTS,
            "hour_aest": hours,
            "mean": mean,
            "p10": p10,
            "p90": p90,
        }))

        label = f"{period} ({len(daily)} days)"
        if daily.empty:
            ax.plot([], [], color=color, label=label)
            continue

        ax.plot(hours, mean, color=color, linewidth=2, label=label)
        ax.fill_between(hours, p10, p90, color=color, alpha=0.18)

    ax.axhline(1, color="0.5", linestyle="--", linewidth=1)
    ax.set_xticks(range(0, 24, 3))
    ax.set(
        title=f"Type {day_type} demand curves across full calendar years",
        xlabel="Market time (interval start, AEST)",
        ylabel="Demand / daily mean",
        xlim=(0, 23.5),
    )
    ax.legend(title="Mean curves with P10–P90 bands")
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    plt.close(fig)
    pd.concat(tables, ignore_index=True).to_csv(stem.with_suffix(".csv"), index=False)


def main():
    labels = load_labels()
    profiles = load_profiles(labels.index)

    calendar = labels.index.to_series(name="market_day")
    coverage = calendar.groupby(labels.index.year).agg(
        observed_days="size",
        first_observed_day="min",
        last_observed_day="max",
    )
    coverage.index.name = "year"
    date_tag = f"{labels.index.min():%Y%m%d}_to_{labels.index.max():%Y%m%d}"

    distance_dir = OUTPUT_DIR / "centroid_distance"
    profile_dir = OUTPUT_DIR / "profile_comparison"
    distance_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)

    annual_tables = []
    for day_type in (1, 2):
        type_tag = f"type{day_type}_{MODEL_TAG}"
        annual = summarize_distances(labels, day_type, coverage)
        distance_stem = distance_dir / f"centroid_distance_{type_tag}_{date_tag}"
        annual.to_csv(distance_stem.with_suffix(".csv"))
        plot_distances(annual, day_type, distance_stem.with_suffix(".png"))
        annual_tables.append(annual.reset_index())

        profile_stem = profile_dir / (
            f"profile_comparison_{type_tag}_2010_2013_vs_2024_2025_full_years"
        )
        compare_profiles(profiles, labels, day_type, profile_stem)

    summary = pd.concat(annual_tables, ignore_index=True)
    summary.sort_values(["year", "day_type"]).to_csv(
        distance_dir / f"centroid_distance_all_types_{MODEL_TAG}_{date_tag}.csv",
        index=False,
    )

    print(f"Distance outputs: {distance_dir}")
    print(f"Profile outputs:  {profile_dir}")


if __name__ == "__main__":
    main()