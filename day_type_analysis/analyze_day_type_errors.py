"""Compare fixed day types with frozen HGBR errors and drift signals."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

DEFAULT_TYPES = (
    HERE / "outputs" / "day_type_clustering"
    / "day_type_assignments_forward_k2_pca95.csv"
)
DEFAULT_P3 = HERE / "inputs" / "p3"
DEFAULT_OUTPUT = HERE / "outputs" / "day_type_errors"

START = pd.Timestamp("2015-01-01")
END = pd.Timestamp("2026-10-31")
TYPE_COLORS = {1: "#0072B2", 2: "#D55E00"}


def read_daily(path):
    frame = pd.read_csv(path, parse_dates=["market_day"])
    if frame["market_day"].isna().any():
        raise ValueError(f"{path}: 日期包含缺失值。")
    if frame["market_day"].duplicated().any():
        raise ValueError(f"{path}: 日期重复。")
    return frame.sort_values("market_day").reset_index(drop=True)


def load_and_merge(types_path, p3_dir):
    types = read_daily(types_path)
    monitors = read_daily(p3_dir / "daily_monitors.csv")

    expected_days = pd.date_range(START, END)
    if not pd.DatetimeIndex(monitors["market_day"]).equals(expected_days):
        raise ValueError("监测表必须完整覆盖2015-01-01至2026-10-31。")

    predictions = pd.read_csv(
        p3_dir / "half_hour_predictions.csv.gz",
        usecols=[
            "timestamp", "market_day",
            "actual_demand_mw", "frozen",
        ],
        parse_dates=["timestamp", "market_day"],
    )
    expected_times = pd.date_range(
        START + pd.Timedelta(minutes=30),
        END + pd.Timedelta(days=1),
        freq="30min",
    )
    if not pd.DatetimeIndex(predictions["timestamp"]).equals(expected_times):
        raise ValueError("半小时预测的时间戳不完整、有重复或顺序错误。")

    actual_days = (
        predictions["timestamp"] - pd.Timedelta(minutes=30)
    ).dt.normalize()
    if not predictions["market_day"].equals(actual_days):
        raise ValueError("预测表的市场日与时间戳不一致。")

    values = predictions[["actual_demand_mw", "frozen"]].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("需求或预测值包含缺失、无穷值。")
    if (predictions["actual_demand_mw"] <= 0).any():
        raise ValueError("真实需求必须大于零。")

    predictions["absolute_error"] = (
        predictions["frozen"] - predictions["actual_demand_mw"]
    ).abs()
    predictions["ape_pct"] = (
        100 * predictions["absolute_error"]
        / predictions["actual_demand_mw"]
    )

    errors = predictions.groupby("market_day", as_index=False).agg(
        n_slots=("absolute_error", "size"),
        frozen_mae_mw=("absolute_error", "mean"),
        checked_mape_pct=("ape_pct", "mean"),
    )

    daily = monitors.merge(
        errors,
        on="market_day",
        how="left",
        validate="one_to_one",
    ).merge(
        types[["market_day", "day_type", "distance_to_centroid"]],
        on="market_day",
        how="left",
        validate="one_to_one",
    )

    required = [
        "day_type", "distance_to_centroid",
        "frozen_mape_pct", "frozen_mae_mw",
        "checked_mape_pct", "input_score_28d", "frozen_cusum",
    ]
    if not np.isfinite(daily[required].to_numpy()).all():
        raise ValueError("合并后存在缺失或无效数值。")
    if not daily["n_slots"].eq(48).all():
        raise ValueError("每个市场日应有48档。")
    if not daily["day_type"].isin([1, 2]).all():
        raise ValueError("本脚本使用原来的k=2日型标签。")
    if (daily["distance_to_centroid"] < 0).any():
        raise ValueError("中心距离不能为负。")

    np.testing.assert_allclose(
        daily["frozen_mape_pct"],
        daily["checked_mape_pct"],
        rtol=0,
        atol=1e-10,
    )

    daily = daily.drop(columns="checked_mape_pct")
    daily["day_type"] = daily["day_type"].astype(int)
    daily["year"] = daily["market_day"].dt.year
    daily["month"] = daily["market_day"].dt.to_period("M").astype(str)

    return daily


def load_detector_files(p3_dir, daily):
    input_config = json.loads(
        (p3_dir / "input_detector.json").read_text()
    )
    cusum_config = json.loads(
        (p3_dir / "cusum_config.json").read_text()
    )
    alarms = pd.read_csv(
        p3_dir / "drift_alarms.csv",
        parse_dates=["observed_day"],
    )

    if alarms["observed_day"].isna().any():
        raise ValueError("告警日期包含缺失值。")

    alarms = alarms.loc[
        alarms["observed_day"].between(START, END)
        & alarms["detector"].isin(
            ["input_only", "frozen_error_cusum"]
        )
    ].copy()

    if alarms.duplicated(["detector", "observed_day"]).any():
        raise ValueError("同类检测器在同一天有重复告警。")

    input_threshold = float(input_config["threshold"])
    cusum_threshold = float(cusum_config["threshold_h"])
    if not np.isfinite([input_threshold, cusum_threshold]).all():
        raise ValueError("检测阈值无效。")

    for detector, threshold, column in [
        ("input_only", input_threshold, "input_score_28d"),
        ("frozen_error_cusum", cusum_threshold, "frozen_cusum"),
    ]:
        selected = alarms.loc[alarms["detector"].eq(detector)]
        np.testing.assert_allclose(
            selected["threshold"], threshold, rtol=0, atol=1e-10
        )
        if not selected["observed_day"].isin(daily["market_day"]).all():
            raise ValueError("告警日期不能与每日表匹配。")

        lookup = daily.set_index("market_day")[column]
        np.testing.assert_allclose(
            selected["score"],
            lookup.reindex(selected["observed_day"]).to_numpy(),
            rtol=0,
            atol=1e-10,
        )

    alarms = alarms.sort_values(
        ["observed_day", "detector"]
    ).reset_index(drop=True)

    return alarms, input_threshold, cusum_threshold


def summarize(daily, alarms, input_threshold):
    # 保留零样本日型：天数和占比为0，误差及距离为NaN。
    years = sorted(daily["year"].unique())
    index = pd.MultiIndex.from_product(
        [years, [1, 2]], names=["year", "day_type"]
    )

    yearly = daily.groupby(["year", "day_type"]).agg(
        days=("market_day", "size"),
        mape_pct=("frozen_mape_pct", "mean"),
        mae_mw=("frozen_mae_mw", "mean"),
        median_distance=("distance_to_centroid", "median"),
        mean_distance=("distance_to_centroid", "mean"),
    ).reindex(index).reset_index()

    yearly["days"] = yearly["days"].fillna(0).astype(int)
    yearly["share_pct"] = (
        100 * yearly["days"]
        / yearly.groupby("year")["days"].transform("sum")
    )

    overall = daily.groupby("year").agg(
        observed_days=("market_day", "size"),
        first_day=("market_day", "min"),
        last_day=("market_day", "max"),
        mape_pct=("frozen_mape_pct", "mean"),
        mae_mw=("frozen_mae_mw", "mean"),
        median_distance=("distance_to_centroid", "median"),
        mean_input_score=("input_score_28d", "mean"),
    ).reset_index()

    # 各日型对当年整体MAPE的算术贡献，不代表因果贡献。
    yearly["mape_contribution_pp"] = np.where(
        yearly["days"].eq(0),
        0.0,
        yearly["share_pct"] * yearly["mape_pct"] / 100,
    )
    reconstructed = yearly.groupby("year")[
        "mape_contribution_pp"
    ].sum()
    np.testing.assert_allclose(
        reconstructed.reindex(overall["year"]).to_numpy(),
        overall["mape_pct"],
        rtol=0,
        atol=1e-10,
    )

    working = daily.assign(
        type1=daily["day_type"].eq(1).astype(float),
        input_above=daily["input_score_28d"].gt(
            input_threshold
        ).astype(float),
    )

    monthly = working.groupby("month").agg(
        days=("market_day", "size"),
        type1_share=("type1", "mean"),
        median_distance=("distance_to_centroid", "median"),
        mape_pct=("frozen_mape_pct", "mean"),
        mae_mw=("frozen_mae_mw", "mean"),
        input_score_mean=("input_score_28d", "mean"),
        input_above_share=("input_above", "mean"),
    ).reset_index()
    monthly["type1_share_pct"] = 100 * monthly.pop("type1_share")
    monthly["input_above_share_pct"] = (
        100 * monthly.pop("input_above_share")
    )

    alarm_months = alarms.assign(
        month=alarms["observed_day"].dt.to_period("M").astype(str)
    )
    for detector in ["input_only", "frozen_error_cusum"]:
        counts = alarm_months.loc[
            alarm_months["detector"].eq(detector)
        ].groupby("month").size()
        monthly[f"{detector}_alarms"] = (
            monthly["month"].map(counts).fillna(0).astype(int)
        )

    return yearly, overall, monthly


def finish_figure(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_yearly(yearly, overall, path):
    fig, axes = plt.subplots(
        3, 1, figsize=(10, 9), sharex=True,
        constrained_layout=True,
    )

    for day_type in [1, 2]:
        rows = yearly.loc[yearly["day_type"].eq(day_type)]
        label = f"Type {day_type}"
        color = TYPE_COLORS[day_type]

        for ax, column in zip(
            axes,
            ["share_pct", "median_distance", "mape_pct"],
        ):
            ax.plot(
                rows["year"], rows[column],
                marker="o", color=color, label=label,
            )

        # 标识小样本年度，避免过度解读。
        sparse = rows.loc[rows["days"].between(1, 29)]
        axes[2].scatter(
            sparse["year"], sparse["mape_pct"],
            s=95, facecolors="none", edgecolors=color,
            linewidths=1.3, zorder=4,
        )

    axes[2].plot(
        overall["year"], overall["mape_pct"],
        color="black", linestyle="--", label="All days",
    )

    axes[0].set_ylabel("Share of days (%)")
    axes[0].set_ylim(0, 105)
    axes[1].set_ylabel("Median distance\nin PCA space")
    axes[2].set_ylabel("Frozen HGBR MAPE (%)")
    axes[2].set_xlabel("Market year")
    axes[0].legend()
    axes[2].legend()

    for ax in axes:
        ax.grid(alpha=0.2)

    axes[2].set_xticks(overall["year"])
    axes[2].tick_params(axis="x", rotation=45)
    fig.suptitle(
        "Fixed day types and frozen HGBR performance\n"
        "2026 covers January-October; outlined points have fewer than 30 days"
    )
    finish_figure(fig, path)


def plot_monthly(daily, monthly, alarms, input_threshold, path):
    dates = pd.to_datetime(monthly["month"])

    fig, axes = plt.subplots(
        5, 1, figsize=(12, 12), sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 1.4, 0.7]},
    )

    axes[0].plot(
        dates, monthly["type1_share_pct"], color=TYPE_COLORS[1]
    )
    axes[0].set_ylabel("Type 1 share (%)")
    axes[0].set_ylim(0, 105)

    axes[1].plot(
        dates, monthly["median_distance"], color="#009E73"
    )
    axes[1].set_ylabel("Monthly median\ncentroid distance")

    axes[2].plot(
        dates, monthly["mape_pct"], color="#CC79A7"
    )
    axes[2].set_ylabel("Monthly frozen\nHGBR MAPE (%)")

    # 展示逐日的28日滚动分数，避免月均值掩盖阈值越界。
    axes[3].plot(
        daily["market_day"], daily["input_score_28d"],
        color="0.65", linewidth=0.6, label="Daily trailing-28-day score",
    )
    axes[3].plot(
        dates, monthly["input_score_mean"],
        color="#0072B2", linewidth=1.5, label="Monthly mean",
    )
    axes[3].axhline(
        input_threshold, color="#D55E00",
        linestyle="--", label=f"Threshold = {input_threshold:.3f}",
    )
    axes[3].set_ylabel("Input drift score")
    axes[3].legend(fontsize=8)

    # 告警单独放在时间轴上，避免219条竖线遮挡曲线。
    for detector, y, color in [
        ("input_only", 1, "#D55E00"),
        ("frozen_error_cusum", 0, "#0072B2"),
    ]:
        selected = alarms.loc[alarms["detector"].eq(detector)]
        axes[4].scatter(
            selected["observed_day"],
            np.full(len(selected), y),
            marker="|", s=100, color=color,
        )

    axes[4].set_yticks([0, 1], ["Error CUSUM", "Input"])
    axes[4].set_ylim(-0.6, 1.6)
    axes[4].set_ylabel("Alerts")
    axes[4].set_xlabel("Market date (AEST)")
    axes[4].set_xlim(START, END)

    for ax in axes:
        ax.grid(alpha=0.2)

    fig.suptitle(
        "Day-type changes, frozen demand errors and drift signals\n"
        "Descriptive time alignment; alerts are not evidence of causation"
    )
    finish_figure(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--types", type=Path, default=DEFAULT_TYPES
    )
    parser.add_argument(
        "--p3-dir", type=Path, default=DEFAULT_P3
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT
    )
    args = parser.parse_args()

    daily = load_and_merge(args.types, args.p3_dir)
    alarms, input_threshold, cusum_threshold = load_detector_files(
        args.p3_dir, daily
    )
    yearly, overall, monthly = summarize(
        daily, alarms, input_threshold
    )

    args.output.mkdir(parents=True, exist_ok=True)

    tables = {
        "daily_types_errors_drift.csv": daily,
        "yearly_errors_by_type.csv": yearly,
        "yearly_overall.csv": overall,
        "monthly_types_errors_drift.csv": monthly,
        "frozen_and_input_alerts.csv": alarms,
    }
    for filename, table in tables.items():
        table.to_csv(args.output / filename, index=False)

    plot_yearly(
        yearly, overall,
        args.output / "yearly_day_types_and_errors.png",
    )
    plot_monthly(
        daily, monthly, alarms, input_threshold,
        args.output / "monthly_day_types_errors_drift.png",
    )

    print(f"合并检查通过：{len(daily):,}天")
    print(yearly.round(3).to_string(index=False))
    print("\n告警数量：")
    print(alarms["detector"].value_counts().to_string())
    print(f"\n输入阈值：{input_threshold:.6f}")
    print(f"CUSUM阈值：{cusum_threshold:.6f}")
    print(f"\n输出目录：{args.output}")

    print(
        "\n解释注意：\n"
        "1. 零样本日型的误差为空，不是0。\n"
        "2. 2026年仅覆盖1—10月，年度比较受到季节覆盖影响。\n"
        "3. 少于30天的分组用空心外圈提示；30不是统计显著性阈值。\n"
        "4. MAPE贡献是占比乘以组内误差的算术分解，不是因果解释。\n"
        "5. 中心距离在固定PCA空间计算，不能捕捉所有被舍弃方向的变化。\n"
        "6. 输入告警只记录由阈值下方越到上方的日期。\n"
        "7. 日期接近、趋势相似不代表因果，也不能单独证明概念漂移。\n"
        "8. 日型来自当天真实需求，只用于事后分析。"
    )


if __name__ == "__main__":
    main()