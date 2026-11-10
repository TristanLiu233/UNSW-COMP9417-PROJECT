import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "nem_weather_halfhourly_2010_2026.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

SLOTS_PER_DAY = 48
HISTORY_DAYS = 28
BOOTSTRAP_REPEATS = 2000

DATA_START = pd.Timestamp("2010-01-01 00:30")
REPLAY_START = pd.Timestamp("2015-01-01")
REPLAY_END = pd.Timestamp("2026-11-01")  # 市场日右边界，不包含当天


def load_data(path):
    data = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    data = data.sort_index()

    # 2026-11-01 00:00 是 2026-10-31 的最后一个半小时档位。
    data = data.loc[DATA_START:REPLAY_END].copy()
    expected = pd.date_range(DATA_START, REPLAY_END, freq="30min")
    if not data.index.equals(expected):
        raise ValueError("课程数据缺少所需时段，或有重复的半小时时间戳。")

    starts = data.index - pd.Timedelta(minutes=30)
    data["market_day"] = starts.normalize()
    data["slot"] = starts.hour * 2 + starts.minute // 30
    return data


def build_ridge_features(data):
    day = data["market_day"]
    slot = data["slot"]
    weekday = day.dt.day_of_week
    year_length = np.where(day.dt.is_leap_year, 366, 365)

    slot_angle = 2 * np.pi * slot / SLOTS_PER_DAY
    season_angle = 2 * np.pi * (day.dt.day_of_year - 1) / year_length

    features = pd.DataFrame({
        "slot_sin": np.sin(slot_angle),
        "slot_cos": np.cos(slot_angle),
        "season_sin": np.sin(season_angle),
        "season_cos": np.cos(season_angle),
        "weekend": (weekday >= 5).astype(int),
    }, index=data.index)

    for i in range(7):
        features[f"weekday_{i}"] = (weekday == i).astype(int)

    for lag in (1, 7):
        features[f"nsw_demand_lag{lag}"] = data["nsw_demand"].shift(SLOTS_PER_DAY * lag)
        features[f"nsw_price_lag{lag}"] = data["nsw_price"].shift(SLOTS_PER_DAY * lag)

    daily_mean_price = data.groupby("market_day")["nsw_price"].mean()
    features["prev_price_mean"] = day.map(daily_mean_price.shift(1))
    features["lag1_price_gap"] = (
        features["nsw_price_lag1"] - features["prev_price_mean"]
    )

    features["qld_demand_lag1"] = data["qld_demand"].shift(SLOTS_PER_DAY)
    features["qld_demand_lag7"] = data["qld_demand"].shift(7 * SLOTS_PER_DAY)
    features["vic_demand_lag7"] = data["vic_demand"].shift(7 * SLOTS_PER_DAY)

    for column in ("nsw_demand", "nsw_price"):
        # 先移一天，再在每个档位上计算过去 28 个市场日的统计量。
        past = data[column].groupby(slot).shift(1)
        by_slot = past.groupby(slot)
        features[f"{column}_mean28"] = by_slot.transform(
            lambda values: values.rolling(HISTORY_DAYS).mean()
        )
        features[f"{column}_std28"] = by_slot.transform(
            lambda values: values.rolling(HISTORY_DAYS).std(ddof=0)
        )

    return features


def run_replay(data, features, train_through):
    day = data["market_day"]
    demand = data["nsw_demand"]
    first_train_day = day.iloc[0] + pd.Timedelta(days=HISTORY_DAYS)

    train = (day >= first_train_day) & (day.dt.year <= train_through)
    replay = (day >= REPLAY_START) & (day < REPLAY_END)
    used = train | replay

    if not np.isfinite(features.loc[used].to_numpy()).all():
        raise ValueError("训练期或回放期的特征存在缺失值或无穷值。")
    if not np.isfinite(demand.loc[used].to_numpy()).all():
        raise ValueError("训练期或回放期的真实需求存在缺失值或无穷值。")
    if (demand.loc[replay] <= 0).any():
        raise ValueError("回放期存在非正需求，无法按课程公式计算 MAPE。")

    # 标准化器与 Ridge 只在训练期拟合一次。
    model = make_pipeline(StandardScaler(), Ridge(alpha=1))
    model.fit(features.loc[train], demand.loc[train])

    # 特征已经按可见历史构造。固定模型对这些行批量预测，
    # 与按市场日依次调用同一个模型等价。
    predictions = pd.Series(
        model.predict(features.loc[replay]),
        index=data.index[replay],
        name="predicted_demand_mw",
    )

    print(f"Ridge 特征数：{features.shape[1]}")
    print(f"训练期：{day.loc[train].min().date()} 至 {day.loc[train].max().date()}")
    print(f"回放期：{day.loc[replay].min().date()} 至 {day.loc[replay].max().date()}")
    return predictions


def calculate_scores(data, predictions):
    results = data.loc[predictions.index, ["market_day", "nsw_demand"]].copy()
    results["absolute_error"] = (predictions - results["nsw_demand"]).abs()
    results["ape_pct"] = 100 * results["absolute_error"] / results["nsw_demand"]

    yearly = (
        results.groupby(results["market_day"].dt.year.rename("year"))
        .agg(
            first_day=("market_day", "min"),
            last_day=("market_day", "max"),
            n_slots=("ape_pct", "size"),
            mape_pct=("ape_pct", "mean"),
            mae_mw=("absolute_error", "mean"),
        )
        .reset_index()
    )

    rng = np.random.default_rng(17)
    monthly_rows = []

    for month, group in results.groupby(results["market_day"].dt.to_period("M")):
        ape = group["ape_pct"]
        mape = ape.mean()
        normal_margin = 1.96 * ape.std(ddof=1) / np.sqrt(len(ape))

        # 每天都是 48 档，因此重采样日均 APE 等价于重采样整天的 48 档。
        daily_ape = group.groupby("market_day")["ape_pct"].mean().to_numpy()
        sampled_days = rng.integers(
            0, len(daily_ape), size=(BOOTSTRAP_REPEATS, len(daily_ape))
        )
        boot_mape = daily_ape[sampled_days].mean(axis=1)
        block_low, block_high = np.quantile(boot_mape, [0.025, 0.975])

        monthly_rows.append({
            "month": str(month),
            "n_days": len(daily_ape),
            "n_slots": len(ape),
            "mape_pct": mape,
            "mae_mw": group["absolute_error"].mean(),
            "normal_ci_low": mape - normal_margin,
            "normal_ci_high": mape + normal_margin,
            "day_bootstrap_ci_low": block_low,
            "day_bootstrap_ci_high": block_high,
        })

    return yearly, pd.DataFrame(monthly_rows)


def plot_monthly(monthly, path):
    dates = pd.to_datetime(monthly["month"], format="%Y-%m")
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.fill_between(
        dates, monthly["day_bootstrap_ci_low"], monthly["day_bootstrap_ci_high"],
        alpha=0.2, label="95% day-block bootstrap interval",
    )
    ax.fill_between(
        dates, monthly["normal_ci_low"], monthly["normal_ci_high"],
        alpha=0.3, label="95% independent half-hour interval",
    )
    ax.plot(dates, monthly["mape_pct"], linewidth=1.2, label="Frozen Ridge")
    ax.set_title("Frozen Ridge NSW demand forecast by market month")
    # ax.set_xlabel("Market month")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(dates.iloc[0], dates.iloc[-1])
    ax.set_xlabel("Market month (Jan 2015 – Oct 2026)")
    ax.set_ylabel("MAPE (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run the frozen Ridge demand replay.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--train-through", type=int, choices=(2013, 2014), default=2014)
    args = parser.parse_args()

    data = load_data(args.data)
    features = build_ridge_features(data)
    predictions = run_replay(data, features, args.train_through)
    yearly, monthly = calculate_scores(data, predictions)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_DIR / "ridge_predictions.csv", index_label="timestamp")
    yearly.to_csv(OUTPUT_DIR / "ridge_yearly_scores.csv", index=False)
    monthly.to_csv(OUTPUT_DIR / "ridge_monthly_scores.csv", index=False)
    plot_monthly(monthly, OUTPUT_DIR / "ridge_monthly_mape.png")

    print(f"已导出 {len(predictions):,} 条半小时预测、{len(monthly)} 个月的指标。")
    print(yearly.round({"mape_pct": 3, "mae_mw": 2}).to_string(index=False))
    print(f"结果目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()