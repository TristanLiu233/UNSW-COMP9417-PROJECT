from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "nem_weather_halfhourly_2010_2026.csv"
SLOTS_PER_DAY = 48


def build_ridge_features(data):
    day = data["market_day"]
    slot = data["slot"]
    weekday = day.dt.day_of_week

    slot_angle = 2 * np.pi * slot / SLOTS_PER_DAY
    year_length = np.where(day.dt.is_leap_year, 366, 365)
    season_angle = season_angle = 2 * np.pi * (day.dt.day_of_year - 1) / year_length

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
        # 先移一天，再按档位计算过去 28 天的统计量。
        past = data[column].groupby(slot).shift(1)
        by_slot = past.groupby(slot)
        features[f"{column}_mean28"] = by_slot.transform(
            lambda values: values.rolling(28).mean()
        )
        features[f"{column}_std28"] = by_slot.transform(
            lambda values: values.rolling(28).std(ddof=0)
        )

    return features


def main():
    data = pd.read_csv(DATA_PATH, parse_dates=["timestamp"], index_col="timestamp")
    data = data.sort_index()

    if data.empty:
        raise ValueError("CSV 没有数据。")

    expected = pd.date_range(data.index[0], periods=len(data), freq="30min")
    if not data.index.equals(expected):
        raise ValueError("半小时时间戳有缺失或重复，无法按行计算滞后特征。")

    # 时间戳是区间结束时间，先减 30 分钟再确定市场日和档位。
    interval_start = data.index - pd.Timedelta(minutes=30)
    data["market_day"] = interval_start.normalize()
    data["slot"] = interval_start.hour * 2 + interval_start.minute // 30

    features = build_ridge_features(data)
    complete = features.dropna()
    if complete.empty:
        raise ValueError("没有特征齐全的记录，请检查数据是否覆盖至少 29 天。")

    print(f"读入 {len(data):,} 条半小时记录")
    print(f"Ridge 特征数：{features.shape[1]}")
    print(f"第一条特征齐全的记录：{complete.index[0]}")


if __name__ == "__main__":
    main()