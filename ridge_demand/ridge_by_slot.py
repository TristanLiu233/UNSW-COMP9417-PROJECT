import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ridge_base import (
    DATA_PATH,
    HISTORY_DAYS,
    OUTPUT_DIR,
    REPLAY_END,
    REPLAY_START,
    SLOTS_PER_DAY,
    build_ridge_features,
    calculate_scores,
    load_data,
    plot_monthly,
)


BY_SLOT_OUTPUT_DIR = OUTPUT_DIR / "by_slot"


def run_by_slot_replay(data, features, train_through):
    day = data["market_day"]
    slot = data["slot"]
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
        raise ValueError("回放期存在非正需求，无法计算 MAPE。")

    predictions = pd.Series(
        index=data.index[replay],
        dtype=float,
        name="predicted_demand_mw",
    )

    for slot_number in range(SLOTS_PER_DAY):
        same_slot = slot.eq(slot_number)
        train_slot = train & same_slot
        replay_slot = replay & same_slot

        # 每套标准化器和 Ridge 都只用该档位的训练样本拟合。
        model = make_pipeline(StandardScaler(), Ridge(alpha=1))
        model.fit(features.loc[train_slot], demand.loc[train_slot])

        # 把各档位的预测按时间戳填回同一列。
        slot_index = data.index[replay_slot]
        predictions.loc[slot_index] = model.predict(features.loc[slot_index])

    if not np.isfinite(predictions.to_numpy()).all():
        raise ValueError("分时段回放存在缺失或无穷预测值。")

    print(f"Ridge 子模型数：{SLOTS_PER_DAY}")
    print(f"每个子模型的特征列数：{features.shape[1]}")
    print(f"训练期：{day.loc[train].min().date()} 至 {day.loc[train].max().date()}")
    print(f"回放期：{day.loc[replay].min().date()} 至 {day.loc[replay].max().date()}")
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Run the per-slot Ridge demand replay.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--train-through", type=int, choices=(2013, 2014), default=2014)
    args = parser.parse_args()

    data = load_data(args.data)
    features = build_ridge_features(data)
    predictions = run_by_slot_replay(data, features, args.train_through)
    yearly, monthly = calculate_scores(data, predictions)

    BY_SLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        BY_SLOT_OUTPUT_DIR / "ridge_predictions_by_slot.csv",
        index_label="timestamp",
    )
    yearly.to_csv(BY_SLOT_OUTPUT_DIR / "ridge_yearly_scores_by_slot.csv", index=False)
    monthly.to_csv(BY_SLOT_OUTPUT_DIR / "ridge_monthly_scores_by_slot.csv", index=False)
    plot_monthly(
        monthly,
        BY_SLOT_OUTPUT_DIR / "ridge_monthly_mape_by_slot.png",
        model_label="Frozen Ridge (by slot)",
    )

    print(f"已导出 {len(predictions):,} 条半小时预测、{len(monthly)} 个月的指标。")
    print(yearly.round({"mape_pct": 3, "mae_mw": 2}).to_string(index=False))
    print(f"结果目录：{BY_SLOT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()