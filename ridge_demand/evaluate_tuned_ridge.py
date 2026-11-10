import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ridge_base import (
    DATA_PATH,
    OUTPUT_DIR,
    REPLAY_END,
    REPLAY_START,
    calculate_scores,
    load_data,
    plot_monthly,
)
from ridge_hdd_cdd import BASE_TEMP_C, build_hdd_cdd_features
from tune_ridge_by_slot_hdd_cdd import predict_by_slot
from tune_ridge_hdd_cdd import predict_shared


TUNE_OUTPUT_DIR = OUTPUT_DIR / "tune"

VARIANTS = (
    ("hdd_cdd", predict_shared, "Ridge + HDD/CDD"),
    ("by_slot_hdd_cdd", predict_by_slot, "By-slot Ridge + HDD/CDD"),
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the saved tuned Ridge models on the historical replay."
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    args = parser.parse_args()

    data = load_data(args.data)
    days = data["market_day"]
    replay = (days >= REPLAY_START) & (days < REPLAY_END)

    if not replay.any():
        raise ValueError("没有找到回放期数据。")

    # 先构造滞后和滚动特征，让回放首日也能用到此前的历史记录。
    features = build_hdd_cdd_features(data).loc[replay]
    slots = data.loc[replay, "slot"].to_numpy()
    actual = data.loc[replay, "nsw_demand"].to_numpy()

    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("回放特征存在缺失值或无穷值。")
    if not np.isfinite(actual).all() or (actual <= 0).any():
        raise ValueError("回放期的真实需求必须为有限正数。")

    for variant, predict, label in VARIANTS:
        output_dir = TUNE_OUTPUT_DIR / variant
        model_path = output_dir / f"ridge_best_model_{variant}.joblib"

        if not model_path.is_file():
            raise FileNotFoundError(
                f"找不到已保存的模型：{model_path}。"
                "请确认已将模型放到对应的 outputs/tune/ 子目录。"
            )

        saved = joblib.load(model_path)

        if (
            saved["variant"] != variant
            or saved["train_through"] != "2014-12-31"
            or saved["base_temp_c"] != BASE_TEMP_C
            or saved["feature_columns"] != features.columns.tolist()
        ):
            raise ValueError(f"保存的模型配置与当前回放设置不一致：{model_path}")

        predicted = predict(saved["model"], features, slots)
        if predicted.shape != actual.shape or not np.isfinite(predicted).all():
            raise ValueError(f"{variant} 的回放预测数量不符或存在无效值。")

        predictions = pd.Series(
            predicted,
            index=features.index,
            name="predicted_demand_mw",
        )
        yearly, monthly = calculate_scores(data, predictions)

        suffix = f"{variant}_tuned"
        predictions.to_csv(
            output_dir / f"ridge_predictions_{suffix}.csv",
            index_label="timestamp",
        )
        yearly.to_csv(
            output_dir / f"ridge_yearly_scores_{suffix}.csv",
            index=False,
        )
        monthly.to_csv(
            output_dir / f"ridge_monthly_scores_{suffix}.csv",
            index=False,
        )
        plot_monthly(
            monthly,
            output_dir / f"ridge_monthly_mape_{suffix}.png",
            model_label=f"{label} (alpha={saved['alpha']:g})",
        )

        absolute_error = np.abs(predicted - actual)
        mape = 100 * (absolute_error / actual).mean()
        mae = absolute_error.mean()

        print(f"\n模型：{variant} | alpha={saved['alpha']:g}")
        print(f"回放档位：{len(actual):,} | 月份：{len(monthly)}")
        print(f"回放期 MAPE：{mape:.4f}% | MAE：{mae:.2f} MW")
        print(f"结果目录：{output_dir}")


if __name__ == "__main__":
    main()