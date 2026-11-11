import argparse
from pathlib import Path

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ridge_base import (
    DATA_PATH,
    OUTPUT_DIR,
    REPLAY_END,
    REPLAY_START,
    build_ridge_features,
    calculate_scores,
    load_data,
    plot_monthly,
)
from tune_ridge_by_slot_hdd_cdd import predict_by_slot
from tune_ridge_hdd_cdd import predict_shared


TUNE_OUTPUT_DIR = OUTPUT_DIR / "tune"
COMPARISON_DIR = OUTPUT_DIR / "comparison_tuned"

# 依次为结果文件标识、模型路径、保存时的 variant、预测函数、图例名称。
MODELS = (
    (
        "hdd_cdd",
        "hdd_cdd/ridge_best_model_hdd_cdd.joblib",
        "hdd_cdd",
        predict_shared,
        "Base Ridge",
    ),
    (
        "by_slot_hdd_cdd",
        "by_slot_hdd_cdd/ridge_best_model_by_slot_hdd_cdd.joblib",
        "by_slot_hdd_cdd",
        predict_by_slot,
        "By-slot Ridge",
    ),
    (
        "base_hdd17_cdd195",
        "hdd17_cdd195/base/ridge_best_model.joblib",
        "base",
        predict_shared,
        "Base Ridge",
    ),
    (
        "by_slot_hdd17_cdd195",
        "hdd17_cdd195/by_slot/ridge_best_model.joblib",
        "by_slot",
        predict_by_slot,
        "By-slot Ridge",
    ),
)


def load_saved_model(model_path, expected_variant):
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到已保存的模型：{model_path}")

    saved = joblib.load(model_path)
    if (
        saved["variant"] != expected_variant
        or saved["train_through"] != "2014-12-31"
    ):
        raise ValueError(f"模型版本或训练截止日期与回放设置不一致：{model_path}")

    # 原模型保存一个共用阈值，新模型分别保存采暖和制冷阈值。
    if "heating_base_c" in saved or "cooling_base_c" in saved:
        if "heating_base_c" not in saved or "cooling_base_c" not in saved:
            raise ValueError(f"模型缺少采暖或制冷阈值：{model_path}")
        heating_base = float(saved["heating_base_c"])
        cooling_base = float(saved["cooling_base_c"])
    else:
        heating_base = cooling_base = float(saved["base_temp_c"])

    if not np.isfinite([heating_base, cooling_base]).all():
        raise ValueError(f"模型保存的温度阈值无效：{model_path}")

    return saved, heating_base, cooling_base


def plot_comparison(monthly_by_model, path):
    fig, ax = plt.subplots(figsize=(13, 5))

    for label, monthly in monthly_by_model.items():
        dates = pd.to_datetime(monthly["month"], format="%Y-%m")
        ax.plot(dates, monthly["mape_pct"], linewidth=1.3, label=label)

    ax.set_title("Monthly NSW demand MAPE across four tuned Ridge models")
    ax.set_xlabel("Market month (Jan 2015 – Oct 2026)")
    ax.set_ylabel("MAPE (%)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate four saved tuned Ridge models on the historical replay."
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    args = parser.parse_args()

    data = load_data(args.data)
    days = data["market_day"]
    replay = (days >= REPLAY_START) & (days < REPLAY_END)

    if not replay.any():
        raise ValueError("没有找到回放期数据。")

    # 先构造滞后和滚动特征，再取回放期，保留首日所需的历史信息。
    base_features = build_ridge_features(data).loc[replay]
    temperature = data.loc[replay, "syd_temp"]
    slots = data.loc[replay, "slot"].to_numpy()
    actual = data.loc[replay, "nsw_demand"].to_numpy()

    if not np.isfinite(actual).all() or (actual <= 0).any():
        raise ValueError("回放期的真实需求必须为有限正数。")

    summary_rows = []
    monthly_by_model = {}

    for variant, relative_path, saved_variant, predict, model_name in MODELS:
        model_path = TUNE_OUTPUT_DIR / relative_path
        saved, heating_base, cooling_base = load_saved_model(
            model_path, saved_variant
        )

        features = base_features.assign(
            hdd=(heating_base - temperature).clip(lower=0),
            cdd=(temperature - cooling_base).clip(lower=0),
        )

        feature_columns = saved["feature_columns"]
        if sorted(features.columns) != sorted(feature_columns):
            raise ValueError(f"回放特征与模型保存的特征不一致：{model_path}")
        features = features.loc[:, feature_columns]

        if not np.isfinite(features.to_numpy()).all():
            raise ValueError(f"{variant} 的回放特征存在缺失值或无穷值。")

        predicted = predict(saved["model"], features, slots)
        if predicted.shape != actual.shape or not np.isfinite(predicted).all():
            raise ValueError(f"{variant} 的回放预测数量不符或存在无效值。")

        predictions = pd.Series(
            predicted,
            index=features.index,
            name="predicted_demand_mw",
        )
        yearly, monthly = calculate_scores(data, predictions)

        alpha = float(saved["alpha"])
        label = (
            f"{model_name} | {heating_base:g}/{cooling_base:g}°C "
            f"| alpha={alpha:g}"
        )
        suffix = f"{variant}_tuned"
        output_dir = model_path.parent

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
            model_label=label,
        )

        absolute_error = np.abs(predicted - actual)
        summary_rows.append({
            "model": variant,
            "heating_base_c": heating_base,
            "cooling_base_c": cooling_base,
            "alpha": alpha,
            "n_slots": len(actual),
            "mape_pct": 100 * (absolute_error / actual).mean(),
            "mae_mw": absolute_error.mean(),
        })
        monthly_by_model[label] = monthly

    summary = pd.DataFrame(summary_rows).sort_values(["mape_pct", "mae_mw"])
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        COMPARISON_DIR / "ridge_tuned_comparison_overall.csv",
        index=False,
    )
    plot_comparison(
        monthly_by_model,
        COMPARISON_DIR / "ridge_tuned_comparison_monthly_mape.png",
    )

    print(f"回放期 {days.loc[replay].min().date()} 至 {days.loc[replay].max().date()}")
    print(summary.round({"mape_pct": 6, "mae_mw": 2}).to_string(index=False))
    print(f"\n四模型对比结果目录 {COMPARISON_DIR}")


if __name__ == "__main__":
    main()