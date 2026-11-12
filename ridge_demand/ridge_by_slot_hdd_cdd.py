import argparse
from pathlib import Path

from ridge_base import (
    DATA_PATH,
    OUTPUT_DIR,
    calculate_scores,
    load_data,
    plot_monthly,
)
from ridge_by_slot import run_by_slot_replay
from ridge_hdd_cdd import BASE_TEMP_C, build_hdd_cdd_features


def main():
    parser = argparse.ArgumentParser(
        description="Run per-slot Ridge demand replay with HDD/CDD features."
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument(
        "--train-through", type=int, choices=(2013, 2014), default=2014
    )
    args = parser.parse_args()

    data = load_data(args.data)
    features = build_hdd_cdd_features(data)
    predictions = run_by_slot_replay(data, features, args.train_through)
    yearly, monthly = calculate_scores(data, predictions)

    output_dir = OUTPUT_DIR / "by_slot_hdd_cdd"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(
        output_dir / "ridge_predictions_by_slot_hdd_cdd.csv",
        index_label="timestamp",
    )
    yearly.to_csv(
        output_dir / "ridge_yearly_scores_by_slot_hdd_cdd.csv", index=False
    )
    monthly.to_csv(
        output_dir / "ridge_monthly_scores_by_slot_hdd_cdd.csv", index=False
    )
    plot_monthly(
        monthly,
        output_dir / "ridge_monthly_mape_by_slot_hdd_cdd.png",
        model_label="Frozen Ridge (by slot + HDD/CDD)",
    )

    print(f"冷暖度差基准温度：{BASE_TEMP_C:g}°C")
    print(f"已导出 {len(predictions):,} 条半小时预测、{len(monthly)} 个月的指标。")
    print(yearly.round({"mape_pct": 3, "mae_mw": 2}).to_string(index=False))
    print(f"结果目录：{output_dir}")


if __name__ == "__main__":
    main()