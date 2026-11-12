import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ridge_base import DATA_PATH, HISTORY_DAYS, OUTPUT_DIR, build_ridge_features, load_data
from tune_ridge_by_slot_hdd_cdd import fit_by_slot, predict_by_slot
from tune_ridge_hdd_cdd import fit_shared, predict_shared


HEATING_BASE_C = 17.0
COOLING_BASE_C = 19.5
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
TRAIN_THROUGH = "2014-12-31"
EXPERIMENT_DIR = OUTPUT_DIR / "tune" / "hdd17_cdd195"

VARIANTS = (
    ("base", fit_shared, predict_shared),
    ("by_slot", fit_by_slot, predict_by_slot),
)


def main():
    parser = argparse.ArgumentParser(
        description="Tune Ridge alpha with HDD 17°C and CDD 19.5°C."
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    args = parser.parse_args()

    data = load_data(args.data)
    data = data.loc[data["market_day"].dt.year <= 2014]
    days = data["market_day"]
    demand = data["nsw_demand"]
    slots = data["slot"]
    temperature = data["syd_temp"]

    features = build_ridge_features(data).assign(
        hdd=(HEATING_BASE_C - temperature).clip(lower=0),
        cdd=(temperature - COOLING_BASE_C).clip(lower=0),
    )

    first_train_day = days.iloc[0] + pd.Timedelta(days=HISTORY_DAYS)
    refit = days >= first_train_day
    train = refit & (days.dt.year <= 2013)
    valid = days.dt.year == 2014

    if not train.any() or not valid.any():
        raise ValueError("缺少 2010–2013 年训练数据或 2014 年验证数据。")
    if not np.isfinite(features.loc[refit].to_numpy()).all():
        raise ValueError("训练或验证特征包含缺失值或无穷值。")
    if not np.isfinite(demand.loc[refit].to_numpy()).all():
        raise ValueError("训练或验证需求包含缺失值或无穷值。")

    x_train = features.loc[train]
    y_train = demand.loc[train]
    x_valid = features.loc[valid]
    actual = demand.loc[valid].to_numpy()
    train_slots = slots.loc[train].to_numpy()
    valid_slots = slots.loc[valid].to_numpy()

    if (actual <= 0).any():
        raise ValueError("验证需求必须为正才能计算 MAPE。")

    search_tables = []
    best_rows = []

    for variant, fit_model, predict_model in VARIANTS:
        rows = []
        for alpha in ALPHAS:
            model = fit_model(x_train, y_train, train_slots, alpha)
            predicted = predict_model(model, x_valid, valid_slots)
            if predicted.shape != actual.shape or not np.isfinite(predicted).all():
                raise ValueError(f"{variant} 在 alpha={alpha:g} 下的验证预测无效。")

            error = np.abs(predicted - actual)
            rows.append({
                "model": variant,
                "alpha": alpha,
                "validation_mape_pct": 100 * (error / actual).mean(),
                "validation_mae_mw": error.mean(),
            })

        scores = pd.DataFrame(rows).sort_values(
            ["validation_mape_pct", "validation_mae_mw", "alpha"],
            ignore_index=True,
        )
        search_tables.append(scores)
        best = scores.iloc[0]
        best_alpha = float(best["alpha"])
        alpha1 = scores.loc[scores["alpha"].eq(1.0)].iloc[0]

        final_model = fit_model(
            features.loc[refit],
            demand.loc[refit],
            slots.loc[refit].to_numpy(),
            best_alpha,
        )

        output_dir = EXPERIMENT_DIR / variant
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "variant": variant,
                "alpha": best_alpha,
                "heating_base_c": HEATING_BASE_C,
                "cooling_base_c": COOLING_BASE_C,
                "feature_columns": features.columns.tolist(),
                "train_through": TRAIN_THROUGH,
                "model": final_model,
            },
            output_dir / "ridge_best_model.joblib",
        )

        best_rows.append({
            "model": variant,
            "best_alpha": best_alpha,
            "validation_mape_alpha1_pct": alpha1["validation_mape_pct"],
            "validation_mape_pct": best["validation_mape_pct"],
            "validation_mae_mw": best["validation_mae_mw"],
        })

    search = pd.concat(search_tables, ignore_index=True)
    best_scores = pd.DataFrame(best_rows)
    search.to_csv(EXPERIMENT_DIR / "ridge_alpha_search.csv", index=False)
    best_scores.to_csv(EXPERIMENT_DIR / "ridge_validation_comparison.csv", index=False)

    print(f"HDD {HEATING_BASE_C:g}°C / CDD {COOLING_BASE_C:g}°C")
    print(f"选型训练 {train.sum():,} 档，2014 验证 {valid.sum():,} 档")
    print(f"重训 {refit.sum():,} 档，特征 {features.shape[1]} 列")
    print(best_scores.round(6).to_string(index=False))
    print(f"结果目录 {EXPERIMENT_DIR}")


if __name__ == "__main__":
    main()