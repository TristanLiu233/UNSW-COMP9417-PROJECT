import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ridge_base import DATA_PATH, HISTORY_DAYS, OUTPUT_DIR, load_data
from ridge_hdd_cdd import BASE_TEMP_C, build_hdd_cdd_features


ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def fit_shared(features, demand, slots, alpha):
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    return model.fit(features, demand)


def predict_shared(model, features, slots):
    return model.predict(features)


def tune_and_save(variant, fit_model, predict_model):
    parser = argparse.ArgumentParser(
        description=f"Tune alpha and save the refitted {variant} Ridge model."
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    args = parser.parse_args()

    data = load_data(args.data)
    data = data.loc[data["market_day"].dt.year <= 2014]
    days = data["market_day"]
    features = build_hdd_cdd_features(data)
    demand = data["nsw_demand"]
    slots = data["slot"].to_numpy()

    first_train_day = days.iloc[0] + pd.Timedelta(days=HISTORY_DAYS)
    ready = days >= first_train_day
    train = ready & (days.dt.year <= 2013)
    valid = days.dt.year == 2014

    if not train.any() or not valid.any():
        raise ValueError("缺少 2010–2013 年训练数据或 2014 年验证数据。")
    if not np.isfinite(features.loc[ready].to_numpy()).all():
        raise ValueError("模型特征存在缺失值或无穷值。")
    if not np.isfinite(demand.loc[ready].to_numpy()).all():
        raise ValueError("真实需求存在缺失值或无穷值。")

    x_train = features.loc[train]
    y_train = demand.loc[train]
    x_valid = features.loc[valid]
    actual = demand.loc[valid].to_numpy()
    train_slots = slots[train.to_numpy()]
    valid_slots = slots[valid.to_numpy()]

    if (actual <= 0).any():
        raise ValueError("验证需求必须为正才能计算 MAPE。")

    rows = []
    for alpha in ALPHAS:
        model = fit_model(x_train, y_train, train_slots, alpha)
        predicted = predict_model(model, x_valid, valid_slots)
        if not np.isfinite(predicted).all():
            raise ValueError(f"alpha={alpha:g} 的验证预测存在缺失值或无穷值。")

        error = np.abs(predicted - actual)
        rows.append({
            "alpha": alpha,
            "mape_pct": 100 * (error / actual).mean(),
            "mae_mw": error.mean(),
        })

    scores = pd.DataFrame(rows).sort_values(
        ["mape_pct", "mae_mw", "alpha"], ignore_index=True
    )
    best = scores.iloc[0]
    best_alpha = float(best["alpha"])
    default_score = scores.loc[scores["alpha"].eq(1.0), "mape_pct"].iloc[0]

    # 用选好的 alpha 在 2010–2014 年重新拟合。
    final_model = fit_model(
        features.loc[ready],
        demand.loc[ready],
        slots[ready.to_numpy()],
        best_alpha,
    )

    output_dir = OUTPUT_DIR / "tune" / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / f"ridge_alpha_search_{variant}.csv"
    model_path = output_dir / f"ridge_best_model_{variant}.joblib"

    scores.to_csv(scores_path, index=False)
    joblib.dump({
        "variant": variant,
        "alpha": best_alpha,
        "base_temp_c": BASE_TEMP_C,
        "feature_columns": features.columns.tolist(),
        "train_through": "2014-12-31",
        "model": final_model,
    }, model_path)

    print(f"模型：{variant}")
    print(f"选型训练样本：{train.sum():,}；2014 验证样本：{valid.sum():,}")
    print(scores.round({"mape_pct": 4, "mae_mw": 2}).to_string(index=False))
    print(f"\n最佳 alpha：{best_alpha:g}")
    print(
        f"相对 alpha=1 的验证 MAPE 变化："
        f"{best['mape_pct'] - default_score:+.4f} 个百分点"
    )
    print(f"重训样本：{ready.sum():,}")
    print(f"调参表：{scores_path}")
    print(f"最佳模型：{model_path}")


if __name__ == "__main__":
    tune_and_save("hdd_cdd", fit_shared, predict_shared)