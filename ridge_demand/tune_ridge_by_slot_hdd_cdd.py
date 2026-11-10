import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ridge_base import SLOTS_PER_DAY
from tune_ridge_hdd_cdd import tune_and_save


def fit_by_slot(features, demand, slots, alpha):
    models = {}

    for slot in range(SLOTS_PER_DAY):
        mask = slots == slot
        if not mask.any():
            raise ValueError(f"档位 {slot} 缺少训练样本。")

        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        models[slot] = model.fit(features.iloc[mask], demand.iloc[mask])

    return models


def predict_by_slot(models, features, slots):
    predictions = np.full(len(features), np.nan)

    for slot, model in models.items():
        mask = slots == slot
        if mask.any():
            predictions[mask] = model.predict(features.iloc[mask])

    return predictions


if __name__ == "__main__":
    tune_and_save("by_slot_hdd_cdd", fit_by_slot, predict_by_slot)