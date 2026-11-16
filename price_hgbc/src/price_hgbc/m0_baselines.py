"""M0.3: the two official baselines on the 2014 validation year (target day D+1 in 2014).

Every number comes from harness/dayahead.py: make_inputs / targets / baseline / score.
Usage: python -m src.price_hgbc.m0_baselines
"""
import json
import os

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from .kit import DATA_CSV, RESULTS, da


def baseline_predictions(df, year):
    """Concatenated truth / b1 / b7 for all complete target days in `year`."""
    days = [d for d in da.complete_days(df) if d.year == year]
    truth, b1, b7 = [], [], []
    for d in days:
        history, weather_next = da.make_inputs(df, d)
        truth.append(da.targets(df, d))
        b1.append(da.baseline(history, weather_next, 1))
        b7.append(da.baseline(history, weather_next, 7))
    return days, pd.concat(truth), pd.concat(b1), pd.concat(b7)


def baseline_scores(df, year=2014):
    days, truth, b1, b7 = baseline_predictions(df, year)
    rows = {}
    for name, pred in (("b1", b1), ("b7", b7)):
        s = da.score(truth, pred)
        y, yhat = truth["nsw_price_up"].to_numpy(), pred["nsw_price_up"].to_numpy()
        s["BalAcc_sklearn"] = float(balanced_accuracy_score(y, yhat) * 100)
        s["Acc_sklearn"] = float(accuracy_score(y, yhat) * 100)
        s["n_days"] = len(days)
        s["n_halfhours"] = int(len(y))
        s["class1_share"] = float(y.mean() * 100)
        rows[name] = s
    return pd.DataFrame(rows).T


def main():
    df = da.load_dataset(DATA_CSV)
    tab = baseline_scores(df, 2014)
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "m0_baselines_2014.csv")
    tab.to_csv(out, float_format="%.6f")
    print(tab.to_string(float_format=lambda v: f"{v:.4f}"))
    print("saved", out)


if __name__ == "__main__":
    main()
