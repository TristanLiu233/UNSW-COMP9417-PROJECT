"""Compare the selected demand recipes with and without an NSW public-holiday flag."""

import argparse
from importlib.metadata import version
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ablation.features import load_study
from ablation.models import predict_raw
from ablation.scoring import metrics
from ablation.study import TOLERANCE
from baseline.replay import REPLAY_END, REPLAY_START
from baseline.train import FROZEN_CUTOFF, ROOT, TRAIN_START, VALIDATION_CUTOFF, selected_recipes
from harness import dayahead
from src.common.features import HOLIDAY_COLUMN, build_features


BOOTSTRAP_SEED = 17
BOOTSTRAP_DRAWS = 2000


def check_environment():
    """Check the course pins used by these fits, without requiring unrelated libraries."""
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Run the ablation under course Python 3.11")
    pins = dict(line.split("==") for line in (ROOT / "requirements.txt").read_text().splitlines()
                if "==" in line and not line.lstrip().startswith("#"))
    versions = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "holidays"):
        versions[package] = version(package)
        if versions[package] != pins[package]:
            raise RuntimeError(f"{package}=={pins[package]} required, found {versions[package]}")
    print("Python", sys.version.split()[0], versions, "numerical threads=1")


def summarize(truth, control, candidate):
    """Paired percentile interval for the MAPE difference, sampling whole market days."""
    baseline = metrics("demand", truth, control)
    holiday = metrics("demand", truth, candidate)
    actual = truth["nsw_demand_mw"].to_numpy()
    change = 100 * (np.abs(candidate - actual) - np.abs(control - actual)) / actual
    daily = change.reshape(-1, dayahead.N_PERIODS).mean(axis=1)
    n = len(daily)
    draws = np.random.default_rng(BOOTSTRAP_SEED).multinomial(n, np.full(n, 1 / n), size=BOOTSTRAP_DRAWS)
    low, high = np.percentile(draws @ daily / n, [2.5, 97.5])
    return {"days": n, "rows": len(actual), "control_mape_pct": baseline["MAPE"],
            "holiday_mape_pct": holiday["MAPE"], "delta_mape_pp": holiday["MAPE"] - baseline["MAPE"],
            "delta_mape_ci95_low_pp": low, "delta_mape_ci95_high_pp": high,
            "control_mae_mw": baseline["MAE"], "holiday_mae_mw": holiday["MAE"],
            "delta_mae_mw": holiday["MAE"] - baseline["MAE"]}


def run(data_path, frozen=False):
    check_environment()
    if frozen:
        train_end = FROZEN_CUTOFF
        eval_start, eval_end = REPLAY_START, REPLAY_END
        expected_rows = (86304, 207456)
    else:
        train_end = VALIDATION_CUTOFF
        eval_start, eval_end = pd.Timestamp("2014-01-01"), pd.Timestamp("2014-12-31")
        expected_rows = (68784, 17520)
    data = load_study(data_path, last_day=eval_end)
    days = dayahead.market_day(data.index)
    train = (days >= TRAIN_START) & (days <= train_end)
    valid = (days >= eval_start) & (days <= eval_end)
    if (train.sum(), valid.sum()) != expected_rows:
        raise ValueError(f"Expected {expected_rows} training/evaluation slots")
    y = data["nsw_demand"].rename("nsw_demand_mw")
    if not (y.loc[train | valid] > 0).all():
        raise ValueError("MAPE requires positive NSW demand in the course dataset")
    truth = y.loc[valid].to_frame()
    recipes = selected_recipes()
    rows = []
    with threadpool_limits(limits=1):
        features = build_features(data)
        if not features.index.equals(data.index) or not np.isfinite(features.loc[train | valid].to_numpy()).all():
            raise ValueError("Expected aligned features with finite values after warm-up")
        flags = features[HOLIDAY_COLUMN].to_numpy().reshape(-1, dayahead.N_PERIODS)
        if not np.isin(flags, [0, 1]).all() or not (flags == flags[:, :1]).all():
            raise ValueError("The public-holiday flag must be binary and constant across a market day")
        holiday_slots = features.loc[valid, HOLIDAY_COLUMN].to_numpy(dtype=bool)
        if not holiday_slots.any() or holiday_slots.all():
            raise ValueError("Expected both holiday and non-holiday days in the evaluation period")
        groups = {"all": np.ones(len(truth), dtype=bool), "holiday": holiday_slots,
                  "non_holiday": ~holiday_slots}
        for name in ("ridge", "boost_demand"):
            recipe = recipes[name]
            base_columns = recipe["columns"]
            if HOLIDAY_COLUMN in base_columns:
                raise ValueError(f"{name} already includes {HOLIDAY_COLUMN} in the control")
            predictions = []
            for cols in (base_columns, [*base_columns, HOLIDAY_COLUMN]):
                predictions.append(predict_raw(name, recipe["parameter"], features.loc[train, cols],
                                               y.loc[train], features.loc[valid, cols]))
            control, candidate = predictions
            for subset, mask in groups.items():
                score = summarize(truth.loc[mask], control[mask], candidate[mask])
                selected = ""
                if not frozen and subset == "all":
                    delta = score["delta_mape_pp"]
                    ci_spans_zero = score["delta_mape_ci95_low_pp"] <= 0 <= score["delta_mape_ci95_high_pp"]
                    near_tie = abs(delta) <= TOLERANCE["demand"] and ci_spans_zero
                    selected = "with_holiday" if delta < 0 and not near_tie else "control"
                rows.append({"model": name, "control_config": recipe["config"], "subset": subset,
                             "control_columns": len(base_columns), "holiday_columns": len(base_columns) + 1,
                             "train_start": TRAIN_START, "train_end": train_end, "train_rows": int(train.sum()),
                             "evaluation": "frozen_replay" if frozen else "2014_selection",
                             "evaluation_start": eval_start.date(), "evaluation_end": eval_end.date(),
                             **score, "annual_selected": selected})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "nem_weather_halfhourly_2010_2026.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "baseline" / "holiday_results",
                        help="Directory for metrics.csv (selection) or frozen_metrics.csv (replay)")
    parser.add_argument("--frozen", action="store_true",
                        help="Refit through 2014 and evaluate both recipes in 2015–2026 with no selection")
    args = parser.parse_args()
    scores = run(args.data, frozen=args.frozen)
    filename = "frozen_metrics.csv" if args.frozen else "metrics.csv"
    args.output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output) as directory:
        staging = Path(directory) / filename
        scores.to_csv(staging, index=False)
        staging.replace(args.output / staging.name)
    print(scores.to_string(index=False))
    print(f"Saved results to {args.output / filename}")


if __name__ == "__main__":
    main()
