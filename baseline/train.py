"""Fit, validate and save the four selected NSW day-ahead models."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from ablation.features import build_targets, load_study
from ablation.models import SPECS, TARGET, make_estimator
from baseline.evaluation import score_period
from baseline.predict import predict_models
from harness import dayahead
from src.common.features import build_features


ROOT = Path(__file__).resolve().parents[1]
TRAIN_START = "2010-01-29"
VALIDATION_CUTOFF = "2013-12-31"
FROZEN_CUTOFF = "2014-12-31"
OUTPUT_FILES = {"validation.joblib", "frozen.joblib", "validation_metrics.csv"}


def selected_recipes():
    rows = pd.read_csv(ROOT / "ablation" / "results" / "selection.csv").set_index("model")
    if not rows.index.is_unique or set(rows.index) != set(SPECS):
        raise ValueError("Expected one selected recipe for each of the four models")
    recipes = {}
    for name, spec in SPECS.items():
        row = rows.loc[name]
        columns = json.loads(row["columns"])
        if row["task"] != spec.task or len(set(columns)) != len(columns) or len(columns) != row["n_features"]:
            raise ValueError(f"Inconsistent task or feature list for {name}")
        threshold = None
        if spec.task == "price":
            threshold = float(row["threshold"])
            if not 0 < threshold < 1:
                raise ValueError(f"Invalid classifier threshold for {name}")
        recipes[name] = {"config": row["config"], "task": spec.task, "columns": columns,
                         "parameter": float(row["parameter"]), "threshold": threshold}
    return recipes


def fit_bundle(features, truth, recipes, last_day):
    days = dayahead.market_day(features.index)
    use = (days >= TRAIN_START) & (days <= last_day)
    models = {}
    for name, recipe in recipes.items():
        X = features.loc[use, recipe["columns"]]
        y = truth.loc[use, TARGET[recipe["task"]]]
        if not np.isfinite(X.to_numpy()).all() or not np.isfinite(y.to_numpy()).all():
            raise ValueError(f"Non-finite training data for {name}")
        estimator = make_estimator(name, recipe["parameter"]).fit(X, y)
        models[name] = {**recipe, "estimator": estimator}
    return {"train_first_day": TRAIN_START, "train_last_day": last_day, "train_rows": int(use.sum()),
            "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                         "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
            "models": models}


def publish(staged, output):
    # Keep a complete generation together; never overwrite the last good weights.
    lock = output.parent / f".{output.name}.publishing"
    previous = None
    # The OS releases this lock even on process termination. Keep the lock file at
    # a stable path so concurrent publishers always lock the same file.
    with lock.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.is_symlink():
            raise ValueError(f"Choose a regular output directory: {output}")
        if output.exists():
            if not output.is_dir() or any(p.name not in OUTPUT_FILES or p.is_symlink() or not p.is_file()
                                          for p in output.iterdir()):
                raise ValueError(f"Output contains unrelated files, choose another --output: {output}")
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            previous = output.with_name(f"{output.name}.backup-{stamp}")
            output.rename(previous)
        try:
            staged.rename(output)
        except BaseException:
            if previous is not None and not output.exists():
                previous.rename(output)
            raise
    if previous is not None:
        print(f"Previous training output retained at {previous}")


def train(data_path, output):
    data = load_study(data_path)  # reads only through the final interval of market year 2014
    features = build_features(data)
    truth = build_targets(data, features)
    valid = dayahead.market_day(data.index).year == 2014
    recipes = selected_recipes()
    output = Path(output).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".baseline-training-", dir=output.parent) as temporary, threadpool_limits(limits=1):
        staged = Path(temporary) / "outputs"
        staged.mkdir()
        for label, end in (("validation", VALIDATION_CUTOFF), ("frozen", FROZEN_CUTOFF)):
            bundle = fit_bundle(features, truth, recipes, end)
            path = staged / f"{label}.joblib"
            joblib.dump(bundle, path, compress=3)
            restored = joblib.load(path)
            if label == "validation":
                forecasts = predict_models(restored, features.loc[valid])
                scores = pd.DataFrame(score_period(features.loc[valid], truth.loc[valid], forecasts))
                scores.insert(0, "validation_rows", int(valid.sum()))
                scores.to_csv(staged / "validation_metrics.csv", index=False)
            else:
                # Smoke-test all four restored estimators before publishing the frozen weights.
                predict_models(restored, features.iloc[-dayahead.N_PERIODS:])
            print(f"{label}: fitted and reloaded four models through {end} on {bundle['train_rows']} slots", flush=True)
        publish(staged, output)
    print(scores.to_string(index=False))
    print(f"Saved training outputs to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "nem_weather_halfhourly_2010_2026.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "baseline" / "outputs")
    args = parser.parse_args()
    train(args.data, args.output)


if __name__ == "__main__":
    main()
