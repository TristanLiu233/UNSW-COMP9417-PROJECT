"""Select one forecasting model per task and compare it with its direct alternatives."""

from pathlib import Path

import numpy as np
import pandas as pd

from .models import PRIMARY, SPECS
from .scoring import Bootstrap, compare_samples, metrics
from .study import TOLERANCE


def write_finalists(result_dir):
    result_dir = Path(result_dir)
    selected = pd.read_csv(result_dir / "selection.csv")
    truth = pd.read_csv(result_dir / "validation.csv.gz", index_col="timestamp", parse_dates=["timestamp"])
    bootstrap = Bootstrap(truth)
    winners, comparisons = [], []
    for task, metric in PRIMARY.items():
        candidates = selected.loc[selected["task"] == task].set_index("model", drop=False)
        expected = {name for name, spec in SPECS.items() if spec.task == task}
        if set(candidates.index) != expected or not candidates.index.is_unique:
            raise ValueError(f"Expected one selected configuration per {task} model")
        samples, points = {}, {}
        for name, row in candidates.iterrows():
            config = row["config"]
            path = result_dir / "predictions" / f"{name}.csv.gz"
            predictions = pd.read_csv(path, usecols=["timestamp", config], index_col="timestamp",
                                      parse_dates=["timestamp"])[config]
            points[name] = metrics(task, truth, predictions)[metric]
            np.testing.assert_allclose(points[name], row[metric], rtol=0, atol=1e-8)
            samples[name] = bootstrap.samples(task, predictions)

        direction = 1 if task == "price" else -1
        best = max(points, key=lambda name: direction * points[name])
        eligible = [best]
        for name in points:
            if name != best and abs(points[name] - points[best]) <= TOLERANCE[task]:
                bounds = compare_samples(samples[name], samples[best])
                if bounds["day_low"] <= 0 <= bounds["day_high"]:
                    eligible.append(name)
        winner = min(eligible, key=lambda name: (candidates.at[name, "n_features"], -direction * points[name], name))
        winner_config = candidates.at[winner, "config"]
        winners.append({**candidates.loc[winner].to_dict(), "point_best_model": best,
                        "point_best_config": candidates.at[best, "config"],
                        "selection_reason": "best_point_score" if best == winner else "near_tie_fewer_features"})

        references = {name: candidates.at[name, "config"] for name in points if name != winner}
        for lag in (1, 7):
            name = f"lag{lag}"
            predictions = truth[f"baseline{lag}_{task}"]
            samples[name] = bootstrap.samples(task, predictions)
            points[name] = metrics(task, truth, predictions)[metric]
            references[name] = name
        for name, config in references.items():
            comparisons.append({"task": task, "metric": metric, "winner_model": winner, "winner_config": winner_config,
                                "reference": name, "reference_config": config, "winner_score": points[winner],
                                "reference_score": points[name], "delta_pp": points[winner] - points[name],
                                **compare_samples(samples[winner], samples[name])})

    pd.DataFrame(winners).to_csv(result_dir / "task_winners.csv", index=False)
    pd.DataFrame(comparisons).to_csv(result_dir / "final_comparisons.csv", index=False)
