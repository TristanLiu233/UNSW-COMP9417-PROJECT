"""Run the three-round ablation study from the repository root with python -m ablation.run."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
import pandas as pd
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from harness import dayahead
from .features import load_study, build_features, validate_study
from .finalists import write_finalists
from .models import SPECS, PRIMARY, TARGET
from .scoring import metrics


ROOT = Path(__file__).resolve().parents[1]
TABLES = ("scores", "contrasts", "tuning", "selection")
STUDY_NAME = "comp9417-three-round-ablation"


def validate_and_write(data_path, destination):
    data = load_study(data_path)
    inputs, truth = build_features(data)
    baselines = validate_study(data, inputs, truth)
    days = dayahead.market_day(data.index)
    train_days = days[(days >= "2010-01-29") & (days < "2014-01-01")]
    valid_days = days[days.year == 2014]
    validation = truth.loc[days.year == 2014].copy()
    rows = []
    for lag, predicted in baselines.items():
        for task, target in TARGET.items():
            validation[f"baseline{lag}_{task}"] = predicted[target]
            rows.append({"method": f"lag{lag}", "task": task,
                         **metrics(task, validation, predicted[target])})
    validation.to_csv(destination / "validation.csv.gz", index_label="timestamp",
                      compression={"method": "gzip", "mtime": 0, "compresslevel": 3})
    pd.DataFrame(rows).to_csv(destination / "baselines.csv", index=False)

    # Identify exactly the release prefix read for the experiment, including its CSV header.
    digest = hashlib.sha256()
    with data_path.open("rb") as source:
        for _ in range(len(data) + 1):
            digest.update(source.readline())
    return {"data_filename": data_path.name, "csv_prefix_sha256": digest.hexdigest(),
            "last_interval_end": str(data.index[-1]), "warmup_days": 28,
            "train_first_day": str(train_days.min().date()), "train_last_day": str(train_days.max().date()),
            "train_days": train_days.nunique(), "train_rows": len(train_days),
            "validation_first_day": str(valid_days.min().date()), "validation_last_day": str(valid_days.max().date()),
            "validation_days": valid_days.nunique(), "validation_rows": len(valid_days)}


def fit_worker(name, data_path, model_dir, threads):
    # Import in the worker so Windows spawn does not import or start any jobs at module load.
    from .study import run_model

    start = time.perf_counter()
    print(f"Starting {name}", flush=True)
    with threadpool_limits(limits=threads):
        run_model(name, load_study(data_path), model_dir)
    elapsed = round(time.perf_counter() - start, 1)
    print(f"Completed {name} in {elapsed} s", flush=True)
    return elapsed


def aggregate(models_dir, destination):
    predictions = destination / "predictions"
    predictions.mkdir()
    for name in SPECS:
        (models_dir / name / "predictions.csv.gz").replace(predictions / f"{name}.csv.gz")
    for table in TABLES:
        parts = []
        for name in SPECS:
            part = pd.read_csv(models_dir / name / f"{table}.csv", dtype=str, keep_default_na=False)
            if "model" not in part.columns:
                part.insert(0, "model", name)
            parts.append(part)
        pd.concat(parts, ignore_index=True).to_csv(destination / f"{table}.csv", index=False)


def check_destination(output):
    """Refuse to replace unrelated files in an existing results directory."""
    if output.resolve() == ROOT or output.resolve() in ROOT.parents or output.is_symlink():
        raise ValueError(f"Unsafe output destination: {output}")
    if not output.exists():
        return
    if not output.is_dir():
        raise ValueError(f"Output must be a directory: {output}")
    entries = {p.name for p in output.iterdir()}
    if not entries:
        return
    expected = {"run.json", "predictions", "validation.csv.gz", "baselines.csv", "task_winners.csv",
                "final_comparisons.csv", *(f"{t}.csv" for t in TABLES)}
    if not entries.issubset(expected) or "run.json" not in entries:
        raise ValueError(f"Output contains files from outside this experiment; use another --output: {output}")
    for path in output.iterdir():
        if path.is_symlink() or (path.is_dir() if path.name != "predictions" else not path.is_dir()):
            raise ValueError(f"Unexpected result file type: {path}")
    if json.loads((output / "run.json").read_text(encoding="utf-8")).get("study") != STUDY_NAME:
        raise ValueError(f"Cannot replace a directory from another study: {output}")
    if "predictions" in entries:
        expected_files = {f"{name}.csv.gz" for name in SPECS}
        if any(p.is_symlink() or not p.is_file() or p.name not in expected_files for p in (output / "predictions").iterdir()):
            raise ValueError(f"The predictions directory contains unrelated files: {output}")


def publish(staged, output, temporary):
    # The short-lived lock serializes concurrent publishers. Workers operate on private temporary paths.
    lock = output.parent / f".{output.name}.publishing"
    with lock.open("x", encoding="utf-8"):
        pass
    try:
        check_destination(output)
        backup = temporary / "previous"
        if output.exists():
            output.rename(backup)
        try:
            staged.rename(output)
        except OSError:
            if backup.exists():
                backup.rename(output)
            raise
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "nem_weather_halfhourly_2010_2026.csv",
                        help="Path to the released half-hourly CSV")
    parser.add_argument("--output", type=Path, default=ROOT / "ablation" / "results")
    parser.add_argument("--jobs", type=int, choices=range(1, len(SPECS) + 1), default=2,
                        help="Number of model workers (default 2)")
    parser.add_argument("--threads", type=int, default=1, help="Numerical library threads per worker (default 1)")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    data_path = args.data.expanduser().resolve()
    if not data_path.is_file():
        parser.error(f"Released CSV not found: {data_path}. Pass --data to select it.")
    output = args.output.expanduser().absolute()
    check_destination(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".ablation-", dir=output.parent) as directory:
        temporary = Path(directory)
        staged = temporary / "results"
        staged.mkdir()
        print("Validating released data, all 2014 targets, both official baselines and feature cutoffs", flush=True)
        with threadpool_limits(limits=args.threads):
            metadata = validate_and_write(data_path, staged)

        models_dir = temporary / "models"
        for name in SPECS:
            (models_dir / name).mkdir(parents=True)
        elapsed = {}
        if args.jobs == 1:
            for name in SPECS:
                elapsed[name] = fit_worker(name, data_path, models_dir / name, args.threads)
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                futures = {pool.submit(fit_worker, name, data_path, models_dir / name, args.threads): name
                           for name in SPECS}
                for future in as_completed(futures):
                    elapsed[futures[future]] = future.result()

        aggregate(models_dir, staged)
        write_finalists(staged)
        metadata.update(study=STUDY_NAME, model_seed=17, bootstrap_seed=17, bootstrap_repetitions=2000,
                        bootstrap_blocks_days=[1, 7], primary_metrics=PRIMARY,
                        classification_thresholds=[0.40, 0.45, 0.50, 0.55, 0.60],
                        tuning_validation_folds=[["2013-01-01", "2013-06-30"], ["2013-07-01", "2013-12-31"]],
                        candidate_grids={name: list(spec.grid) for name, spec in SPECS.items()},
                        jobs=args.jobs, threads_per_worker=args.threads, worker_seconds=elapsed,
                        versions={"python": platform.python_version(), "numpy": np.__version__,
                                  "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__})
        (staged / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        publish(staged, output, temporary)
    print(f"Results written to {output}", flush=True)


if __name__ == "__main__":
    main()
