"""Rebuild only the final demand artifact from the released CSV and locked configuration."""
import argparse
import json
from importlib.metadata import version
from pathlib import Path

from threadpoolctl import threadpool_limits

from .demand import DATA_STOP, fit_period, load_released, DemandData, save_model


def package_versions():
    return {p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "joblib")}


def train(path, config, artifact_dir):
    data = DemandData.from_frame(load_released(path))
    start = config["final_train_start"]
    # Match the Stage 3 fit exactly; threaded numerical reductions can change tree fits.
    with threadpool_limits(limits=1):
        estimator = fit_period(data, start, DATA_STOP, config)
    save_model(Path(artifact_dir) / "demand_hgbr.joblib", estimator, config,
               start, DATA_STOP, package_versions())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    args = parser.parse_args()
    train(args.data, json.loads(args.config.read_text()), args.artifact_dir)


if __name__ == "__main__":
    main()
