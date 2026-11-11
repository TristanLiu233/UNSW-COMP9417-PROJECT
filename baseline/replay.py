"""Score the four frozen models and both persistence rules by market month and year."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd

from ablation.features import build_targets, load_study
from baseline.evaluation import score_period
from baseline.predict import predict_models
from harness import dayahead
from src.common.features import build_features


FROZEN_END = pd.Timestamp("2014-12-31")
REPLAY_START = pd.Timestamp("2015-01-01")
REPLAY_END = pd.Timestamp("2026-10-31")
PACKAGE_DIR = Path(__file__).resolve().parent
HALF_HOUR = pd.Timedelta(minutes=30)


def score_periods(inputs, truth, predicted):
    """Report pooled scores, including partial 2026 and the last slot of each market day."""
    if not truth.index.equals(inputs.index) or not predicted.index.equals(inputs.index):
        raise ValueError("Forecast, input and target timestamps must match exactly")
    days = dayahead.market_day(truth.index)
    periods = [("overall", "all", slice(0, len(truth)))]
    for grouping, frequency in (("year", "Y"), ("month", "M")):
        labels = days.to_period(frequency)
        for label in labels.unique():
            bounds = slice(labels.searchsorted(label), labels.searchsorted(label, side="right"))
            periods.append((grouping, str(label), bounds))

    rows = []
    for grouping, period, bounds in periods:
        actual = truth.iloc[bounds]
        period_scores = score_period(inputs.iloc[bounds], actual, predicted.iloc[bounds])
        for score in period_scores:
            rows.append({"grouping": grouping, "period": period,
                         "first_day": days[bounds.start].date(), "last_day": days[bounds.stop - 1].date(),
                         "days": len(actual) // dayahead.N_PERIODS, "half_hours": len(actual), **score})
    return pd.DataFrame(rows)


def replay(data_path, bundle_path, output):
    bundle = joblib.load(bundle_path)
    if pd.Timestamp(bundle["train_last_day"]) != FROZEN_END:
        raise ValueError("Replay expects frozen models fitted through market day 2014-12-31")
    data = load_study(data_path, last_day=REPLAY_END)
    inputs = build_features(data)
    truth = build_targets(data, inputs)
    expected = pd.date_range(REPLAY_START + HALF_HOUR, REPLAY_END + pd.Timedelta(days=1), freq="30min")
    inputs = inputs.loc[expected[0]:expected[-1]]
    truth = truth.loc[expected[0]:expected[-1]]
    if not inputs.index.equals(expected) or not truth.index.equals(expected):
        raise ValueError("Feature construction changed timestamps within the replay period")
    if not np.isfinite(inputs.to_numpy(dtype=float)).all() or not np.isfinite(truth.to_numpy()).all():
        raise ValueError("Replay features and targets must be finite")
    if (truth["nsw_demand_mw"] <= 0).any() or not truth["nsw_price_up"].isin([0, 1]).all():
        raise ValueError("Expected positive demand and binary price targets")
    predicted = predict_models(bundle, inputs)
    scores = score_periods(inputs, truth, predicted)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as temporary:
        csv_path = Path(temporary) / "metrics.csv"
        scores.to_csv(csv_path, index=False)
        csv_path.replace(output / csv_path.name)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Path to the released half-hourly CSV")
    parser.add_argument("--bundle", type=Path, default=PACKAGE_DIR / "outputs" / "frozen.joblib")
    parser.add_argument("--output", type=Path, default=PACKAGE_DIR / "replay_results")
    args = parser.parse_args()
    scores = replay(args.data, args.bundle, args.output)
    print(scores.loc[scores["grouping"] == "overall"].to_string(index=False))
    print(f"Saved {len(scores)} score rows to {args.output / 'metrics.csv'}")


if __name__ == "__main__":
    main()
