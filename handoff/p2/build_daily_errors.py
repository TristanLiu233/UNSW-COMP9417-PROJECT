"""Rebuild and validate P3 frozen demand errors for P2's day-type analysis."""

import argparse
from io import BytesIO
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
P3_COMMIT = "4fab1bd2aa3978e0332c763a327d68d6b732cd9f"
P2_COMMIT = "f5aeff7acc0e9019eba321c6c92ce689ef139588"
P3_DIR = "hgbr_demand/results/stage2"
ASSIGNMENTS = "day_type_analysis/outputs/day_type_clustering/day_type_assignments_loyo_k2_pca95.csv"
START = pd.Timestamp("2015-01-01")
END = pd.Timestamp("2026-10-31")
STEP = pd.Timedelta(minutes=30)
SLOTS = 48


def git_csv(commit, path, **kwargs):
    result = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{path}"],
                            check=True, capture_output=True)
    compression = "gzip" if path.endswith(".gz") else None
    return pd.read_csv(BytesIO(result.stdout), compression=compression, **kwargs)


def build_daily_errors(data_path, p3_results=None, assignment_path=None):
    def p3_table(name, **kwargs):
        if p3_results is not None:
            return pd.read_csv(Path(p3_results) / name, **kwargs)
        return git_csv(P3_COMMIT, f"{P3_DIR}/{name}", **kwargs)

    expected_days = pd.date_range(START, END, freq="D")
    expected_slots = pd.date_range(START + STEP, END + pd.Timedelta(days=1), freq=STEP)
    predictions = p3_table("half_hour_predictions.csv.gz",
                          usecols=["timestamp", "market_day", "actual_demand_mw", "frozen", "frozen_residual_b1"],
                          parse_dates=["timestamp", "market_day"], index_col="timestamp")
    data = pd.read_csv(data_path, usecols=["timestamp", "nsw_demand"],
                       parse_dates=["timestamp"], index_col="timestamp")
    actual = data.loc[(data.index >= expected_slots[0]) & (data.index <= expected_slots[-1]), "nsw_demand"]
    if not actual.index.equals(expected_slots) or not predictions.index.equals(expected_slots):
        raise ValueError("Expected ordered, unique timestamps for all 207,456 released replay slots")
    days = (predictions.index - STEP).normalize()
    if not pd.DatetimeIndex(predictions["market_day"]).equals(days):
        raise ValueError("Published market days differ from the interval-start dates")
    np.testing.assert_array_equal(actual, predictions["actual_demand_mw"])
    np.testing.assert_array_equal(predictions["frozen"], predictions["frozen_residual_b1"])
    if not (actual > 0).all() or not np.isfinite(actual).all() or not np.isfinite(predictions["frozen"]).all():
        raise ValueError("Expected finite demand predictions and strictly positive demand observations")

    absolute = (actual - predictions["frozen"]).abs()
    daily = pd.DataFrame({"ape": 100 * absolute / actual, "ae": absolute}).groupby(days).agg(
        n_slots=("ape", "size"), frozen_mape_pct=("ape", "mean"), frozen_mae_mw=("ae", "mean"))
    daily.index.name = "market_day"
    if not daily.index.equals(expected_days) or not daily["n_slots"].eq(SLOTS).all():
        raise ValueError("Expected exactly 48 half-hours on each of the 4,322 market days")

    monitors = p3_table("daily_monitors.csv", index_col="market_day",
                       parse_dates=["market_day"])
    if not daily.index.equals(monitors.index):
        raise ValueError("Daily records do not align with the P3 monitoring dates")
    np.testing.assert_allclose(daily["frozen_mape_pct"], monitors["frozen_mape_pct"], rtol=0, atol=1e-10)
    monthly = p3_table("monthly.csv").query("model == 'frozen'").set_index("month")
    grouped = daily.groupby(daily.index.strftime("%Y-%m"))
    if not grouped.size().index.equals(monthly.index):
        raise ValueError("Monthly records do not align with the P3 replay months")
    np.testing.assert_array_equal(grouped.size(), monthly["n_days"])
    np.testing.assert_array_equal(grouped["n_slots"].sum(), monthly["n_slots"])
    np.testing.assert_allclose(grouped[["frozen_mape_pct", "frozen_mae_mw"]].mean(),
                               monthly[["mape_pct", "mae_mw"]], rtol=0, atol=1e-10)
    overall = p3_table("strategy_comparison.csv").set_index("model").loc["frozen"]
    np.testing.assert_equal(overall["retrains"], 0)
    np.testing.assert_allclose(daily[["frozen_mape_pct", "frozen_mae_mw"]].mean(),
                               overall[["mape_pct", "mae_mw"]].to_numpy(dtype=float), rtol=0, atol=1e-10)

    type_options = dict(usecols=["market_day", "day_type"], parse_dates=["market_day"], index_col="market_day")
    assignments = (pd.read_csv(assignment_path, **type_options) if assignment_path is not None
                   else git_csv(P2_COMMIT, ASSIGNMENTS, **type_options))
    labels = assignments.loc[(assignments.index >= START) & (assignments.index <= END)].sort_index()
    if not labels.index.equals(daily.index) or labels["day_type"].isna().any():
        raise ValueError("P2 day-type labels must cover all replay days exactly once")
    return daily


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Course half-hourly CSV")
    parser.add_argument("--output", type=Path, default=HERE / "frozen_demand_daily.csv")
    parser.add_argument("--p3-results", type=Path, help="Local copy of P3's pinned stage2 directory")
    parser.add_argument("--assignments", type=Path, help="Local copy of P2's pinned day-type CSV")
    args = parser.parse_args()
    daily = build_daily_errors(args.data, args.p3_results, args.assignments)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=args.output.parent, delete=False) as temp:
        staged = Path(temp.name)
    try:
        daily.to_csv(staged, date_format="%Y-%m-%d", lineterminator="\n")
        staged.replace(args.output)
    finally:
        staged.unlink(missing_ok=True)
    print(f"Validated {len(daily)} days, {daily.n_slots.sum()} slots, "
          f"MAPE {daily.frozen_mape_pct.mean():.6f}%, MAE {daily.frozen_mae_mw.mean():.6f} MW")


if __name__ == "__main__":
    main()
