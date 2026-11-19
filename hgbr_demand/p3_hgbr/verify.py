"""Independent checks of the exported results and the official next-day interface."""
import argparse
from hashlib import file_digest
from importlib import util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .demand import (DATA_STOP, DAY, MARKET, REPLAY_START, SLOTS, VALID_START, WEATHER,
                     DemandData, DemandForecaster, build_features, market_days,
                     next_day_features, predict)
from .stage2 import MAX_RETRAINS


def official_helpers(kit):
    spec = util.spec_from_file_location("official_dayahead", kit / "harness" / "dayahead.py")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(kit, data_path, results):
    da = official_helpers(kit)
    frame = da.load_dataset(data_path)
    x = build_features(frame)
    stage1, stage2 = results / "stage1", results / "stage2"
    config = json.loads((stage1 / "p3_stage1_candidate.json").read_text())
    with open(data_path, "rb") as source:
        assert file_digest(source, "sha256").hexdigest() == config["dataset_sha256"]
    validated = pd.read_csv(stage1 / "validation_predictions.csv.gz", parse_dates=["timestamp"]).set_index("timestamp")
    scored = pd.read_csv(stage1 / "validation_2014.csv").set_index("model")
    dates = market_days(validated.index)
    assert len(validated) == 17_520 and dates.min() == VALID_START and dates.max() == REPLAY_START-DAY
    for name, row in scored.iterrows():
        np.testing.assert_allclose([row.mape_pct, row.mae_mw],
                                   [da.mape(validated.actual_mw, validated[name]), da.mae(validated.actual_mw, validated[name])],
                                   rtol=0, atol=1e-9)
    tuning = pd.read_csv(stage1 / "tuning_2014.csv")
    assert len(tuning) == 72 and tuning.groupby("target").size().to_dict() == {"level": 36, "residual_b1": 36}
    assert tuning.iloc[0]["target"] == config["selected"]["target"]
    # Independent sum for each score and matched official baseline on hand-picked boundary days.
    pred = pd.read_csv(stage2 / "half_hour_predictions.csv.gz", parse_dates=["timestamp","market_day"]).set_index("timestamp")
    assert len(pred) == 207_456 and pred.index.is_unique and pred.index.is_monotonic_increasing
    assert pred.index[0] == REPLAY_START + pd.Timedelta(minutes=30)
    assert pred.index[-1] == DATA_STOP and pred.market_day.iloc[-1] == DATA_STOP-DAY
    assert pred.groupby("market_day").size().eq(SLOTS).all()
    np.testing.assert_array_equal(pred.actual_demand_mw, frame.loc[pred.index, "nsw_demand"])
    np.testing.assert_array_equal(pred.b1, frame["nsw_demand"].shift(48).loc[pred.index])
    np.testing.assert_array_equal(pred.b7, frame["nsw_demand"].shift(336).loc[pred.index])
    summary = pd.read_csv(stage2 / "strategy_comparison.csv").set_index("model")
    for name, row in summary.iterrows():
        np.testing.assert_allclose([row.mape_pct, row.mae_mw],
                                   [da.mape(pred.actual_demand_mw, pred[name]), da.mae(pred.actual_demand_mw, pred[name])],
                                   rtol=0, atol=1e-9)
    monthly = pd.read_csv(stage2 / "monthly.csv")
    assert monthly.groupby("model").size().eq(142).all()
    assert (monthly.normal_ci_low <= monthly.normal_ci_high).all()
    assert (monthly.day_bootstrap_ci_low <= monthly.day_bootstrap_ci_high).all()
    updates = pd.read_csv(stage2 / "retraining_log.csv",
                          parse_dates=["alarm_day","first_prediction_day","train_first_day","train_last_day"])
    assert updates.groupby("strategy").size().le(MAX_RETRAINS).all()
    assert (updates.train_last_day == updates.first_prediction_day-DAY).all()
    triggered = updates.loc[updates.reason == "cusum"]
    assert (triggered.alarm_day == triggered.train_last_day).all()
    assert triggered.first_prediction_day.diff().dropna().dt.days.ge(365).all()

    frozen = DemandForecaster(stage1 / "frozen")
    final = DemandForecaster(stage2 / "final_artifacts")
    probe_days = sorted({pd.Timestamp(d) for d in
                         ("2014-01-01","2014-12-31","2015-01-01","2016-02-29","2021-10-01",
                          "2022-06-15","2025-01-01","2026-10-31")} | set(updates.first_prediction_day))
    for day in probe_days:
        history, weather = da.make_inputs(frame, day)
        baseline = da.baseline(history, weather, 1)["nsw_demand_mw"]
        b7 = da.baseline(history, weather, 7)["nsw_demand_mw"]
        scored_day = (validated if day < REPLAY_START else pred).loc[weather.index]
        np.testing.assert_array_equal(baseline, scored_day.b1)
        np.testing.assert_array_equal(b7, scored_day.b7)
        online = next_day_features(history, weather)
        np.testing.assert_allclose(online, x.loc[weather.index], atol=1e-5, rtol=1e-9)
        # Exercise the same bulk feature function after removing all target-day market observations.
        masked = frame.loc[history.index[-7*SLOTS]:weather.index[-1], [*MARKET,*WEATHER]].copy()
        for poison in (np.nan, 987654.0):
            masked.loc[weather.index, list(MARKET)] = poison
            pd.testing.assert_frame_equal(build_features(masked).loc[weather.index], online)
        if day >= REPLAY_START:
            prediction = frozen.predict(history, weather)
            np.testing.assert_allclose(prediction.nsw_demand_mw, scored_day.frozen, atol=1e-7, rtol=0)
        check = final.predict(history, weather)
        assert check.index.equals(weather.index) and list(check) == ["nsw_demand_mw"]
        assert check.nsw_demand_mw.between(0, 20_000).all()
    print(f"PASS: official scores, 72 fits, 142 months, 207456 replay slots and {len(probe_days)} official-interface/leakage probes")
    print("PASS: at most six updates per strategy, every fit stops before its first prediction day")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kit", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    a = p.parse_args()
    run(a.kit, a.data, a.results)


if __name__ == "__main__":
    main()
