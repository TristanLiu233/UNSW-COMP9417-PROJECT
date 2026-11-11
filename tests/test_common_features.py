"""Run with python -m unittest discover -s tests -v. Set COMP9417_DATA to also test the released CSV."""

import json
import os
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from harness import dayahead
from src.common.features import build_features, build_features_for_day


ROOT = Path(__file__).resolve().parents[1]
SLOTS = 48
HOLIDAY = "is_nsw_public_holiday"


def assert_same_features(actual, expected):
    pd.testing.assert_frame_equal(actual, expected, check_exact=True, check_freq=False)
    # Exact numeric equality above treats +0.0 and -0.0 as equal. Compare the bits too.
    for name in expected.select_dtypes(include="floating"):
        if actual[name].to_numpy().tobytes() != expected[name].to_numpy().tobytes():
            raise AssertionError(f"Float representation differs in {name} on {actual.index[0]}")


def check_days(data, batch, dates):
    for date in dates:
        history, weather = dayahead.make_inputs(data, date)
        actual = build_features_for_day(history, weather)
        pd.testing.assert_index_equal(actual.index, weather.index, exact=True)
        assert_same_features(actual, batch.loc[weather.index])


def check_future_market_shuffle(data, batch, day):
    day = pd.Timestamp(day)
    future = dayahead.market_day(data.index) >= day
    altered = data.copy()
    source = altered.loc[future, dayahead.MARKET_COLS].to_numpy(copy=True)
    altered.loc[future, dayahead.MARKET_COLS] = np.random.default_rng(17).permutation(source)
    changed = build_features(altered)
    _, today = dayahead.make_inputs(data, day)
    assert_same_features(changed.loc[today.index], batch.loc[today.index])
    _, tomorrow = dayahead.make_inputs(data, day + pd.Timedelta(days=1))
    lag = changed.loc[tomorrow.index, "nsw_demand_lag1"]
    np.testing.assert_array_equal(lag, altered.loc[today.index, "nsw_demand"])
    if lag.equals(batch.loc[tomorrow.index, "nsw_demand_lag1"]):
        raise AssertionError("Shuffling today's market observations did not change tomorrow's lag")


class FeatureContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        index = pd.date_range("2021-11-01 00:30", periods=95 * SLOTS, freq="30min", name="timestamp")
        rng = np.random.default_rng(9417)
        columns = dayahead.MARKET_COLS + dayahead.WEATHER_COLS
        data = pd.DataFrame(rng.normal(size=(len(index), len(columns))), index=index, columns=columns)
        for name in ("nsw_demand", "qld_demand", "vic_demand"):
            data[name] = 5000 + 250 * data[name]
        cls.data = data
        cls.batch = build_features(data)

    def test_schema_warmup_and_source_values(self):
        recipes = pd.read_csv(ROOT / "ablation/results/selection.csv")["columns"]
        expected = list(dict.fromkeys(col for recipe in recipes for col in json.loads(recipe))) + [HOLIDAY]
        self.assertEqual(len(expected), 36)
        self.assertEqual(self.batch.columns.tolist(), expected)
        for name in expected:
            dtype = "float64"
            if name == HOLIDAY:
                dtype = "int8"
            elif name == "weekend" or name.startswith("weekday_"):
                dtype = "int64"
            self.assertEqual(self.batch[name].dtype, dtype, name)
        pd.testing.assert_index_equal(self.data.index, self.batch.index, exact=True)
        self.assertTrue(self.batch["nsw_demand_mean28"].iloc[:28 * SLOTS].isna().all())
        self.assertTrue(np.isfinite(self.batch.iloc[28 * SLOTS:].to_numpy()).all())

        history, weather = dayahead.make_inputs(self.data, "2021-12-27")
        rows = self.batch.loc[weather.index]
        yesterday = history.iloc[-SLOTS:]
        np.testing.assert_array_equal(rows["prev_price_mean"], np.repeat(yesterday["nsw_price"].mean(), SLOTS))
        np.testing.assert_array_equal(rows["lag1_price_gap"], yesterday["nsw_price"] - yesterday["nsw_price"].mean())
        for name in rows.filter(regex=r"_lag[17]$"):
            source, lag = name.rsplit("_lag", 1)
            values = history[source].to_numpy().reshape(-1, SLOTS)
            np.testing.assert_array_equal(rows[name], values[-int(lag)])
        for name in ("syd_temp", "syd_rh", "syd_cloud", "syd_solar"):
            np.testing.assert_array_equal(rows[name], weather[name])
        for name in ("nsw_demand", "nsw_price"):
            for window in (7, 28):
                past = history[name].to_numpy().reshape(-1, SLOTS)[-window:]
                np.testing.assert_allclose(rows[f"{name}_mean{window}"], past.mean(axis=0), rtol=0, atol=1e-8)
                np.testing.assert_allclose(rows[f"{name}_std{window}"], past.std(axis=0, ddof=0), rtol=0, atol=1e-8)

    def test_batch_and_day_match_on_seeded_days_and_boundaries(self):
        eligible = dayahead.market_day(self.data.index).unique()[28:]
        picked = np.random.default_rng(17).choice(eligible, 5, replace=False)
        check_days(self.data, self.batch, [*picked, "2021-11-29", "2021-12-27", "2022-01-01"])

    def test_market_midnight_and_observed_public_holiday(self):
        # Christmas 2021 fell on Saturday. Monday 27 December was an additional NSW public holiday.
        for day, flag in (("2021-12-24", 0), ("2021-12-25", 1), ("2021-12-27", 1), ("2021-12-29", 0)):
            _, weather = dayahead.make_inputs(self.data, day)
            np.testing.assert_array_equal(self.batch.loc[weather.index, HOLIDAY], np.repeat(flag, SLOTS))
        self.assertEqual(self.batch.at[pd.Timestamp("2021-12-25 00:00"), HOLIDAY], 0)
        self.assertEqual(self.batch.at[pd.Timestamp("2021-12-25 00:30"), HOLIDAY], 1)

    def test_future_market_values_cannot_change_target_day(self):
        check_future_market_shuffle(self.data, self.batch, "2021-12-27")

    def test_reject_broken_market_days(self):
        history, weather = dayahead.make_inputs(self.data, "2021-12-27")
        with self.assertRaises(ValueError):
            build_features_for_day(history.tail(27 * SLOTS), weather)
        with self.assertRaises(ValueError):
            build_features_for_day(history, weather.iloc[1:])
        with self.assertRaises(ValueError):
            build_features_for_day(history, weather.shift(freq="30min"))
        with self.assertRaises(ValueError):
            build_features(self.data.drop(self.data.index[175]))


class ReleasedDataTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("COMP9417_DATA"), "set COMP9417_DATA to the released CSV")
    def test_17_seeded_years_and_six_boundary_days_match_bit_for_bit(self):
        data = pd.read_csv(os.environ["COMP9417_DATA"], parse_dates=["timestamp"], index_col="timestamp")
        batch = build_features(data)
        eligible = dayahead.market_day(data.index).unique()[28:]
        self.assertEqual(eligible[0], pd.Timestamp("2010-01-29"))
        self.assertEqual(eligible[-1], pd.Timestamp("2026-10-31"))
        edges = pd.to_datetime(["2010-01-29", "2014-06-15", "2015-01-01", "2020-02-29",
                                "2021-12-27", "2026-10-31"])
        candidates = eligible.difference(edges)
        rng = np.random.default_rng(17)
        sample = [rng.choice(candidates[candidates.year == year]) for year in range(2010, 2027)]
        check_days(data, batch, [*edges, *sample])
        check_future_market_shuffle(data, batch, "2014-06-15")


if __name__ == "__main__":
    unittest.main()
