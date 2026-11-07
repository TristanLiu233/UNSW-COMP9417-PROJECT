"""Information available before each target market day, for the 2014 ablation study."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

from harness import dayahead


SLOTS = dayahead.N_PERIODS
PRICE_GAP = "lag1_price_gap"
NSW_MARKET = ("nsw_demand", "nsw_price")
NEIGHBOURS = ("qld_demand", "qld_price", "vic_demand", "vic_price")
CORE = ["slot_sin", "slot_cos", "season_sin", "season_cos", "weekend",
        *(f"weekday_{i}" for i in range(7)),
        *(f"{name}_lag{lag}" for lag in (1, 7) for name in NSW_MARKET),
        "prev_price_mean", PRICE_GAP]
WEATHER = [name for name in dayahead.WEATHER_COLS if name.startswith("syd_")]
REGION = [f"{name}_lag{lag}" for lag in (1, 7) for name in NEIGHBOURS]
EXTRA_WEATHER = [name for name in dayahead.WEATHER_COLS if name not in WEATHER]
ROLLING = {window: [f"{name}_{stat}{window}" for name in NSW_MARKET for stat in ("mean", "std")]
           for window in (7, 28)}
PRICE_SPIKES = ["prev_price_max", "prev_negative_slots"]
DAY_TYPE = [*(f"prev_day_type_{i}" for i in range(4)), "prev_day_type_distance"]


def load_study(path):
    """Read only the study prefix and reject a missing, duplicated or unordered slot."""
    expected = pd.date_range("2010-01-01 00:30", "2015-01-01 00:00", freq="30min")
    columns = [*dayahead.MARKET_COLS, *dayahead.WEATHER_COLS]
    data = pd.read_csv(path, nrows=len(expected), usecols=["timestamp", *columns],
                       parse_dates=["timestamp"], index_col="timestamp")
    if not data.index.equals(expected):
        raise ValueError("Expected complete 30-minute intervals from 2010-01-01 00:30 to 2015-01-01 00:00")
    if not np.isfinite(data[columns].to_numpy(dtype=float)).all():
        raise ValueError("A released market or weather column contains missing or non-finite values")
    return data


def _profiles(data, days, name):
    """One row per market day and one column per interval, in end-timestamp order."""
    return pd.DataFrame(data[name].to_numpy().reshape(-1, SLOTS), index=days.unique())


def _fit_day_types(demand_profiles, train_end):
    training = demand_profiles.loc[:pd.Timestamp(train_end)]
    if len(training) < 4 or not (training.mean(axis=1) > 0).all():
        raise ValueError("Day types need at least four complete training profiles with positive mean demand")
    normalized = training.div(training.mean(axis=1), axis=0)
    model = make_pipeline(PCA(n_components=0.95, svd_solver="full"),
                          KMeans(n_clusters=4, random_state=17, n_init=10))
    return model.fit(normalized)


def _construct(data, days, day_type_model):
    starts = data.index - pd.Timedelta(minutes=30)
    slot_angle = 2 * np.pi * (starts.hour * 2 + starts.minute // 30) / SLOTS
    season_angle = 2 * np.pi * (days.dayofyear - 1) / np.where(days.is_leap_year, 366, 365)
    columns = {"slot_sin": np.sin(slot_angle), "slot_cos": np.cos(slot_angle),
               "season_sin": np.sin(season_angle), "season_cos": np.cos(season_angle),
               "weekend": (days.dayofweek >= 5).astype(int)}
    for weekday in range(7):
        columns[f"weekday_{weekday}"] = (days.dayofweek == weekday).astype(int)
    for lag in (1, 7):
        for name in (*NSW_MARKET, *NEIGHBOURS):
            columns[f"{name}_lag{lag}"] = data[name].shift(lag * SLOTS)
    for name in dayahead.WEATHER_COLS:
        columns[name] = data[name]

    # Keep the same reduction as dayahead.targets (Series.mean) even in price ties.
    prices_by_day = data["nsw_price"].groupby(days)
    previous_mean = prices_by_day.agg(lambda day_prices: day_prices.mean()).shift(1)
    columns["prev_price_mean"] = days.map(previous_mean)
    columns[PRICE_GAP] = columns["nsw_price_lag1"] - columns["prev_price_mean"]
    columns["prev_price_max"] = days.map(prices_by_day.max().shift(1))
    negative_count = data["nsw_price"].lt(0).groupby(days).sum()
    columns["prev_negative_slots"] = days.map(negative_count.shift(1))

    for name in NSW_MARKET:
        prior_profiles = _profiles(data, days, name).shift(1)
        for window in ROLLING:
            rolling = prior_profiles.rolling(window, min_periods=window)
            columns[f"{name}_mean{window}"] = rolling.mean().to_numpy().ravel()
            columns[f"{name}_std{window}"] = rolling.std(ddof=0).to_numpy().ravel()

    # At target T the last visible profile is T-1. No T value participates in its type.
    prior_demand = _profiles(data, days, "nsw_demand").shift(1).iloc[1:]
    prior_demand = prior_demand.div(prior_demand.mean(axis=1), axis=0)
    distances = day_type_model.transform(prior_demand)
    memberships = distances.argmin(axis=1)
    for cluster, name in enumerate(DAY_TYPE[:-1]):
        columns[name] = days.map(pd.Series((memberships == cluster).astype(float), index=prior_demand.index))
    columns[DAY_TYPE[-1]] = days.map(pd.Series(distances.min(axis=1), index=prior_demand.index))

    inputs = pd.DataFrame(columns, index=data.index)
    truth = pd.DataFrame({"nsw_demand_mw": data["nsw_demand"],
                          "nsw_price_up": (data["nsw_price"] > inputs["prev_price_mean"]).astype(int)})
    return inputs, truth


def build_features(data, day_type_train_end="2013-12-31"):
    """Return all feature blocks and targets; the initial 28 days are for lag warm-up."""
    cutoff = pd.Timestamp(day_type_train_end)
    if cutoff > pd.Timestamp("2013-12-31"):
        raise ValueError("The 2014 validation year cannot be used to fit the day-type model")
    days = dayahead.market_day(data.index)
    day_type_model = _fit_day_types(_profiles(data, days, "nsw_demand"), cutoff)
    return _construct(data, days, day_type_model)


def validate_study(data, inputs, truth):
    """Verify eligible features, all 2014 official targets/baselines, and as-of consistency."""
    days = dayahead.market_day(data.index)
    expected_columns = CORE + WEATHER + REGION + EXTRA_WEATHER + PRICE_SPIKES + DAY_TYPE
    expected_columns += ROLLING[7] + ROLLING[28]
    if set(inputs.columns) != set(expected_columns) or not inputs.columns.is_unique:
        raise ValueError("Feature catalog and constructed columns differ")
    if not data.index.equals(inputs.index) or not data.index.equals(truth.index):
        raise ValueError("Input or target index differs from the released timestamps")
    if not np.isfinite(inputs.loc[days >= "2010-01-29"].to_numpy()).all():
        raise ValueError("Non-finite features after the common 28-day warm-up")

    predictions = {1: [], 7: []}
    for day in days[days.year == 2014].unique():
        block = data.loc[day - pd.Timedelta(days=7) + pd.Timedelta(minutes=30):day + pd.Timedelta(days=1)]
        history, weather = dayahead.make_inputs(block, day)
        target = dayahead.targets(block, day)
        pd.testing.assert_frame_equal(truth.loc[target.index], target)
        rows = inputs.loc[weather.index]
        for lag, parts in predictions.items():
            official = dayahead.baseline(history, weather, lag_days=lag)
            reconstructed = pd.DataFrame({"nsw_demand_mw": rows[f"nsw_demand_lag{lag}"],
                "nsw_price_up": (rows[f"nsw_price_lag{lag}"] > rows["prev_price_mean"]).astype(int)})
            pd.testing.assert_frame_equal(reconstructed, official)
            parts.append(official)

    day_type_model = _fit_day_types(_profiles(data, days, "nsw_demand"), "2013-12-31")
    for day in ("2014-01-01", "2014-07-01", "2014-12-31"):
        history, weather = dayahead.make_inputs(data, day)
        as_of = pd.concat([history, weather])
        for value in (np.nan, 123456.0):
            as_of.loc[weather.index, dayahead.MARKET_COLS] = value
            replay, _ = _construct(as_of, dayahead.market_day(as_of.index), day_type_model)
            pd.testing.assert_frame_equal(replay.loc[weather.index], inputs.loc[weather.index])
    return {lag: pd.concat(parts) for lag, parts in predictions.items()}
