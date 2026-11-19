"""One feature implementation for fitting, backtesting, and live demand prediction."""
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

SLOTS = 48
STEP = pd.Timedelta(minutes=30)
DAY = pd.Timedelta(days=1)
START = pd.Timestamp("2010-01-01 00:30")
WARM = pd.Timestamp("2010-01-29")  # shared 28-day warmup, including the Ridge comparison
VALID_START = pd.Timestamp("2014-01-01")
REPLAY_START = pd.Timestamp("2015-01-01")
DATA_STOP = pd.Timestamp("2026-11-01")  # exclusive market day, inclusive interval end
MARKET = ("nsw_demand", "nsw_price", "vic_demand", "vic_price")
WEATHER = ("syd_temp", "syd_rh", "syd_cloud", "syd_solar")
FEATURES = (
    "slot_sin", "slot_cos", "season_sin", "season_cos", "weekend",
    *(f"weekday_{i}" for i in range(7)),
    "nsw_demand_lag1", "nsw_price_lag1", "nsw_demand_lag7", "nsw_price_lag7",
    "prev_price_mean", "lag1_price_gap", *WEATHER,
    "vic_demand_lag1", "vic_demand_lag7", "vic_price_lag7",
    "nsw_demand_mean7", "nsw_demand_std7", "nsw_price_mean7", "nsw_price_std7",
)
REF_PARAMS = dict(max_leaf_nodes=31, learning_rate=0.1, max_iter=100)


def market_days(index):
    return (index - STEP).normalize()


def require_complete(index):
    if (not isinstance(index, pd.DatetimeIndex) or index.tz is not None
            or not len(index) or len(index) % SLOTS
            or index[0] != index[0].normalize() + STEP
            or not index.equals(pd.date_range(index[0], periods=len(index), freq="30min"))):
        raise ValueError("Expected contiguous complete market days, timestamped at interval end in fixed AEST")


def load_released(path):
    df = pd.read_csv(path, usecols=["timestamp", *MARKET, *WEATHER],
                     parse_dates=["timestamp"], index_col="timestamp")
    require_complete(df.index)
    if df.index[0] != START or df.index[-1] != DATA_STOP:
        raise ValueError("Expected the fixed released 2010-01-01 to 2026-10-31 course CSV")
    if not np.isfinite(df.to_numpy(dtype=float)).all() or not df["nsw_demand"].gt(0).all():
        raise ValueError("Course input contains non-finite values or nonpositive NSW demand")
    return df


def build_features(df):
    require_complete(df.index)
    starts = df.index - STEP
    day = starts.normalize()
    slot = starts.hour * 2 + starts.minute // 30
    hour_angle = 2 * np.pi * slot / SLOTS
    year_angle = 2 * np.pi * (day.dayofyear - 1) / np.where(day.is_leap_year, 366, 365)
    x = pd.DataFrame(
        {"slot_sin": np.sin(hour_angle), "slot_cos": np.cos(hour_angle),
         "season_sin": np.sin(year_angle), "season_cos": np.cos(year_angle),
         "weekend": (day.dayofweek >= 5).astype(int)},
        index=df.index,
    )
    for i in range(7):
        x[f"weekday_{i}"] = (day.dayofweek == i).astype(int)
    for col, lags in {"nsw_demand": (1, 7), "nsw_price": (1, 7),
                      "vic_demand": (1, 7), "vic_price": (7,)}.items():
        for lag in lags:
            x[f"{col}_lag{lag}"] = df[col].shift(lag * SLOTS)
    previous_mean = df["nsw_price"].groupby(day).agg(lambda s: s.mean()).shift(1)
    x["prev_price_mean"] = day.map(previous_mean)
    x["lag1_price_gap"] = x["nsw_price_lag1"] - x["prev_price_mean"]
    for col in WEATHER:
        x[col] = df[col]
    for col in ("nsw_demand", "nsw_price"):
        # One row per market day; shift BEFORE rolling so day D never enters its own inputs.
        prior = pd.DataFrame(df[col].to_numpy().reshape(-1, SLOTS)).shift(1)
        roll = prior.rolling(7, min_periods=7)
        x[f"{col}_mean7"] = roll.mean().to_numpy().ravel()
        x[f"{col}_std7"] = roll.std(ddof=0).to_numpy().ravel()
    return x.loc[:, list(FEATURES)]


def next_day_features(history, weather_next):
    if len(weather_next) != SLOTS or len(history) < 7 * SLOTS:
        raise ValueError("Expected 48 weather slots and at least seven complete historical days")
    past = history.iloc[-7 * SLOTS:].loc[:, list(MARKET)]
    future = weather_next.loc[:, list(WEATHER)]
    if past.index[-1] + STEP != future.index[0]:
        raise ValueError("History must stop at the end of the previous market day")
    window = pd.concat([past, future])
    result = build_features(window).iloc[-SLOTS:]
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("Next-day feature matrix has missing/non-finite values")
    return result


@dataclass
class DemandData:
    """The study keeps outcome and feature arrays separate; only prefix slices are fitted."""
    index: pd.DatetimeIndex
    days: pd.DatetimeIndex
    x: np.ndarray
    actual: np.ndarray
    b1: np.ndarray
    b7: np.ndarray

    @classmethod
    def from_frame(cls, df):
        x = build_features(df).to_numpy(dtype=float)
        y = df["nsw_demand"].to_numpy(dtype=float)
        days = market_days(df.index)
        if not np.isfinite(x[days >= WARM]).all():
            raise ValueError("Non-finite inputs after common warmup")
        return cls(df.index, days, x, y, df["nsw_demand"].shift(48).to_numpy(),
                   df["nsw_demand"].shift(336).to_numpy())

    def interval(self, start, stop):
        """Half-open [start, stop) on market days, not on end timestamps."""
        a = int(self.days.searchsorted(pd.Timestamp(start)))
        b = int(self.days.searchsorted(pd.Timestamp(stop)))
        if a >= b:
            raise ValueError(f"Empty market-day period: {start} to {stop}")
        return slice(a, b)


def fit(x, actual, b1, target, params, seed=0):
    if target not in {"level", "residual_b1"}:
        raise ValueError(f"Unrecognised target: {target}")
    y = actual if target == "level" else actual - b1
    estimator = HistGradientBoostingRegressor(
        loss="squared_error", early_stopping=False, random_state=seed, **params)
    estimator.fit(x, y)
    return estimator


def predict(estimator, x, b1, target):
    y = estimator.predict(x)
    if target == "residual_b1":
        y = y + b1
    # The official runner requires physical bounds; use the same rule for every experiment.
    return np.clip(y, 0.0, 20_000.0)


def fit_period(data, start, stop, config):
    q = data.interval(max(WARM, pd.Timestamp(start)), stop)
    return fit(data.x[q], data.actual[q], data.b1[q],
               config["target"], config["params"], config["seed"])


def save_model(path, estimator, config, start, stop, versions):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": estimator, "features": FEATURES, "config": config,
                 "first_market_day": str(pd.Timestamp(start).date()),
                 "last_market_day": str((pd.Timestamp(stop) - DAY).date()),
                 "versions": versions}, path, compress=3)


class DemandForecaster:
    """P3 integration component. The team's wrapper should add the price-direction column."""

    def __init__(self, artifact_dir):
        self.bundle = joblib.load(Path(artifact_dir) / "demand_hgbr.joblib")
        if tuple(self.bundle["features"]) != FEATURES:
            raise ValueError("Saved feature order does not match the demand predictor")

    def predict(self, history, weather_next):
        x = next_day_features(history, weather_next)
        with threadpool_limits(limits=1):
            y = predict(self.bundle["model"], x.to_numpy(dtype=float),
                        x["nsw_demand_lag1"].to_numpy(), self.bundle["config"]["target"])
        return pd.DataFrame({"nsw_demand_mw": y}, index=weather_next.index)
