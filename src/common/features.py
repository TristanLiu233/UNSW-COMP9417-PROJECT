"""Feature table v1 for the NSW day-ahead models (fixed AEST, interval-end timestamps)."""

import holidays
import numpy as np
import pandas as pd


SLOTS = 48
STEP = pd.Timedelta(minutes=30)
HOLIDAY_COLUMN = "is_nsw_public_holiday"
_MARKET = ("nsw_demand", "nsw_price", "qld_demand", "vic_demand", "vic_price")
_SYDNEY = ("syd_temp", "syd_rh", "syd_cloud", "syd_solar")
_NSW = ("nsw_demand", "nsw_price")

# First use in the selected Ridge, demand tree, logistic and price tree recipes.
FEATURE_COLUMNS = (
    "slot_sin", "slot_cos", "season_sin", "season_cos", "weekend",
    *(f"weekday_{i}" for i in range(7)),
    *(f"{name}_lag{lag}" for lag in (1, 7) for name in _NSW),
    "prev_price_mean", "lag1_price_gap",
    "qld_demand_lag1", "qld_demand_lag7", "vic_demand_lag7",
    *(f"{name}_{stat}28" for name in _NSW for stat in ("mean", "std")),
    *_SYDNEY, "vic_demand_lag1", "vic_price_lag7",
    *(f"{name}_{stat}7" for name in _NSW for stat in ("mean", "std")),
    HOLIDAY_COLUMN,
)


def _check_intervals(index):
    if not isinstance(index, pd.DatetimeIndex) or index.tz is not None:
        raise ValueError("Use a DatetimeIndex in timezone-naive fixed AEST, as in the released CSV")
    if index.empty or len(index) % SLOTS or index[0] != index[0].normalize() + STEP:
        raise ValueError("Expected complete market days from 00:30 through the next midnight")
    if not index.equals(pd.date_range(index[0], periods=len(index), freq=STEP)):
        raise ValueError("Half-hourly timestamps must be ordered, unique and consecutive")


def _numeric(frame, names):
    if not frame.columns.is_unique:
        raise ValueError("Duplicate input column names")
    values = frame.loc[:, list(names)].astype(np.float64)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"Missing or non-finite input in {names}")
    return values


def _build(market, weather):
    """Compute the trailing weather-indexed rows; market includes the same target slots."""
    starts = weather.index - STEP
    days = starts.normalize()
    n_days = len(days) // SLOTS
    slot_angle = 2 * np.pi * (starts.hour * 2 + starts.minute // 30) / SLOTS
    season_angle = 2 * np.pi * (days.day_of_year - 1) / np.where(days.is_leap_year, 366, 365)
    weekday_number = days.day_of_week
    values = {"slot_sin": np.sin(slot_angle), "slot_cos": np.cos(slot_angle),
              "season_sin": np.sin(season_angle), "season_cos": np.cos(season_angle),
              "weekend": (weekday_number >= 5).astype(np.int64)}
    for weekday in range(7):
        values[f"weekday_{weekday}"] = (weekday_number == weekday).astype(np.int64)
    for lag, names in ((1, (*_NSW, "qld_demand", "vic_demand")), (7, _MARKET)):
        for name in names:
            values[f"{name}_lag{lag}"] = market[name].shift(lag * SLOTS).iloc[-len(days):].to_numpy()
    for name in _SYDNEY:
        values[name] = weather[name]

    # Use exactly the course target's 48-slot Series.mean, including negative and spike prices.
    prices = market["nsw_price"].iloc[-(len(days) + SLOTS):]
    prices_by_day = prices.groupby((prices.index - STEP).normalize())
    previous_means = prices_by_day.agg(lambda group: group.mean()).shift(1)
    values["prev_price_mean"] = days.map(previous_means).to_numpy()
    values["lag1_price_gap"] = values["nsw_price_lag1"] - values["prev_price_mean"]

    # Retain the entire supplied prefix: pandas rolling variance accumulates round-off
    # along the time axis, so truncating to 28 days can change an existing tree split.
    for name in _NSW:
        previous = pd.DataFrame(market[name].to_numpy().reshape(-1, SLOTS)).shift(1)
        for window in (7, 28):
            rolling = previous.rolling(window, min_periods=window)
            values[f"{name}_mean{window}"] = rolling.mean().iloc[-n_days:].to_numpy().ravel()
            values[f"{name}_std{window}"] = rolling.std(ddof=0).iloc[-n_days:].to_numpy().ravel()

    dates = days[::SLOTS]
    calendar = holidays.country_holidays("AU", subdiv="NSW", years=dates.year.unique().tolist(),
                                         observed=True, categories=holidays.PUBLIC)
    flags = np.fromiter((day in calendar for day in dates.date), dtype=np.int8, count=n_days)
    values[HOLIDAY_COLUMN] = np.repeat(flags, SLOTS)
    return pd.DataFrame(values, index=weather.index, columns=FEATURE_COLUMNS)


def build_features(df):
    """Return the 36 v1 inputs on the supplied index, including lag warm-up NaNs.

    Provide consecutive, complete market days, indexed by the course's half-hour end
    timestamps. The maximum lookback is 28 days. Targets belong to the caller.
    """
    _check_intervals(df.index)
    return _build(_numeric(df, _MARKET), _numeric(df, _SYDNEY))


def build_features_for_day(history, weather_next):
    """Return the 48 rows for the day immediately after the end of history.

    Supply the same historical prefix as the batch run (at least 28 complete days)
    for bit-identical rolling inputs. Only past market values and forecast Sydney
    weather are read; the returned index is weather_next.index.
    """
    _check_intervals(history.index)
    _check_intervals(weather_next.index)
    if len(history) < 28 * SLOTS:
        raise ValueError("At least 28 complete market days of history are required")
    if len(weather_next) != SLOTS or weather_next.index[0] != history.index[-1] + STEP:
        raise ValueError("Weather must contain the 48 slots immediately after the last history day")
    market = _numeric(history, _MARKET)
    # The forecast slots have no observable market values, even if the caller supplies them.
    market = market.reindex(history.index.append(weather_next.index))
    return _build(market, _numeric(weather_next, _SYDNEY))
