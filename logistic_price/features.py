"""The 22 Logistic features from 消融说明.md; naive timestamps mean fixed AEST."""
import numpy as np
import pandas as pd

GROUPS = {
    'calendar': ['slot_sin', 'slot_cos', 'season_sin', 'season_cos', 'weekend',
                 *[f'weekday_{i}' for i in range(7)]],
    'history': ['nsw_demand_lag1', 'nsw_price_lag1', 'nsw_demand_lag7',
                'nsw_price_lag7', 'prev_price_mean'],
    'cross_state': ['qld_demand_lag1'],
    'rolling28': ['nsw_demand_mean28', 'nsw_demand_std28', 'nsw_price_mean28', 'nsw_price_std28'],
}
FEATURES = [col for group in GROUPS.values() for col in group]
RAW = ['nsw_demand', 'nsw_price', 'qld_demand']


def validate_index(index):
    if not isinstance(index, pd.DatetimeIndex) or index.tz is not None or len(index) == 0:
        raise ValueError('Expected nonempty timezone-naive fixed-AEST timestamps')
    expected = pd.date_range(index[0].normalize() + pd.Timedelta(minutes=30), periods=len(index), freq='30min')
    if len(index) % 48 or not index.equals(expected):
        raise ValueError('Expected ordered, continuous, complete market days (00:30 through next 00:00)')


def build_features(data):
    validate_index(data.index)
    starts = data.index - pd.Timedelta(minutes=30)
    days = starts.normalize()
    slot = 2 * np.pi * (starts.hour * 2 + starts.minute // 30) / 48
    season = 2 * np.pi * (days.dayofyear - 1) / np.where(days.is_leap_year, 366, 365)
    x = pd.DataFrame(index=data.index)
    x['slot_sin'], x['slot_cos'] = np.sin(slot), np.cos(slot)
    x['season_sin'], x['season_cos'] = np.sin(season), np.cos(season)
    x['weekend'] = (days.dayofweek >= 5).astype(int)
    for i in range(7):
        x[f'weekday_{i}'] = (days.dayofweek == i).astype(int)
    for name, lags in [('nsw_demand', (1, 7)), ('nsw_price', (1, 7)), ('qld_demand', (1,))]:
        for lag in lags:
            x[f'{name}_lag{lag}'] = data[name].shift(48 * lag)
    means = data['nsw_price'].groupby(days).agg(lambda v: v.mean()).shift(1)
    x['prev_price_mean'] = days.map(means)
    for name in ['nsw_demand', 'nsw_price']:
        prior = pd.DataFrame(data[name].to_numpy().reshape(-1, 48)).shift(1)
        rolling = prior.rolling(28, min_periods=28)
        x[f'{name}_mean28'] = rolling.mean().to_numpy().ravel()
        x[f'{name}_std28'] = rolling.std(ddof=0).to_numpy().ravel()
    return x[FEATURES]


def build_features_for_day(history, weather_next):
    validate_index(weather_next.index)
    if len(weather_next) != 48:
        raise ValueError('Expected exactly 48 target intervals')
    day = weather_next.index[0].normalize()
    expected = pd.date_range(day - pd.Timedelta(days=28) + pd.Timedelta(minutes=30), day, freq='30min')
    # Only the last 28 complete past market days are visible. Ignore all future rows.
    past = history.reindex(expected)[RAW]
    if not np.isfinite(past.to_numpy(dtype=float)).all():
        raise ValueError('Need 28 complete days of finite past NSW/QLD data')
    future = pd.DataFrame(np.nan, index=weather_next.index, columns=RAW)
    result = build_features(pd.concat([past, future])).loc[weather_next.index]
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError('Nonfinite day-ahead features')
    return result
