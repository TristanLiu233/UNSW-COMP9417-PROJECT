"""Two feature sets:
  (1) the group's 18-column "core" recipe (top of file; historical diagnosis, kept unchanged);
  (2) the M1 feature set (`M1_*`, `build_m1_features`, ... at the bottom of the file).

(1) The group's 18-column "core" inputs for the selected price HGBC (`boost_price`, `tuned_C`).

Source of truth: comp9417_project_kit/ablation/消融说明.md §B2 (column list) and §C (formulas),
which match ablation/features.py `_construct`. The arithmetic below follows that code
step for step, so values are bit-identical, with two differences:
  * day boundaries come from `da.market_day` (identical to the group's `(ts-30min).normalize()`);
  * labels come from `da.targets`, one call per target day (hard rule: no own target code).

Every feature of target day T uses only (a) calendar information of T and (b) NSW market data of
days <= T-1. No weather column and no D+1 (= T) market value enters the matrix.
Nothing here is fitted, so there is nothing to restrict to a training window.
"""
import numpy as np
import pandas as pd

from .kit import da

N = da.N_PERIODS  # 48

BOOST_PRICE_FEATURES = [
    "slot_sin", "slot_cos", "season_sin", "season_cos", "weekend",
    "weekday_0", "weekday_1", "weekday_2", "weekday_3", "weekday_4", "weekday_5", "weekday_6",
    "nsw_demand_lag1", "nsw_price_lag1", "nsw_demand_lag7", "nsw_price_lag7",
    "prev_price_mean", "lag1_price_gap",
]
# The 18-column recipe encodes period and weekday numerically (sin/cos, one-hot), so the HGBC
# gets no native categorical features. Kept explicit so the fitted model can be checked.
CATEGORICAL_FEATURES = []

WARMUP_DAYS = 28                 # group protocol: common 28-day warm-up (only 7 are needed here)
TRAIN_FIRST_DAY = pd.Timestamp("2010-01-29")


def _check_contiguous(df):
    """shift(48 * lag) == "same period lag days earlier" only on a gap-free half-hour grid."""
    idx = df.index
    if not (idx.is_monotonic_increasing and idx.is_unique):
        raise ValueError("index must be sorted and unique")
    if len(idx) % N or (idx[0] - pd.Timedelta("30min")) != idx[0].normalize() \
            or not (idx[1:] - idx[:-1] == pd.Timedelta("30min")).all():
        raise ValueError("need complete market days on a regular 30-min grid")


def build_features(df):
    """18-column feature matrix for every row of `df` (row = target half-hour of its market day).

    Rows whose lags are not yet available (first 7 days) contain NaN; callers drop them.
    """
    _check_contiguous(df)
    days = da.market_day(df.index)
    starts = df.index - pd.Timedelta(minutes=30)
    slot_angle = 2 * np.pi * (starts.hour * 2 + starts.minute // 30) / N
    season_angle = 2 * np.pi * (days.dayofyear - 1) / np.where(days.is_leap_year, 366, 365)
    cols = {"slot_sin": np.sin(slot_angle), "slot_cos": np.cos(slot_angle),
            "season_sin": np.sin(season_angle), "season_cos": np.cos(season_angle),
            "weekend": (days.dayofweek >= 5).astype(int)}
    for i in range(7):
        cols[f"weekday_{i}"] = (days.dayofweek == i).astype(int)
    for lag in (1, 7):
        for name in ("nsw_demand", "nsw_price"):
            cols[f"{name}_lag{lag}"] = df[name].shift(lag * N)
    # Series.mean per day: the same reduction da.targets / da.baseline use for P̄_D.
    prev_mean = df["nsw_price"].groupby(days).agg(lambda p: p.mean()).shift(1)
    cols["prev_price_mean"] = days.map(prev_mean)
    cols["lag1_price_gap"] = cols["nsw_price_lag1"] - cols["prev_price_mean"]
    X = pd.DataFrame(cols, index=df.index)
    return X[BOOST_PRICE_FEATURES]


def features_for_day(history, weather_next):
    """Prediction-time features: exactly what Forecaster.predict will see for day D+1.

    The D+1 rows are appended with NaN market columns, so any use of D+1 demand/price would
    show up as NaN or as a mismatch against `build_features` on the full data.
    """
    nxt = pd.DataFrame(np.nan, index=weather_next.index, columns=history.columns)
    nxt[da.WEATHER_COLS] = weather_next[da.WEATHER_COLS]
    return build_features(pd.concat([history, nxt])).loc[weather_next.index]


def labels(df, target_days):
    """nsw_price_up for each target day, straight from da.targets (concatenated, time order)."""
    out = []
    for d in target_days:
        d = pd.Timestamp(d)
        # da.targets only reads days d-1 and d; slicing keeps each call cheap.
        block = df.loc[d - pd.Timedelta("1D") + pd.Timedelta("30min"): d + pd.Timedelta("1D")]
        out.append(da.targets(block, d)["nsw_price_up"])
    return pd.concat(out)


def day_range(df, first, last):
    """Complete market days of df in [first, last], in order."""
    first, last = pd.Timestamp(first), pd.Timestamp(last)
    return [d for d in da.complete_days(df) if first <= d <= last]


def design(df, first, last, X=None):
    """(X, y) for target days in [first, last]; asserts no NaN/inf in the selected rows."""
    X = build_features(df) if X is None else X
    days = day_range(df, first, last)
    rows = da.market_day(X.index).isin(days)
    Xs = X.loc[rows]
    y = labels(df, days)
    if not y.index.equals(Xs.index):
        raise ValueError("label and feature rows differ")
    if not np.isfinite(Xs.to_numpy(dtype=float)).all():
        raise ValueError("NaN/inf in selected feature rows")
    return Xs, y


# =====================================================================================
# (2) M1 feature set (my scheme). Independent of the group recipe above; nothing above changes.
#
# Row = target half-hour t of target day T = D+1. With P̄(D) = mean NSW price of day D
# (computed with Series.mean per day, the same reduction da.targets / da.baseline use):
#   up_lag1         1[P(D,t)   > P̄(D)]          (= baseline b1's prediction)
#   up_lag7         1[P(D-6,t) > P̄(D)]          (= baseline b7's prediction)
#   12 weather cols of (D+1, t)                  (da.WEATHER_COLS order; allowed by the spec)
#   period          0..47, interval start        (categorical)
#   weekday         0=Mon..6=Sun of D+1          (categorical)
#   price_gap_lag1  P(D,t)   - P̄(D)
#   price_gap_lag7  P(D-6,t) - P̄(D)
# Market data enter only through days D and D-6 (both <= D); no D+1 demand/price is read.
# Nothing is fitted here, so there is no training window to respect.
# =====================================================================================
M1_FEATURES = (["up_lag1", "up_lag7"] + list(da.WEATHER_COLS)
               + ["period", "weekday", "price_gap_lag1", "price_gap_lag7"])
M1_CATEGORICAL = ["period", "weekday"]
M1_LAG_DAYS = 7          # rows of the first 7 market days have no D-6 lag and are dropped


def build_m1_features(df):
    """M1 feature matrix for every row of `df` (row = target half-hour of its market day).

    Rows of the first M1_LAG_DAYS days contain NaN (no lag-7 day yet); `m1_design` drops them.
    """
    _check_contiguous(df)
    days = da.market_day(df.index)
    starts = df.index - pd.Timedelta(minutes=30)
    p = df["nsw_price"]
    # P̄(D) for target day T is the mean of day T-1; Series.mean per day as in dayahead.py.
    day_mean = p.groupby(days).agg(lambda s: s.mean())
    prev_mean = pd.Series(days.map(day_mean.shift(1)), index=df.index, dtype="float64")
    p1, p7 = p.shift(N), p.shift(M1_LAG_DAYS * N)
    gap1, gap7 = p1 - prev_mean, p7 - prev_mean
    cols = {
        # NaN comparisons give False; such rows are dropped via the NaN in the gap columns.
        "up_lag1": (p1 > prev_mean).astype("int64"),
        "up_lag7": (p7 > prev_mean).astype("int64"),
    }
    for c in da.WEATHER_COLS:
        cols[c] = df[c].astype("float64")
    cols["period"] = (starts.hour * 2 + starts.minute // 30).astype("int64")
    cols["weekday"] = days.day_of_week.astype("int64")
    cols["price_gap_lag1"] = gap1
    cols["price_gap_lag7"] = gap7
    X = pd.DataFrame(cols, index=df.index)
    return X[M1_FEATURES]


def m1_features_for_day(history, weather_next):
    """Prediction-time M1 features for day D+1 from exactly what Forecaster.predict receives.

    Only the last 8 market days of `history` are needed (D-6 .. D, plus a margin); the D+1 rows
    get NaN market columns, so any use of D+1 demand/price would surface as NaN / mismatch.
    """
    days = da.market_day(history.index)
    hist = history[days >= days.max() - pd.Timedelta(f"{M1_LAG_DAYS}D")]
    nxt = pd.DataFrame(np.nan, index=weather_next.index, columns=hist.columns)
    nxt[da.WEATHER_COLS] = weather_next[da.WEATHER_COLS]
    X = build_m1_features(pd.concat([hist, nxt])).loc[weather_next.index]
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("NaN/inf in prediction-time features")
    return X


def m1_design(df, first, last, X=None):
    """(X, y, dropped) for complete target days in [first, last].

    Days whose feature rows are not all finite are dropped and returned in `dropped`
    (list of days) so the caller can record them; labels come from da.targets.
    """
    X = build_m1_features(df) if X is None else X
    days = day_range(df, first, last)
    sel = X.loc[da.market_day(X.index).isin(days)]
    finite = pd.Series(np.isfinite(sel.to_numpy(dtype=float)).all(axis=1), index=sel.index)
    ok_by_day = finite.groupby(da.market_day(sel.index)).all()
    dropped = [d for d in days if not ok_by_day.get(d, False)]
    keep = [d for d in days if ok_by_day.get(d, False)]
    Xs = sel.loc[da.market_day(sel.index).isin(keep)]
    y = labels(df, keep)
    if not y.index.equals(Xs.index):
        raise ValueError("label and feature rows differ")
    if not np.isfinite(Xs.to_numpy(dtype=float)).all():
        raise ValueError("NaN/inf in selected feature rows")
    return Xs, y.astype("int64"), dropped
