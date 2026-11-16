"""M2 extended features on top of the M1 set (features_simple.M1_*). Nothing in features_simple
or in the group's ablation code is changed.

Relation to P1 (group ablation, comp9417_project_kit/ablation/features.py, commit 116e296):
  * no artefact is labelled "feature table v1"; the closest is `ablation.features.build_features`.
    It cannot be called for prediction: `load_study` reads only up to 2015-01-01 and it fits a
    day-type model, so it has no day-by-day interface. Its definitions are reused instead:
      prev_price_max  == P1 `prev_price_max`
      prev_neg_share  == P1 `prev_negative_slots` / 48
    (checked in tests/price_hgbc/test_m2.py).
  * the M1 columns already match P1's: price_gap_lag1 == P1 `lag1_price_gap`.

Row = target half-hour t of target day T = D+1. New columns (all known at the end of day D or
calendar of D+1):
  Price distribution (day D)
    prev_price_max     max_t P(D,t)
    prev_price_std     population std (ddof=0) of the 48 prices of D
    prev_neg_share     #{t: P(D,t) < 0} / 48                        in [0, 1]
  Price spike (day D)
    prev_spike_ratio   P_max(D) / max(P̄(D), SPIKE_FLOOR)          finite by construction
                       Rule: the denominator is floored at SPIKE_FLOOR = 1 $/MWh, so a zero or
                       negative daily mean (possible since ~2020) never gives inf / sign flips.
    prev_n_gt300       #{t: P(D,t) > 300}                            in [0, 48]
    prev_n_gt1000      #{t: P(D,t) > 1000}                           in [0, 48]
  Weather
    syd_temp_diff      syd_temp(D+1,t) - syd_temp(D,t)   (D+1 weather is allowed; D weather is history)
  Calendar
    is_holiday         1 if D+1 is an NSW public holiday in `holidays` (country "AU", subdiv "NSW",
                       default category "public", observed days included as the package lists them)
"""
import holidays
import numpy as np
import pandas as pd

from . import features_simple as fs
from .kit import da

N = da.N_PERIODS
SPIKE_FLOOR = 1.0
PRICE_DIST = ["prev_price_max", "prev_price_std", "prev_neg_share"]
PRICE_SPIKE = ["prev_spike_ratio", "prev_n_gt300", "prev_n_gt1000"]
WEATHER_EXTRA = ["syd_temp_diff"]
CALENDAR_EXTRA = ["is_holiday"]
EXTRA_COLUMNS = PRICE_DIST + PRICE_SPIKE + WEATHER_EXTRA + CALENDAR_EXTRA
EXT_FEATURES = list(fs.M1_FEATURES) + EXTRA_COLUMNS          # 18 + 8 = 26 columns
EXT_CATEGORICAL = list(fs.M1_CATEGORICAL)                     # is_holiday is a 0/1 numeric column

FEATURE_SETS = {"M1": list(fs.M1_FEATURES), "M1_ext": EXT_FEATURES}


def nsw_holidays(years):
    return holidays.country_holidays("AU", subdiv="NSW", years=list(years))


def build_extra(df):
    """The 8 extra columns for every row of df (first day: NaN, dropped by the design step)."""
    fs._check_contiguous(df)
    days = da.market_day(df.index)
    p = df["nsw_price"]
    g = p.groupby(days)
    # Same per-day reduction as dayahead.py for the mean (Series.mean).
    mean = g.agg(lambda s: s.mean())
    stats = pd.DataFrame({
        "prev_price_max": g.max(),
        "prev_price_std": g.agg(lambda s: s.std(ddof=0)),
        "prev_neg_share": p.lt(0).groupby(days).sum() / N,
        "prev_spike_ratio": g.max() / np.maximum(mean, SPIKE_FLOOR),
        "prev_n_gt300": p.gt(300).groupby(days).sum().astype("float64"),
        "prev_n_gt1000": p.gt(1000).groupby(days).sum().astype("float64"),
    }).shift(1)                                   # value for target day T comes from day T-1
    cols = {c: pd.Series(days.map(stats[c]), index=df.index, dtype="float64") for c in stats.columns}
    cols["syd_temp_diff"] = df["syd_temp"] - df["syd_temp"].shift(N)
    hol = nsw_holidays(range(days.min().year, days.max().year + 1))
    cols["is_holiday"] = pd.Series([int(d.date() in hol) for d in days], index=df.index, dtype="int64")
    return pd.DataFrame(cols, index=df.index)[EXTRA_COLUMNS]


def build_ext_features(df):
    """M1 columns + extra columns, in EXT_FEATURES order."""
    return pd.concat([fs.build_m1_features(df), build_extra(df)], axis=1)[EXT_FEATURES]


def ext_features_for_day(history, weather_next):
    """Prediction-time features for D+1 from exactly what Forecaster.predict receives."""
    days = da.market_day(history.index)
    hist = history[days >= days.max() - pd.Timedelta(f"{fs.M1_LAG_DAYS}D")]
    nxt = pd.DataFrame(np.nan, index=weather_next.index, columns=hist.columns)
    nxt[da.WEATHER_COLS] = weather_next[da.WEATHER_COLS]
    X = build_ext_features(pd.concat([hist, nxt])).loc[weather_next.index]
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("NaN/inf in prediction-time features")
    return X


def build(df, feature_set):
    if feature_set == "M1":
        return fs.build_m1_features(df)
    if feature_set == "M1_ext":
        return build_ext_features(df)
    raise KeyError(feature_set)


def feature_groups():
    """One row per column of the extended set (reused by the M3 ablation)."""
    src_m1 = "src/price_hgbc/features_simple.py (M1)"
    src_ex = "src/price_hgbc/extra_features.py (M2)"
    rows = []

    def add(name, group, source, date, categorical=False, in_m1=False):
        rows.append({"feature": name, "group": group, "source": source, "date": date,
                     "categorical": categorical, "allowed_at_prediction": True, "in_M1": in_m1,
                     "in_M1_ext": True})
    for c in ("up_lag1", "price_gap_lag1"):
        add(c, "Price lag", src_m1, "D (same period t) and P̄(D)", in_m1=True)
    for c in ("up_lag7", "price_gap_lag7"):
        add(c, "Price lag", src_m1, "D-6 (same period t) and P̄(D)", in_m1=True)
    for c in da.WEATHER_COLS:
        add(c, "Weather", src_m1, "D+1 (period t), released weather", in_m1=True)
    add("period", "Calendar", src_m1, "D+1 (period t)", categorical=True, in_m1=True)
    add("weekday", "Calendar", src_m1, "D+1", categorical=True, in_m1=True)
    for c in PRICE_DIST:
        add(c, "Price distribution", src_ex, "D (all 48 periods)")
    for c in PRICE_SPIKE:
        add(c, "Price spike", src_ex, "D (all 48 periods)")
    add("syd_temp_diff", "Weather", src_ex, "D+1 and D (period t)")
    add("is_holiday", "Calendar", src_ex, "D+1 (holidays 0.107, AU/NSW public)")
    out = pd.DataFrame(rows).set_index("feature").loc[EXT_FEATURES].reset_index()
    out.insert(0, "order", range(len(out)))
    return out
