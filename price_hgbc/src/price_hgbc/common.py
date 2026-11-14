"""Shared data helpers: load the released CSV, tag each half-hour with its market day and
period, and split chronologically by year.

Day boundaries come from harness/dayahead.py (`market_day`): a timestamp marks the END of a
half-hour, so the row stamped (D+1) 00:00 belongs to day D. Period 0..47 is the index of the
interval start within its day, i.e. (ts - 30min).hour * 2 + (ts - 30min).minute // 30.
"""
import pandas as pd

from .kit import DATA_CSV, da

STAGE1_TRAIN_YEARS = range(2010, 2014)   # 2010-2013
STAGE1_VALID_YEARS = range(2014, 2015)   # 2014
STAGE2_YEARS = range(2015, 2027)         # 2015-01 .. 2026-10


def load_data(path=DATA_CSV):
    """The released CSV via the official loader, with `day` and `period` columns added."""
    return add_day_period(da.load_dataset(path))


def add_day_period(df):
    out = df.copy()
    start = out.index - pd.Timedelta("30min")
    out["day"] = da.market_day(out.index)
    out["period"] = (start.hour * 2 + start.minute // 30).astype("int16")
    return out


def incomplete_days(df):
    """Market days that do not have exactly 48 half-hours (complement of da.complete_days)."""
    counts = pd.Series(1, index=df.index).groupby(da.market_day(df.index)).sum()
    return list(counts[counts != da.N_PERIODS].index)


def split_by_year(df, years):
    """Rows whose market day falls in `years` (years are market-day years, not timestamp years)."""
    day = pd.DatetimeIndex(df["day"] if "day" in df.columns else da.market_day(df.index))
    return df[day.year.isin(list(years))]


def stage_splits(df):
    """Chronological Stage 1 train / Stage 1 validation / Stage 2 simulation row splits."""
    return {"train": split_by_year(df, STAGE1_TRAIN_YEARS),
            "valid": split_by_year(df, STAGE1_VALID_YEARS),
            "sim": split_by_year(df, STAGE2_YEARS)}
