"""Course data, batch features and labels for the LR price model (P4 conventions).

Batch features are computed once on the released data (P4 verified batch = daily interface on all 4,322 replay
days; each row of target day d uses market data of d-1 and earlier plus d's weather and calendar).
Labels y = 1[nsw_price > prev_price_mean] are checked against the official harness (`da.targets`).
"""
import joblib
import numpy as np
import pandas as pd

from .env import DATA_CSV, FROZEN_ARTIFACT, RECIPE, da, lf, lt

START = pd.Timestamp("2010-01-01")
FIRST_TRAIN_DAY = pd.Timestamp("2010-01-29")
LAST_DAY = pd.Timestamp("2026-10-31")
D1 = pd.Timedelta("1D")


def market_days(index):
    return (pd.DatetimeIndex(index) - pd.Timedelta(minutes=30)).normalize()


def load_raw():
    """P4's read: timestamp + RAW columns, complete 2010-01-01 00:30 .. 2026-11-01 00:00."""
    data = pd.read_csv(DATA_CSV, usecols=["timestamp", *lf.RAW], index_col="timestamp", parse_dates=True)
    expected = pd.date_range("2010-01-01 00:30", "2026-11-01 00:00", freq="30min")
    assert data.index.equals(expected) and np.isfinite(data.to_numpy()).all()
    return data


def prefix(data, cutoff):
    """Rows of market days <= cutoff (ends at cutoff + 1 day 00:00)."""
    cutoff = pd.Timestamp(cutoff)
    out = data.loc[data.index <= cutoff + D1]
    assert out.index[-1] == cutoff + D1
    return out


def frozen_bundle():
    return joblib.load(FROZEN_ARTIFACT)


class Frame:
    """X (31 columns, frozen artifact order), y, day of every half-hour from 2010-01-29 00:30."""

    def __init__(self, data=None):
        self.data = load_raw() if data is None else data
        self.columns = list(frozen_bundle()["columns"])
        base = lf.build_features(self.data)
        x = lt.transform_features(base, RECIPE)[self.columns]
        y = (self.data.nsw_price > base.prev_price_mean).astype(int)
        days = market_days(x.index)
        keep = days >= FIRST_TRAIN_DAY
        self.X, self.y, self.day = x.loc[keep], y.loc[keep], days[keep]
        assert np.isfinite(self.X.to_numpy()).all()

    def rows(self, first, last):
        return (self.day >= pd.Timestamp(first)) & (self.day <= pd.Timestamp(last))


def official_labels(df_full, first, last):
    """da.targets for every day in [first, last] (two-day slices)."""
    out = []
    for d in pd.date_range(first, last):
        block = df_full.loc[d - D1 + pd.Timedelta("30min"): d + D1]
        out.append(da.targets(block, d)["nsw_price_up"])
    return pd.concat(out)


def truth_and_baselines(df_full, first, last):
    """Official truth, b1 and b7 for days [first, last] (same construction as price_hgbc m6_eval)."""
    y, b1, b7 = [], [], []
    for d in pd.date_range(first, last):
        block = df_full.loc[d - pd.Timedelta("7D") + pd.Timedelta("30min"): d + D1]
        h, w = da.make_inputs(block, d)
        y.append(da.targets(block, d)["nsw_price_up"])
        b1.append(da.baseline(h, w, 1)["nsw_price_up"])
        b7.append(da.baseline(h, w, 7)["nsw_price_up"])
    return pd.DataFrame({"y": pd.concat(y), "b1": pd.concat(b1), "b7": pd.concat(b7)})
