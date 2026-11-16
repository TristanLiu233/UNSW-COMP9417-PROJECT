"""M3 (explanation only) shared pieces: REF, data, feature groups, paired whole-day bootstrap, figure style.

Data come from P1's `load_study` (rows up to 2015-01-01 00:00 only), so nothing after 2014 is ever read.
Nothing here tunes or changes the official model; REF = results/price_hgbc/official/boost_price_ref.json.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd

from . import official_ref as ofr
from .kit import RESULTS, da

OUT = os.path.join(RESULTS, "m3")
N_BOOT, BOOT_SEED = 1000, 0
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a3a29c"        # validated categorical pair (light surface)

# Mutually exclusive, exhaustive groups of the 18 REF columns (there is no weather column among them).
GROUPS = {
    "price_lag": ("电价滞后", ["nsw_price_lag1", "nsw_price_lag7"]),
    "threshold_gap": ("阈值与价差", ["prev_price_mean", "lag1_price_gap"]),
    "demand_lag": ("需求滞后", ["nsw_demand_lag1", "nsw_demand_lag7"]),
    "slot": ("时段编码", ["slot_sin", "slot_cos"]),
    "season": ("季节编码", ["season_sin", "season_cos"]),
    "weekday": ("星期", ["weekend"] + [f"weekday_{i}" for i in range(7)]),
}
NOTES = {   # why a single column's permutation number should not be read on its own
    "slot_sin": "sin/cos pair", "slot_cos": "sin/cos pair", "season_sin": "sin/cos pair", "season_cos": "sin/cos pair",
    "weekend": "one-hot / redundant with weekday_5,6", **{f"weekday_{i}": "one-hot" for i in range(7)},
    "lag1_price_gap": "linear: = nsw_price_lag1 - prev_price_mean", "nsw_price_lag1": "linear: lag1_price_gap + prev_price_mean",
    "prev_price_mean": "linear: nsw_price_lag1 - lag1_price_gap",
}


def ref():
    return json.load(open(ofr.REF_JSON))


def feature_groups_table(r=None):
    r = r or ref()
    rows = []
    for g, (zh, cols) in GROUPS.items():
        for c in cols:
            rows.append({"feature": c, "group": g, "group_zh": zh, "ref_order": r["features"].index(c),
                         "single_column_note": NOTES.get(c, "")})
    return pd.DataFrame(rows).sort_values("ref_order").reset_index(drop=True)


def load():
    """(data <= 2015-01-01 00:00, X, truth, REF columns, train mask, 2014 mask) via P1's code."""
    data, X, truth, cols, tr, va = ofr.design()
    assert data.index.max() == pd.Timestamp("2015-01-01 00:00")
    return data, X, truth, cols, tr, va


# ---------------- paired whole-day bootstrap ----------------
def day_counts(y, pred, index):
    days = np.asarray(da.market_day(pd.DatetimeIndex(index)))
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    f = pd.DataFrame({"tp": (y == 1) & (pred == 1), "fn": (y == 1) & (pred == 0),
                      "tn": (y == 0) & (pred == 0), "fp": (y == 0) & (pred == 1)}).astype("int64")
    c = f.groupby(days).sum()
    assert (c.sum(axis=1) == 48).all()
    return c


def balacc_from_counts(tp, fn, tn, fp):
    """Mean recall over the classes present (kit convention, = da.balanced_accuracy)."""
    tp, fn, tn, fp = (np.asarray(a, dtype="float64") for a in (tp, fn, tn, fp))
    pos, neg = tp + fn, tn + fp
    r1 = np.where(pos > 0, tp / np.where(pos > 0, pos, 1), np.nan)
    r0 = np.where(neg > 0, tn / np.where(neg > 0, neg, 1), np.nan)
    return np.nanmean(np.vstack([r1, r0]), axis=0) * 100


def boot_indices(n_days, n_boot=N_BOOT, seed=BOOT_SEED):
    return np.random.default_rng(seed).integers(0, n_days, size=(n_boot, n_days))


def boot_balacc(y, pred, index, B):
    c = day_counts(y, pred, index)
    s = c[["tp", "fn", "tn", "fp"]].to_numpy()[B].sum(axis=1)
    return balacc_from_counts(s[:, 0], s[:, 1], s[:, 2], s[:, 3])


def ci(a):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(lo), float(hi)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def sha_arr(a, dtype):
    return sha(np.ascontiguousarray(np.asarray(a, dtype=dtype)).tobytes())


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_axisbelow(True)


def mpl():
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "price_hgbc_mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    return plt
