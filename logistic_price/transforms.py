"""Predeclared Logistic feature recipes; deterministic, past-market values only."""
import numpy as np
from .features import FEATURES

RECIPES = ('raw22', 'log22', 'log_gaps25', 'log_interactions31')
PRICE_COLUMNS = ('nsw_price_lag1', 'nsw_price_lag7', 'prev_price_mean',
                 'nsw_price_mean28', 'nsw_price_std28')


def transform_features(base, recipe='raw22'):
    if recipe not in RECIPES:
        raise ValueError(f'Unknown recipe: {recipe}')
    x = base.loc[:, FEATURES].copy()
    if recipe == 'raw22':
        return x
    # Fixed unit scale 1 $/MWh; preserves zero and negative prices, no fitted transform.
    for col in PRICE_COLUMNS:
        x[col] = np.sign(x[col]) * np.log1p(np.abs(x[col]))
    if recipe in ('log_gaps25', 'log_interactions31'):
        for source, name in [('nsw_price_lag1', 'gap1'), ('nsw_price_lag7', 'gap7'),
                             ('nsw_price_mean28', 'gap_mean28')]:
            gap = base[source] - base['prev_price_mean']
            x[name] = np.sign(gap) * np.log1p(np.abs(gap))
    if recipe == 'log_interactions31':
        for gap in ('gap1', 'gap7', 'gap_mean28'):
            for slot in ('slot_sin', 'slot_cos'):
                x[f'{gap}_x_{slot}'] = x[gap] * x[slot]
    return x
