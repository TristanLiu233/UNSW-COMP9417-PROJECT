"""P4 expanding-prefix training; nested probability-threshold (tau) selection.

Run from the repository root. This module does not schedule Stage 2 retrains or
implement the group's demand model. Tau is the probability decision threshold.
"""
import argparse
import hashlib
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from .features import RAW, build_features
from .transforms import transform_features
from .tuning import estimator, metrics

START = pd.Timestamp('2010-01-01')
LAST_RELEASED_DAY = pd.Timestamp('2026-10-31')
SELECTION = Path(__file__).parent / 'handoff/selected_price_model.json'


def market_date(value):
    day = pd.Timestamp(value)
    if pd.isna(day) or day.tz is not None or day != day.normalize():
        raise ValueError('Expected a timezone-naive AEST market date at midnight')
    return day


def training_prefix(data, cutoff):
    """Discard future rows before inspecting values or building any features."""
    cutoff = market_date(cutoff)
    if cutoff > LAST_RELEASED_DAY or cutoff < START + pd.Timedelta(days=28):
        raise ValueError('Cutoff must be an eligible day within the released period')
    if not isinstance(data.index, pd.DatetimeIndex) or data.index.tz is not None:
        raise ValueError('Expected timezone-naive AEST DatetimeIndex')
    prefix = data.loc[data.index <= cutoff + pd.Timedelta(days=1), RAW].copy()
    expected = pd.date_range(START + pd.Timedelta(minutes=30), cutoff + pd.Timedelta(days=1), freq='30min')
    if not prefix.index.equals(expected):
        raise ValueError('Need the complete ordered 2010-01-01 through cutoff market-day prefix')
    return prefix


def fit_final(data, cutoff, *, validation_folds=None, thresholds=None, fixed_threshold=False):
    """Return a PriceClassifier-compatible bundle and auditable selection records.

    By default validate on the last calendar year and search the existing P4 grid.
    Set fixed_threshold=True to retain the Stage 1 threshold. With explicit folds,
    each (start, end) is an inclusive validation market-day range, wholly within
    cutoff. Fit on ALL eligible days before start, pool out-of-fold predictions,
    select threshold, then independently refit scaler and model through cutoff.
    Custom folds/grid override the documented last-year policy.
    """
    prefix = training_prefix(data, cutoff)
    cutoff = market_date(cutoff)
    config = json.loads(SELECTION.read_text(encoding='utf-8'))['model']
    base = build_features(prefix)
    x = transform_features(base, config['recipe']).iloc[28 * 48:]
    y = (prefix.nsw_price > base.prev_price_mean).astype(int).reindex(x.index)
    if not np.isfinite(x.to_numpy()).all() or y.nunique() != 2:
        raise ValueError('Invalid training features or missing training class')
    days = (x.index - pd.Timedelta(minutes=30)).normalize()
    if (validation_folds is None) != (thresholds is None):
        raise ValueError('Provide both validation_folds and thresholds, or neither')
    if fixed_threshold and validation_folds is not None:
        raise ValueError('Fixed threshold cannot be combined with a selection plan')
    if validation_folds is None and not fixed_threshold:
        validation_folds = [(cutoff - pd.DateOffset(years=1) + pd.Timedelta(days=1), cutoff)]
        thresholds = [i / 100 for i in range(25, 61)]
    folds, search = [], []
    threshold = float(config['threshold'])
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter('error', ConvergenceWarning)
        if validation_folds is not None:
            grid = np.asarray(list(thresholds), dtype=float)
            if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all() or ((grid < 0) | (grid > 1)).any():
                raise ValueError('Probability thresholds must be a nonempty finite grid in [0, 1]')
            if not validation_folds:
                raise ValueError('At least one validation fold is required')
            # Validate all boundaries before doing any fitting.
            previous_end = None
            for start, end in validation_folds:
                start, end = market_date(start), market_date(end)
                if not days[0] < start <= end <= cutoff or (previous_end is not None and start <= previous_end):
                    raise ValueError('Validation folds must be ordered, disjoint, and within cutoff, with earlier training data')
                train, valid = days < start, (days >= start) & (days <= end)
                if y.loc[train].nunique() != 2 or not valid.any():
                    raise ValueError('Insufficient fold data or missing training class')
                folds.append({'train_start': str(days[0].date()), 'train_end': str((start-pd.Timedelta(days=1)).date()),
                              'validation_start': str(start.date()), 'validation_end': str(end.date()),
                              'train_rows': int(train.sum()), 'validation_rows': int(valid.sum())})
                previous_end = end
            truths, probabilities = [], []
            for fold in folds:
                train = days < fold['validation_start']
                valid = (days >= fold['validation_start']) & (days <= fold['validation_end'])
                fitted = estimator(config['C'], config['class_weight']).fit(x.loc[train], y.loc[train])
                truths.append(y.loc[valid].to_numpy())
                probabilities.append(fitted.predict_proba(x.loc[valid])[:, 1])
            truth, probability = np.concatenate(truths), np.concatenate(probabilities)
            search = [{'threshold': float(t), **metrics(truth, probability >= t)} for t in sorted(set(grid))]
            # Existing P4 tie preference: closest to 0.5; lower threshold for an exact tie.
            best = min(search, key=lambda row: (-row['BalAcc'], round(abs(row['threshold']-.5), 12), row['threshold']))
            threshold = best['threshold']
        pipeline = estimator(config['C'], config['class_weight']).fit(x, y)
    audit = {'training_policy': 'expanding_from_2010-01-29', 'train_end': str(cutoff.date()),
             'train_rows': len(x), 'recipe': config['recipe'], 'C': config['C'],
             'class_weight': config['class_weight'], 'threshold': threshold,
             'threshold_policy': 'explicit_nested_temporal' if folds else 'selected_stage1_fixed',
             'folds': folds, 'search': search, 'random_state': 17,
             'selection_scores_are_not_outer_test_scores': True,
             'prefix_sha256': hashlib.sha256(pd.util.hash_pandas_object(prefix, index=True).values.tobytes()).hexdigest()}
    bundle = {'pipeline': pipeline, 'columns': list(x.columns), 'recipe': config['recipe'],
              'threshold': threshold, 'train_end': audit['train_end'], 'train_rows': len(x), 'parameters': audit}
    return bundle, audit


def train_final(data, cutoff, output, **kwargs):
    """Train and save to a NEW directory; never replace frozen or existing models."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    bundle, audit = fit_final(data, cutoff, **kwargs)
    output.mkdir(parents=True, exist_ok=False)
    joblib.dump(bundle, output / 'logistic.joblib')
    (output / 'training.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--cutoff', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection-plan', type=Path, help='Override default last-year validation with explicit folds/grid')
    parser.add_argument('--fixed-threshold', action='store_true', help='Retain Stage 1 threshold 0.42 instead of tuning tau')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    cutoff = market_date(args.cutoff)
    if not START + pd.Timedelta(days=28) <= cutoff <= LAST_RELEASED_DAY:
        raise ValueError('Cutoff is outside the eligible released period')
    # Read only the requested prefix, including its final midnight interval.
    nrows = ((cutoff - START).days + 1) * 48
    data = pd.read_csv(args.data, usecols=['timestamp', *RAW], parse_dates=['timestamp'], index_col='timestamp', nrows=nrows)
    plan = json.loads(args.selection_plan.read_text()) if args.selection_plan else {}
    if args.selection_plan and set(plan) != {'validation_folds', 'thresholds'}:
        raise ValueError('Selection plan must contain exactly validation_folds and thresholds')
    print(json.dumps(train_final(data, cutoff, args.output, fixed_threshold=args.fixed_threshold, **plan), indent=2))


if __name__ == '__main__':
    main()
