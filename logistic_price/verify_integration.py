"""Verify shared v1 against the saved P4 results, without replacing model artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.common import features as shared
from .classifier import PriceClassifier
from .features import FEATURES, RAW, WEATHER, build_features
from .transforms import transform_features
from .tuning import estimator, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError('Choose a new report path')
    data = pd.read_csv(args.data, usecols=['timestamp', *RAW],
                       index_col='timestamp', parse_dates=True)
    base = build_features(data)
    days = (data.index - pd.Timedelta(minutes=30)).normalize()
    train = (days >= '2010-01-29') & (days < '2014-01-01')
    truth = (data.nsw_price > base.prev_price_mean).astype(int)
    validation = pd.read_csv(args.results / 'validation_predictions.csv.gz',
                             index_col='timestamp', parse_dates=True)
    replay = pd.read_csv(args.results / 'replay_predictions.csv.gz',
                         index_col='timestamp', parse_dates=True)
    scores = pd.read_csv(args.results / 'validation_2014.csv').set_index('model')
    selected = json.loads((args.results / 'selection_2014.json').read_text())['model']
    checks = []
    for folder in sorted((args.results / 'artifacts').iterdir()):
        if not folder.is_dir():
            continue
        bundle = joblib.load(folder / 'logistic.joblib')
        name = folder.name
        x = transform_features(base, bundle['recipe']).loc[:, bundle['columns']]
        frozen_probability = bundle['pipeline'].predict_proba(x.loc[replay.index])[:, 1]
        np.testing.assert_allclose(frozen_probability, replay[name + '_probability'], rtol=0, atol=1e-14)
        np.testing.assert_array_equal(frozen_probability >= bundle['threshold'], replay[name])
        # Refit only the validation estimator to check the training path through v1.
        # Saved frozen artifacts stay untouched.
        cfg = bundle['parameters']
        fitted = estimator(cfg['C'], cfg['class_weight']).fit(x.loc[train], truth.loc[train])
        probability = fitted.predict_proba(x.loc[validation.index])[:, 1]
        np.testing.assert_allclose(probability, validation[name + '_probability'], rtol=0, atol=1e-14)
        prediction = probability >= bundle['threshold']
        np.testing.assert_array_equal(prediction, validation[name])
        actual = metrics(validation.truth, prediction)
        np.testing.assert_allclose([actual['BalAcc'], actual['Accuracy']],
                                   scores.loc[name, ['BalAcc', 'Accuracy']].to_numpy(dtype=float),
                                   rtol=0, atol=1e-12)
        checks.append({'model': name, 'validation_rows': len(validation), 'replay_rows': len(replay),
                       'max_replay_probability_error': float(np.max(np.abs(
                           frozen_probability - replay[name + '_probability'].to_numpy()))),
                       'validation_BalAcc': actual['BalAcc']})
        print(f'PASS: {name}, all validation and replay predictions', flush=True)

    model = PriceClassifier(args.results / 'artifacts' / selected)
    replay_days = (replay.index - pd.Timedelta(minutes=30)).normalize().unique()
    start = time.perf_counter()
    worst = 0.
    for i, day in enumerate(replay_days):
        idx = pd.date_range(day + pd.Timedelta(minutes=30), periods=48, freq='30min')
        tick = time.perf_counter()
        predicted = model.predict(data.loc[:day], data.loc[idx, WEATHER])
        worst = max(worst, time.perf_counter() - tick)
        pd.testing.assert_index_equal(predicted.index, replay.loc[idx].index)
        np.testing.assert_array_equal(predicted.nsw_price_up, replay.loc[idx, selected])
        if i % 365 == 0 or i == len(replay_days) - 1:
            print(f'PASS: shared daily interface {i + 1}/{len(replay_days)}', flush=True)
    report = {
        'status': 'passed', 'feature_module': 'src.common.features',
        'feature_module_sha256': hashlib.sha256(Path(shared.__file__).read_bytes()).hexdigest(),
        'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'base_columns': FEATURES, 'model_checks': checks,
        'selected': selected, 'daily_checks': len(replay_days),
        'daily_total_seconds': time.perf_counter() - start, 'worst_predict_seconds': worst,
        'probability_absolute_tolerance': 1e-14,
        'saved_artifacts_modified': False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: shared-v1 validation training, frozen replay and every selected-model day', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
