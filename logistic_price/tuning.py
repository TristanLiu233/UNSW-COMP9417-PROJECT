"""python -m logistic_price.tuning --data COURSE.csv --output NEW_DIRECTORY"""
import argparse
import hashlib
import json
import platform
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from .features import FEATURES, GROUPS, RAW, build_features


def estimator(c, weight):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight=weight, max_iter=2000, random_state=17))


def metrics(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    return {'BalAcc': float(100 * np.mean([np.mean(prediction[y == c] == c) for c in np.unique(y)])),
            'Accuracy': float(100 * np.mean(prediction == y))}


def tune(x, y, search_profile="standard"):
    if search_profile not in ("standard", "extended"):
        raise ValueError("Unknown search profile")
    cs = (0.01, 0.1, 1.0) if search_profile == "standard" else (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
    weights = (None, "balanced") if search_profile == "standard" else (None, "balanced", {0: 1, 1: .75}, {0: 1, 1: 1.25}, {0: 1, 1: 1.5})
    thresholds = (0.40, 0.45, 0.50, 0.55, 0.60) if search_profile == "standard" else tuple(i / 100 for i in range(20, 66))
    days = (x.index - pd.Timedelta(minutes=30)).normalize()
    folds = [('2013-01-01', '2013-07-01'), ('2013-07-01', '2014-01-01')]
    rows = []
    for c in cs:
        for weight in weights:
            truths, probabilities = [], []
            for start, stop in folds:
                train, valid = days < start, (days >= start) & (days < stop)
                if not train.any() or not valid.any() or y.loc[train].nunique() != 2:
                    raise ValueError('Insufficient temporal fold data or missing training class')
                fitted = estimator(c, weight).fit(x.loc[train], y.loc[train])
                truths.append(y.loc[valid].to_numpy())
                probabilities.append(fitted.predict_proba(x.loc[valid])[:, 1])
            truth, probability = np.concatenate(truths), np.concatenate(probabilities)
            for threshold in thresholds:
                rows.append({'C': c, 'class_weight': weight, 'threshold': threshold,
                             **metrics(truth, probability >= threshold)})
    best = min(rows, key=lambda r: (-r['BalAcc'], abs(round(r['threshold'] * 100) - 50),
                                   r['C'] != 1, r['class_weight'] is not None, r['C']))
    return dict(best), pd.DataFrame([{**r, 'chosen': r is best} for r in rows])


def run(data_path, output, search_profile="standard"):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}; choose a new output directory')
    expected = pd.date_range('2010-01-01 00:30', '2015-01-01 00:00', freq='30min')
    data = pd.read_csv(data_path, nrows=len(expected), usecols=['timestamp', *RAW],
                       parse_dates=['timestamp'], index_col='timestamp')
    if not data.index.equals(expected) or not np.isfinite(data[RAW].to_numpy(dtype=float)).all():
        raise ValueError('Need the complete finite 2010–2014 course CSV prefix in timestamp order')
    x = build_features(data)
    days = (data.index - pd.Timedelta(minutes=30)).normalize()
    y = (data['nsw_price'] > x['prev_price_mean']).astype(int)
    train = (days >= '2010-01-29') & (days < '2014-01-01')
    valid = days.year == 2014
    if train.sum() != 68784 or valid.sum() != 17520 or not np.isfinite(x.loc[train | valid]).all().all():
        raise ValueError('Unexpected eligible rows or invalid features')
    best, search = tune(x.loc[train], y.loc[train], search_profile)
    variants = {'document_reference': FEATURES, 'full': FEATURES}
    variants.update({f'without_{name}': [c for c in FEATURES if c not in group] for name, group in GROUPS.items()})
    rows, predictions, artifact = [], pd.DataFrame({'truth': y.loc[valid]}), None
    for name, cols in variants.items():
        c, weight, threshold = (1, None, .40) if name == 'document_reference' else (best['C'], best['class_weight'], best['threshold'])
        fitted = estimator(c, weight).fit(x.loc[train, cols], y.loc[train])
        probability = fitted.predict_proba(x.loc[valid, cols])[:, 1]
        predicted = (probability >= threshold).astype(int)
        predictions[name] = predicted
        predictions[name + '_probability'] = probability
        rows.append({'experiment': name, 'n_features': len(cols), 'columns': json.dumps(cols),
                     'C': c, 'class_weight': weight, 'threshold': threshold, **metrics(y.loc[valid], predicted)})
        if name == 'full':
            artifact = {'pipeline': fitted, 'columns': cols, 'threshold': threshold,
                        'train_end': '2013-12-31', 'parameters': best}
    results = pd.DataFrame(rows)
    full = results.loc[results.experiment == 'full', 'BalAcc'].iloc[0]
    results['delta_vs_full_pp'] = results.BalAcc - full
    digest = hashlib.sha256()
    with data_path.open('rb') as source:
        for _ in range(len(expected) + 1):
            digest.update(source.readline())
    metadata = {'status': 'completed', 'search_profile': search_profile, 'search_candidates': len(search), 'csv_prefix_sha256': digest.hexdigest(), 'train_rows': int(train.sum()),
                'validation_rows': int(valid.sum()), 'selection': best,
                'tuning': 'expanding training; held-out Jan-Jun and Jul-Dec 2013; pooled scores',
                'ablation': 'fixed full-model C, class_weight and threshold; refit each reduced feature set',
                'caveat': '22-column recipe previously selected on 2014; 2014 is selection validation, not independent test',
                'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                             'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'joblib': joblib.__version__}}
    output.mkdir(parents=True, exist_ok=False)
    search.to_csv(output / 'search.csv', index=False)
    results.to_csv(output / 'ablation.csv', index=False)
    predictions.to_csv(output / 'predictions.csv.gz', index_label='timestamp')
    joblib.dump(artifact, output / 'logistic.joblib')
    (output / 'run.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(results[['experiment', 'BalAcc', 'Accuracy', 'delta_vs_full_pp']].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--search-profile', choices=['standard', 'extended'], default='standard')
    args = parser.parse_args()
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter('error', ConvergenceWarning)
        run(args.data, args.output, args.search_profile)


if __name__ == '__main__':
    main()
