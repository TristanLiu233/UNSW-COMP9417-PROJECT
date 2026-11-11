"""Reproducible feature comparison, refit through 2014 and frozen 2015-2026 replay."""
import argparse
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .features import RAW, GROUPS, build_features, build_features_for_day
from .transforms import RECIPES, transform_features
from .tuning import estimator, metrics
from .classifier import PriceClassifier
from .scoring import summarize, plots, intervals


def search(x, y):
    days = (x.index - pd.Timedelta(minutes=30)).normalize()
    rows = []
    for c in (.01, .1, 1., 10.):
        for weight in (None, 'balanced'):
            probs, truths = [], []
            for start, end in [('2013-01-01','2013-07-01'), ('2013-07-01','2014-01-01')]:
                train, valid = days < start, (days >= start) & (days < end)
                if y.loc[train].nunique() != 2 or not valid.any():
                    raise ValueError('Invalid temporal folds')
                fitted = estimator(c, weight).fit(x.loc[train], y.loc[train])
                probs.append(fitted.predict_proba(x.loc[valid])[:,1]); truths.append(y.loc[valid].to_numpy())
            for threshold in [i/100 for i in range(25,61)]:
                fold_scores = [metrics(t,p>=threshold)['BalAcc'] for t,p in zip(truths,probs)]
                rows.append({'C':c, 'class_weight':weight, 'threshold':threshold,
                             'fold1_BalAcc':fold_scores[0], 'fold2_BalAcc':fold_scores[1],
                             **metrics(np.concatenate(truths),np.concatenate(probs)>=threshold)})
    best = min(rows,key=lambda r:(-r['BalAcc'],abs(r['threshold']-.5),r['C']!=1,r['class_weight'] is not None,r['C']))
    return dict(best), pd.DataFrame([{**r, 'chosen':r is best} for r in rows])


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def write_ablations(base, y, train, valid, chosen, output):
    x = transform_features(base, chosen['recipe'])
    full_model = estimator(chosen['C'], chosen['class_weight']).fit(x.loc[train], y.loc[train])
    full = metrics(y.loc[valid], full_model.predict_proba(x.loc[valid])[:,1] >= chosen['threshold'])['BalAcc']
    rows = []
    for group, removed in GROUPS.items():
        cols = [c for c in x.columns if c not in removed]
        if group == 'history':
            cols = [c for c in cols if not c.startswith('gap')]
        if group == 'calendar':
            cols = [c for c in cols if '_x_slot_' not in c]
        if group == 'rolling28':
            cols = [c for c in cols if not c.startswith('gap_mean28')]
        fitted = estimator(chosen['C'], chosen['class_weight']).fit(x.loc[train,cols], y.loc[train])
        pred = fitted.predict_proba(x.loc[valid,cols])[:,1] >= chosen['threshold']
        rows.append({'removed': group, 'n_features': len(cols), 'columns': json.dumps(cols), **metrics(y.loc[valid], pred)})
    table = pd.DataFrame(rows)
    table['delta_vs_full_pp'] = table.BalAcc - full
    table.to_csv(output/'selected_ablation_2014.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output directory; existing results are never overwritten')
    expected = pd.date_range('2010-01-01 00:30', '2026-11-01 00:00', freq='30min')
    data = pd.read_csv(args.data, usecols=['timestamp',*RAW], index_col='timestamp',parse_dates=True)
    if not data.index.equals(expected) or not np.isfinite(data.to_numpy()).all():
        raise ValueError('Expected complete finite released data through October 2026')
    base = build_features(data)
    days = (data.index-pd.Timedelta(minutes=30)).normalize()
    y = (data.nsw_price > base.prev_price_mean).astype(int)
    train = (days >= '2010-01-29') & (days < '2014-01-01')
    valid = days.year == 2014
    refit = (days >= '2010-01-29') & (days < '2015-01-01')
    replay = days >= '2015-01-01'
    assert [int(m.sum()) for m in (train,valid,refit,replay)] == [68784,17520,86304,207456]
    output = args.output; output.mkdir(parents=True,exist_ok=False)
    write_json(output/'protocol.json', {'recipes':list(RECIPES),'C':[.01,.1,1,10],
        'class_weight':[None,'balanced'],'thresholds':[i/100 for i in range(25,61)],
        'tuning':'pooled 2013 held-out BalAcc per recipe',
        'selection':'2014 BalAcc among four predefined tuned recipes, then fewer columns, then name',
        'validation':'2014 final recipe selection; not used by the internal hyperparameter search',
        'replay':'2015-2026 frozen; no selection or retuning based on replay'})
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter('error',ConvergenceWarning)
        winners, grids, frames = [], [], {}
        for recipe in RECIPES:
            print(f'Searching {recipe}',flush=True)
            x = transform_features(base,recipe); frames[recipe]=x
            best, grid = search(x.loc[train],y.loc[train])
            grids.append(grid.assign(recipe=recipe))
            winners.append({'recipe':recipe,'n_features':x.shape[1],**best})
        chosen = min(winners,key=lambda r:(-r['BalAcc'],r['n_features'],r['recipe']))
        selected = chosen['recipe']
        # Persist selection BEFORE any 2014 scoring or future replay.
        write_json(output/'selection_2013.json',chosen)
        pd.concat(grids,ignore_index=True).to_csv(output/'search_2013.csv',index=False)
        pd.DataFrame(winners).to_csv(output/'candidates_2013.csv',index=False)
        configs = {r['recipe']:r for r in winners}
        configs.update({'document_reference':{'recipe':'raw22','C':1.,'class_weight':None,'threshold':.4},
                        'previous_raw22':{'recipe':'raw22','C':1.,'class_weight':'balanced','threshold':.4}})
        validation = pd.DataFrame({'truth':y.loc[valid]})
        predictions = pd.DataFrame({'truth':y.loc[replay]})
        validation_rows = []
        for name,cfg in configs.items():
            print(f'Validating/refitting {name}',flush=True)
            x=frames[cfg['recipe']]
            fitted=estimator(cfg['C'],cfg['class_weight']).fit(x.loc[train],y.loc[train])
            prob=fitted.predict_proba(x.loc[valid])[:,1]
            validation[name]=(prob>=cfg['threshold']).astype(int)
            validation[name+'_probability']=prob
            validation_rows.append({'model':name,'recipe':cfg['recipe'], 'n_features':x.shape[1],
                                    'C':cfg['C'],'class_weight':cfg['class_weight'],'threshold':cfg['threshold'],
                                    **metrics(y.loc[valid],prob>=cfg['threshold'])})
            frozen=estimator(cfg['C'],cfg['class_weight']).fit(x.loc[refit],y.loc[refit])
            target=output/'artifacts'/name;target.mkdir(parents=True)
            joblib.dump({'pipeline':frozen,'columns':list(x.columns),'threshold':cfg['threshold'],
                         'recipe':cfg['recipe'],'parameters':cfg,'train_end':'2014-12-31',
                         'train_rows':86304},target/'logistic.joblib')
            prob=frozen.predict_proba(x.loc[replay])[:,1]
            predictions[name]=(prob>=cfg['threshold']).astype(int)
            predictions[name+'_probability']=prob
        final_choice = min((r for r in validation_rows if r['model'] in RECIPES),
                           key=lambda r: (-r['BalAcc'], r['n_features'], r['model']))
        selected = final_choice['model']
        chosen = configs[selected]
        write_json(output/'selection_2014.json', final_choice)
        for lag in (1,7):
            bp=(base[f'nsw_price_lag{lag}']>base.prev_price_mean).astype(int)
            predictions[f'lag{lag}']=bp.loc[replay]
            validation[f'lag{lag}']=bp.loc[valid]
            validation_rows.append({'model':f'lag{lag}',**metrics(y.loc[valid],bp.loc[valid])})
        pd.DataFrame(validation_rows).to_csv(output/'validation_2014.csv',index=False)
        validation.to_csv(output/'validation_predictions.csv.gz',index_label='timestamp')
        write_ablations(base, y, train, valid, chosen, output)
        # Entire selected-model replay is checked via the public daily interface.
        classifier=PriceClassifier(output/'artifacts'/selected)
        worst_seconds=0.; start=time.perf_counter()
        for i,day in enumerate(days[replay].unique()):
            idx=pd.date_range(day+pd.Timedelta(minutes=30),periods=48,freq='30min')
            past=data.loc[day-pd.Timedelta(days=28)+pd.Timedelta(minutes=30):day]
            weather=pd.DataFrame(index=idx)
            tick=time.perf_counter()
            daily=classifier.predict(past,weather)
            worst_seconds=max(worst_seconds,time.perf_counter()-tick)
            np.testing.assert_array_equal(daily.nsw_price_up,predictions.loc[idx,selected])
            if i%365==0:print(f'Daily interface verified: {i+1}/4322 days',flush=True)
        daily_seconds=time.perf_counter()-start
        predictions.to_csv(output/'replay_predictions.csv.gz',index_label='timestamp')
        print('Computing overall/yearly/monthly intervals',flush=True)
        tables=summarize(predictions,selected,output)
        plots(tables,selected,output)
        # Pair selected against fixed reference and prior P4 on the same 2014 days.
        rows,samples=intervals(validation.truth.to_numpy(),{n:validation[n].to_numpy() for n in [selected,'document_reference','previous_raw22']})
        points={r['model']:r['BalAcc'] for r in rows}
        pair=[]
        for n in ['document_reference','previous_raw22']:
            lo,hi=np.quantile(samples[selected]-samples[n],[.025,.975])
            pair.append({'candidate':selected,'reference':n,'delta_BalAcc_pp':points[selected]-points[n],'day_low':lo,'day_high':hi})
        pd.DataFrame(pair).to_csv(output/'paired_validation_2014.csv',index=False)
        write_json(output/'run.json',{'status':'completed','selected':selected,
          'csv_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),
          'train_rows':68784,'validation_rows':17520,'refit_rows':86304,'replay_rows':207456,
          'replay_days':4322,'replay_months':142,'bootstrap_repetitions':2000,'seed':17,
          'daily_checks':4322,'daily_total_seconds':daily_seconds,'worst_predict_seconds':worst_seconds,
          'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                      'scipy':scipy.__version__,'sklearn':sklearn.__version__,'joblib':joblib.__version__}})
        from .diagnostics import generate
        generate(output)
        print(tables['overall'][['model','BalAcc','Accuracy']].to_string(index=False),flush=True)


if __name__=='__main__':
    main()
