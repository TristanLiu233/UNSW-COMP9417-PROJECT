"""Verify completed replay against timestamps, original labels, lags and public interfaces."""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .features import RAW, build_features, build_features_for_day
from .transforms import RECIPES, transform_features
from .classifier import PriceClassifier
from .tuning import metrics
from .scoring import intervals
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True,type=Path);parser.add_argument('--results',required=True,type=Path)
    args=parser.parse_args();out=args.results
    metadata=json.loads((out/'run.json').read_text())
    assert metadata['status']=='completed' and metadata['daily_checks']==4322
    data=pd.read_csv(args.data,usecols=['timestamp',*RAW],parse_dates=True,index_col='timestamp')
    base=build_features(data)
    pred=pd.read_csv(out/'replay_predictions.csv.gz',parse_dates=True,index_col='timestamp')
    expected=pd.date_range('2015-01-01 00:30','2026-11-01 00:00',freq='30min')
    assert pred.index.equals(expected) and pred.index.is_unique
    # Derive yesterday means independently from complete 48-slot arrays.
    prices=data.nsw_price.to_numpy().reshape(-1,48)
    yesterday=np.repeat(np.r_[np.nan,prices.mean(axis=1)[:-1]],48)
    truth=(data.nsw_price.to_numpy()>yesterday).astype(int)
    np.testing.assert_array_equal(pred.truth,truth[-len(pred):])
    for lag in (1,7):
        past=data.nsw_price.shift(lag*48).to_numpy()
        np.testing.assert_array_equal(pred[f'lag{lag}'],(past>yesterday)[-len(pred):])
    for frequency in ['overall','yearly','monthly']:
        table=pd.read_csv(out/f'{frequency}.csv',dtype={'period':str})
        days=pred.index-pd.Timedelta(minutes=30)
        labels = None if frequency=='overall' else days.strftime('%Y' if frequency=='yearly' else '%Y-%m')
        for _,row in table.iterrows():
            block=pred if frequency=='overall' else pred.loc[labels==row.period]
            m=metrics(block.truth,block[row.model])
            np.testing.assert_allclose([row.BalAcc,row.Accuracy],[m['BalAcc'],m['Accuracy']],atol=1e-10)
            assert row.n_slots==len(block)
        if frequency=='monthly': assert table.period.nunique()==142
    for model_dir in (out/'artifacts').iterdir():
        cfg=joblib.load(model_dir/'logistic.joblib')
        model=PriceClassifier(model_dir)
        for date in ['2015-01-01','2020-02-29','2026-10-31']:
            day=pd.Timestamp(date);idx=pd.date_range(day+pd.Timedelta(minutes=30),periods=48,freq='30min')
            weather=pd.DataFrame(index=idx)
            single=build_features_for_day(data.loc[:day],weather)
            np.testing.assert_allclose(transform_features(single,cfg['recipe']),transform_features(base.loc[idx],cfg['recipe']),atol=1e-8)
            np.testing.assert_array_equal(model.predict(data.loc[:day],weather).nsw_price_up,pred.loc[idx,model_dir.name])
            poisoned=data.loc[day-pd.Timedelta(days=28)+pd.Timedelta(minutes=30):day+pd.Timedelta(days=1)].copy()
            poisoned.loc[poisoned.index>day,RAW]=1e12
            np.testing.assert_allclose(single,build_features_for_day(poisoned,weather),atol=1e-8)
    ab=pd.read_csv(out/'selected_ablation_2014.csv')
    rolling=json.loads(ab.loc[ab.removed=='rolling28','columns'].iloc[0])
    assert not any('mean28' in c or 'std28' in c for c in rolling)
    hist=json.loads(ab.loc[ab.removed=='history','columns'].iloc[0])
    assert not any(c.startswith('gap') for c in hist)
    rows,_=intervals(np.tile([0,1],48),{'perfect':np.tile([0,1],48),'zeros':np.zeros(96,dtype=int)},200)
    assert rows[0]['BalAcc']==100 and rows[1]['BalAcc']==50
    rows,_=intervals(np.zeros(48,dtype=int),{'zeros':np.zeros(48,dtype=int)},200)
    assert rows[0]['BalAcc']==100
    print('PASS: full replay timestamps/targets/baselines; all overall/yearly/monthly metrics; all saved model interfaces; future poisoning; derived-feature ablations; metric edge cases.')


if __name__=='__main__':
    with threadpool_limits(limits=1): main()
