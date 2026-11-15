"""Synthetic smoke checks only: python -m logistic_price.selfcheck (no course scores)."""
from pathlib import Path
import tempfile
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .features import MARKET, WEATHER, build_features, build_features_for_day
from .classifier import PriceClassifier
from .tuning import run


def main():
    rng = np.random.default_rng(17)
    index = pd.date_range('2010-01-01 00:30', '2015-01-01 00:00', freq='30min')
    t = np.arange(len(index))
    data = pd.DataFrame({'nsw_demand': 6500 + 800*np.sin(t*2*np.pi/48) + rng.normal(0,200,len(t)),
                         'nsw_price': 50 + 20*np.sin(t*2*np.pi/48) + rng.normal(0,15,len(t)),
                         'qld_demand': 5000 + 500*np.cos(t*2*np.pi/48) + rng.normal(0,200,len(t))}, index=index)
    for name in ['vic_demand', 'vic_price', *WEATHER]:
        data[name] = rng.normal(size=len(data))
    x = build_features(data)
    for date in ['2012-02-29', '2014-01-01', '2014-12-31']:
        day = pd.Timestamp(date)
        weather = data.loc[pd.date_range(day + pd.Timedelta(minutes=30), periods=48, freq='30min'), WEATHER]
        single = build_features_for_day(data.loc[:day], weather)
        pd.testing.assert_frame_equal(single, x.loc[weather.index], check_exact=True)
        for fill in (np.nan, 1e12):
            poisoned = weather.copy()
            poisoned[MARKET] = fill
            pd.testing.assert_frame_equal(single, build_features_for_day(data.loc[:day], poisoned), check_exact=True)
    for bad in [data.iloc[:-1], data.drop(index[100]), data.iloc[::-1], pd.concat([data.iloc[:48], data.iloc[:48]])]:
        try:
            build_features(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Malformed timestamps accepted')
    with tempfile.TemporaryDirectory(prefix='p4-synthetic-') as temporary:
        source, output = Path(temporary)/'synthetic.csv', Path(temporary)/'results'
        data.to_csv(source, index_label='timestamp')
        print('SYNTHETIC DATA ONLY — the following numbers are NOT course results.')
        with warnings.catch_warnings(), threadpool_limits(limits=1):
            warnings.simplefilter('error', ConvergenceWarning)
            run(source, output)
        assert len(pd.read_csv(output/'search.csv')) == 30
        assert len(pd.read_csv(output/'ablation.csv')) == 6
        predicted = PriceClassifier(output).predict(data.loc[:day], weather)
        saved = pd.read_csv(output/'predictions.csv.gz', index_col='timestamp', parse_dates=True)
        np.testing.assert_array_equal(predicted.nsw_price_up, saved.loc[weather.index, 'full'])
        try:
            run(source, output)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing output overwritten')
    print('PASS: synthetic feature parity, future isolation, invalid inputs, tuning, ablations, serialization, overwrite protection')


if __name__ == '__main__':
    main()
