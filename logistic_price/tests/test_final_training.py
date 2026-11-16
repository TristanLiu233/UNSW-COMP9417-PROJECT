import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from logistic_price import final_training as ft
from logistic_price.classifier import PriceClassifier
from logistic_price.features import RAW, WEATHER, build_features
from logistic_price.transforms import transform_features


class FinalTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(17)
        index = pd.date_range('2010-01-01 00:30', periods=110*48, freq='30min')
        cls.data = pd.DataFrame(rng.normal(size=(len(index), len(RAW))), index=index, columns=RAW)
        cls.plan = {'validation_folds': [('2010-03-01', '2010-03-15'), ('2010-03-16', '2010-03-31')],
                    'thresholds': [.4, .42, .5, .6]}

    def test_nested_scalers_and_refit_boundaries(self):
        fitted = []
        original = ft.estimator
        def record(*args):
            model = original(*args)
            fitted.append(model)
            return model
        with patch.object(ft, 'estimator', side_effect=record):
            bundle, audit = ft.fit_final(self.data, '2010-03-31', **self.plan)
        x = transform_features(build_features(ft.training_prefix(self.data, '2010-03-31')), bundle['recipe']).iloc[28*48:]
        days = (x.index-pd.Timedelta(minutes=30)).normalize()
        self.assertEqual(len(fitted), 3)
        for model, end in zip(fitted, ['2010-02-28', '2010-03-15', '2010-03-31']):
            subset = x.loc[days <= end]
            scaler = model.named_steps['standardscaler']
            self.assertEqual(scaler.n_samples_seen_, len(subset))
            np.testing.assert_allclose(scaler.mean_, subset.mean().to_numpy(), atol=1e-14)
        self.assertEqual(audit['train_rows'], 62*48)
        self.assertEqual(len(audit['search']), 4)

    def test_future_isolation_and_serialized_daily_prediction(self):
        altered = self.data.copy()
        altered.loc[altered.index > '2010-04-01', :] = np.nan
        a, aa = ft.fit_final(self.data, '2010-03-31', **self.plan)
        b, bb = ft.fit_final(altered, '2010-03-31', **self.plan)
        self.assertEqual(aa, bb)
        np.testing.assert_array_equal(a['pipeline'].named_steps['logisticregression'].coef_, b['pipeline'].named_steps['logisticregression'].coef_)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'new'
            ft.train_final(self.data, '2010-03-31', output, **self.plan)
            weather = self.data.loc['2010-04-01 00:30':'2010-04-02 00:00', WEATHER]
            predicted = PriceClassifier(output).predict(self.data.loc[:pd.Timestamp('2010-04-01')], weather)
            x = transform_features(build_features(self.data), a['recipe']).loc[weather.index]
            expected = (a['pipeline'].predict_proba(x)[:, 1] >= a['threshold']).astype(int)
            np.testing.assert_array_equal(predicted.nsw_price_up, expected)
            self.assertTrue(predicted.index.equals(weather.index))
            with self.assertRaises(FileExistsError):
                ft.train_final(self.data, '2010-03-31', output)

    def test_default_last_year_policy(self):
        index = pd.date_range('2010-01-01 00:30', '2012-03-01 00:00', freq='30min')
        data = pd.DataFrame(np.random.default_rng(9).normal(size=(len(index), len(RAW))), index=index, columns=RAW)
        _, audit = ft.fit_final(data, '2012-02-29')
        self.assertEqual(audit['folds'][0]['train_end'], '2011-02-28')
        self.assertEqual(audit['folds'][0]['validation_start'], '2011-03-01')
        self.assertEqual(audit['folds'][0]['validation_rows'], 366*48)
        self.assertEqual(len(audit['search']), 36)

    def test_fixed_selected_threshold(self):
        _, audit = ft.fit_final(self.data, '2010-03-31', fixed_threshold=True)
        self.assertEqual(audit['threshold'], .42)
        self.assertEqual(audit['threshold_policy'], 'selected_stage1_fixed')

    def test_reject_invalid_prefixes_and_plans(self):
        for data in [self.data.iloc[1:], self.data.drop(self.data.index[100]), self.data.iloc[::-1]]:
            with self.subTest(prefix=len(data)), self.assertRaises(ValueError):
                ft.fit_final(data, '2010-03-31')
        for plan in [
            {'validation_folds': [], 'thresholds': [.5]},
            {'validation_folds': [('2010-03-01', '2010-04-01')], 'thresholds': [.5]},
            {'validation_folds': [('2010-03-01', '2010-03-20'), ('2010-03-15', '2010-03-31')], 'thresholds': [.5]},
            {'validation_folds': self.plan['validation_folds'], 'thresholds': [np.nan]},
            {'validation_folds': self.plan['validation_folds'], 'thresholds': [-1]},
            {'thresholds': [.5]},
        ]:
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                ft.fit_final(self.data, '2010-03-31', **plan)
        for cutoff in ['2026-11-01', '2010-03-31 00:30', '2010-03-31T00:00:00+10:00']:
            with self.subTest(cutoff=cutoff), self.assertRaises(ValueError):
                ft.fit_final(self.data, cutoff)


if __name__ == '__main__':
    unittest.main()
