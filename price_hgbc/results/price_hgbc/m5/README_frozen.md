# Stage 1 frozen price-direction model (M5)

Frozen for the Stage 2 replay (2015-01 … 2026-10). Do not edit these files; every change needs a new freeze.

## Model

- P1 boost_price, refit on 2010-01-29..2014-12-31 (Stage 1 frozen model). All HGBC parameters are read from REF (`results/price_hgbc/official/boost_price_ref.json`); nothing is restated in code.
- HistGradientBoostingClassifier: loss=log_loss, max_leaf_nodes=15, max_iter=100, learning_rate=0.1, early_stopping=False, random_state=17, categorical_features='from_dtype'; decision 1 iff p_up ≥ τ = 0.45.
- Training: target days 2010-01-29 … 2014-12-31 (1,798 days, 86,304 half-hours), via P1 ablation.features.load_study (rows up to 2015-01-01 00:00). The first day follows P1's 28-day warm-up (the lag-7 columns alone would allow 2010-01-08).
- sklearn 1.9.1, numpy 2.4.6, pandas 3.1.0, Python 3.11.8; trained 2026-11-11 22:46:43 AEDT in 1.377 s.
- Feature code: P1 `comp9417_project_kit/ablation/features.py` (`_construct`, `_fit_day_types`, `CORE`), commit `116e296`, imported unchanged. P1's day-type model is fitted on ≤ 2013 profiles; only the 18 CORE columns reach the model.

### Features (order = REF = training = prediction)

1. `slot_sin` 2. `slot_cos` 3. `season_sin` 4. `season_cos` 5. `weekend` 6. `weekday_0` 7. `weekday_1` 8. `weekday_2` 9. `weekday_3` 10. `weekday_4` 11. `weekday_5` 12. `weekday_6` 13. `nsw_demand_lag1` 14. `nsw_price_lag1` 15. `nsw_demand_lag7` 16. `nsw_price_lag7` 17. `prev_price_mean` 18. `lag1_price_gap`

## Files and hashes

| File | SHA-256 |
|---|---|
| `results/price_hgbc/m5/stage1_model.joblib` | `bca5291ba0e143db518f220f112b5bcfad127a42d3de7bd3a167edbea246a1d6` |
| `results/price_hgbc/m5/stage1_config.json` | `f7cea1483d46159402593745ad0a26f7c70700d3ab74c20a022daa5243706504` |
| `results/price_hgbc/m5/predictions.parquet` | `9014a23e12df76f1885dd00176c480a8d6a1ab645cc23b00791a81e4226ba8d9` |
| `results/price_hgbc/m5/p1_day_type_model.joblib` | `555cb2fbfcabdd440482260ed858e7fc810e28808f66082eed6f0442c33383a2` |

Prediction arrays: p_up `654fb8007b974719134f3d87e6697619d2bfed2a0d23c7da28d3e56e129729e0`, pred `8640eb8e0575db145785e1fa12dc6aeba3d077b93b97be12f0f86d48267718a4`; tree fingerprint `51ec055f7f67310f1289efafccac53ab7ba72194114ebad550bdaef1ffec6572`. Data CSV `c13711002906140e…`, REF `06b022774a10dad5…`.

## Prediction file (`predictions.parquet`)

- Index: timestamp (end of half-hour, as in the released CSV). Columns: `day` (market day = timestamp − 30 min, normalised), `period` (0–47), `p_up`, `pred` (= 1[p_up ≥ τ]). 4,322 target days 2015-01-01 … 2026-10-31, 207,456 rows. **No labels are stored**; score with `da.targets`.
- Ownership: the frozen replay of the **price** model (boost_price) is produced here, by the price_hgbc workstream (HGBC / CUSUM / S0–S3). P1 supplies the feature code and configuration and does not need to re-run it.
- **To confirm with P1 / P3:** the project files contain no written format agreement. This layout (timestamp index, day, period, p_up, pred) is a proposal until P1 and P3 confirm it.

## Day-by-day interface (for M6 / P3)

`predict_price(model, tau, history, weather_next) -> DataFrame[p_up, nsw_price_up] indexed by weather_next.index` in `src/price_hgbc/m5.py`. Inputs are exactly the kit's `Forecaster.predict` inputs (`history` up to the end of day D, `weather_next` = 48 rows of D+1 weather). It keeps the last 35 days of history, sets D+1 market columns to NaN, and calls P1's builder.

Batch = day-by-day evidence (`daily_vs_batch.csv`):

| Day | history rows | seconds | p_up bitwise equal | pred equal |
|---|---|---|---|---|
| 2015-01-01 | 87,648 | 0.072 | True | True |
| 2015-07-15 | 97,008 | 0.030 | True | True |
| 2016-01-01 | 105,168 | 0.030 | True | True |
| 2016-02-29 | 108,000 | 0.031 | True | True |
| 2017-12-31 | 140,208 | 0.029 | True | True |
| 2018-01-01 | 140,256 | 0.038 | True | True |
| 2019-06-30 | 166,416 | 0.030 | True | True |
| 2020-02-29 | 178,128 | 0.035 | True | True |
| 2021-01-01 | 192,864 | 0.032 | True | True |
| 2022-05-08 | 216,480 | 0.031 | True | True |
| 2023-08-01 | 238,080 | 0.032 | True | True |
| 2024-02-29 | 248,256 | 0.032 | True | True |
| 2024-12-31 | 262,944 | 0.032 | True | True |
| 2025-01-01 | 262,992 | 0.035 | True | True |
| 2026-03-15 | 284,016 | 0.031 | True | True |
| 2026-10-31 | 295,056 | 0.033 | True | True |

## Development scores 2015–2024 (official `da` metrics; 2025–2026 not scored)

| Year | BalAcc model | BalAcc b1 | BalAcc b7 | Acc model | Acc b1 | Acc b7 |
|---|---|---|---|---|---|---|
| 2015 | 78.492 | 74.721 | 72.059 | 78.550 | 74.789 | 72.357 |
| 2016 | 73.388 | 69.835 | 67.560 | 75.632 | 71.101 | 68.670 |
| 2017 | 66.181 | 68.706 | 68.290 | 70.331 | 69.258 | 68.642 |
| 2018 | 68.195 | 72.256 | 69.103 | 72.803 | 73.333 | 69.949 |
| 2019 | 68.845 | 71.527 | 69.100 | 73.293 | 72.483 | 69.475 |
| 2020 | 74.903 | 72.257 | 69.936 | 76.685 | 73.793 | 71.027 |
| 2021 | 72.845 | 72.850 | 69.264 | 75.303 | 73.870 | 69.949 |
| 2022 | 62.468 | 70.816 | 67.413 | 63.556 | 70.856 | 67.511 |
| 2023 | 68.361 | 76.005 | 73.310 | 70.303 | 76.142 | 73.487 |
| 2024 | 68.890 | 76.394 | 72.476 | 72.131 | 76.702 | 72.854 |
| 2015–2024 | 70.082 | 72.668 | 69.894 | 72.860 | 73.233 | 70.392 |

Monthly values: `dev_monthly_2015_2024.csv`. These scores were not used for any choice.

## CUSUM for M6 (M4, option A)

- Config `results/price_hgbc/m4/cusum_config.json`: ref_update = **A**; q99 = 1.0494324087377636, μ₀ = 0.4434939030076151, σ₀ = 0.19881467766062375, k = 0.5, h = 6. **Read these from the config, not from this text.** S resets to 0 after a retrain. Historical 2014 alarm kept: 2014-09-26 (results/price_hgbc/m4/cusum_path_2014.csv).
- **μ₀ and σ₀ come from an earlier model** (REF, trained to 2013) and its 2014 out-of-sample errors. The Stage 1 model here is refit to 2014, so its normal error distribution may differ. M6 must check S3's alarm frequency, any cooldown and the budget of at most 6 retrains. It must not assume the M4 ARL₀ holds for every new model.

## Known limitations

- τ = 0.45 was chosen on 2014 for models trained on 2010–2013. After refitting to 2010–2014 the probability distribution can shift, so τ = 0.45 is not re-optimised for this model. This is a known limitation, kept on purpose (no re-tuning).
- In M6, S3's CUSUM must read the 2025–2026 truth to update its statistic. That is a legitimate part of the simulation, but **2025–2026 scores must not be used for any choice** (they are the M8 hold-out).
- The batch builds features on the whole file. Causality is shown by tests: shuffling D+1-onward market data leaves D+1 unchanged, and batch = day-by-day bitwise.
