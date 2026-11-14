# M6 strategy design (pre-declared before any strategy was run; not changed during execution)

Model: P1 `boost_price`. Every HGBC parameter, **including random_state = 17**, is read from REF
(`official/boost_price_ref.json`); the user authorised this (M6 rule A). S0 is the M5 frozen model.
Replay period 2015-01-01 … 2026-10-31 (4,322 days). Development scoring 2015-01-01 … 2024-12-31 only; 2025–2026 are
predicted and simulated causally but not scored or ranked.

## Common rules
1. **Features:** the M5 batch builder (P1 `_construct`, CORE, REF order). Every row of target day d uses market data of days ≤ d−1
   and the weather of d. Labels are kept outside the feature table.
2. **Labels:** `da.targets` per day, through a gate that releases a day's labels only at the end of that day.
   Training at cutoff D can read labels of days ≤ D only. The CUSUM reads day d's labels only after d's predictions are stored.
3. **Retrain at the end of day D:** the new model forecasts from D+1 (effective day). It never predicts day D or earlier.
4. **τ selection (S1, S2, S3, identical):**
   * validation = the last 12 months of the training window, (D − 1 year, D] by calendar offset;
   * fit a tuning model (REF params) on the training-window days ≤ D − 1 year, and predict the validation days out of sample;
   * pick τ with P1's protocol (thresholds 0.40, 0.45, 0.50, 0.55, 0.60; highest BalAcc; ties → closest to 0.5), the
     protocol that produced REF's τ = 0.45;
   * fix τ, refit the final model on the full training window, and use it from D+1.
   * The validation days are never part of the tuning model's training data.
5. **Windows:** S1 and S3 expanding (2010-01-29 … D); S2 sliding, the three calendar years (D − 3 years, D]
   (2016 retrain: 2014-01-01 … 2016-12-31, 1,096 days). Older days may feed lag features but are not training targets.
6. **Training size:** every fit logs its row count and whether it exceeds 200,000 rows (above that, sklearn's binning
   subsamples using random_state = 17).
7. **Model ids:** `M5_frozen` (S0, and every strategy before its first retrain); otherwise `<S>_r<n>_<cutoff>`.

## Strategies
* **S0:** the M5 frozen model for every day, no retraining. It must equal `m5/predictions.parquet` bitwise.
* **S1:** expanding-window retrain at the end of 2016, 2018, 2020, 2022 and 2024; effective 1 January of the next year (5 retrains).
* **S2:** same dates as S1, three-calendar-year sliding window (5 retrains).
* **S3:** CUSUM-triggered, M4 option A. Each day d:
  1. predict with the model in force;
  2. store the predictions;
  3. release d's labels and compute d's mean log loss (clip 1e-6);
  4. `Cusum.update` with the full-precision config values; S resets to 0 on **every** alarm day, including suppressed alarms;
  5. if alarm: retrain iff (no earlier S3 retrain **or** d − alarm day of the last actual retrain ≥ 180 days) **and** fewer
     than 6 retrains so far; otherwise record the suppression reason (`cooldown`, else `budget`);
  6. retrain = S1's window and τ method with cutoff d; effective d+1;
  7. μ₀, σ₀, q99, k, h never change (option A).
  S = 0 on 2015-01-01; the 2014-09-26 alarm is a calibration record only.

## Evaluation (2015–2024)
* Acc and BalAcc (official `da`) per strategy, per year, per month; b1 and b7 via `da.baseline`; the same days for every strategy.
* Paired whole-day bootstrap: n_boot 1000, seed 0, one index matrix (3,653 days) for all strategies and b1.
  Differences: S1/S2/S3 − S0, S3 − S1, each strategy − b1.
* Batch vs day-by-day: S1/S2 at ≥ 3 segments × 3 days via `m5.predict_price`; S3 via a full day-by-day replay
  (features from `m5.features_for_day` for every day), with alarm days and predictions compared.
