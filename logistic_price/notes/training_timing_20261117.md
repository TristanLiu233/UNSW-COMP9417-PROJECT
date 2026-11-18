# P4 training time and reproducibility, 17 November 2026
One complete final-training CLI run took **3.345846 seconds** (about 3.346 seconds). It reproduced the 16 November artifact byte for byte and all 672 predictions on the 14 days checked. See the [machine-readable record](training_timing_20261117.json).

## Timing conditions
- Commit `c23c95956eea95dd56e311ccb22a25fdbf2b5457`. Apple M2, eight logical cores, 16 GiB RAM, macOS 26.3.1 arm64, Python 3.11.18 and one fitting thread. The JSON lists dependency versions.
- The parent process used `time.perf_counter()` around a fresh Python CLI process. Timing includes startup, imports, CSV prefix reading, feature construction, nested threshold selection, final fitting, audit digest and file writes. Exit code was 0.
- Artifact comparison and prediction checks ran after the timer stopped. This single local measurement used the existing OS file cache.

## Configuration
The selected log_interactions31 model used 31 features, C=1, class_weight=None and seed 17. Inner training covered 2010-01-29 to 2025-10-31 (276,240 rows). Validation covered 2025-11-01 to 2026-10-31 (17,520 rows). Of 36 thresholds from 0.25 to 0.60 in steps of 0.01, τ=0.49 was selected. The final fit used 293,760 rows through 2026-10-31.

## Reproducibility checks
The two training.json files matched. The scaler's mean_, scale_, var_ and n_samples_seen_ matched exactly, as did the classifier's coef_, intercept_, classes_ and n_iter_. Both bundles recorded identical column order, recipe, threshold, cutoff, row count and settings. Both logistic.joblib files had SHA256 `7e898b2aebd146b83f40df869a9ffbc728bc447ca22970e872b51b103131dd50`. PriceClassifier.predict returned identical indexes and predictions on all 14 complete market days from 18 to 31 October 2026.

## Repeat the timing
Run from the repository root with the course environment and a fresh output path.
```sh
/usr/bin/time -p .venv311/bin/python -m logistic_price.final_training \
  --data /absolute/path/nem_weather_halfhourly_2010_2026.csv \
  --cutoff 2026-10-31 \
  --output /absolute/path/new-price-artifact
```

The system timer's `real` field gives elapsed wall time. The JSON contains the precise parent-process measurement, versions and hashes for the data, code and artifacts.

## Handoff
P1 still needs to time the combined train.py in the target environment against the two-hour limit and run check_submission on the full package. The schedule reserved 18 November for reviewing 2025–2026. The 16 November fit already used this period for threshold selection. The 17 November run checked timing and reproducibility using the same settings. The 14-day check measured agreement between runs. This timing release contained the note and its JSON record.
