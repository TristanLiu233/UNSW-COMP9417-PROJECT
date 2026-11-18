# Experiment verification
The original experiment used the released `nem_weather_halfhourly_2010_2026.csv` with SHA256
```text
c13711002906140eaf7cc67c1ce17a49facd359cf3d1a84722aaa3ccd647752c
```

The environment used Python 3.11.18, NumPy 2.4.6, pandas 3.1.0, SciPy 1.17.1, scikit-learn 1.9.1 and joblib 1.6.0. See the [file checksums](reference_sha256.json) and the [shared v1 integration check](v1_integration.md) dated 15 November 2026.

## Checks passed
- The training, validation, frozen refit and replay sets contain 68,784, 17,520, 86,304 and 207,456 intervals respectively. Timestamps are consecutive, unique and assigned to their market day.
- Daily predictions match batch predictions for all 4,322 replay days. The slowest original call took about 0.155 seconds locally. Timing on the marking server may differ.
- Labels and both lag baselines match independent calculations from raw prices. Accuracy and balanced accuracy match interval-level calculations for every month, year and the full period.
- All six artifacts reproduce saved predictions on the first day, leap day and last day tested.
- Extreme future market values leave the current forecast unchanged. Input validation rejects missing, duplicated, reversed and incomplete days.
- Dropping rolling statistics also removes `gap_mean28` and its interactions. Dropping NSW history removes all dependent price gaps.
- Metric tests cover perfect predictions, constant-zero predictions and periods containing a single class. The study requires a fresh output directory.

Two independent runs in the same environment reproduced all 14 result CSVs, including compressed predictions, searches, ablations and intervals, and the same 2014 selection. After packaging the code as `logistic_price`, real-data verification and synthetic checks passed again. The reference result files matched byte for byte.

## Reproduce the checks
Run from the repository root.
```sh
python -m logistic_price.verify --data data/nem_weather_halfhourly_2010_2026.csv --results logistic_price/results/frozen
python -m logistic_price.selfcheck
```

The self-check uses synthetic data. See the main README for the full experiment. The [schedule](p4_schedule_check.md) and [check log](final_check_log.txt) cover the 14-day course example, starter notebook cells, annual class shares and days 37–43 after training. The combined demand and price submission requires its own official check.
