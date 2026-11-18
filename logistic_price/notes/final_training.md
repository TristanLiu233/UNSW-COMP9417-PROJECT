# P4 expanding-window training and handoff, 16 November 2026
Pages 4–7 of `specs/project_spec.pdf` allow Stage 3 training through 31 October 2026. Each retrain uses the history available on its date. Stage 2 allows six replacements per model and strategy after initial training. The live package loads its fitted model and keeps parameters fixed during prediction.

`handoff/selected_price_model.json` specifies log_interactions31, C=1, class_weight=None and the Stage 1 threshold of 0.42. P4 takes 22 of the 36 shared features and transforms them into 31 inputs. The symbol τ denotes the probability threshold. T in protocol.md denotes a market day.

The default final-training procedure fixes the recipe, C and class weights, and searches thresholds from 0.25 to 0.60 in steps of 0.01. Following `price_hgbc/src/price_hgbc/m6.py::retrain`, the inner fit ends one calendar year before the cutoff. The following year supplies validation predictions. The one-year window is a project choice within the course's chronological protocol.

## Python interface
`logistic_price.final_training.fit_final(data, cutoff, validation_folds=None, thresholds=None, fixed_threshold=False)` returns `(bundle, audit)`. Use `train_final(data, cutoff, output, **kwargs)` to write a fresh artifact directory. The fixed-AEST cutoff includes its final interval ending at midnight the next day.

The function truncates future rows, checks for complete consecutive market days from 1 January 2010, constructs shared features and drops 28 warm-up days. Eligible rows begin on 29 January 2010. Later cutoffs expand the training set.

Supply both `validation_folds` and `thresholds` to override the default. Fold boundaries are inclusive, chronological and non-overlapping. To reproduce the original 0.42 threshold, pass `fixed_threshold=True` on its own. For each fold, fit the scaler and classifier on all eligible rows before validation. Pool validation probabilities and maximise balanced accuracy, breaking ties by distance to 0.5 and then by the smaller threshold. Reinitialise the pipeline and refit on all eligible rows through the cutoff. The audit reports inner selection scores.

Example of a custom plan using the two historical 2013 folds:
```json
{"validation_folds": [["2013-01-01", "2013-06-30"], ["2013-07-01", "2013-12-31"]], "thresholds": [0.40, 0.42, 0.50]}
```

## Command line
```sh
python -m logistic_price.final_training --data /absolute/path/nem_weather_halfhourly_2010_2026.csv --cutoff 2026-10-31 --output /absolute/path/new-price-artifact
```

Use `--selection-plan /absolute/path/plan.json` for a custom plan or `--fixed-threshold` for the original threshold. The CLI reads the required CSV prefix and requires a new output directory. `PriceClassifier(output)` loads `logistic.joblib`. The companion `training.json` records the cutoff, row counts, folds, candidates, scores, settings, seed and prefix digest.

The caller manages deployment dates and the six-retrain budget. For each historical evaluation segment, use a cutoff preceding its first day. Group integration combines the price and demand models and runs the submission checker.

## Verification
```sh
python -m unittest discover -s logistic_price/tests -v
COMP9417_DATA=/absolute/path/nem_weather_halfhourly_2010_2026.csv python -m unittest discover -s tests -v
python -m logistic_price.selfcheck
```

Tests cover isolation from future data, sample counts and means used by fold and final scalers, saved daily versus batch predictions, market-day boundaries, invalid validation ranges and overwrite protection.

## Released-data run
For cutoff 2026-10-31, the inner fit through 2025-10-31 used 276,240 rows. The 2025-11-01 to 2026-10-31 validation used 17,520 rows and selected τ=0.49 from 36 candidates. Inner balanced accuracy was 78.896947% and accuracy was 78.835616%. These scores include threshold selection. The final fit used 293,760 rows and was saved separately from the Stage 1 artifacts.
