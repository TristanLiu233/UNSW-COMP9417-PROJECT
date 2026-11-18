# Live-week analysis at model age 37–43 days
Model age is the target market date minus the most recent training cutoff date. A cutoff of 31 October 2026 places 7–13 December 2026 at ages 37–43 days. Midnight marks the end of the preceding market day's final interval.

For the frozen model fitted through 31 December 2014, the matching window is 6–12 February 2015 (7 days, 336 intervals). `handoff/age_37_43_frozen_window.csv` gives balanced accuracy, accuracy and both intervals for each candidate and both baselines. The interval-level data are in `age_37_43_predictions.csv.gz`.

All six models predict the same historical week. The 2,000 bootstrap resamples describe uncertainty within that sample. Estimating December 2026 performance requires evidence across comparable historical windows.

## Using the group's retraining logs
1. Record each deployed model's ID, training cutoff, activation and replacement dates. Retain the predictions and probabilities generated at the time.
2. Select days 37–43 following each actual training cutoff.
3. Include complete seven-day windows with known outcomes and the same model deployed throughout. Score its original forecasts.
4. Report class counts, accuracy, balanced accuracy, yesterday and last-week baselines, and paired differences. Document overlapping windows and the retraining budget.
5. Compare training length, season, class shares and market conditions with the expected live week. Other December weeks can provide seasonal comparisons. Record their actual model ages.
6. Use the number and dependence of the windows to justify the live-week estimate and interval. Recalculate model age if the final cutoff changes.

The original framework used the existing frozen run. At that stage, the final forecast depended on the group's strategy and retraining logs.

## Reproduce the handoff files
```sh
python -m logistic_price.milestones --data data/nem_weather_halfhourly_2010_2026.csv --results logistic_price/results/frozen --output logistic_price/_scratch/handoff
```
