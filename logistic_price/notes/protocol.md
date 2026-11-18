# P4 experimental protocol

## Target and market day
For each half-hour h of market day T, the label is 1 when `nsw_price[T,h] > mean(nsw_price[T-1,:])`. Use the original prices, including spikes, zeros and negatives, to compute yesterday's mean and the label. The feature transformations apply to the inputs only.

Timestamps mark interval ends in fixed AEST (UTC+10). Subtract 30 minutes to identify the market date. Day T spans T 00:30 through T+1 00:00. History available when forecasting T ends at T 00:00.

## Inputs
The group baseline ablation selected the 22 inputs listed in order in `features.FEATURES`:

- Slot and season sine/cosine pairs, weekend and weekday_0 through weekday_6.
- Yesterday's and last week's same-slot NSW demand and price, and prev_price_mean.
- Yesterday's same-slot Queensland demand (qld_demand_lag1).
- Same-slot NSW demand and price means and population standard deviations over the previous 28 complete days.

The one-day and seven-day lags span 48 and 336 rows. Check for consecutive complete days before shifting. Rolling standard deviations use ddof=0. The seasonal angle is 2π(dayofyear−1)/N, with N=366 in leap years and 365 otherwise. Batch and daily calls use src.common.features. Supply the complete historical prefix to reproduce floating-point rolling statistics in daily predictions. Select the ordered 22 columns before applying transforms.py.

| Recipe | Inputs | Transformation |
|---|---:|---|
| raw22 | 22 | Original group recipe |
| log22 | 22 | Transform five historical price columns |
| log_gaps25 | 25 | Add three signed-log gaps |
| log_interactions31 | 31 | Add six gap-by-slot interactions |

Use slog(v)=sign(v)×log(1+abs(v)) on prices in $/MWh. No additional scale is fitted. The five columns are nsw_price_lag1, nsw_price_lag7, prev_price_mean, nsw_price_mean28 and nsw_price_std28. They retain their names, so the saved recipe identifies their scale.

For gap1, gap7 and gap_mean28, subtract yesterday's mean from yesterday's same-slot price, last week's same-slot price and the trailing 28-day same-slot mean respectively. Apply slog to each difference, then multiply each transformed gap separately by slot_sin and slot_cos. These terms use historical market observations and the target calendar.

## Chronological selection
| Step | Training | Evaluation |
|---|---|---|
| Fold 1 | 2010-01-29 to 2012-12-31 | First half of 2013 |
| Fold 2 | 2010-01-29 to 2013-06-30 | Second half of 2013 |
| Recipe selection | 2010-01-29 to 2013-12-31 (68,784 intervals) | 2014 (17,520 intervals) |
| Frozen refit | 2010-01-29 to 2014-12-31 (86,304 intervals) | 2015-01-01 to 2026-10-31 (207,456 intervals, 4,322 days, 142 months) |

The final timestamp, 2026-11-01 00:00, belongs to October. For each recipe search C in {0.01, 0.1, 1, 10}, class_weight in {None, balanced}, and thresholds from 0.25 to 0.60 in steps of 0.01. Each combination of C and class weight uses two fold-specific fits, including a fresh scaler for each fold. Reusing probabilities across thresholds gives 1,152 candidate scores from 64 fits.

Choose each recipe's parameters by pooled 2013 balanced accuracy. Break ties by distance from threshold 0.5, preference for C=1, preference for unweighted classes and then smaller C. Refit each recipe and select by 2014 balanced accuracy. Break remaining ties by fewer inputs and then recipe name. selection_2013.json records log_gaps25 as the internal winner. selection_2014.json records log_interactions31 as the selected recipe. The original reference and early 30-candidate model serve as fixed comparators. Stage 1 selection ends in 2014.

| Configuration | C | class_weight | Threshold |
|---|---:|---|---:|
| raw22 | 10 | balanced | 0.33 |
| log22 | 10 | balanced | 0.46 |
| log_gaps25 | 1 | balanced | 0.49 |
| log_interactions31 | 1 | None | 0.42 |
| Original reference | 1 | None | 0.40 |
| Early weighted model | 1 | balanced | 0.40 |

## Ablation
Refit each reduced model on the same rows with the selected 31-feature model's C, class weight and threshold. Removing calendar inputs drops 12 columns and six slot interactions, leaving 13. Removing NSW history drops five columns and all gaps and interactions derived from yesterday's mean, leaving 17. Removing Queensland demand leaves 30. Removing rolling statistics drops four columns, gap_mean28 and its two interactions, leaving 24. These comparisons estimate group contributions conditional on the fixed settings.

## Baselines and uncertainty
Both baselines compare a historical same-slot price (yesterday or last week) with yesterday's raw daily mean. The target day's price supplies the observed label. Accuracy counts correctly classified intervals. Balanced accuracy averages recall across classes present, using that class's recall in single-class periods. Compute each full-period, annual or monthly score directly from its intervals.

Monthly accuracy has two 95% intervals. The independent approximation uses 100 × [p ± 1.96√(p(1−p)/n)], clipped to [0,100]. The day bootstrap resamples whole 48-interval days 2,000 times with seed 17 and takes the 2.5th and 97.5th percentiles. Balanced accuracy combines class-specific recall variances for the independent approximation and uses the same day resamples for bootstrap intervals.

Paired comparisons use the same sampled days for both models. paired_validation_2014.csv compares the selected recipe with the two earlier configurations. paired_comparisons.csv covers replay comparisons with other recipes and baselines.

Daily resampling preserves within-day dependence and assumes exchangeable days. Persistent weather and changing market conditions limit this assumption. Intervals describe expected performance under comparable conditions. Use paired differences to assess model gaps. The 2014 data also informed selection, and the comparisons have no multiplicity adjustment. Full-period intervals summarise several market regimes.

## Verification and scope
The shared v1 integration was checked on 15 November 2026. See [the integration record](v1_integration.md). Original timing and repeat-run records are retained. run.json contains the full dataset SHA256, versions, sample counts and daily check timings. Daily PriceClassifier.predict output matched batch results on all 4,322 days for the selected model. The remaining artifacts were checked on the first day, a leap day and the last day. Frozen replay keeps coefficients and scaling fixed while updating historical lags and rolling inputs.

monthly_class_balance.csv and its figure describe class proportions. The observation/probability figures use the first and last replay weeks. Full submission requires the selected training strategy, demand model, combined two-column forecast and official package check.
