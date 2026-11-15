# Shared feature-table v1 integration

Checked on 2026-11-15 against main snapshot `5e8161224f248009d5ed577ba7fb09752acf1399`.

## Implementation

`logistic_price/features.py` calls `src.common.features.build_features` and `build_features_for_day`, then selects the original 22 columns in their recorded order. The four P4 recipes remain in `transforms.py`.

Training reads the five market and four Sydney weather columns required by the shared module. Daily prediction receives the full historical prefix and the following 48 weather rows. The shared function returns 36 columns; P4 retains its original 22-column input schema before transformation. The public-holiday feature remains outside these recipes.

The shared module, baseline and Ridge files are unchanged. All 118 files supplied by the main ZIP were compared byte for byte. All 35 original files under P4 `results/`, including six fitted models, were preserved.

## Verification

- The shared 22-column batch inputs exactly match the previous P4 implementation across all 295,104 course rows, including warm-up rows.
- All six configurations reproduce their saved 17,520 validation labels after refitting through 2013 with the shared inputs. Probabilities agree within an absolute tolerance of 1e-14.
- All six frozen models reproduce their 207,456 saved replay labels. Probabilities agree within the same tolerance.
- The selected `log_interactions31` model matches every saved daily prediction across 4,322 replay days through 2026-10-31. These checks use the full-prefix shared daily interface.
- Shared-module tests: six passed, including the released-data comparison across 17 sampled years and six boundary days.
- P4 result verification and synthetic checks passed. These include exact batch/day feature equality on checked dates, targets, baseline predictions, pooled metrics, future market fields filled with NaN or extreme values in forecast inputs, malformed timestamps, model loading, ablation dependencies and output protection.
- Daily interface checks took 164.64 seconds locally; the slowest prediction took 0.187 seconds.

The search grid and fitted weights are unchanged. The full parameter search was previously reproduced; this integration check refits six validation models and checks saved replay predictions. Original experiment metadata stays in `results/frozen/run.json`; current integration evidence is in [v1_integration_check.json](v1_integration_check.json) and [v1_check_log.txt](v1_check_log.txt).

Shared module SHA256: `49a0b02495e60466bc5b2295056f845b84b3a849ec2e43ac895b6ece576a8666`.

## Recheck

From a repository containing the shared module and P4 directory:

```sh
python -m pip install -r logistic_price/requirements.txt
python -m logistic_price.verify --data data/nem_weather_halfhourly_2010_2026.csv --results logistic_price/results/frozen
python -m logistic_price.selfcheck
python -m logistic_price.verify_integration --data data/nem_weather_halfhourly_2010_2026.csv --results logistic_price/results/frozen --report logistic_price/_scratch/v1_check.json
COMP9417_DATA=data/nem_weather_halfhourly_2010_2026.csv python -m unittest discover -s tests -v
```

## Group handoff

The supplied main snapshot contains the original Logistic baseline and no `logistic_price/` directory. This update is ready for review on a branch based on that main snapshot. Group approval of P4 feature recipes, final model selection, the shared Stage 1 configuration and the combined two-target submission check remain open.

`handoff/main_comparison.csv` places the main baseline scores alongside the P4 configurations. The selected P4 model has 2014 BalAcc 79.1249% and frozen replay BalAcc 74.6210%. The P4 `log22` model has 77.0323% and 75.6977%, respectively. Main's price tree has 79.2999% and 70.3369%. These are historical comparisons; the original 2014 selection remains unchanged.
