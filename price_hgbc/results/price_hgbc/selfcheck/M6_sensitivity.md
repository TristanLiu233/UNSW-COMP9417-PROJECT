# M6 sensitivity: pre-registered S3 variants (exploratory)

Date: 2026-11-11/12 · same environment as M6.
Pre-registration: `results/price_hgbc/m6/s3_sensitivity_preregistration.md`. You approved it as drafted ("Approve and
run": four variants, default usage rule, 730-day cooldown). It was frozen before any variant ran: SHA-256 `0b713240…`,
pinned in `test_m6_sensitivity.py` and recorded in `sensitivity/sensitivity_meta.json`.
**The official M6 S3 result is unchanged.** All 16 official M6 output files are byte-identical before and after this run.
These results are exploratory: the variants were chosen after M6 showed S3's budget running out early.

## 1. What was run

| Variant | Changed factor | Setting |
|---|---|---|
| S3-A | none (control) | official S3: option A, k = 0.5, h = 6, cooldown 180, budget 6 |
| S3-B | reference after retrain | option B: q99, μ₀, σ₀ re-estimated from the tuning model's out-of-sample daily log loss on the last 12 months (365–366 days, the same inner fit that picks τ, so no extra fit) |
| S3-K | allowance / threshold | k = 0.75; h = **5**, chosen by the unchanged M4 rule on 2014 only (`sensitivity/s3k_calibration.csv`, written and hashed before the replay; h = 4 → ARL₀ 174 days, h = 5 → median > 3650) |
| S3-C | cooldown | 730 days |

Code: `src/price_hgbc/m6.py` `run_s3(..., variant=...)` (the default call is still bitwise the official S3) and
`src/price_hgbc/m6_sensitivity.py`.

## 2. Results, 2015–2024 (3,653 days; paired whole-day bootstrap, same index matrix as M6)

| Variant | BalAcc | − S3-A (primary; Holm p) | − S1 (secondary) | Retrains / budget gone | Alarms in 2015–24 (cooldown / budget suppressed) |
|---|---|---|---|---|---|
| S3-A | 74.710 | — | **−0.670 [−0.883, −0.455]** | 6 / 2019-03-17 | 80 (11 / 63) |
| **S3-B** | **75.262** | **+0.552 [+0.350, +0.796]; p < 0.001** | −0.118 [−0.291, +0.079] | 6 / 2021-12-22 | 23 (12 / 5) |
| **S3-K** | 75.195 | **+0.485 [+0.319, +0.658]; p < 0.001** | −0.186 [−0.368, +0.005] | 6 / 2021-01-05 | 41 (7 / 28) |
| S3-C | 74.723 | +0.013 [−0.237, +0.283]; p = 0.83 | **−0.657 [−0.908, −0.410]** | 6 / 2026-05-26 (5 in 2015–24) | 81 (76 / 0) |
| S1 (M6) | 75.380 | +0.670 | — | 5 | — |
| S0 (M6) | 70.082 | −4.628 | −5.298 | 0 | — |

All variants beat S0 and b1 (CIs exclude 0; `sensitivity_bootstrap_dev.csv`).
"p < 0.001" means no bootstrap resample out of 1000 fell on the other side of 0.

Retrain days:
* S3-B: 2015-09-23, 2016-03-24, 2016-10-24, 2020-04-25, 2021-01-06, 2021-12-22;
* S3-K: 2015-10-01, 2016-04-28, 2017-02-05, 2019-03-17, 2020-02-08, 2021-01-05;
* S3-C: 2015-09-23, 2017-10-06, 2019-12-13, 2021-12-12, 2024-01-10, 2026-05-26.

**Mean standardised daily error z under each variant's own reference** (`sensitivity_z_by_year.csv`):

| Year | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|---|---|---|---|
| S3-A | 0.03 | 0.34 | 0.39 | 0.11 | 0.16 | 0.22 | 0.47 | 0.98 | 0.48 | 0.38 |
| S3-B | 0.01 | 0.21 | 0.11 | −0.05 | 0.02 | −0.02 | 0.10 | 0.47 | −0.09 | −0.19 |
| S3-K | 0.04 | 0.29 | 0.40 | 0.19 | 0.17 | 0.21 | 0.37 | 0.87 | 0.35 | 0.27 |
| S3-C | 0.03 | 0.36 | 0.72 | 0.11 | 0.23 | 0.21 | 0.41 | 0.87 | 0.27 | 0.01 |

S3-B's reference updates (`s3b_reference_updates.csv`): after the 6 retrains, μ₀ moves from 0.443 (2014) to 0.462 … 0.518
and q99 from 1.049 to 1.06 … 1.24. In later years the "normal" error level is clearly higher than in 2014.

## 3. Interpretation

1. **The stale 2014 reference explains most of S3's shortfall.** Re-basing after each retrain (S3-B) cuts alarms from 80
   to 23. It spreads the budget to 2021 and gains +0.55 pp over the control. Mean z falls to about 0 in most years, so
   alarms again mean "worse than this model's own normal".
2. **Over-sensitivity is a second, overlapping cause.** Raising k to 0.75 with h re-calibrated by the same ARL₀ rule
   (S3-K) gives +0.49 pp. It still fires 41 times because the 2014 reference stays stale.
3. **Pacing alone does not help.** A 730-day cooldown (S3-C) leaves BalAcc unchanged and still trails S1. It holds an
   outdated model through most of 2016–2017 (2017 BalAcc 67.0 %). Many alarms are wasted inside the cooldown (76 suppressed).
4. **Against S1, the fixed variants come close but do not win:** S3-B −0.12 [−0.29, +0.08], S3-K −0.19 [−0.37, +0.005].
   Detector-triggered retraining can match scheduled expanding retraining once it is configured sensibly. On this data it
   does not show that detecting drift chooses *better* retraining times than a two-yearly calendar.
5. 2022 is hard for every model (z up even under option B). This looks like a genuine shift in that year rather than a detector artefact.

## 4. Pre-registered usage rule (§6, default)

A variant can become the Stage 3 strategy only if it beats S1 by ≥ 0.5 pp with a CI that excludes 0 and stays within
budget. **No variant meets the rule** (`meets_adoption_rule = False` for all; best S3-B −0.12 pp vs S1). So:
* the official M6 result and the S3-A/S1 comparison stand;
* this run is reported as an explanation of why S3 trailed S1;
* S1 remains the strongest candidate for M7's Stage 3 choice. If a detector-triggered strategy is required in Stage 3,
  the evidence points to option B (S3-B) as the better-configured version, but it would be a post-hoc choice and must be
  labelled as such.

## 5. Checks (`tests/price_hgbc/test_m6_sensitivity.py`, 14 tests incl. 1 slow)

| # | Check | Status |
|---|---|---|
| frozen | pre-registration hash pinned; official M6 outputs, REF, M4 config and M5 files unchanged | PASS |
| 1 | S3-A reproduces the official S3 bitwise (p_up, pred, alarm days, retrain flags, S at alarm, daily S) | PASS |
| 2 | every variant: ≤ 6 retrains; its own cooldown respected; effective = alarm + 1; equals S0 before its first retrain; suppression reasons valid | PASS |
| 3 | S3-B: market data shuffled after 2017-06-30 → identical alarms and predictions up to that day (slow) | PASS |
| 4 | S3-B: each reference uses validation days ≤ the retrain day (≥ 300 days); the first update reproduces from a fresh retrain; the new reference is in force from the next day | PASS |
| 5 | S3-K: calibration CSV hashed and written before the replay (mtime); k = 0.75; ARL₀ non-decreasing; h = smallest meeting 730 days | PASS |
| 6 | batch = day-by-day (`m5.predict_price`) at the first, middle and last segment of S3-B/K/C, 3 days each | PASS |
| 7–8 | no 2025–2026 scores (yearly table 2015–2024; daily log loss masked after 2024); Holm p-values and adoption flags recomputed; bootstrap index = M6's | PASS |

**Acceptance runs:**
* `pytest tests/price_hgbc -q -m "not archive and not historical_repro"` (incl. slow): **201 passed, 2 skipped, 0 failed**
  (`M6_sensitivity_pytest_full.txt`);
* `-m historical_repro`: the 3 authorised failures, unchanged.

The 2 skips are the opt-in Docker re-runs.

Fix during the run: the M4 config hash in the new test file was typed incorrectly (only its first 16 characters were
right). It was corrected to the file's real hash. The file itself never changed (it matches every earlier check).

## 6. Files (`results/price_hgbc/m6/sensitivity/`)

`s3{a,b,k,c}_predictions.parquet`, `s3{a,b,k,c}_events.csv`, `s3{a,b,k,c}_daily_cusum.csv`, `models/` (18 variant models),
`models_log_sensitivity.csv`, `s3b_reference_updates.csv`, `s3k_calibration.csv`, `sensitivity_comparison_dev.csv`,
`sensitivity_yearly_dev.csv`, `sensitivity_bootstrap_dev.csv`, `sensitivity_z_by_year.csv`, `sensitivity_meta.json`,
`fig_sensitivity_cusum_paths.png`, `fig_sensitivity_yearly.png`.

## 7. Spot checks (suggested)

1. `sensitivity/s3a_predictions.parquet` vs `m6/s3_predictions.parquet`: `p_up` identical byte for byte (the control reproduces M6).
2. `sensitivity/s3b_reference_updates.csv`: 6 rows; `val_end` = `alarm_day` each time; μ₀ between 0.46 and 0.52;
   `s3b_daily_cusum.csv` shows that μ₀ from the next day on.
3. `sensitivity/sensitivity_comparison_dev.csv`: `meets_adoption_rule` False for every variant; S3-B
   `BalAcc_minus_S1` = −0.118.
