# PR draft (not opened, not pushed): price HGBC M2 — extended features, 48-config grid, decision "retain M1"

## Summary

- Adds 8 extended price-direction features (`src/price_hgbc/extra_features.py`), the M2 experiment pipeline
  (`src/price_hgbc/m2.py`), M2 figures (`src/price_hgbc/m2_figures.py`) and 61 self-checks (`tests/price_hgbc/test_m2.py`).
- Every M2 choice uses only 2013, through the group's two expanding folds (`ablation/study.py` `temporal_folds`), with training starting 2010-01-29 as in P1.
- **Outcome: keep M1.** Neither the extended features nor any of the 48 HGBC configurations clears the
  pre-declared adoption bar (≥ 0.5 pp BalAcc over M1_ref on 2013 **and** a paired day-bootstrap 95 % interval that excludes 0).

## Numbers (BalAcc %, 2013 = pooled out-of-fold, decides; 2014 = one fixed confirmation)

| Step | Candidate | Reference | Gain (pp) | 95 % CI (pp) | Rule result |
|---|---|---|---|---|---|
| 2a features | M1 + 8 extended: 79.947 | M1 features: 79.465 | +0.482 | [−0.200, +1.232] | keep M1 features |
| 2b best of 48 | g21 (31 leaves, lr 0.05, 200 iter, l2 0.1): 79.843 | M1_ref: 79.465 | +0.378 | [+0.036, +0.721] | gain < 0.5 → retain M1 |
| 2014 check | g21: 80.324 (Acc 80.314) | M1_ref: 80.127 | +0.197 | [−0.078, +0.485] | consistent with "no clear gain" |

The top five configurations are within 0.197 pp of each other on 2013, so their ranking may be sampling noise.

## New features (`m2_feature_groups.csv` lists all 26 columns for M3)

`prev_price_max`, `prev_price_std` (day D); `prev_spike_ratio` = max / max(P̄(D), 1 $/MWh);
`prev_n_gt300`, `prev_n_gt1000`, `prev_neg_share` (day D); `syd_temp_diff` (D+1 vs D, same period);
`is_holiday` (D+1, `holidays` 0.107 AU/NSW). `prev_price_max` and `prev_neg_share` equal P1's `prev_price_max`
and `prev_negative_slots`/48 bit for bit.

## Verification

- Leakage: shuffling the true D+1 price or demand leaves the features bitwise unchanged. Bulk features equal day-by-day `make_inputs` features.
- Labels / `up_lag1`: 0 mismatches against the official definition; holidays checked around 2014-01-26/27 and on observed days.
- The grid is resumable and idempotent: one-shot and interrupted-then-resumed runs (including a simulated crash) give identical rows, and a changed definition is refused.
- No 2014 leakage: replacing every market value from 2014 onwards with noise leaves all validation results bitwise identical.
- Repeatability: fresh-process re-runs reproduce every M2 output file byte for byte.
- `pytest tests/price_hgbc -q -m "not historical_repro"`: see `results/price_hgbc/selfcheck/M2.md`.

## Risks / limitations

- 2014 was already used by the group's ablation, so it is not a fresh test year; it was used here once, for confirmation only.
- The bootstrap intervals assume days are exchangeable within 2013. Each config's τ is tuned on the same 2013 rows, so the 2013 numbers are mildly optimistic for every config, M1_ref included.
- g21's 2013 interval excludes 0 although its gain is below 0.5 pp. Under the pre-declared rule that is still "retain M1" and must not be re-argued after seeing 2014.
- No shared group results table exists outside the read-only kit, so nothing was appended to group files.
