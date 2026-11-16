# S3 sensitivity analysis: pre-registration (draft for approval; nothing has been run)

Status: **APPROVED by the user on 2026-11-11 ("Approve and run"), as drafted: all four variants in §2, the default
usage rule in §6, and the 730-day cooldown for S3-C.** Frozen from this version on: its SHA-256 is pinned in
`tests/price_hgbc/test_m6_sensitivity.py` before any variant runs. Variants, rules and outputs may not change.

## 1. Why, and what this can and cannot show

M6 found that the official S3 (CUSUM, option A, h = 6, k = 0.5, cooldown 180 days, budget 6) spent its whole budget by
2019-03-17. It then trailed S1 by −0.67 pp BalAcc on 2015–2024 [−0.88, −0.46]. The question is whether that comes from
**how the detector was configured** (reference never updated, threshold too sensitive, cooldown too short) or from
**detector-triggered retraining as such**.

Honest limits:
* The variants were chosen **after** seeing the M6 result (budget used by 2019), so they are post-hoc hypotheses tested
  on the same 2015–2024 data. Their results are **exploratory sensitivity evidence**: they explain S3's behaviour.
  They do **not** replace the official S3 result.
* Any later "adopt variant X" decision would be optimistic for the same reason. Section 6 restricts this.
* 2025–2026 stay untouched: simulated causally, never scored, never used.

## 2. Variants (one factor changed at a time vs the official S3)

Everything not listed here is identical to official S3:
* model: P1 `boost_price`, REF params (random_state 17);
* features and label gate as in M6;
* expanding window and nested τ;
* S = 0 on 2015-01-01; S reset on every alarm day and after every retrain;
* budget 6; error = daily mean log loss with clip 1e-6.

| ID | Factor changed | Setting | Where its value comes from | Hypothesis tested |
|---|---|---|---|---|
| **S3-A** (control) | none | official S3 | M6 | must reproduce M6's S3 bitwise |
| **S3-B** | reference after each retrain | **option B**: after each retrain, q99, μ₀, σ₀ are re-estimated from the **tuning model's out-of-sample daily log loss on the last 12 months** of the training window. That is the same inner fit that selects τ, so there is no extra fit and no extra budget use. Before the first retrain the reference is the M4 one (it equals option A until then). h stays 6, k 0.5 | M4 design §B12 (already implemented in `cusum.reference_for_retrain`) | the alarm flood comes from a stale 2014 reference |
| **S3-K** | allowance and threshold | **k = 0.75**; h re-selected with the **unchanged M4 rule** (smallest h ∈ {3, 4, 5, 6, 7, 8, 10} with bootstrap ARL₀ ≥ 730 days) on the **2014** errors (same weekly block bootstrap, 1000 reps, seed 0). Reference stays option A | 2014 only, computed before any S3-K replay | a less sensitive detector (same calibration rule) spends the budget more slowly |
| **S3-C** | cooldown | **730 days** between retrain alarms (instead of 180). Reference A, h 6, k 0.5 | design choice: 730 days ≈ S1's two-year spacing | with S1-like pacing, does *detector timing* beat *calendar timing*? |

Pre-declared extra details:
1. **S3-B** inner-error definition: daily mean log loss of the tuning model's `predict_proba` on the validation days
   (clip 1e-6), then `reference_stats` (numpy linear q99; ddof = 1). If fewer than 300 validation days exist, that
   retrain keeps the previous reference (cannot happen with 12-month windows; asserted).
2. **S3-K** h search: if no candidate meets 730 days, take h = 10 and flag it (same fallback as M4). The resulting
   (k, h, ARL₀ and its 5–95 % band) is written to `sensitivity/s3k_calibration.csv` **before** the S3-K replay starts.
3. **S3-C** keeps the budget at 6. With 730-day spacing, at most 6 retrains fit in 2015–2026 by construction.
4. No variant combines factors. That avoids a garden of forking paths; a combination would need a new pre-registration.

## 3. What is computed (2015–2024 only, 3,653 days, the same rows as M6)

For each of S3-A, S3-B, S3-K, S3-C:
* Acc, BalAcc (official `da`), per year and per month;
* retrain dates; alarms; suppressed alarms (cooldown / budget); date the budget runs out (if it does);
* for S3-B: the reference values (q99, μ₀, σ₀) after each retrain;
* yearly mean standardised error z under the variant's own reference.

Paired whole-day bootstrap: n_boot 1000, seed 0, **the same index matrix as M6** (its SHA-256 is pinned in M6's `eval_meta.json`).

Pre-declared comparisons:
* **primary:** each variant − S3-A;
* **secondary:** each variant − S1 (S1 = M6's best strategy);
* **descriptive only:** each variant − S0, − b1.

Report raw 95 % CIs, and in addition Holm-adjusted p-values for the 3 primary comparisons (bootstrap p = share of
resampled differences on the other side of 0, two-sided). No other comparison is tested.

## 4. Checks the run must pass (tests in `tests/price_hgbc/test_m6_sensitivity.py`)

1. S3-A reproduces M6's `s3_predictions.parquet` and `s3_events.csv` bitwise.
2. Every variant: ≤ 6 retrains; cooldown respected; effective day = alarm day + 1; equals S0 before its first retrain.
3. Label gate: no label read before its day ends; a shuffle of market data after 2017-06-30 leaves earlier alarms
   unchanged (S3-B, the variant that reads extra OOS errors).
4. S3-B: each reference update uses only errors from validation days ≤ the retrain day. Its q99, μ₀, σ₀ are reproducible
   from the logged tuning-model predictions.
5. S3-K: the calibration CSV exists and was written before the replay (file time and hash logged); h follows the M4 rule;
   ARL₀ is non-decreasing in h.
6. Batch = day-by-day for 3 days in each variant's first, middle and last model segment.
7. The official M6 outputs (`m6/*.parquet`, `*.csv`) and all frozen files (REF, M4, M5) are unchanged (hash pins).
8. No metric of 2025–2026 appears in any output; daily log loss after 2024 is masked as in M6.

## 5. Outputs (`results/price_hgbc/m6/sensitivity/`)

* `s3{a,b,k,c}_predictions.parquet`, `s3{a,b,k,c}_events.csv`, `models_log_sensitivity.csv`;
* `s3b_reference_updates.csv`, `s3k_calibration.csv`;
* `sensitivity_comparison_dev.csv`, `sensitivity_yearly_dev.csv`, `sensitivity_bootstrap_dev.csv`;
* `fig_sensitivity_cusum_paths.png` (four S paths, 2015–2024), `fig_sensitivity_yearly.png`;
* `selfcheck/M6_sensitivity.md`.

Code: a parameterised copy path in `src/price_hgbc/m6.py` (`run_s3(..., variant=...)`, defaults = official S3).
The official call and its outputs must stay bitwise identical. Estimated run time: about 1.5 minutes per variant
plus about 1 minute for the S3-K calibration.

## 6. How results may be used (needs your decision; default proposed)

**Proposed default:** the sensitivity results are **reported only**. The official S3 stays the M6 result. The Stage 3
(M7) strategy choice uses the pre-declared M6 comparison (S0–S3) plus this section as explanation. A variant can
become the Stage 3 strategy **only** if **all** of these hold:
* it beats S1 on 2015–2024 by ≥ 0.5 pp BalAcc;
* its paired CI against S1 excludes 0;
* it stays within budget;
* and the report states that it was chosen post hoc.

Otherwise S1 (or the official S3, if preferred for the "detector-triggered" requirement) remains the candidate.

**Alternatives you might prefer:**
* (a) purely descriptive: no variant can ever be adopted;
* (b) allow adoption with a stricter bar (e.g. ≥ 1.0 pp vs S1).

## 7. Open points (resolved at approval: as drafted; default rule; 730 days)

1. Approve the four rows in §2 (or drop / replace one). Adding a variant later means a new pre-registration.
2. Choose the usage rule in §6 (default / a / b).
3. Confirm that S3-C's 730-day cooldown is acceptable as "S1-like pacing" (alternative: 365 days).
