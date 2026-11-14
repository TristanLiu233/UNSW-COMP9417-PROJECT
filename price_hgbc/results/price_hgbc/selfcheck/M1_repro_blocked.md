# M1: stopped at the reproduction gate (difference > 0.01)

Date: 2026-11-11 · Machine: macOS 26.6.2 arm64, Python 3.11.8, sklearn 1.9.1, numpy 2.4.6, pandas 3.1.0
(all pins match). The group's run (`ablation/results/run.json`): Python 3.11.13, the same numpy/pandas/scipy/sklearn
versions, platform not recorded, `threads_per_worker` 1, `model_seed` 17.

## Pre-stage check

`pytest tests/price_hgbc -q -m "not slow"` → 28 passed. The new slow test `test_check_submission_rerun_live`,
which really re-runs `check_submission.py example_package` and checks the kit is unchanged, → 1 passed (8.4 s).

## Reproduction result

| | BalAcc % | Acc % | 0/1 rows differing from the group's saved `tuned_C` |
|---|---|---|---|
| Group (selection.csv / predictions file, re-scored with `da`) | 79.29988 | 79.21804 | — |
| This env, `src/price_hgbc` (random_state=0) | 79.18680 | 79.10959 | 157 / 17,520 (on 103 of 365 days) |
| **Difference** | **−0.11309** (> 0.01 → STOP) | −0.10845 | |

## What was ruled out (each one is evidence, not a fix)

| Hypothesis | Test | Result |
|---|---|---|
| Different data | SHA-256 of the CSV prefix read by the study vs `run.json` `csv_prefix_sha256` | identical |
| My features differ | `ablation.features.build_features` (group code) vs `src/price_hgbc/features_simple.build_features`, 18 cols, train + valid rows | **bitwise equal**, same dtypes |
| My labels differ | group labels vs `da.targets`-based labels, all training rows | equal |
| random_state 0 (rule) vs 17 (group) | fit both, compare 2014 probabilities | **bitwise equal**: with early_stopping=False, max_features=1, n=68,784 < 200,000 the seed is unused (sklearn source, gradient_boosting.py L484–600, L975) |
| OpenMP thread count | threadpool_limits = default(8) / 1 / 2 / 4 | identical result every time (79.18680) |
| Column order | `study.py` orders by `ORDER` = documented list | same order |
| My pipeline vs group pipeline | group's own `ablation.models.predict_raw("boost_price", 15, …)` run here | 79.18680, the same 157 mismatches |
| Doc §D script | `消融说明.md` §D assembled verbatim from its code blocks, run here | ridge 3.30928 ✓ and logistic 75.79293 ✓ match the doc; **both HGB models differ**: boost_price 79.18680 (doc 79.300), boost_demand MAPE 2.43820 / MAE 192.70 (doc 2.436 / 192.50) |

The mismatched rows are not only threshold ties: |p − 0.45| has median 0.011, p90 0.039 and max 0.091. So the
fitted trees differ, which is not rounding at the cut-off. Group code and data are byte-identical and the linear
models reproduce exactly. The difference is specific to HistGradientBoosting across machines: it builds
float32 gradient/hessian histograms in compiled code, so a near-tie split can resolve differently on another
CPU/OS build, and the trees diverge from there.

Re-run on any machine: `python -m src.price_hgbc.m1_repro_check` (prints platform + BalAcc + mismatches).

Not started (by instruction): the M1 tasks 1–4 self-checks, eval.py, m1.csv.
