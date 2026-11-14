# M1 reproduction blocker: root-cause diagnosis

Date: 2026-11-11 · This machine: macOS 26.6.2, **arm64 (Apple M3)**, Python 3.11.8 (conda-forge, clang 16),
numpy 2.4.6 / pandas 3.1.0 / scipy 1.17.1 / sklearn 1.9.1, wheels `macosx_*_arm64`, BLAS = Accelerate,
OpenMP = libomp, no bottleneck/numexpr/numba. Full fingerprint: `fp_macos_arm64_m3.json`.

Goal: find where 79.29988 (group) vs 79.18680 (here) comes from, **not** to improve the score.
Nothing was changed in the kit, the saved group results, the assertions/tolerances, `random_state=0` or `early_stopping=False`.

## 1. Provenance of the saved result

| Item | Finding | Evidence |
|---|---|---|
| Git history | Code, results and guide all arrive in **one squashed commit** `116e296` (bizhe3, 2026-11-07). No commit records the run itself, so the repo cannot show which code revision produced `predictions/boost_price.csv.gz` | `git log --all -- ablation/…` |
| Generating command | `python -m ablation.run --data … --jobs 3 --threads 1` (README; `run.json`: jobs 3, threads_per_worker 1) | `ablation/README.md`, `run.json` |
| Recorded env | Python **3.11.13**, numpy 2.4.6, pandas 3.1.0, scipy 1.17.1, sklearn 1.9.1; **OS/CPU not recorded** | `run.json` |
| Model config | `HistGradientBoostingClassifier(loss="log_loss", max_leaf_nodes=15, max_iter=100, learning_rate=0.1, early_stopping=False, random_state=17)`, τ = 0.45, 18 cols in `ORDER` order | `ablation/models.py` `predict_raw`; `study.py` `evaluate`; `selection.csv` |
| Defaults used | `categorical_features="from_dtype"` (no category dtypes → `is_categorical_=None`), `max_bins=255`, `max_features=1.0`, `min_samples_leaf=20`, `l2_regularization=0` (sklearn 1.9.1 defaults, same version both sides) | fingerprint `model.params` |
| Sample order / dtypes | training rows in time order (monotonic index), float64 lags/means, int64 calendar dummies; identical to the group's frame | `test_our_features_bitwise_equal_group_features`, `test_model_params_match_group_spec` |
| Saved files | SHA-256 pinned, and `git diff HEAD` is clean | `test_saved_group_results_unchanged`, `test_kit_matches_git_head` |

## 2. Root-cause diagnosis table

| # | Hypothesis | Status | Evidence |
|---|---|---|---|
| H1 | Different data | **Excluded** | CSV prefix SHA-256 = `run.json` `csv_prefix_sha256` |
| H2 | Different features | **Excluded** | 18-col train/valid matrices bitwise equal to the group's `ablation.features` (values + dtypes + names) |
| H3 | Different labels | **Excluded** | train labels equal; the group's saved 2014 truth (`validation.csv.gz`) equals ours; **0 exact price ties** in 2010–2014 (exact decimal arithmetic; the smallest non-zero gap is 1/4800), so no summation order can flip a label |
| H4 | Different baseline / scoring | **Excluded** | saved `baseline1_price` equals ours; the saved `tuned_C` re-scored with `da` gives exactly 79.29988 |
| H5 | random_state 0 vs 17 | **Excluded** | probabilities and per-tree hashes bitwise equal (seed is unused: early_stopping=False, n=68,784 < 200k, max_features=1) |
| H6 | Thread count / OpenMP scheduling (here) | **Excluded on this machine** | threads 1 / 2 / 4 / 8 and fresh processes → identical trees, bins and probabilities |
| H7 | Column order, parameter defaults, categorical handling, early stopping | **Excluded** | `ORDER` = documented order; params dump; `is_categorical_=None`; `n_iter_=100` |
| H8 | My pipeline differs from the group's | **Excluded** | the group's own `predict_raw` and the `消融说明.md` §D script run verbatim give the same 79.18680 / 157 mismatches |
| H9 | Saved results were not produced by this code + inputs at all | **Largely excluded** | the group's own code re-fits: **logistic 85/85** and **ridge 83/83** configs bit-exact; **boost_price 28/85** exact on all 17,520 labels (incl. tuned_CW/CR/F/dimension_mix); boost_demand 1/87. Exact HGB matches on 28 configs would not happen with different code/data |
| H10 | Generic floating-point noise (ULP-level rounding) is enough to change this HGB fit | **Not supported (here)** | permuting the training row order (10×) and nudging up to **10 %** of float32 gradients by 1 ULP every iteration (3 seeds × 4 fractions) both change **0** labels, for tuned_C and tuned_CR (`m1_diag_sensitivity.csv`, `m1_diag_ulp.csv`) |
| H11 | Platform/build-dependent behaviour of compiled HGB code (OS, CPU arch, compiler, SIMD, libm) larger than random ULP noise | **Unverified** | consistent with: only HGB affected; which configs match looks arbitrary (tuned 4/5 exact; non-exact Δ is symmetric, median 0.14 pp). Weakened by H10. Needs a second machine (A/B) or Linux (C/D) |
| H12 | The original machine is itself non-deterministic (e.g. its OpenMP runtime) | **Unverified** | needs a run on the original machine |
| H13 | The saved run used a different library state than `run.json` records (e.g. a mixed env, a stale worker, an un-committed code edit for some stages) | **Unverified** | `run.json` versions come from the main process; history is squashed; H9 makes a wholesale code difference unlikely, but a partial one is not excluded |
| H14 | Python 3.11.13 vs 3.11.8 | **Unverified, low prior** | the interpreter patch level does not enter compiled sklearn numerics; tested in C/D by using 3.11.13 images |

**Not concluded:** that "cross-platform difference" is the root cause. The evidence localises the
difference to the HistGradientBoosting fits, and shows those fits are stable under random ULP
perturbation on this machine. That makes a plain rounding story *less* likely, not confirmed.

## 3. Scenarios A–D: what each would show and what to do next

| Scenario | How to test | Evidence it produces | Interpretation | Next step |
|---|---|---|---|---|
| **A** original machine reproduces, this one does not | bizhe3 runs the fingerprint on the original machine | their `model.BalAcc` = 79.29988, `mismatch_vs_saved_tuned_C` = 0; `--compare` shows the env differences and the **first differing tree** | the result is machine-dependent; H11 supported, H12/H13 excluded for that machine | decide the canonical training/eval environment (preferably one matching marking: Linux x86-64); re-baseline the M1 gate in that env (your decision) |
| **B** original machine also cannot reproduce | same run | their BalAcc ≠ 79.29988 (or differs between two runs there → H12) | the saved file does not come from the recorded code/env state (H13) or that machine is non-deterministic | group regenerates `ablation/results` with the committed code and records platform + fingerprint; check whether the selection (core 18, 15 leaves, τ = 0.45) still holds; re-baseline the gate on the regenerated numbers (your decision) |
| **C** Linux/amd64 reproduces | Docker linux/amd64, Python 3.11.13, pinned wheels (needs your OK, §5) | 0 mismatches in the container | supports arch/build dependence (macOS-arm64 vs x86-64); also matches the marking platform | run M1 inside that container as the canonical env; macOS only for development, with the gap documented |
| **D** Linux/amd64 does not reproduce | same container | ≠ 79.29988; also run linux/**arm64** to split OS from CPU | the platform story is weaker (at least for Linux x86-64); points to B/H13 or to a specific original setup (e.g. Windows/MSVC) | the A/B test on the original machine becomes the deciding experiment |

Caveat for C/D: on this Mac, linux/amd64 runs **emulated** (QEMU/Rosetta). Its CPU feature set (for
example no AVX-512) can steer numpy/sklearn SIMD dispatch differently from real x86 hardware. A match
is informative; a mismatch under emulation is weaker evidence than one from a real x86 Linux box.

## 4. Local Linux environments (checked, nothing downloaded)

Docker Desktop is installed (engine 29.8.0, **linux/arm64**). I started it only to list local images,
then quit it. Local images: `caddy:2`, `pgvector/pgvector:pg16` (both arm64); neither has Python
3.11 or the pinned stack. No colima / lima / multipass / OrbStack / podman / UTM.
→ **No usable local Linux environment without downloads.**

## 5. Operations that need your authorization (not done)

1. Start Docker Desktop (local, no download).
2. `docker pull --platform linux/amd64 python:3.11.13-slim-bookworm` (Docker Hub, about 45 MB compressed).
3. Optional, to separate OS from arch: `docker pull --platform linux/arm64 python:3.11.13-slim-bookworm` (about 45 MB).
4. In each container (container only, nothing installed on the Mac):
   `pip install numpy==2.4.6 pandas==3.1.0 scipy==1.17.1 scikit-learn==1.9.1 joblib==1.6.0` from PyPI
   (plus their deps threadpoolctl, python-dateutil, tzdata; about 100–150 MB per arch).
5. Run with the project mounted **read-only** and only `results/price_hgbc/selfcheck/` writable:
   `env_fingerprint.py --kit /work/comp9417_project_kit --out …/fp_linux_<arch>.json`, plus `m1_diag_replay`.
6. Clean-up afterwards: `docker rm` the containers; `docker rmi` the images if you want.

## 6. For the original machine (no download on our side)

Send `src/price_hgbc/env_fingerprint.py` (single file) to bizhe3 and ask them to run it from their kit checkout:
```
python env_fingerprint.py --kit . --out fp_original.json
```
Then here: `python src/price_hgbc/env_fingerprint.py --compare results/price_hgbc/selfcheck/fp_macos_arm64_m3.json fp_original.json`.
Running it twice there also tests H12.

## 7. Saved scripts, tests and evidence

| Kind | Path |
|---|---|
| Fingerprint (self-contained, `--compare`) | `src/price_hgbc/env_fingerprint.py` → `fp_macos_arm64_m3.json` |
| Replay of all saved configs + 2013 tuning | `src/price_hgbc/m1_diag_replay.py` → `m1_diag_replay_{configs,summary,tuning}.csv` |
| Sensitivity probes (row order, 1-ULP gradients) | `src/price_hgbc/m1_diag_sensitivity.py` → `m1_diag_sensitivity.csv`, `m1_diag_ulp.csv` |
| Single-number probe | `src/price_hgbc/m1_repro_check.py` |
| Tests | `tests/price_hgbc/test_m1_diag.py` (15 tests, 2 slow), `tests/price_hgbc/test_m1_gate.py` (the gate) |
| Test output | `M1_diag_pytest_output.txt`: **43 passed, 1 failed** (the gate: \|Δ\| = 0.1131 > 0.01) |

The 2013 tuning replay picks the same (15 leaves, τ = 0.45) as the group, though individual fold scores
differ by up to 0.74 pp (`m1_diag_replay_tuning.csv`). The *selection* is reproducible; the exact
score is not.

## 8. Can M1 resume?

**No.** The gate test fails and the cause is not confirmed. M1 can resume when either:
1. a scenario (A or C) yields an environment where `test_m1_gate.py` passes, and you name that environment canonical; or
2. you explicitly decide to re-baseline the gate (for example to regenerated group results in scenario B).
   That changes an assertion, so it is your decision, not mine.
