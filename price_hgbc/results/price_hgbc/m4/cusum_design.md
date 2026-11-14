# M4 CUSUM design (pre-declared; written before any M4 number was computed; not changed during execution)

Model: official `boost_price` (REF = `results/price_hgbc/official/boost_price_ref.json`). The 2014 out-of-sample
predictions come from `official/preds_2014.parquet` (model trained up to 2013-12-31). **No data after 2015-01-01 00:00 is read.**

## A. Design given by the user (verbatim content)

* **Daily error e_D**
  * main: mean log loss over the 48 half-hours of day D, probabilities clipped to [1e-6, 1 − 1e-6];
  * comparison: misclassification rate of day D at τ = 0.45.
* **Truncation:** q99 = 99 % quantile of 2014 e_D; before CUSUM, e_D ← min(e_D, q99). q99 is estimated on 2014 only and then fixed.
* **Reference:** μ₀ and σ₀ are estimated from the truncated 2014 e_D.
* **Statistic:** upper CUSUM, S_D = max(0, S_{D−1} + (e_D − μ₀)/σ₀ − k), k = 0.5. Alarm when S_D > h; S resets to 0 on the alarm day.
* **Timing:** the truth of day D is known at the end of day D. S_D is updated at the end of day D; after an alarm,
  the retrained model is used from the forecast of D+1 onwards.
* **Choosing h:** target in-control ARL₀ ≥ 730 days. The candidates are h ∈ {3, 4, 5, 6, 7, 8, 10}; take the smallest
  that meets the target.
* **ARL₀ estimate:** weekly block bootstrap of the truncated standardised 2014 errors (block length 7 days), concatenated
  to 3650 days. Run CUSUM, count alarm intervals, repeat 1000 times with seed 0, and report the median (ARL₀) plus the
  5 % and 95 % quantiles.
* **Reference after retraining:** both options are implemented behind `ref_update`; no decision is made in M4.
  * A: keep the 2014 μ₀, σ₀, q99;
  * B: re-estimate them from the new model's out-of-sample errors on the last 12 months of its training window
    (computed within the same retraining).

## B. Implementation details fixed here (the user's text leaves these open)

1. **Quantile:** `numpy.quantile(e, 0.99)` with the default linear interpolation, on the 365 daily values of 2014.
2. **μ₀, σ₀:** mean and sample standard deviation (ddof = 1) of the truncated 2014 e_D.
3. **Standardised error:** z_D = (min(e_D, q99) − μ₀) / σ₀. The CUSUM increment is z_D − k.
4. **Alarm-day value:** `update` returns S_D before the reset (the value compared with h) and the alarm flag.
   The internal state is then set to 0, so S_{D+1} = max(0, z_{D+1} − k).
5. **Weekly blocks:** to keep the weekday pattern, a block is a full calendar week Monday–Sunday inside 2014.
   That gives 51 non-overlapping weeks (2014-01-06 … 2014-12-28); 2014-01-01 … 01-05 and 2014-12-29 … 12-31 are
   not used as bootstrap material (8 days). Each replicate draws ⌈3650 / 7⌉ = 522 weeks with replacement,
   concatenates them in drawn order (weekday phase preserved), and keeps the first 3650 days.
6. **Common random numbers:** one week-index matrix (1000 × 522), `numpy.random.default_rng(0)`, is shared by every
   h and by every error variant, so ARL₀ across h compares like with like.
7. **ARL₀ per replicate:** 3650 / (number of alarms in the 3650 days). This is the average spacing between false alarms,
   with the open interval after the last alarm counted in. With no alarm the value is +∞ (reported as "> 3650").
   ARL₀ = median over the 1000 replicates; the 5 % / 95 % quantiles are reported the same way (+∞ allowed).
8. **No candidate meets the target:** report "target not met", provisionally take the largest candidate (h = 10),
   and flag it for the user. No candidates are added.
9. **Comparison scheme (misclassification rate):** the same steps 1–8 with e_D = 1 − daily accuracy at τ = 0.45.
10. **No-truncation scheme:** the same pipeline with no clipping; μ₀ and σ₀ come from the untruncated 2014 e_D.
11. **Detection-delay material (selected h only, not used for any choice):** zero-state delay = days from the start of a
    shift to the first alarm, averaged over 1000 replicates (seed 0), for shifts δ ∈ {0.5, 1, 2, 3} σ₀. There are two
    noise models: (i) iid N(0, 1) standardised errors; (ii) the weekly block bootstrap of the real 2014 z. In both,
    the shift is added on the error scale and then truncated at q99 (as in deployment). The delay is capped at 3650 days.
12. **Option B estimator:** for a retrain whose window ends at day T_end, fit the same configuration on
    [window start, T_end − 12 months], predict the last 12 months (T_end − 12 months, T_end], compute daily e_D,
    then apply steps 1–2. M4 demonstrates it once, for a window ending 2013-12-31 (inner fit 2010-01-29 … 2012-12-31,
    predicted 2013). This uses no 2014+ labels and no 2015+ data.
13. **Code version:** the project root is not a git repository, so there is no commit. The config records the SHA-256
    of the M4 code files instead.
