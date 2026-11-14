"""Write results/price_hgbc/m5/README_frozen.md from the frozen M5 files (numbers are read, not typed).
Usage: python -m src.price_hgbc.m5_readme
"""
import json

import pandas as pd

from . import m5


def build():
    cfg = json.load(open(m5.P["config"]))
    man = json.load(open(m5.P["manifest"]))
    yearly = pd.read_csv(m5.P["yearly"])
    allrow = json.load(open(m5.P["yearly"].replace("dev_yearly_2015_2024.csv", "dev_all_2015_2024.json")))
    dv = pd.read_csv(m5.P["daily"])
    c = cfg["cusum"]
    L = []
    w = L.append
    w("# Stage 1 frozen price-direction model (M5)\n")
    w("Frozen for the Stage 2 replay (2015-01 … 2026-10). Do not edit these files; every change needs a new freeze.\n")
    w("## Model\n")
    w(f"- {cfg['model']}. All HGBC parameters are read from REF (`{cfg['params_from'].split(' ')[0]}`); nothing is restated in code.")
    p = cfg["params"]
    w(f"- HistGradientBoostingClassifier: loss={p['loss']}, max_leaf_nodes={p['max_leaf_nodes']}, max_iter={p['max_iter']}, "
      f"learning_rate={p['learning_rate']}, early_stopping={p['early_stopping']}, random_state={p['random_state']}, "
      f"categorical_features={p['categorical_features']!r}; decision 1 iff p_up ≥ τ = {cfg['tau']}.")
    w(f"- Training: target days {cfg['train']['first_day']} … {cfg['train']['last_day']} ({cfg['train']['days']:,} days, "
      f"{cfg['train']['rows']:,} half-hours), via {cfg['train']['source']}. The first day follows P1's 28-day warm-up "
      "(the lag-7 columns alone would allow 2010-01-08).")
    w(f"- sklearn {man['versions']['sklearn']}, numpy {man['versions']['numpy']}, pandas {man['versions']['pandas']}, "
      f"Python {man['versions']['python']}; trained {man['trained_at']} in {man['train_seconds']} s.")
    w(f"- Feature code: P1 `{cfg['feature_code']['module']}` (`_construct`, `_fit_day_types`, `CORE`), commit "
      f"`{cfg['feature_code']['p1_commit'][:7]}`, imported unchanged. P1's day-type model is fitted on ≤ 2013 profiles; "
      "only the 18 CORE columns reach the model.\n")
    w("### Features (order = REF = training = prediction)\n")
    w(" ".join(f"{i + 1}. `{f}`" for i, f in enumerate(cfg["features"])) + "\n")
    w("## Files and hashes\n")
    w("| File | SHA-256 |")
    w("|---|---|")
    for k, v in man["files"].items():
        w(f"| `{v['path']}` | `{v['sha256']}` |")
    w(f"\nPrediction arrays: p_up `{man['fingerprint']['p_up_sha256']}`, pred `{man['fingerprint']['pred_sha256']}`; "
      f"tree fingerprint `{man['fingerprint']['trees']['trees_sha256']}`. Data CSV `{man['data_csv_sha256'][:16]}…`, "
      f"REF `{man['ref_sha256'][:16]}…`.\n")
    w("## Prediction file (`predictions.parquet`)\n")
    w(f"- Index: timestamp (end of half-hour, as in the released CSV). Columns: `day` (market day = timestamp − 30 min, normalised), "
      f"`period` (0–47), `p_up`, `pred` (= 1[p_up ≥ τ]). {man['pred_days']:,} target days "
      f"{cfg['predictions']['first_day']} … {cfg['predictions']['last_day']}, {man['pred_rows']:,} rows. "
      "**No labels are stored**; score with `da.targets`.")
    w("- Ownership: the frozen replay of the **price** model (boost_price) is produced here, by the price_hgbc workstream "
      "(HGBC / CUSUM / S0–S3). P1 supplies the feature code and configuration and does not need to re-run it.")
    w("- **To confirm with P1 / P3:** the project files contain no written format agreement. This layout (timestamp index, "
      "day, period, p_up, pred) is a proposal until P1 and P3 confirm it.\n")
    w("## Day-by-day interface (for M6 / P3)\n")
    w(f"`{cfg['interface']}` in `src/price_hgbc/m5.py`. Inputs are exactly the kit's `Forecaster.predict` inputs "
      "(`history` up to the end of day D, `weather_next` = 48 rows of D+1 weather). It keeps the last "
      f"{cfg['feature_code']['window_days_per_call']} days of history, sets D+1 market columns to NaN, and calls P1's builder.\n")
    w("Batch = day-by-day evidence (`daily_vs_batch.csv`):\n")
    w("| Day | history rows | seconds | p_up bitwise equal | pred equal |")
    w("|---|---|---|---|---|")
    for _, r in dv.iterrows():
        w(f"| {r['day']} | {r['history_rows']:,} | {r['seconds']:.3f} | {r['p_up_bitwise_equal']} | {r['pred_equal']} |")
    w("\n## Development scores 2015–2024 (official `da` metrics; 2025–2026 not scored)\n")
    w("| Year | BalAcc model | BalAcc b1 | BalAcc b7 | Acc model | Acc b1 | Acc b7 |")
    w("|---|---|---|---|---|---|---|")
    for _, r in yearly.iterrows():
        w(f"| {int(r['year'])} | {r['model_BalAcc']:.3f} | {r['b1_BalAcc']:.3f} | {r['b7_BalAcc']:.3f} | "
          f"{r['model_Acc']:.3f} | {r['b1_Acc']:.3f} | {r['b7_Acc']:.3f} |")
    w(f"| 2015–2024 | {allrow['model_BalAcc']:.3f} | {allrow['b1_BalAcc']:.3f} | {allrow['b7_BalAcc']:.3f} | "
      f"{allrow['model_Acc']:.3f} | {allrow['b1_Acc']:.3f} | {allrow['b7_Acc']:.3f} |")
    w("\nMonthly values: `dev_monthly_2015_2024.csv`. These scores were not used for any choice.\n")
    w("## CUSUM for M6 (M4, option A)\n")
    w(f"- Config `{c['config']}`: ref_update = **{c['ref_update']}**; q99 = {c['q99']!r}, μ₀ = {c['mu0']!r}, σ₀ = {c['sigma0']!r}, "
      f"k = {c['k']}, h = {c['h']}. **Read these from the config, not from this text.** S resets to 0 after a retrain. "
      f"Historical 2014 alarm kept: {c['historical_alarm_2014']}.")
    w("- **μ₀ and σ₀ come from an earlier model** (REF, trained to 2013) and its 2014 out-of-sample errors. The Stage 1 model "
      "here is refit to 2014, so its normal error distribution may differ. M6 must check S3's alarm frequency, any cooldown "
      "and the budget of at most 6 retrains. It must not assume the M4 ARL₀ holds for every new model.\n")
    w("## Known limitations\n")
    w("- τ = 0.45 was chosen on 2014 for models trained on 2010–2013. After refitting to 2010–2014 the probability "
      "distribution can shift, so τ = 0.45 is not re-optimised for this model. This is a known limitation, kept on purpose (no re-tuning).")
    w("- In M6, S3's CUSUM must read the 2025–2026 truth to update its statistic. That is a legitimate part of the simulation, "
      "but **2025–2026 scores must not be used for any choice** (they are the M8 hold-out).")
    w("- The batch builds features on the whole file. Causality is shown by tests: shuffling D+1-onward market data leaves "
      "D+1 unchanged, and batch = day-by-day bitwise.")
    return "\n".join(L) + "\n"


def main():
    open(m5.P["readme"], "w", encoding="utf-8").write(build())
    print("wrote", m5.P["readme"])


if __name__ == "__main__":
    main()
