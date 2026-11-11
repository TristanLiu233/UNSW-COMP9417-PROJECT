"""Run chronological demand-only simulation, drift detection, and adaptation."""
import argparse
from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .demand import (DATA_STOP, DAY, REPLAY_START, SLOTS, VALID_START, WARM,
                     DemandData, fit_period, load_released, save_model)
from .stage1 import errors, metrics, predict_slice, write_json
from .train_demand import package_versions

HOLDOUT = pd.Timestamp("2025-01-01")
PERIODIC_DATES = {pd.Timestamp(f"{y}-01-01") for y in (2017, 2019, 2021, 2023, 2025)}
MAX_RETRAINS = 6
COOLDOWN_DAYS = 365
WINDOW_YEARS = 4
BOOT_N = 2000
STRATEGIES = ("frozen", "periodic_expanding", "periodic_recent4y", "cusum_expanding")
INPUT_NAMES = ("yesterday_load_mean", "yesterday_load_shape", "yesterday_price_slog",
               "vic_yesterday_load_mean", "sydney_temp_mean", "sydney_solar_mean")


def monthly_scores(index_days, truth, predictions):
    month = index_days.to_period("M")
    rng = np.random.default_rng(0)
    groups = []
    for date in month.unique():
        mask = month == date
        n = int(mask.sum())
        if n % SLOTS:
            raise ValueError(f"Incomplete market days in month {date}")
        days = n // SLOTS
        # Reuse the same daily bootstrap draw for all models to make intervals comparable.
        sample = rng.integers(0, days, size=(BOOT_N, days))
        for name, p in predictions.items():
            abs_err, ape = errors(truth[mask], p[mask])
            mean = float(ape.mean())
            margin = 1.96 * float(ape.std(ddof=1)) / np.sqrt(n)
            boot = ape.reshape(days, SLOTS).mean(axis=1)[sample].mean(axis=1)
            lo, hi = np.quantile(boot, [0.025, 0.975])
            groups.append({"month": str(date), "model": name, "n_days": days, "n_slots": n,
                           "mape_pct": mean, "mae_mw": abs_err.mean(),
                           "normal_ci_low": mean - margin, "normal_ci_high": mean + margin,
                           "day_bootstrap_ci_low": lo, "day_bootstrap_ci_high": hi})
    return pd.DataFrame(groups)


def yearly_scores(index_days, truth, predictions):
    return pd.DataFrame([
        {"year": year, "model": name, "n_slots": int((mask := (index_days.year == year)).sum()),
         **metrics(truth[mask], p[mask])}
        for year in index_days.year.unique()
        for name, p in predictions.items()])


def upper_cusum_update(previous, value, config):
    return max(0.0, previous + value - config["mean_daily_mape_pct"] - config["allowance_k_pct"])


def error_alerts(days, daily_mape, config):
    s = 0.0
    values, alerts = [], []
    for day, err in zip(days, daily_mape):
        s = upper_cusum_update(s, err, config)
        values.append(s)
        if s > config["threshold_h"]:
            alerts.append({"detector": "frozen_error_cusum", "observed_day": day,
                           "score": s, "threshold": config["threshold_h"]})
            s = 0.0
    return np.array(values), alerts


def input_drift(data):
    """Reference on 2010–13; threshold on 2014; trailing score only uses current/past inputs."""
    cols = {"nsw_demand_lag1": 12, "prev_price_mean": 16,
            "syd_temp": 18, "syd_solar": 21, "vic_demand_lag1": 22}
    daily_x = data.x.reshape(-1, SLOTS, data.x.shape[1])
    cols = {name: j for name, j in cols.items() if j < data.x.shape[1]}
    from .demand import FEATURES
    if any(FEATURES[j] != name for name, j in cols.items()):
        raise ValueError("Unexpected input schema")
    nsw = daily_x[:, :, cols["nsw_demand_lag1"]]
    prices = daily_x[:, 0, cols["prev_price_mean"]]
    nsw_mean = nsw.mean(axis=1)
    # Compare afternoon against evening, normalized by the overall historical-day load.
    shape = (nsw[:, 26:32].mean(axis=1) - nsw[:, 34:40].mean(axis=1)) / nsw_mean
    z = pd.DataFrame(
        np.column_stack((nsw_mean, shape, np.sign(prices) * np.log1p(np.abs(prices)),
                         daily_x[:, :, cols["vic_demand_lag1"]].mean(axis=1),
                         daily_x[:, :, cols["syd_temp"]].mean(axis=1),
                         daily_x[:, :, cols["syd_solar"]].mean(axis=1))),
        index=data.days[::SLOTS], columns=INPUT_NAMES,
    )
    ref = z.loc[(z.index >= WARM) & (z.index < VALID_START)]
    cent = ref.groupby(ref.index.month).mean()
    sd = ref.groupby(ref.index.month).std(ddof=1)
    if not (sd > 0).all().all():
        raise ValueError("Input reference has a zero variance")
    scale = (z.to_numpy() - cent.loc[z.index.month].to_numpy()) / sd.loc[z.index.month].to_numpy()
    dist = pd.Series(np.sqrt(np.mean(scale ** 2, axis=1)), index=z.index)
    rolling = dist.rolling(28, min_periods=28).mean()
    threshold = float(rolling.loc["2014"].quantile(0.95))
    replay = rolling.loc[rolling.index >= REPLAY_START]
    prev_above = bool(rolling.loc["2014"].iloc[-1] > threshold)
    alerts = []
    for day, value in replay.items():
        above = value > threshold
        if above and not prev_above:
            alerts.append({"detector": "input_only", "observed_day": day, "score": value,
                           "threshold": threshold})
        prev_above = above
    profile = pd.concat([z, dist.rename("input_distance"), rolling.rename("input_score_28d")], axis=1)
    return profile, alerts, {"reference": "2010-01-29 through 2013-12-31, grouped by calendar month",
                             "calibration": "95th percentile of 2014 trailing 28-market-day scores",
                             "statistic": "trailing mean of RMS standardized distances in six daily predictors",
                             "variables": INPUT_NAMES, "threshold": threshold}


def replay_adapted(data, q, config, cusum, initial, strategy):
    pred = np.empty(q.stop - q.start)
    current_model = initial
    target = config["target"]
    events, alarms = [], []
    s = 0.0
    last_fit = REPLAY_START
    for start in range(q.start, q.stop, SLOTS):
        day = data.days[start]
        # Scheduled updates predict their due day, fitting only completed days.
        if strategy.startswith("periodic") and day in PERIODIC_DATES:
            fit_start = WARM if strategy == "periodic_expanding" else max(WARM, day - pd.DateOffset(years=WINDOW_YEARS))
            current_model = fit_period(data, fit_start, day, config)
            events.append({"strategy": strategy, "reason": "scheduled", "alarm_day": None,
                           "first_prediction_day": day, "train_first_day": fit_start, "train_last_day": day-DAY})
        window = slice(start, start + SLOTS)
        p = predict_slice(current_model, data, window, target)
        pred[start-q.start:start-q.start+SLOTS] = p
        if strategy == "cusum_expanding":
            s = upper_cusum_update(s, metrics(data.actual[window], p)["mape_pct"], cusum)
            if s > cusum["threshold_h"]:
                next_day = day + DAY
                eligible = (len(events) < MAX_RETRAINS and next_day < DATA_STOP
                            and (next_day-last_fit).days >= COOLDOWN_DAYS)
                alarms.append({"detector": "adaptive_error_cusum", "observed_day": day,
                               "score": s, "threshold": cusum["threshold_h"], "triggered_retrain": eligible})
                s = 0.0
                if eligible:
                    # Observing this day's actuals happens after its prediction.
                    current_model = fit_period(data, WARM, next_day, config)
                    events.append({"strategy": strategy, "reason": "cusum", "alarm_day": day,
                                   "first_prediction_day": next_day, "train_first_day": WARM, "train_last_day": day})
                    last_fit = next_day
    if len(events) > MAX_RETRAINS or any(e["train_last_day"] >= e["first_prediction_day"] for e in events):
        raise AssertionError("A replay update violated the budget or chronological cutoff")
    return pred, events, alarms


def alarm_context(daily, alarms, width=30):
    rows = []
    frozen = daily.set_index("market_day")["frozen_mape_pct"]
    for record in alarms:
        date = record["observed_day"]
        before = frozen.loc[date - pd.Timedelta(days=width):date - DAY]
        after = frozen.loc[date + DAY:date + pd.Timedelta(days=width)]
        rows.append({**record, "frozen_mape_previous_30d_pct": before.mean(),
                     "frozen_mape_next_30d_pct": after.mean(), "previous_days": len(before), "next_days": len(after),
                     "frozen_mape_change_pp": after.mean() - before.mean()})
    return pd.DataFrame(rows)


def fit_start_for_release(strategy, exclusive_stop):
    if strategy == "periodic_recent4y":
        return max(WARM, exclusive_stop - pd.DateOffset(years=WINDOW_YEARS))
    return WARM


def live_week_analogs(data, config, strategy):
    """Season/lead matched analogs: Dec 7–13, 37–43 days after each Oct 31 training end."""
    records = []
    for year in range(2015, 2026):
        train_stop = pd.Timestamp(f"{year}-11-01")
        q = data.interval(f"{year}-12-07", f"{year}-12-14")
        fit_start = fit_start_for_release(strategy, train_stop)
        model = fit_period(data, fit_start, train_stop, config)
        p = predict_slice(model, data, q, config["target"])
        base = metrics(data.actual[q], data.b1[q])
        records.append({"year": year, "train_first_day": fit_start, "train_last_day": train_stop-DAY,
                        "week_start": f"{year}-12-07", "week_end": f"{year}-12-13",
                        "n_slots": q.stop-q.start, **metrics(data.actual[q], p),
                        "b1_mape_pct": base["mape_pct"],
                        "b7_mape_pct": metrics(data.actual[q], data.b7[q])["mape_pct"]})
    return pd.DataFrame(records)


def run(data_path, stage1, out, ridge_dir=None):
    stage1, out = Path(stage1), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    s1 = json.loads((stage1 / "p3_stage1_candidate.json").read_text())
    cusum = json.loads((stage1 / "cusum_config.json").read_text())
    with open(data_path, "rb") as source:
        if file_digest(source, "sha256").hexdigest() != s1["dataset_sha256"]:
            raise ValueError("Stage 2 dataset differs from the Stage 1 dataset")
    config = s1["selected"]
    frame = load_released(data_path)
    data = DemandData.from_frame(frame)
    q = data.interval(REPLAY_START, DATA_STOP)
    assert q.stop - q.start == 207_456
    days = data.days[q]
    truth = data.actual[q]
    initial = joblib.load(stage1 / "frozen" / "demand_hgbr.joblib")["model"]
    predictions = {"b1": data.b1[q], "b7": data.b7[q]}
    events, alarms = [], []
    with threadpool_limits(limits=1):
        for name, choice in {"group_reference": s1["group_reference"], **s1["best_by_target"]}.items():
            m = initial if choice == config else fit_period(data, WARM, REPLAY_START, choice)
            predictions[f"frozen_{name}"] = predict_slice(m, data, q, choice["target"])
        predictions["frozen"] = predictions[f"frozen_{config['target']}"]
        for strategy in STRATEGIES[1:]:
            p, e, a = replay_adapted(data, q, config, cusum, initial, strategy)
            predictions[strategy] = p
            events.extend(e)
            alarms.extend(a)
            print(strategy, metrics(truth, p), "retrains", len(e), flush=True)

    full = pd.DataFrame({"timestamp": data.index[q], "market_day": days,
                         "actual_demand_mw": truth, **predictions})
    if not np.isfinite(full[list(predictions)].to_numpy()).all():
        raise AssertionError("Non-finite historical prediction")
    full.to_csv(out / "half_hour_predictions.csv.gz", index=False, compression={"method":"gzip", "mtime":0})
    # Avoid listing the selected frozen configuration twice in monthly/yearly tables.
    metrics_predictions = {k:v for k,v in predictions.items() if k != f"frozen_{config['target']}"}
    monthly = monthly_scores(days, truth, metrics_predictions)
    yearly = yearly_scores(days, truth, metrics_predictions)
    monthly.to_csv(out / "monthly.csv", index=False)
    yearly.to_csv(out / "yearly.csv", index=False)

    counts = Counter(e["strategy"] for e in events)
    split = days < HOLDOUT
    compare = []
    for name, p in metrics_predictions.items():
        a, b, c = metrics(truth, p), metrics(truth[split], p[split]), metrics(truth[~split], p[~split])
        compare.append({"model": name, "retrains": counts[name], **a,
                        "selection_2015_2024_mape_pct": b["mape_pct"], "selection_2015_2024_mae_mw": b["mae_mw"],
                        "holdout_2025_2026_mape_pct": c["mape_pct"], "holdout_2025_2026_mae_mw": c["mae_mw"]})
    comparison = pd.DataFrame(compare)
    comparison.to_csv(out / "strategy_comparison.csv", index=False)
    selection = comparison.loc[comparison["model"].isin(STRATEGIES)].sort_values(
        ["selection_2015_2024_mape_pct", "selection_2015_2024_mae_mw", "retrains", "model"])
    strategy = str(selection.iloc[0]["model"])
    pd.DataFrame(events, columns=["strategy", "reason", "alarm_day", "first_prediction_day",
                                  "train_first_day", "train_last_day"]).to_csv(out / "retraining_log.csv", index=False)

    unique = days[::SLOTS]
    frozen_day = errors(truth, predictions["frozen"])[1].reshape(-1, SLOTS).mean(axis=1)
    cusum_scores, frozen_alarms = error_alerts(unique, frozen_day, cusum)
    input_scores, input_alarms, input_cfg = input_drift(data)
    write_json(out / "input_detector.json", input_cfg)
    daily = pd.DataFrame({"market_day": unique, "frozen_mape_pct": frozen_day, "frozen_cusum": cusum_scores,
                          "input_score_28d": input_scores.loc[unique, "input_score_28d"].to_numpy()})
    for name in STRATEGIES[1:]:
        daily[f"{name}_mape_pct"] = errors(truth, predictions[name])[1].reshape(-1, SLOTS).mean(axis=1)
    daily.to_csv(out / "daily_monitors.csv", index=False)
    alarm_context(daily, frozen_alarms+input_alarms+alarms).to_csv(out / "drift_alarms.csv", index=False)

    if ridge_dir is not None:
        ridge = pd.read_csv(Path(ridge_dir) / "ridge_monthly.csv")
        ours = monthly.loc[monthly["model"] == "frozen", ["month", "n_slots", "mape_pct", "mae_mw"]]
        joined = ours.merge(ridge[["month", "n_slots", "mape_pct", "mae_mw"]], on=["month", "n_slots"],
                            how="outer", suffixes=("_hgbr", "_p2_ridge"), validate="one_to_one", indicator=True)
        if len(joined) != 142 or not joined["_merge"].eq("both").all():
            raise ValueError("P2 Ridge monthly summaries differ in window or sample count")
        joined.drop(columns="_merge").to_csv(out / "monthly_comparison_with_p2_ridge.csv", index=False)

    start = fit_start_for_release(strategy, DATA_STOP)
    final_config = {**config, "strategy_from_2015_2024": strategy,
                    "final_train_start": str(start.date()), "final_train_through": "2026-10-31",
                    "holdout_period": "2025-01-01 through 2026-10-31 (not used to rank strategies)",
                    "status": "P3 demand candidate for group review"}
    write_json(out / "p3_final_demand_config.json", final_config)
    write_json(out / "replay_protocol.json",
               {"candidate": config, "initial_fit_through": "2014-12-31", "replay_slots": len(truth),
                "periodic_dates": sorted(str(d.date()) for d in PERIODIC_DATES),
                "retrain_budget_per_strategy": MAX_RETRAINS, "cusum_min_days_between_refits": COOLDOWN_DAYS,
                "window_years": WINDOW_YEARS, "monthly_bootstrap_replicates": BOOT_N,
                "selection_period": "2015-01-01 through 2024-12-31",
                "alarm_context": "descriptive 30-day windows around alerts, excluding the alert day",
                "ridge_monthly_comparison": "P2 branch published aggregation, aligned by month and n_slots; not rerun"})
    with threadpool_limits(limits=1):
        analog = live_week_analogs(data, final_config, strategy)
        estimator = fit_period(data, start, DATA_STOP, final_config)
    analog.to_csv(out / "live_week_demand_analogs.csv", index=False)
    write_json(out / "live_week_demand_analog_summary.json",
               {"point_median_mape_pct": float(analog["mape_pct"].median()),
                "empirical_middle_80_pct": [float(analog["mape_pct"].quantile(p)) for p in (0.10, 0.90)],
                "n_weeks": len(analog), "lead_days": [37, 43], "window": "Dec 7–13 for 2015–2025",
                "interpretation": "descriptive historical analog spread, not a guaranteed prediction interval or live score"})
    save_model(out / "final_artifacts" / "demand_hgbr.joblib", estimator,
               final_config, start, DATA_STOP, package_versions())
    print(comparison.to_string(index=False), flush=True)
    print("Stage 3 strategy:", strategy, "training starts", start.date(), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--stage1", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ridge-dir", type=Path)
    args = p.parse_args()
    run(args.data, args.stage1, args.out, args.ridge_dir)


if __name__ == "__main__":
    main()
