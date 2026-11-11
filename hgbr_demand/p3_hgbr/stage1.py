"""Stage 1: reproduce the group's HGBR reference, tune two targets, and lock CUSUM."""
import argparse
from hashlib import file_digest
from itertools import product
import json
from pathlib import Path
import platform

import holidays
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from .demand import (DemandData, FEATURES, REF_PARAMS, REPLAY_START, VALID_START, WARM,
                     SLOTS, fit, fit_period, load_released, predict, save_model)
from .train_demand import package_versions

SEED = 0
REF_SEED = 17
TARGETS = ("level", "residual_b1")
GRID = {"max_leaf_nodes": (7, 15, 31, 63), "learning_rate": (0.05, 0.1, 0.2),
        "max_iter": (100, 200, 300)}
METRIC_LABELS = {"mape_pct": "MAPE (%)", "mae_mw": "MAE (MW)"}


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def errors(actual, prediction):
    abs_err = np.abs(np.asarray(actual, dtype=float) - prediction)
    return abs_err, 100 * abs_err / np.abs(actual)


def metrics(actual, prediction):
    abs_err, ape = errors(actual, prediction)
    return {"mape_pct": float(np.mean(ape)), "mae_mw": float(np.mean(abs_err))}


def predict_slice(estimator, data, q, target):
    return predict(estimator, data.x[q], data.b1[q], target)


def score_candidate(data, train, valid, target, params):
    model = fit(data.x[train], data.actual[train], data.b1[train], target, params, SEED)
    p = predict_slice(model, data, valid, target)
    return {"target": target, **params, **metrics(data.actual[valid], p)}


def day_group_metrics(index, actual, pred):
    days = (index - pd.Timedelta(minutes=30)).normalize()
    cal = holidays.country_holidays("AU", subdiv="NSW", years=sorted(set(days.year)))
    is_holiday = np.array([d.date() in cal for d in days], dtype=bool)
    day_type = np.where(is_holiday, "Public holiday", np.where(days.dayofweek >= 5, "Non-holiday weekend", "Non-holiday weekday"))
    rows = []
    for kind, groups in (("weekday", days.day_name()), ("holiday_status", day_type)):
        for label in pd.unique(groups):
            mask = groups == label
            base = metrics(actual[mask], pred["b1"][mask])
            for name, p in pred.items():
                score = metrics(actual[mask], p[mask])
                rows.append({"grouping": kind, "group": label, "model": name,
                             "n_days": int(mask.sum()) // SLOTS, "n_slots": int(mask.sum()), **score,
                             "mape_improvement_vs_b1_pp": base["mape_pct"] - score["mape_pct"]})
    return rows


def calibrate_cusum(daily_ape, n_draws=2000, block_days=7):
    """Upper CUSUM reference: 95th percentile of one-year maxima under the 2014 errors."""
    v = np.asarray(daily_ape, dtype=float)
    if v.shape != (365,) or not np.isfinite(v).all():
        raise ValueError("CUSUM calibration requires every 2014 market day")
    mean = float(v.mean())
    std = float(v.std(ddof=1))
    k = 0.5 * std
    rng = np.random.default_rng(SEED)
    n_blocks = (len(v) + block_days - 1) // block_days
    starts = rng.integers(0, len(v) - block_days + 1, size=(n_draws, n_blocks))
    samples = v[(starts[..., None] + np.arange(block_days)).reshape(n_draws, -1)[:, :len(v)]]
    walk = np.column_stack((np.zeros(n_draws), np.cumsum(samples - mean - k, axis=1)))
    maxima = (walk - np.minimum.accumulate(walk, axis=1)).max(axis=1)
    return {"statistic": "market-day MAPE in percentage points; upper CUSUM", "reference": "2014-01-01 through 2014-12-31",
            "calibration_fit_through": "2013-12-31", "mean_daily_mape_pct": mean,
            "sd_daily_mape_pct": std, "allowance_k_pct": k, "threshold_h": float(np.quantile(maxima, 0.95)),
            "calibration": "95th percentile of 365-day CUSUM maxima under moving-block resampling of 2014 errors",
            "bootstrap_block_days": block_days, "bootstrap_replicates": n_draws, "seed": SEED}


def permutation_table(data, q, estimator, config, n_repeat=5):
    x = data.x[q].copy()
    ref = metrics(data.actual[q], predict_slice(estimator, data, q, config["target"]))["mape_pct"]
    rng = np.random.default_rng(SEED)
    rows = []
    for j, name in enumerate(FEATURES):
        original = x[:, j].copy()
        delta = []
        for _ in range(n_repeat):
            x[:, j] = rng.permutation(original)
            # b1 remains the true known yesterday baseline when predicting a residual.
            p = predict(estimator, x, data.b1[q], config["target"])
            delta.append(metrics(data.actual[q], p)["mape_pct"] - ref)
        x[:, j] = original
        rows.append({"feature": name, "mean_mape_increase_pp": np.mean(delta),
                     "sd_increase_pp": np.std(delta, ddof=1), "repeats": n_repeat})
    return pd.DataFrame(rows).sort_values("mean_mape_increase_pp", ascending=False)


def run(data_path, output, jobs):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    frame = load_released(data_path)
    # Physically exclude any market days after 2014 from the Stage 1 feature/selection pipeline.
    frame = frame.loc[:REPLAY_START]
    data = DemandData.from_frame(frame)
    train = data.interval(WARM, VALID_START)
    valid = data.interval(VALID_START, REPLAY_START)
    assert (train.stop - train.start, valid.stop - valid.start) == (68_784, 17_520)
    old = {"target": "level", "params": REF_PARAMS, "seed": REF_SEED}
    with threadpool_limits(limits=1):
        old_model = fit_period(data, WARM, VALID_START, old)
        pred = {"b1": data.b1[valid], "b7": data.b7[valid],
                "group_reference": predict_slice(old_model, data, valid, "level")}
    reference = metrics(data.actual[valid], pred["group_reference"])
    print("Group direct reference:", reference, flush=True)
    reported = {"mape_pct": 2.4355253151006484, "mae_mw": 192.49681716073613}
    with open(data_path, "rb") as source:
        from hashlib import sha256
        digest = sha256()
        for _ in range(len(frame) + 1):  # include the CSV header
            digest.update(source.readline())
    group_prefix_sha256 = "8243fff94d730db8c221cd21b9d59413c40dfe4cd6621c50e0b35233aca84a4b"
    if digest.hexdigest() != group_prefix_sha256:
        raise RuntimeError("2010–2014 course CSV prefix differs from the group's ablation run")
    write_json(output / "group_reference_check.json",
               {"published": reported, "local": reference,
                "local_minus_published": {k: reference[k]-reported[k] for k in reported},
                "group_csv_prefix_sha256": group_prefix_sha256, "local_csv_prefix_match": True,
                "identical_metrics": all(abs(reference[k]-reported[k]) < 1e-10 for k in reported)})
    if abs(reference["mape_pct"] - reported["mape_pct"]) > 0.01 or abs(reference["mae_mw"] - reported["mae_mw"]) > 1.0:
        raise RuntimeError(f"Material difference from group reference: {reference}")
    configs = [dict(zip(GRID, values)) for values in product(*GRID.values())]
    with threadpool_limits(limits=1):
        results = Parallel(n_jobs=jobs, prefer="threads", verbose=10)(
            delayed(score_candidate)(data, train, valid, target, params)
            for target, params in product(TARGETS, configs))
    ranking = pd.DataFrame(results).sort_values(
        ["mape_pct", "mae_mw", "max_iter", "max_leaf_nodes", "learning_rate", "target"], ignore_index=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking)+1))
    ranking.to_csv(output / "tuning_2014.csv", index=False)
    choices = {}
    versions = package_versions()
    with threadpool_limits(limits=1):
        for target in TARGETS:
            row = ranking.loc[ranking["target"] == target].iloc[0]
            params = {key: (float(row[key]) if key == "learning_rate" else int(row[key])) for key in GRID}
            choices[target] = {"target": target, "params": params, "seed": SEED}
            model = fit_period(data, WARM, VALID_START, choices[target])
            pred[target] = predict_slice(model, data, valid, target)
        selected_target = str(ranking.iloc[0]["target"])
        selected = choices[selected_target]
        selected_model = fit_period(data, WARM, VALID_START, selected)
        permutation_table(data, valid, selected_model, selected).to_csv(output / "validation_feature_importance.csv", index=False)
        frozen = fit_period(data, WARM, REPLAY_START, selected)
        save_model(output / "frozen" / "demand_hgbr.joblib", frozen, selected, WARM, REPLAY_START, versions)

    names = ["group_reference", "level", "residual_b1", "b1", "b7"]
    val_rows = [{"model": name, "n_slots": len(data.actual[valid]), **metrics(data.actual[valid], pred[name])}
                for name in names]
    pd.DataFrame(val_rows).to_csv(output / "validation_2014.csv", index=False)
    pd.DataFrame([{"member": "P3", "model": "HistGradientBoostingRegressor", "n_features": len(FEATURES),
                   "target": selected_target, **selected["params"], "seed": SEED,
                   **metrics(data.actual[valid], pred[selected_target])}]).to_csv(output / "p3_shared_row.csv", index=False)
    table = pd.DataFrame({"timestamp": data.index[valid], "actual_mw": data.actual[valid], **pred})
    table.to_csv(output / "validation_predictions.csv.gz", index=False, compression={"method":"gzip", "mtime":0})
    pd.DataFrame(day_group_metrics(data.index[valid], data.actual[valid], pred)).to_csv(
        output / "validation_weekday_holiday.csv", index=False)
    abs_err, ape = errors(data.actual[valid], pred[selected_target])
    daily = pd.DataFrame({"market_day": data.days[valid][::SLOTS],
                          "mape_pct": ape.reshape(-1, SLOTS).mean(axis=1),
                          "mae_mw": abs_err.reshape(-1, SLOTS).mean(axis=1)})
    daily.to_csv(output / "cusum_reference_daily_2014.csv", index=False)
    write_json(output / "cusum_config.json", calibrate_cusum(daily["mape_pct"]))
    with open(data_path, "rb") as source:
        dataset_hash = file_digest(source, "sha256").hexdigest()
    write_json(output / "p3_stage1_candidate.json",
               {"selected": selected, "best_by_target": choices, "group_reference": old,
                "features": FEATURES, "validation": "2014-01-01 through 2014-12-31",
                "frozen_fit_through": "2014-12-31", "training_start": str(WARM.date()),
                "selection": "minimum 2014 MAPE; ties by MAE, iterations, leaves, rate and target",
                "validation_metrics": metrics(data.actual[valid], pred[selected_target]),
                "dataset_sha256": dataset_hash, "python": platform.python_version(),
                "versions": {**versions, "holidays": holidays.__version__},
                "status": "P3 handoff candidate; not the team's approved configuration"})
    print(pd.DataFrame(val_rows).to_string(index=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--jobs", type=int, default=3)
    a = p.parse_args()
    run(a.data, a.out, a.jobs)


if __name__ == "__main__":
    main()
