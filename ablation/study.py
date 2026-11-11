"""Paired source, dimension and feature-construction experiments for one algorithm."""

import json
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from harness import dayahead
from . import features as f
from .models import PRIMARY, SPECS, TARGET, predict_raw
from .scoring import Bootstrap, compare_samples, metrics
from .tuning import tune


ORDER = [*f.CORE, *f.WEATHER, *f.REGION, *f.EXTRA_WEATHER,
         *f.ROLLING[7], *f.ROLLING[28], *f.PRICE_SPIKES, *f.DAY_TYPE]
TOLERANCE = {"demand": 0.05, "price": 0.5}  # percentage points


@dataclass
class Evaluation:
    columns: tuple
    metrics: dict
    samples: np.ndarray
    predictions: np.ndarray


class Experiment:
    def __init__(self, name, data):
        self.name = name
        self.spec = SPECS[name]
        self.task = self.spec.task
        self.primary = PRIMARY[self.task]
        self.direction = -1 if self.task == "demand" else 1
        self.days = dayahead.market_day(data.index)
        self.features, truth = f.build_features(data)
        train = (self.days >= "2010-01-29") & (self.days < "2014-01-01")
        valid = self.days.year == 2014
        self.train = self.features.loc[train]
        self.y_train = truth.loc[train, TARGET[self.task]]
        self.valid = self.features.loc[valid]
        self.truth = truth.loc[valid]
        self.bootstrap = Bootstrap(self.truth)
        self.raw_cache = {}
        self.evaluation_cache = {}
        self.runs = {}
        self.score_rows = []
        self.contrasts = []
        self.baselines = {}
        for lag in (1, 7):
            pred = (self.valid[f"nsw_demand_lag{lag}"] if self.task == "demand" else
                    (self.valid[f"nsw_price_lag{lag}"] > self.valid["prev_price_mean"]).astype(int))
            self.baselines[lag] = metrics(self.task, self.truth, pred)[self.primary]

    def evaluate(self, key, stage, columns, param=None, threshold=None):
        if key in self.runs:
            raise ValueError(f"Duplicate configuration: {key}")
        requested = set(columns)
        cols = tuple(c for c in ORDER if c in requested)
        if not cols or len(cols) != len(requested):
            raise ValueError(f"Empty or unknown features in {key}: {requested - set(cols)}")
        param = self.spec.default if param is None else param
        if self.task == "price" and threshold is None:
            threshold = 0.5
        fit_key = (cols, param)
        if fit_key not in self.raw_cache:
            start = perf_counter()
            raw = predict_raw(self.name, param, self.train.loc[:, list(cols)], self.y_train,
                              self.valid.loc[:, list(cols)])
            self.raw_cache[fit_key] = (raw, perf_counter() - start)
        raw, seconds = self.raw_cache[fit_key]
        evaluation_key = (fit_key, threshold)
        if evaluation_key not in self.evaluation_cache:
            pred = (raw >= threshold).astype(np.int8) if self.task == "price" else raw
            self.evaluation_cache[evaluation_key] = Evaluation(
                cols, metrics(self.task, self.truth, pred), self.bootstrap.samples(self.task, pred), pred)
        run = self.runs[key] = self.evaluation_cache[evaluation_key]
        score = run.metrics[self.primary]
        self.score_rows.append({"model": self.name, "task": self.task, "stage": stage, "config": key,
            "n_features": len(cols), "columns": json.dumps(cols), "parameter": param, "threshold": threshold,
            "train_rows": len(self.train), "validation_rows": len(self.valid), "primary": self.primary,
            **run.metrics, "delta_vs_yesterday": score - self.baselines[1],
            "delta_vs_last_week": score - self.baselines[7], "fit_predict_seconds": seconds})
        return key

    def point(self, key):
        return self.runs[key].metrics[self.primary]

    def contrast(self, stage, question, candidate, reference):
        delta = self.point(candidate) - self.point(reference)
        bounds = compare_samples(self.runs[candidate].samples, self.runs[reference].samples)
        self.contrasts.append({"model": self.name, "task": self.task, "stage": stage, "contrast": question,
            "candidate": candidate, "reference": reference, "metric": self.primary, "delta_pp": delta, **bounds})
        return self.direction * delta

    def choose(self, candidates):
        """Prefer fewer fields if near the point winner AND its paired CI spans zero."""
        candidates = list(dict.fromkeys(candidates))
        winner = max(candidates, key=lambda key: self.direction * self.point(key))
        eligible = [winner]
        for key in candidates:
            if key != winner and abs(self.point(key) - self.point(winner)) <= TOLERANCE[self.task]:
                ci = compare_samples(self.runs[key].samples, self.runs[winner].samples)
                if ci["day_low"] <= 0 <= ci["day_high"]:
                    eligible.append(key)
        return min(eligible, key=lambda k: (len(self.runs[k].columns), -self.direction * self.point(k), k))


def sources(exp):
    for name, added in (("C", []), ("CW", f.WEATHER), ("CR", f.REGION), ("F", f.WEATHER + f.REGION)):
        exp.evaluate(name, "1_source", f.CORE + added)
    for candidate, reference in (("CW", "C"), ("CR", "C"), ("F", "CW"), ("F", "CR"), ("F", "C")):
        exp.contrast("1_source", f"{candidate}_minus_{reference}", candidate, reference)
    samples = {k: exp.runs[k].samples for k in ("F", "CW", "CR", "C")}
    exp.contrasts.append({"model": exp.name, "task": exp.task, "stage": "1_source",
        "contrast": "weather_region_interaction", "candidate": "F-CW", "reference": "CR-C",
        "metric": exp.primary, "delta_pp": exp.point("F")-exp.point("CW")-exp.point("CR")+exp.point("C"),
        **compare_samples(samples["F"]-samples["CW"], samples["CR"]-samples["C"])})


def dimensions(exp):
    units = [(col, [col], "CW") for col in f.WEATHER]
    # States, market variables, their intersection, and individual same-slot lags.
    for prefix in ("qld", "vic", "demand", "price", *f.NEIGHBOURS):
        cols = [c for c in f.REGION if prefix in c]
        units.append((prefix, cols, "CR"))
    units += [(c, [c], "CR") for c in f.REGION]
    retained = set(f.CORE)
    for unit, cols, source in units:
        add = exp.evaluate("add_" + unit, "2_dimension", f.CORE + cols)
        gain_alone = exp.contrast("2_dimension", "add_alone_" + unit, add, "C")
        for context in (source, "F"):
            drop = exp.evaluate(f"drop_{context}_{unit}", "2_dimension", set(exp.runs[context].columns) - set(cols))
            gain_given_others = exp.contrast("2_dimension", f"retain_in_{context}_{unit}", context, drop)
            if context == "F" and len(cols) == 1 and max(gain_alone, gain_given_others) > TOLERANCE[exp.task]:
                retained.update(cols)
    mix = exp.evaluate("dimension_mix", "2_dimension", retained)
    exp.contrast("2_dimension", "screened_vs_core", mix, "C")
    exp.contrast("2_dimension", "screened_vs_full", mix, "F")
    return exp.choose(exp.runs)


def constructions(exp, anchor):
    base = set(exp.runs[anchor].columns)
    exp.evaluate("anchor", "3_construction", base)
    changes = {}

    def option(key, add=(), remove=()):
        changes[key] = (set(add), set(remove))
        exp.evaluate(key, "3_construction", (base - set(remove)) | set(add))
        exp.contrast("3_construction", key, key, "anchor")
        return key

    options = [[option("only_lag1", remove=[c for c in base if c.endswith("_lag7")])],
               [option("window7", add=f.ROLLING[7]), option("window28", add=f.ROLLING[28])],
               [option("extra_city_weather", add=f.EXTRA_WEATHER)]]
    exp.contrast("3_construction", "28_days_vs_7_days", "window28", "window7")
    if exp.task == "price":
        options += [[option("no_explicit_gap", remove=[f.PRICE_GAP])],
                    [option("spike_max", add=f.PRICE_SPIKES[:1]), option("negative_count", add=f.PRICE_SPIKES[1:]),
                     option("both_spike_stats", add=f.PRICE_SPIKES)]]
    else:
        options.append([option("previous_day_profile", add=f.DAY_TYPE)])
    # Keep a common regional/weather context available when anchors differ across algorithms.
    exp.evaluate("F_extra_city_weather", "3_construction", set(exp.runs["F"].columns) | set(f.EXTRA_WEATHER))
    exp.contrast("3_construction", "extra_cities_given_F", "F_extra_city_weather", "F")
    chosen_changes = [exp.choose(["anchor", *variants]) for variants in options]
    combined = base.copy()
    for change in chosen_changes:
        if change != "anchor":
            added, removed = changes[change]
            combined = (combined - removed) | added
    exp.evaluate("combined", "3_construction", combined)
    exp.contrast("3_construction", "combined_vs_anchor", "combined", "anchor")
    for change in chosen_changes:
        if change != "anchor":
            added, removed = changes[change]
            undo = exp.evaluate("undo_" + change, "3_construction", (combined - added) | removed)
            exp.contrast("3_construction", "confirm_" + change, "combined", undo)
    for col in sorted(combined.intersection(f.WEATHER + f.REGION)):
        undo = exp.evaluate("combined_without_" + col, "3_construction", combined - {col})
        exp.contrast("3_construction", "confirm_dimension_" + col, "combined", undo)
    return exp.choose(exp.runs)


def temporal_folds(data, task):
    days = dayahead.market_day(data.index)
    folds = []
    for end, valid_end in (("2012-12-31", "2013-06-30"), ("2013-06-30", "2013-12-31")):
        prefix = data.loc[days <= valid_end]
        fold_days = dayahead.market_day(prefix.index)
        X, y = f.build_features(prefix, day_type_train_end=end)
        train = (fold_days >= "2010-01-29") & (fold_days <= end)
        valid = (fold_days > end) & (fold_days <= valid_end)
        target = y[TARGET[task]]
        folds.append((X.loc[train], target.loc[train], X.loc[valid], target.loc[valid]))
    return folds


def run_model(model_name, data, out_dir):
    exp = Experiment(model_name, data)
    sources(exp)
    print(f"{model_name}: source comparisons complete", flush=True)
    anchor = dimensions(exp)
    print(f"{model_name}: dimensions complete, construction anchor={anchor}", flush=True)
    fixed = constructions(exp, anchor)
    folds = temporal_folds(data, exp.task)
    tuning_rows = []
    tuned = []
    seen = set()
    for source in ("C", "CW", "CR", "F", fixed):
        cols = exp.runs[source].columns
        if cols in seen:
            continue
        seen.add(cols)
        param, threshold, rows = tune(model_name, list(cols), folds)
        for row in rows:
            tuning_rows.append({"model": model_name, "task": exp.task, "source": source, **row})
        key = exp.evaluate("tuned_" + source, "4_tuning", cols, param=param, threshold=threshold)
        tuned.append(key)
        exp.contrast("4_tuning", "tuned_vs_fixed_" + source, key, source)
    selected = exp.choose(tuned)
    ref = next(row for row in exp.score_rows if row["config"] == selected)
    selected_row = {**ref, "dimension_anchor": anchor, "fixed_winner": fixed}
    for key in tuned:
        if key != selected:
            exp.contrast("4_tuning", "selected_vs_" + key, selected, key)
    pd.DataFrame(exp.score_rows).to_csv(out_dir / "scores.csv", index=False)
    pd.DataFrame(exp.contrasts).to_csv(out_dir / "contrasts.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(out_dir / "tuning.csv", index=False)
    pd.DataFrame([selected_row]).to_csv(out_dir / "selection.csv", index=False)
    predictions = pd.DataFrame({key: record.predictions for key, record in exp.runs.items()}, index=exp.valid.index)
    predictions.to_csv(out_dir / "predictions.csv.gz", index_label="timestamp", compression={"method": "gzip", "compresslevel": 3})
    print(f"{model_name}: {len(exp.raw_cache)} distinct full-period fits; selected {selected}", flush=True)
