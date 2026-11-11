"""Load and cross-check the published experiments for a standalone matrix view."""

import csv
import json
import math
from pathlib import Path


STATES = {"nsw": "新南威尔士州", "qld": "昆州", "vic": "维州"}
MARKET = {"demand": ("需求", "MW"), "price": ("电价", "澳元/MWh")}
CITIES = {"syd": "悉尼", "mel": "墨尔本", "bne": "布里斯班"}
WEATHER = {"temp": ("气温", "°C"), "rh": ("相对湿度", "%"),
           "cloud": ("云量", "%"), "solar": ("太阳辐射", "W/m²")}
LAGS = {1: "昨日", 7: "上周"}


def _fields():
    labels = {"slot_sin": "半小时档位正弦", "slot_cos": "半小时档位余弦",
              "season_sin": "年内周期正弦", "season_cos": "年内周期余弦", "weekend": "周末指示"}
    labels.update({f"weekday_{i}": f"周{day}指示" for i, day in enumerate("一二三四五六日")})

    def market_lags(states):
        for lag, day in LAGS.items():
            for state in states:
                for kind, (measure, unit) in MARKET.items():
                    labels[f"{state}_{kind}_lag{lag}"] = f"{STATES[state]}{day}同档{measure}（{unit}）"

    def city_weather(cities):
        for city in cities:
            for kind, (measure, unit) in WEATHER.items():
                labels[f"{city}_{kind}"] = f"{CITIES[city]}{measure}（{unit}）"

    market_lags(["nsw"])
    labels["prev_price_mean"] = "NSW 昨日全天均价（澳元/MWh）"
    labels["lag1_price_gap"] = "NSW 昨日同档价减昨日全天均价（澳元/MWh）"
    city_weather(["syd"])
    market_lags(["qld", "vic"])
    city_weather(["mel", "bne"])
    for days in (7, 28):
        for kind, (measure, unit) in MARKET.items():
            for stat, title in (("mean", "均值"), ("std", "标准差")):
                labels[f"nsw_{kind}_{stat}{days}"] = f"NSW 过去 {days} 日同档{measure}{title}（{unit}）"
    labels.update({"prev_price_max": "NSW 昨日最高电价（澳元/MWh）",
                   "prev_negative_slots": "NSW 昨日负价半小时数（档）"})
    labels.update({f"prev_day_type_{i}": f"昨日日型 {i} 指示" for i in range(4)})
    labels["prev_day_type_distance"] = "昨日日型至最近聚类中心的距离（归一化 PCA 空间）"
    return labels


def _units():
    units = [{"id": f"syd_{kind}", "label": f"悉尼{measure}", "group": "悉尼天气"}
             for kind, (measure, _) in WEATHER.items()]
    for state in ("qld", "vic"):
        units.append({"id": state, "label": f"{STATES[state]}需求及电价", "group": "按州"})
    for kind, (measure, _) in MARKET.items():
        units.append({"id": kind, "label": f"两邻州{measure}", "group": "按市场变量"})
    for state in ("qld", "vic"):
        for kind, (measure, _) in MARKET.items():
            units.append({"id": f"{state}_{kind}", "label": f"{STATES[state]}{measure}", "group": "州与变量"})
    for lag, day in LAGS.items():
        for state in ("qld", "vic"):
            for kind, (measure, _) in MARKET.items():
                units.append({"id": f"{state}_{kind}_lag{lag}", "label": f"{STATES[state]}{day}同档{measure}",
                              "group": "单个滞后字段"})
    return units


CONSTRUCTION = {
    "only_lag1": "移除上周同档市场字段", "window7": "加入 7 日同档统计", "window28": "加入 28 日同档统计",
    "28_days_vs_7_days": "28 日窗口对比 7 日窗口", "extra_city_weather": "模型起点增加两城市天气",
    "extra_cities_given_F": "完整输入增加两城市天气", "previous_day_profile": "加入昨日日型和距离",
    "no_explicit_gap": "去掉昨日显式价差", "spike_max": "加入昨日最高电价",
    "negative_count": "加入昨日负价档数", "both_spike_stats": "加入昨日最高价与负价档数",
    "combined_vs_anchor": "组合方案对比模型起点",
}

INTEGERS = {"n_features", "train_rows", "validation_rows"}
NUMERIC = {"param", "parameter", "threshold", "score", "MAPE", "MAE", "Acc", "BalAcc", "Recall0", "Recall1",
           "delta_pp", "day_low", "day_high", "week_low", "week_high", "delta_vs_yesterday", "delta_vs_last_week",
           "fit_predict_seconds", "winner_score", "reference_score"}


def _value(key, text):
    if text == "":
        return None
    if key in INTEGERS:
        return int(text)
    if key in NUMERIC:
        val = float(text)
        if not math.isfinite(val):
            raise ValueError(f"Non-finite value for {key}: {text}")
        return val
    if key == "columns":
        cols = json.loads(text)
        if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
            raise ValueError("Expected JSON list of feature identifiers")
        return cols
    if key == "chosen":
        return {"true": True, "false": False}[text.lower()]
    return text


def collect_results(result_dir: Path) -> dict:
    result_dir = Path(result_dir)
    tables = {}
    for key, stem in (("scores", "scores"), ("contrasts", "contrasts"), ("tuning", "tuning"),
                      ("selection", "selection"), ("winners", "task_winners"), ("baselines", "baselines"),
                      ("final_comparisons", "final_comparisons")):
        with (result_dir / f"{stem}.csv").open(encoding="utf-8-sig", newline="") as fh:
            tables[key] = [{k: _value(k, v) for k, v in row.items()} for row in csv.DictReader(fh)]
    run = json.loads((result_dir / "run.json").read_text(encoding="utf-8"))
    labels, units = _fields(), _units()
    scores = {(r["model"], r["config"]): r for r in tables["scores"]}
    if len(scores) != len(tables["scores"]) or {c for r in scores.values() for c in r["columns"]} != set(labels):
        raise ValueError("Duplicate score keys or mismatch with the feature catalog")
    for row in scores.values():
        if (len(row["columns"]) != len(set(row["columns"])) or len(row["columns"]) != row["n_features"]
                or row["train_rows"] != run["train_rows"] or row["validation_rows"] != run["validation_rows"]):
            raise ValueError(f"Counts disagree for {row['model']}/{row['config']}")
    for row in tables["selection"] + tables["winners"]:
        base = scores[(row["model"], row["config"])]
        if row["columns"] != base["columns"] or row[row["primary"]] != base[row["primary"]]:
            raise ValueError("Selected configuration does not match its recorded score")

    contrasts = {(r["model"], r["contrast"]): r for r in tables["contrasts"]}
    if len(contrasts) != len(tables["contrasts"]):
        raise ValueError("Duplicate contrast keys")
    for model in run["candidate_grids"]:
        for unit in units:
            key = unit["id"]
            context = "CW" if key.startswith("syd_") else "CR"
            for prefix in ("add_alone_", f"retain_in_{context}_", "retain_in_F_"):
                if (model, prefix + key) not in contrasts:
                    raise ValueError(f"Missing dimension contrast: {model} {prefix}{key}")

    names = dict(CONSTRUCTION)
    for row in tables["contrasts"]:
        ident, model, metric = row["contrast"], row["model"], row["metric"]
        if ident == "weather_region_interaction":
            delta = (scores[model, "F"][metric] - scores[model, "CW"][metric]
                     - scores[model, "CR"][metric] + scores[model, "C"][metric])
        else:
            delta = scores[model, row["candidate"]][metric] - scores[model, row["reference"]][metric]
        if not math.isclose(delta, row["delta_pp"], abs_tol=1e-8):
            raise ValueError(f"Contrast differs from paired scores: {ident}")
        if row["stage"] != "3_construction":
            continue
        if ident.startswith("confirm_dimension_"):
            names[ident] = "组合中保留" + labels[ident.removeprefix("confirm_dimension_")]
        elif ident.startswith("confirm_"):
            names[ident] = "组合中复核：" + CONSTRUCTION[ident.removeprefix("confirm_")]
        elif ident not in names:
            raise ValueError(f"Unknown construction contrast: {ident}")
    for row in tables["tuning"]:
        if (row["model"], row["source"]) not in scores:
            raise ValueError(f"Unknown tuning source: {row['source']}")
    result = {**tables, "run": run, "features": list(labels), "units": units,
              "feature_labels": labels, "contrast_labels": names}
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    return result
