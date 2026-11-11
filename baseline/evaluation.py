"""Score the four fitted algorithms and both persistence rules on identical slots."""

from ablation.models import SPECS
from ablation.scoring import metrics


def score_period(inputs, truth, predictions):
    if truth.empty or not (inputs.index.equals(truth.index) and predictions.index.equals(truth.index)):
        raise ValueError("Expected the same non-empty time interval for inputs, targets and predictions")
    rows = []
    for name, spec in SPECS.items():
        rows.append({"model": name, "task": spec.task, **metrics(spec.task, truth, predictions[name])})
    for lag in (1, 7):
        demand = inputs[f"nsw_demand_lag{lag}"]
        price = (inputs[f"nsw_price_lag{lag}"] > inputs["prev_price_mean"]).astype(int)
        for task, values in (("demand", demand), ("price", price)):
            rows.append({"model": f"lag{lag}", "task": task, **metrics(task, truth, values)})
    return rows
