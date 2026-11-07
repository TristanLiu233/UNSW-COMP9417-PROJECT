"""Official point scores and paired bootstrap intervals for ablation comparisons."""

import numpy as np
import pandas as pd

from harness import dayahead


def _predictions(task, predictions, index):
    if task not in ("demand", "price"):
        raise ValueError(f"Unknown ablation task: {task}")
    if isinstance(predictions, pd.Series) and not predictions.index.equals(index):
        raise ValueError("Prediction timestamps do not match truth")
    values = np.asarray(predictions, dtype=float)
    if values.shape != (len(index),) or not np.isfinite(values).all():
        raise ValueError("Expected one finite prediction for each interval")
    if task == "price" and not np.isin(values, (0, 1)).all():
        raise ValueError("Price predictions must be class labels 0 or 1")
    return values


def metrics(task, truth, predictions):
    """Pool all half-hours in the period. Recalls of absent classes are undefined."""
    values = _predictions(task, predictions, truth.index)
    if task == "demand":
        actual = truth["nsw_demand_mw"]
        return {"MAPE": dayahead.mape(actual, values), "MAE": dayahead.mae(actual, values)}

    actual = truth["nsw_price_up"].to_numpy()
    scores = {"BalAcc": dayahead.balanced_accuracy(actual, values),
              "Acc": dayahead.accuracy(actual, values)}
    for label in (0, 1):
        present = actual == label
        scores[f"Recall{label}"] = 100 * np.mean(values[present] == label) if present.any() else np.nan
    return scores


class Bootstrap:
    """Resample the same validation days for every model and construction.

    Each draw contains exactly the original number of complete market days.
    For seven-day blocks, sample overlapping consecutive weeks with replacement
    and truncate the final block. No block wraps around the year boundary.
    """

    def __init__(self, truth, seed=17, repetitions=2000):
        self.index = truth.index
        if not isinstance(self.index, pd.DatetimeIndex) or self.index.empty:
            raise ValueError("Expected timestamp-indexed validation truth")
        first_day = dayahead.market_day(self.index[:1])[0]
        expected = pd.date_range(first_day + pd.Timedelta(minutes=30), periods=len(truth), freq="30min")
        if not self.index.equals(expected) or len(truth) % dayahead.N_PERIODS:
            raise ValueError("Bootstrap requires consecutive, complete 48-slot market days")
        if repetitions < 1:
            raise ValueError("Bootstrap repetitions must be positive")

        self.demand = truth["nsw_demand_mw"].to_numpy(dtype=float)
        if not np.isfinite(self.demand).all() or (self.demand == 0).any():
            raise ValueError("Demand must be finite and nonzero for MAPE")
        self.labels = truth["nsw_price_up"].to_numpy().reshape(-1, dayahead.N_PERIODS)
        if not np.isin(self.labels, (0, 1)).all():
            raise ValueError("Price truth must contain class labels 0 or 1")
        self.n_days = self.labels.shape[0]
        self.repetitions = repetitions
        rng = np.random.default_rng(seed)
        days = rng.multinomial(self.n_days, np.full(self.n_days, 1 / self.n_days), size=repetitions)

        block = min(7, self.n_days)
        n_blocks = (self.n_days + block - 1) // block
        starts = rng.integers(self.n_days - block + 1, size=(repetitions, n_blocks))
        selected = (starts[:, :, None] + np.arange(block)).reshape(repetitions, -1)[:, :self.n_days]
        # Encode each draw in its own bin range to count days in all draws at once.
        bins = selected + self.n_days * np.arange(repetitions)[:, None]
        weeks = np.bincount(bins.ravel(), minlength=self.n_days * repetitions).reshape(repetitions, self.n_days)
        self._weights = np.vstack([days, weeks]).astype(float)
        label_counts = np.column_stack([(self.labels == label).sum(axis=1) for label in (0, 1)])
        self._label_counts = self._weights @ label_counts

    def samples(self, task, predictions):
        """Return the primary metric for each draw, with day then week rows."""
        values = _predictions(task, predictions, self.index)
        if task == "demand":
            ape = 100 * np.abs((values - self.demand) / self.demand)
            daily = ape.reshape(self.n_days, dayahead.N_PERIODS).mean(axis=1)
            scores = self._weights @ daily / self.n_days
        else:
            correct = values.reshape(self.labels.shape) == self.labels
            daily_hits = np.column_stack([(correct & (self.labels == label)).sum(axis=1) for label in (0, 1)])
            hits = self._weights @ daily_hits
            recalls = np.divide(100 * hits, self._label_counts, out=np.full_like(hits, np.nan),
                                where=self._label_counts > 0)
            scores = np.nanmean(recalls, axis=1)
        return scores.reshape(2, self.repetitions)


def compare_samples(candidate, reference):
    """Percentile 95% intervals for candidate minus reference, in percentage points."""
    candidate, reference = np.asarray(candidate), np.asarray(reference)
    if candidate.shape != reference.shape or candidate.ndim != 2 or candidate.shape[0] != 2:
        raise ValueError("Expected paired day and week replicate arrays with identical shapes")
    low, high = np.percentile(candidate - reference, [2.5, 97.5], axis=1)
    return {"day_low": float(low[0]), "day_high": float(high[0]),
            "week_low": float(low[1]), "week_high": float(high[1])}
