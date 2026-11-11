"""Error-based drift detector for the price-direction model: upper CUSUM on daily errors (M4).

PRE-DECLARED DESIGN (see results/price_hgbc/m4/cusum_design.md; not changed during execution)
  e_D   main: mean log loss of day D's 48 half-hours, p clipped to [1e-6, 1 - 1e-6];
        comparison: misclassification rate of day D at tau = 0.45.
  Truncation: e_D <- min(e_D, q99), q99 = 99 % quantile of 2014 e_D (numpy default linear), fixed afterwards.
  Reference: mu0 = mean, sigma0 = sample sd (ddof 1) of truncated 2014 e_D.
  S_D = max(0, S_{D-1} + (min(e_D, q99) - mu0) / sigma0 - k), k = 0.5; alarm iff S_D > h; S resets to 0 on the
  alarm day. Timing: S_D is updated at the END of day D (truth of D known then); after an alarm the retrained
  model is used from the forecast of D+1.
  h: smallest of {3, 4, 5, 6, 7, 8, 10} with bootstrap ARL0 (median) >= 730 days.
  ref_update: "A" keep 2014 mu0 / sigma0 / q99; "B" re-estimate them from the new model's out-of-sample
  errors on the last 12 months of its training window (same retraining).

One function, `step`, implements the recursion. The online class, the batch runner and the vectorised
simulation all call it, so they cannot drift apart.
"""
import json
import math

import numpy as np
import pandas as pd

from .kit import da

LOGLOSS_CLIP = (1e-6, 1 - 1e-6)
K_DEFAULT = 0.5
H_CANDIDATES = (3, 4, 5, 6, 7, 8, 10)
ARL0_TARGET = 730
REF_UPDATE_OPTIONS = ("A", "B")


# ---------------- daily errors ----------------
def _by_day(values, index):
    days = np.asarray(da.market_day(pd.DatetimeIndex(index)))
    g = pd.Series(values, index=days).groupby(level=0)
    if not (g.size() == da.N_PERIODS).all():
        raise ValueError("every day must have 48 half-hours")
    return g.mean()


def daily_logloss(y, proba, index, clip=LOGLOSS_CLIP):
    p = np.clip(np.asarray(proba, dtype=float), *clip)
    y = np.asarray(y, dtype=float)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return _by_day(ll, index)


def daily_misclass(y, pred, index):
    return _by_day((np.asarray(y, int) != np.asarray(pred, int)).astype(float), index)


def reference_stats(e, truncate=True):
    """q99, mu0, sigma0 (ddof 1) of the (truncated) daily errors; q99 = +inf when truncate is False."""
    e = np.asarray(e, dtype=float)
    q99 = float(np.quantile(e, 0.99)) if truncate else math.inf
    et = np.minimum(e, q99)
    return {"q99": q99, "mu0": float(et.mean()), "sigma0": float(et.std(ddof=1)),
            "n_days": int(len(e)), "n_truncated": int((e > q99).sum())}


# ---------------- the recursion ----------------
def step(S_prev, e, mu0, sigma0, q99, k, h):
    """One day (scalars or numpy arrays). Returns (S_D before reset, alarm, state carried to D+1)."""
    z = (np.minimum(e, q99) - mu0) / sigma0
    S = np.maximum(0.0, S_prev + z - k)
    alarm = S > h
    return S, alarm, np.where(alarm, 0.0, S)


class Cusum:
    """Online detector: call update(e_D) once at the end of each day D."""

    def __init__(self, mu0, sigma0, q99, k=K_DEFAULT, h=5.0):
        if not (sigma0 > 0 and k >= 0 and h > 0):
            raise ValueError("need sigma0 > 0, k >= 0, h > 0")
        self.mu0, self.sigma0, self.q99, self.k, self.h = float(mu0), float(sigma0), float(q99), float(k), float(h)
        self.state = 0.0

    @classmethod
    def from_config(cls, cfg, scheme="main"):
        cfg = json.load(open(cfg)) if isinstance(cfg, str) else cfg
        validate_config(cfg)
        s = cfg["schemes"][scheme]
        return cls(s["mu0"], s["sigma0"], s["q99"], s["k"], s["h"])

    def reset(self):
        self.state = 0.0

    def update(self, e):
        S, alarm, nxt = step(self.state, float(e), self.mu0, self.sigma0, self.q99, self.k, self.h)
        self.state = float(nxt)
        return float(S), bool(alarm)

    def set_reference(self, mu0, sigma0, q99):
        """Used after a retrain (ref_update); also resets S."""
        self.mu0, self.sigma0, self.q99 = float(mu0), float(sigma0), float(q99)
        self.reset()


def run_batch(errors, detector):
    """Batch interface = the online update applied day by day (same code path)."""
    detector.reset()
    rows = []
    for day, e in pd.Series(errors).items():
        S, alarm = detector.update(e)
        rows.append({"day": day, "e": float(e), "e_trunc": min(float(e), detector.q99),
                     "z": (min(float(e), detector.q99) - detector.mu0) / detector.sigma0, "S": S, "alarm": alarm})
    return pd.DataFrame(rows)


def simulate_alarm_counts(E, mu0, sigma0, q99, k, h):
    """Vectorised over replicates: E is (n_rep, n_days) raw daily errors. Returns alarms per replicate."""
    S = np.zeros(E.shape[0])
    count = np.zeros(E.shape[0], dtype=int)
    for t in range(E.shape[1]):
        _, alarm, S = step(S, E[:, t], mu0, sigma0, q99, k, h)
        count += alarm
    return count


def simulate_first_alarm(E, mu0, sigma0, q99, k, h):
    """Zero-state run length: 1-based day of the first alarm per replicate (np.inf if none)."""
    S = np.zeros(E.shape[0])
    first = np.full(E.shape[0], np.inf)
    for t in range(E.shape[1]):
        _, alarm, S = step(S, E[:, t], mu0, sigma0, q99, k, h)
        newly = alarm & np.isinf(first)
        first[newly] = t + 1
        if np.isfinite(first).all():
            break
    return first


def quantile_with_inf(a, q):
    """numpy's default (linear) quantile, but a value bracketed by +inf is +inf instead of NaN (inf - inf)."""
    a = np.asarray(a, dtype=float)
    big = 1e300
    v = float(np.quantile(np.where(np.isinf(a), big, a), q))
    return math.inf if v >= 1e299 else v


def arl_from_counts(count, n_days):
    """Average spacing between alarms in an n_days run: n_days / count (+inf without alarms)."""
    count = np.asarray(count, dtype=float)
    return np.where(count > 0, n_days / np.where(count > 0, count, 1), np.inf)


# ---------------- reference after a retrain ----------------
def reference_for_retrain(ref_update, current, oos_daily_errors=None):
    """New (mu0, sigma0, q99) after a retrain.
    A: keep `current`; B: re-estimate from the new model's out-of-sample daily errors on the last 12 months
    of its training window (the caller computes them inside the same retraining)."""
    if ref_update == "A":
        return {k: current[k] for k in ("mu0", "sigma0", "q99")}
    if ref_update == "B":
        if oos_daily_errors is None or len(oos_daily_errors) < 300:
            raise ValueError("option B needs about 12 months of out-of-sample daily errors")
        s = reference_stats(oos_daily_errors, truncate=math.isfinite(current["q99"]))
        return {k: s[k] for k in ("mu0", "sigma0", "q99")}
    raise ValueError(f"ref_update must be one of {REF_UPDATE_OPTIONS}")


# ---------------- config ----------------
REQUIRED = {
    "error_definitions": dict, "logloss_clip": list, "k": (int, float), "h_candidates": list, "arl0_target_days": (int, float),
    "schemes": dict, "ref_update_options": dict, "ref": dict, "code_sha256": dict, "code_commit": (str, type(None)),
    "bootstrap": dict, "data_period": dict,
}
SCHEME_KEYS = {"error": str, "truncate": bool, "q99": (int, float), "mu0": (int, float), "sigma0": (int, float),
               "k": (int, float), "h": (int, float), "arl0_median": (int, float, str), "arl0_q05": (int, float, str),
               "arl0_q95": (int, float, str), "meets_target": bool, "role": str}
NO_ALARM = ">3650"     # JSON stand-in for an infinite ARL (no alarm in the 3650-day replicate); strict JSON has no inf


def validate_config(cfg):
    """Structural check of cusum_config.json (raises ValueError)."""
    for key, typ in REQUIRED.items():
        if key not in cfg or not isinstance(cfg[key], typ):
            raise ValueError(f"config: missing or wrong type for {key!r}")
    if "main" not in cfg["schemes"]:
        raise ValueError("config: no 'main' scheme")
    for name, s in cfg["schemes"].items():
        for key, typ in SCHEME_KEYS.items():
            if key not in s or not isinstance(s[key], typ) or isinstance(s[key], bool) != (typ is bool):
                raise ValueError(f"config: scheme {name!r} key {key!r}")
            if isinstance(s[key], str) and key.startswith("arl0") and s[key] != NO_ALARM:
                raise ValueError(f"config: scheme {name!r} key {key!r} must be a number or {NO_ALARM!r}")
        if not (s["sigma0"] > 0 and s["h"] in cfg["h_candidates"] and s["k"] == cfg["k"]):
            raise ValueError(f"config: scheme {name!r} values out of range")
    if cfg.get("ref_update") not in (None, *REF_UPDATE_OPTIONS):
        raise ValueError("config: ref_update must be null, 'A' or 'B'")
    if set(cfg["ref_update_options"]) != set(REF_UPDATE_OPTIONS):
        raise ValueError("config: ref_update_options must be exactly A and B")
    if cfg["logloss_clip"] != list(LOGLOSS_CLIP):
        raise ValueError("config: logloss clip differs from the design")
    return True
