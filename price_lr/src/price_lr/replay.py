"""M7 step 2: S0 (frozen), S1 (expanding, scheduled), S2 (3-year sliding, scheduled), S3 (CUSUM-triggered) for LR.

Design (frozen 2026-11-17): results/price_lr/strategy_design.md sections 2 and 4.
  * retrains: S1 / S3 = P4 `fit_final(data <= D, D)` (default last-calendar-year tau selection);
              S2 = `fit_sliding(data <= D, D, start = D - 3 years + 1 day)`.
  * CUSUM: results/price_lr/cusum/cusum_config.json (option A, full precision), cooldown 180 days, budget 6.
  * Labels sit behind price_hgbc's LabelGate (released at the end of each day).
2025-2026 are simulated causally but nothing scored is written for them (daily log loss / S are masked).
Usage: python -m src.price_lr.replay [run|s3-daily]
"""
import hashlib
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

from src.price_hgbc import cusum as cs
from src.price_hgbc.m6 import LabelGate

from . import data as D
from .env import REPLAY_CSV, RESULTS, ROOT, check_inputs, ft, lf, lt, sha_file
from .fit_sliding import fit_sliding

OUT = RESULTS
MODELS = os.path.join(OUT, "models")
REPLAY = (pd.Timestamp("2015-01-01"), pd.Timestamp("2026-10-31"))
DEV_END = pd.Timestamp("2024-12-31")
SCHEDULE = [pd.Timestamp(f"{y}-12-31") for y in (2016, 2018, 2020, 2022, 2024)]
COOLDOWN_DAYS, BUDGET = 180, 6
CUSUM_CONFIG = os.path.join(OUT, "cusum", "cusum_config.json")
D1 = pd.Timedelta("1D")
FROZEN_ID = "LR_frozen"


def coef_sha(pipeline):
    sc, lr = pipeline[0], pipeline[-1]
    b = b"".join(np.ascontiguousarray(a, dtype="float64").tobytes() for a in (sc.mean_, sc.scale_, lr.coef_, lr.intercept_))
    return hashlib.sha256(b).hexdigest()


class Context:
    def __init__(self):
        self.inputs = check_inputs()
        self.raw = D.load_raw()
        self.f = D.Frame(self.raw)
        self.df_full = pd.read_csv(D.DATA_CSV, index_col="timestamp", parse_dates=True)
        off = D.official_labels(self.df_full, D.FIRST_TRAIN_DAY, REPLAY[1])
        assert off.index.equals(self.f.y.index), "label index mismatch"
        assert np.array_equal(off.to_numpy(), self.f.y.to_numpy()), "P4 label differs from da.targets"
        self.labels = off
        self.bundle0 = D.frozen_bundle()
        assert self.bundle0["columns"] == self.f.columns and self.bundle0["threshold"] == 0.42
        self.cusum_cfg = json.load(open(CUSUM_CONFIG))
        assert self.cusum_cfg["ref_update"] == "A"

    def rows(self, first, last):
        return self.f.rows(first, last)

    def replay_days(self, end=None):
        return pd.date_range(REPLAY[0], end or REPLAY[1])


def retrain(ctx, gate, cutoff, window):
    cutoff = pd.Timestamp(cutoff)
    gate.until(cutoff)                                        # raises unless the end of `cutoff` has been reached
    pre = D.prefix(ctx.raw, cutoff)                           # no row after cutoff + 1 day 00:00
    t0 = time.perf_counter()
    if window == "expanding":
        bundle, audit = ft.fit_final(pre, cutoff)
        start = D.FIRST_TRAIN_DAY
    else:
        start = cutoff - pd.DateOffset(years=3) + D1
        bundle, audit = fit_sliding(pre, cutoff, start)
    sec = time.perf_counter() - t0
    fold = audit["folds"][0]
    best = max(audit["search"], key=lambda r: (r["BalAcc"], -round(abs(r["threshold"] - .5), 12), -r["threshold"]))
    assert best["threshold"] == audit["threshold"]
    assert bundle["columns"] == ctx.f.columns
    log = {"cutoff": str(cutoff.date()), "train_start": str(pd.Timestamp(start).date()), "train_end": str(cutoff.date()),
           "train_rows": int(audit["train_rows"]), "train_days": int(audit["train_rows"]) // 48,
           "tune_fit_start": fold["train_start"], "tune_fit_end": fold["train_end"], "tune_rows": fold["train_rows"],
           "val_start": fold["validation_start"], "val_end": fold["validation_end"], "val_rows": fold["validation_rows"],
           "tau": float(audit["threshold"]), "val_BalAcc_at_tau": best["BalAcc"], "total_seconds": round(sec, 3),
           "trainer": "fit_final" if window == "expanding" else "fit_sliding", "coef_sha256": coef_sha(bundle["pipeline"]),
           "prefix_sha256": audit["prefix_sha256"]}
    return bundle, log


def _frame(ctx, mask, p, tau, strategy, model_id):
    idx = ctx.f.X.index[mask]
    start = idx - pd.Timedelta("30min")
    return pd.DataFrame({"day": ctx.f.day[mask], "period": (start.hour * 2 + start.minute // 30).astype("int64"),
                         "p_up": p, "pred": (p >= tau).astype("int64"), "strategy": strategy, "model_id": model_id}, index=idx)


def save_model(bundle, model_id, save):
    if not save:
        return {}
    os.makedirs(MODELS, exist_ok=True)
    path = os.path.join(MODELS, f"{model_id}.joblib")
    joblib.dump(bundle, path)
    return {"model_path": os.path.relpath(path, ROOT), "model_sha256": sha_file(path)}


def initial_log(ctx, strategy):
    return {"strategy": strategy, "model_id": FROZEN_ID, "trigger": "initial (P4 frozen artifact)", "alarm_day": None,
            "cutoff": "2014-12-31", "train_start": "2010-01-29", "train_end": "2014-12-31", "train_rows": 86304,
            "train_days": 1798, "tau": 0.42, "effective_from": "2015-01-01", "trainer": "P4 study.py (frozen)",
            "coef_sha256": coef_sha(ctx.bundle0["pipeline"]),
            "model_path": "_group_main/logistic_price/results/frozen/artifacts/log_interactions31/logistic.joblib",
            "model_sha256": sha_file(os.path.join(ROOT, "_group_main/logistic_price/results/frozen/artifacts/log_interactions31/logistic.joblib"))}


def _close(logs, last_day):
    for i, l in enumerate(logs):
        nxt = logs[i + 1]["effective_from"] if i + 1 < len(logs) else None
        l["effective_to"] = str((pd.Timestamp(nxt) - D1).date()) if nxt else str(last_day.date())
    return logs


def run_scheduled(ctx, strategy, window, schedule=SCHEDULE, end=None, save=True):
    gate = LabelGate(ctx.labels)
    days = ctx.replay_days(end)
    bundle, mid = ctx.bundle0, FROZEN_ID
    logs, parts, seg_start = [initial_log(ctx, strategy)], [], days[0]
    for k, cut in enumerate(list(schedule) + [None]):
        seg_end = days[-1] if cut is None or cut >= days[-1] else cut
        m = ctx.rows(seg_start, seg_end)
        parts.append(_frame(ctx, m, bundle["pipeline"].predict_proba(ctx.f.X.loc[m])[:, 1], bundle["threshold"], strategy, mid))
        if cut is None or cut >= days[-1]:
            break
        gate.advance(cut)
        bundle, log = retrain(ctx, gate, cut, window)
        mid = f"{strategy}_r{k + 1}_{cut.date()}"
        log.update({"strategy": strategy, "model_id": mid, "trigger": "scheduled (year end)", "alarm_day": None,
                    "effective_from": str((cut + D1).date()), **save_model(bundle, mid, save)})
        logs.append(log)
        seg_start = cut + D1
    n = len(logs) - 1
    assert n <= BUDGET
    return pd.concat(parts), pd.DataFrame(_close(logs, days[-1]))


def features_for_day(ctx, d):
    """P4's public daily path: history through the end of d-1, d's 48 weather rows (as PriceClassifier.predict)."""
    d = pd.Timestamp(d)
    hist = ctx.raw.loc[:d]
    idx = pd.date_range(d + pd.Timedelta(minutes=30), periods=48, freq="30min")
    weather = ctx.raw.loc[idx, lf.WEATHER]
    return lt.transform_features(lf.build_features_for_day(hist, weather), D.RECIPE)[ctx.f.columns]


def run_s3(ctx, mode="batch", end=None, save=True):
    gate = LabelGate(ctx.labels)
    det = cs.Cusum.from_config(ctx.cusum_cfg, "main")
    det.reset()
    bundle, mid = ctx.bundle0, FROZEN_ID
    logs, parts, events, daily = [initial_log(ctx, "S3")], [], [], []
    last_retrain_alarm, n_retrain = None, 0
    for d in ctx.replay_days(end):
        m = ctx.f.day == d
        Xd = ctx.f.X.loc[m] if mode == "batch" else features_for_day(ctx, d)
        assert Xd.index.equals(ctx.f.X.index[m])
        p = bundle["pipeline"].predict_proba(Xd)[:, 1]
        parts.append(_frame(ctx, m, p, bundle["threshold"], "S3", mid))   # stored before d's labels are seen
        gate.advance(d)
        y = gate.day(d)
        assert y.index.equals(Xd.index)
        e = float(cs.daily_logloss(y.to_numpy(), p, Xd.index).iloc[0])
        S, alarm = det.update(e)
        row = {"day": d.date(), "model_id": mid, "logloss": e, "S": S, "alarm": alarm, "retrained": False}
        if alarm:
            cool_ok = last_retrain_alarm is None or (d - last_retrain_alarm).days >= COOLDOWN_DAYS
            ev = {"alarm_day": d.date(), "S_at_alarm": S, "retrained": False, "suppressed_reason": "",
                  "new_model_id": "", "effective_from": "", "retrain_index": None}
            if cool_ok and n_retrain < BUDGET:
                n_retrain += 1
                bundle, log = retrain(ctx, gate, d, "expanding")
                mid = f"S3_r{n_retrain}_{d.date()}"
                log.update({"strategy": "S3", "model_id": mid, "trigger": "cusum alarm", "alarm_day": str(d.date()),
                            "effective_from": str((d + D1).date()), **save_model(bundle, mid, save)})
                logs.append(log)
                last_retrain_alarm = d
                det.reset()
                ev.update({"retrained": True, "new_model_id": mid, "effective_from": str((d + D1).date()),
                           "retrain_index": n_retrain})
                row["retrained"] = True
            else:
                ev["suppressed_reason"] = "cooldown" if not cool_ok else "budget"
            events.append(ev)
        daily.append(row)
    assert n_retrain <= BUDGET
    days = ctx.replay_days(end)
    return (pd.concat(parts), pd.DataFrame(_close(logs, days[-1])),
            pd.DataFrame(events, columns=["alarm_day", "S_at_alarm", "retrained", "suppressed_reason", "new_model_id",
                                          "effective_from", "retrain_index"]), pd.DataFrame(daily))


def monitor_s0(ctx, preds0):
    """CUSUM on the frozen model's path without retraining (for report section 4.3); 2015-2024 only."""
    det = cs.Cusum.from_config(ctx.cusum_cfg, "main")
    p = preds0[preds0["day"] <= DEV_END]
    y = ctx.labels.loc[p.index]
    e = cs.daily_logloss(y.to_numpy(), p["p_up"].to_numpy(), p.index)
    out = cs.run_batch(e, det)
    out["day"] = pd.DatetimeIndex(out["day"]).date
    return out


def mask_holdout(daily):
    out = daily.copy()
    late = pd.to_datetime(out["day"]) > DEV_END
    out.loc[late, ["logloss", "S"]] = np.nan
    return out


def check_s0(preds0):
    r = pd.read_csv(REPLAY_CSV, index_col="timestamp", parse_dates=True, float_precision="round_trip")
    assert r.index.equals(preds0.index) and len(r) == 207456
    assert np.array_equal(preds0["p_up"].to_numpy(), r["log_interactions31_probability"].to_numpy()), "S0 proba != P4 replay"
    assert np.array_equal(preds0["pred"].to_numpy(), r["log_interactions31"].to_numpy()), "S0 class != P4 replay"
    return {"rows": len(r), "proba_bitwise_equal": True, "class_equal": True}


def main(argv=None):
    mode = (argv or sys.argv[1:] or ["run"])[0]
    os.makedirs(OUT, exist_ok=True)
    ctx = Context()
    if mode == "s3-daily":
        t0 = time.perf_counter()
        p, l, e, dly = run_s3(ctx, mode="daily", save=False)
        p.to_parquet(os.path.join(OUT, "s3_daily_mode_predictions.parquet"))
        mask_holdout(dly).to_csv(os.path.join(OUT, "s3_daily_mode_cusum.csv"), index=False, float_format="%.17g")
        l.to_csv(os.path.join(OUT, "s3_daily_mode_models_log.csv"), index=False)
        print(f"daily-mode S3 done in {time.perf_counter() - t0:.0f}s; retrains {int(dly['retrained'].sum())}")
        return
    alll = []
    for name, window, sched in (("S0", "expanding", []), ("S1", "expanding", SCHEDULE), ("S2", "sliding3y", SCHEDULE)):
        t0 = time.perf_counter()
        p, l = run_scheduled(ctx, name, window, schedule=sched)
        if name == "S0":
            s0chk = check_s0(p)
            mon = monitor_s0(ctx, p)
            mon.to_csv(os.path.join(OUT, "s0_monitor_cusum.csv"), index=False, float_format="%.17g")
        p.to_parquet(os.path.join(OUT, f"{name.lower()}_predictions.parquet"))
        alll.append(l)
        print(f"{name}: {len(p)} rows, {len(l) - 1} retrains, {time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    p, l, ev, dly = run_s3(ctx)
    p.to_parquet(os.path.join(OUT, "s3_predictions.parquet"))
    ev.to_csv(os.path.join(OUT, "s3_events.csv"), index=False, float_format="%.17g")
    mask_holdout(dly).to_csv(os.path.join(OUT, "s3_daily_cusum.csv"), index=False, float_format="%.17g")
    alll.append(l)
    print(f"S3: {len(p)} rows, {len(l) - 1} retrains, {len(ev)} alarms, {time.perf_counter() - t0:.1f}s")
    log = pd.concat(alll, ignore_index=True)
    log.to_csv(os.path.join(OUT, "models_log.csv"), index=False)
    json.dump({"inputs": ctx.inputs, "s0_check": s0chk, "labels_checked_against_da": [str(D.FIRST_TRAIN_DAY.date()), str(REPLAY[1].date())],
               "s0_monitor_alarms_2015_2024": [str(x) for x in mon.loc[mon["alarm"], "day"]]},
              open(os.path.join(OUT, "replay_meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
