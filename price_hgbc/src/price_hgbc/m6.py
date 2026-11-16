"""M6 strategy runner: S0 (frozen), S1 (expanding, scheduled), S2 (3-year sliding, scheduled), S3 (CUSUM-triggered).

Design: results/price_hgbc/m6/strategy_design.md (pre-declared). No P3 `run_strategy` exists in the project, so this
is a standalone runner; the output fields to confirm with P3 are listed in M6.md.
All HGBC parameters come from REF; CUSUM parameters come from m4/cusum_config.json (option A, full precision).
Labels sit behind `LabelGate`, which releases a day's labels only at the end of that day.
Usage: python -m src.price_hgbc.m6 [run|s3-daily]
"""
import hashlib
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import cusum as cs
from . import m5
from . import official_ref as ofr
from .ablate import p1_retune
from .kit import DATA_CSV, RESULTS, ROOT, da

OUT = os.path.join(RESULTS, "m6")
MODELS = os.path.join(OUT, "models")
REPLAY = (pd.Timestamp("2015-01-01"), pd.Timestamp("2026-10-31"))
DEV_END = pd.Timestamp("2024-12-31")
EXPANDING_START = pd.Timestamp("2010-01-29")
SCHEDULE = [pd.Timestamp(f"{y}-12-31") for y in (2016, 2018, 2020, 2022, 2024)]
COOLDOWN_DAYS, BUDGET = 180, 6
CUSUM_CONFIG = os.path.join(RESULTS, "m4", "cusum_config.json")
BIN_SUBSAMPLE_ROWS = 200_000
D1 = pd.Timedelta("1D")


class LabelGate:
    """Official labels (da.targets), released day by day. `now` = the last day whose end has been reached."""

    def __init__(self, labels):
        self._y = labels
        self._day = da.market_day(labels.index)
        self.now = None

    def advance(self, day):
        self.now = pd.Timestamp(day)

    def until(self, last_day):
        if self.now is None or pd.Timestamp(last_day) > self.now:
            raise RuntimeError(f"labels of {last_day} requested at the end of {self.now}")
        return self._y[self._day <= pd.Timestamp(last_day)]

    def day(self, d):
        if self.now is None or pd.Timestamp(d) > self.now:
            raise RuntimeError(f"labels of {d} requested at the end of {self.now}")
        return self._y[self._day == pd.Timestamp(d)]


class Context:
    def __init__(self, df=None):
        self.ref = json.load(open(ofr.REF_JSON))
        self.df = da.load_dataset(DATA_CSV) if df is None else df
        self.cols = self.ref["features"]
        self.X = m5.batch_features(self.df)[self.cols]
        self.xday = da.market_day(self.X.index)
        days = [d for d in da.complete_days(self.df) if d >= pd.Timestamp("2010-01-02")]
        self.labels = ofr.official_labels(self.df, days)
        self.model0 = joblib.load(m5.P["model"])
        self.tau0 = json.load(open(m5.P["config"]))["tau"]
        self.cusum_cfg = json.load(open(CUSUM_CONFIG))
        assert self.cusum_cfg["ref_update"] == "A"

    def rows(self, first, last):
        return (self.xday >= pd.Timestamp(first)) & (self.xday <= pd.Timestamp(last))

    def replay_days(self, end=None):
        return pd.date_range(REPLAY[0], end or REPLAY[1])


def tree_sha(model):
    t = [hashlib.sha256(p.nodes.tobytes()).hexdigest() for it in model._predictors for p in it]
    return hashlib.sha256("".join(t).encode()).hexdigest()


def fit(ctx, gate, first, last):
    m = ctx.rows(first, last)
    X = ctx.X.loc[m]
    y = gate.until(last).reindex(X.index)
    assert y.notna().all() and len(X) == ctx.rows(first, last).sum()
    t0 = time.perf_counter()
    model = HistGradientBoostingClassifier(**ctx.ref["params"]).fit(X, y.astype(int))
    return model, {"rows": int(len(X)), "days": int(da.market_day(X.index).nunique()), "seconds": time.perf_counter() - t0}


def retrain(ctx, gate, cutoff, window):
    """Nested tau selection on the last 12 months, then the final fit on the whole window (design §4-5)."""
    cutoff = pd.Timestamp(cutoff)
    start = EXPANDING_START if window == "expanding" else cutoff - pd.DateOffset(years=3) + D1
    tune_end = cutoff - pd.DateOffset(years=1)
    val_start = tune_end + D1
    assert start <= tune_end < val_start <= cutoff
    tm, tinfo = fit(ctx, gate, start, tune_end)
    vm = ctx.rows(val_start, cutoff)
    pv = tm.predict_proba(ctx.X.loc[vm])[:, 1]
    yv = gate.until(cutoff).reindex(ctx.X.index[vm]).astype(int).to_numpy()
    tau, bal, _ = p1_retune(yv, pv)
    model, finfo = fit(ctx, gate, start, cutoff)
    log = {"cutoff": cutoff.date(), "train_start": start.date(), "train_end": cutoff.date(), "train_days": finfo["days"],
           "train_rows": finfo["rows"], "rows_gt_200k": finfo["rows"] > BIN_SUBSAMPLE_ROWS,
           "tune_fit_start": start.date(), "tune_fit_end": tune_end.date(), "tune_rows": tinfo["rows"],
           "tune_rows_gt_200k": tinfo["rows"] > BIN_SUBSAMPLE_ROWS, "val_start": val_start.date(), "val_end": cutoff.date(),
           "val_days": int(da.market_day(ctx.X.index[vm]).nunique()), "tau": tau, "val_BalAcc_at_tau": bal,
           "fit_seconds": round(finfo["seconds"], 3), "tune_seconds": round(tinfo["seconds"], 3),
           "total_seconds": round(finfo["seconds"] + tinfo["seconds"], 3), "trees_sha256": tree_sha(model),
           # out-of-sample daily log loss of the tuning model on the validation days (option B input; popped by callers)
           "_val_daily_logloss": cs.daily_logloss(yv, pv, ctx.X.index[vm])}
    return model, tau, log


def _frame(ctx, idx, p, tau, strategy, model_id):
    start = idx - pd.Timedelta("30min")
    return pd.DataFrame({"day": da.market_day(idx), "period": (start.hour * 2 + start.minute // 30).astype("int64"),
                         "p_up": p, "pred": (p >= tau).astype("int64"), "strategy": strategy, "model_id": model_id}, index=idx)


def save_model(model, model_id, models_dir=MODELS):
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"{model_id}.joblib")
    joblib.dump(model, path)
    return os.path.relpath(path, ROOT), m5.sha_file(path)


def initial_log(ctx, strategy):
    return {"strategy": strategy, "model_id": "M5_frozen", "trigger": "initial (M5 frozen)", "alarm_day": None,
            "cutoff": "2014-12-31", "train_start": "2010-01-29", "train_end": "2014-12-31", "train_days": 1798,
            "train_rows": 86304, "rows_gt_200k": False, "tau": ctx.tau0, "effective_from": "2015-01-01",
            "trees_sha256": tree_sha(ctx.model0), "model_path": os.path.relpath(m5.P["model"], ROOT),
            "model_sha256": m5.sha_file(m5.P["model"])}


def run_scheduled(ctx, strategy, window, schedule=SCHEDULE, end=None, save=True):
    """S0 (empty schedule), S1 (expanding) or S2 (sliding): segment-wise prediction with causal features."""
    gate = LabelGate(ctx.labels)
    days = ctx.replay_days(end)
    model, tau, mid = ctx.model0, ctx.tau0, "M5_frozen"
    logs, parts = [initial_log(ctx, strategy)], []
    seg_start = days[0]
    for k, cut in enumerate(list(schedule) + [None]):
        seg_end = days[-1] if cut is None or cut >= days[-1] else cut
        m = ctx.rows(seg_start, seg_end)
        parts.append(_frame(ctx, ctx.X.index[m], model.predict_proba(ctx.X.loc[m])[:, 1], tau, strategy, mid))
        if cut is None or cut >= days[-1]:
            break
        gate.advance(cut)                                          # end of the retrain day: its labels are known
        model, tau, log = retrain(ctx, gate, cut, window)
        log.pop("_val_daily_logloss")
        mid = f"{strategy}_r{k + 1}_{cut.date()}"
        log.update({"strategy": strategy, "model_id": mid, "trigger": "scheduled (year end)", "alarm_day": None,
                    "effective_from": (cut + D1).date()})
        if save:
            log["model_path"], log["model_sha256"] = save_model(model, mid)
        logs.append(log)
        seg_start = cut + D1
    preds = pd.concat(parts)
    logs = _close_segments(logs, days[-1])
    return preds, pd.DataFrame(logs)


def _close_segments(logs, last_day):
    for i, l in enumerate(logs):
        nxt = logs[i + 1]["effective_from"] if i + 1 < len(logs) else None
        l["effective_to"] = str((pd.Timestamp(nxt) - D1).date()) if nxt else str(last_day.date())
    return logs


def _daily_inputs(ctx, d):
    pos = int((pd.Timestamp(d) - pd.Timestamp("2010-01-01")).days) * da.N_PERIODS
    return ctx.df.iloc[:pos], ctx.df.iloc[pos:pos + da.N_PERIODS][da.WEATHER_COLS]


def run_s3(ctx, mode="batch", end=None, save=True, variant=None, models_dir=MODELS):
    """CUSUM-triggered strategy, simulated day by day (design: S3).
    variant=None is the official S3. A variant dict (pre-registered sensitivity run, m6/s3_sensitivity_preregistration.md)
    may set: name, ref_update ("A"/"B"), k, h, cooldown. Variants also log the reference in force each day."""
    v = variant
    name = v["name"] if v else "S3"
    cooldown = v["cooldown"] if v else COOLDOWN_DAYS
    gate = LabelGate(ctx.labels)
    det = cs.Cusum.from_config(ctx.cusum_cfg, "main")
    if v:
        det.k, det.h = float(v["k"]), float(v["h"])
        assert v["ref_update"] in cs.REF_UPDATE_OPTIONS
    det.reset()                                                     # S = 0 on 2015-01-01
    model, tau, mid = ctx.model0, ctx.tau0, "M5_frozen"
    logs, parts, events, daily = [initial_log(ctx, name)], [], [], []
    last_retrain_alarm, n_retrain = None, 0
    for d in ctx.replay_days(end):
        if mode == "batch":
            m = ctx.xday == d
            idx, Xd = ctx.X.index[m], ctx.X.loc[m]
        else:
            h, w = _daily_inputs(ctx, d)
            Xd = m5.features_for_day(h, w, ctx.cols)
            idx = Xd.index
        p = model.predict_proba(Xd)[:, 1]
        parts.append(_frame(ctx, idx, p, tau, name, mid))            # predictions stored before any label is seen
        gate.advance(d)                                              # end of day d: its labels are revealed
        y = gate.day(d)
        assert y.index.equals(idx)
        e = float(cs.daily_logloss(y.to_numpy(), p, idx).iloc[0])
        S, alarm = det.update(e)
        row = {"day": d.date(), "model_id": mid, "logloss": e, "S": S, "alarm": alarm, "retrained": False}
        if v:
            row.update({"mu0": det.mu0, "sigma0": det.sigma0, "q99": det.q99, "k": det.k, "h": det.h})
        if alarm:
            cool_ok = last_retrain_alarm is None or (d - last_retrain_alarm).days >= cooldown
            ev = {"alarm_day": d.date(), "S_at_alarm": S, "retrained": False, "suppressed_reason": "",
                  "new_model_id": "", "effective_from": "", "retrain_index": None}
            if cool_ok and n_retrain < BUDGET:
                n_retrain += 1
                model, tau, log = retrain(ctx, gate, d, "expanding")
                val_errors = log.pop("_val_daily_logloss")
                mid = f"{name}_r{n_retrain}_{d.date()}"
                log.update({"strategy": name, "model_id": mid, "trigger": "cusum alarm", "alarm_day": d.date(),
                            "effective_from": (d + D1).date()})
                if v and v["ref_update"] == "B":                     # option B: reference from this retrain's OOS errors
                    assert pd.DatetimeIndex(val_errors.index).max() <= d
                    new = cs.reference_for_retrain("B", {"mu0": det.mu0, "sigma0": det.sigma0, "q99": det.q99},
                                                   val_errors.to_numpy())
                    det.set_reference(new["mu0"], new["sigma0"], new["q99"])     # also resets S
                    log.update({"ref_q99_after": new["q99"], "ref_mu0_after": new["mu0"], "ref_sigma0_after": new["sigma0"],
                                "ref_from_days": int(len(val_errors))})
                if save:
                    log["model_path"], log["model_sha256"] = save_model(model, mid, models_dir)
                logs.append(log)
                last_retrain_alarm = d
                det.reset()                                          # S reset after every retrain
                ev.update({"retrained": True, "new_model_id": mid, "effective_from": (d + D1).date(), "retrain_index": n_retrain})
                row["retrained"] = True
            else:
                ev["suppressed_reason"] = "cooldown" if not cool_ok else "budget"
            events.append(ev)
        daily.append(row)
    assert n_retrain <= BUDGET
    days = ctx.replay_days(end)
    return (pd.concat(parts), pd.DataFrame(_close_segments(logs, days[-1])),
            pd.DataFrame(events, columns=["alarm_day", "S_at_alarm", "retrained", "suppressed_reason", "new_model_id",
                                          "effective_from", "retrain_index"]), pd.DataFrame(daily))


def mask_holdout(daily):
    """The CUSUM may read 2025-2026 truth inside the simulation, but daily scores of the hold-out are not published."""
    out = daily.copy()
    out.loc[pd.to_datetime(out["day"]) > DEV_END, "logloss"] = np.nan
    return out


def main(argv=None):
    mode = (argv or sys.argv[1:] or ["run"])[0]
    os.makedirs(OUT, exist_ok=True)
    ctx = Context()
    if mode == "s3-daily":                                          # independent day-by-day replay of S3 (evidence)
        t0 = time.perf_counter()
        p, l, e, dly = run_s3(ctx, mode="daily", save=False)
        p.to_parquet(os.path.join(OUT, "s3_daily_mode_predictions.parquet"))
        mask_holdout(dly).to_csv(os.path.join(OUT, "s3_daily_mode_cusum.csv"), index=False, float_format="%.17g")
        print(f"daily-mode S3 done in {time.perf_counter() - t0:.0f}s; retrains {int(dly['retrained'].sum())}")
        return
    allp, alll = [], []
    for name, fn in (("S0", lambda: run_scheduled(ctx, "S0", "expanding", schedule=[])),
                     ("S1", lambda: run_scheduled(ctx, "S1", "expanding")),
                     ("S2", lambda: run_scheduled(ctx, "S2", "sliding3y"))):
        t0 = time.perf_counter()
        p, l = fn()
        p.to_parquet(os.path.join(OUT, f"{name.lower()}_predictions.parquet"))
        allp.append(p), alll.append(l)
        print(f"{name}: {len(p)} rows, {len(l) - 1} retrains, {time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    p, l, ev, dly = run_s3(ctx)
    p.to_parquet(os.path.join(OUT, "s3_predictions.parquet"))
    ev.to_csv(os.path.join(OUT, "s3_events.csv"), index=False, float_format="%.17g")
    mask_holdout(dly).to_csv(os.path.join(OUT, "s3_daily_cusum.csv"), index=False, float_format="%.17g")
    alll.append(l)
    print(f"S3: {len(p)} rows, {len(l) - 1} retrains, {len(ev)} alarms, {time.perf_counter() - t0:.1f}s")
    pd.concat(alll, ignore_index=True).to_csv(os.path.join(OUT, "models_log.csv"), index=False)


if __name__ == "__main__":
    main()
