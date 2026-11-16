"""M5: refit + freeze the Stage 1 price model (P1 boost_price), causal day-by-day prediction, 2015-2026 batch.

Rules (user, M5):
  * every HGBC parameter (incl. random_state) is read from REF; features = REF's 18 columns in REF order; tau = REF tau;
  * training window 2010-01-29 .. 2014-12-31 (1,798 days, 86,304 rows), read through P1's `load_study`, which stops at
    2015-01-01 00:00; nothing after 2014 enters training or any choice;
  * features use P1's own code: `ablation.features._construct` (the as-of builder P1's `validate_study` uses) with P1's
    day-type model fitted on <= 2013 profiles (`_fit_day_types(..., "2013-12-31")`). Only CORE (= BOOST_PRICE_FEATURES)
    is used, so the day-type columns never reach the model. Kit code is imported, never copied or edited;
  * the batch may build features on the whole file because every CORE column uses shifted past data only. This is
    proved by tests: (1) shuffling the market data of D+1 onwards leaves D+1's features and predictions unchanged;
    (2) batch = day-by-day, bitwise;
  * predictions.parquet stores p_up and pred only (no labels); scores are recomputed with da.targets;
  * development scores cover 2015-2024 only; 2025-2026 scores are not computed in M5.
Usage: python -m src.price_hgbc.m5 [--fingerprint f.json]
"""
import argparse
import hashlib
import json
import os
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from . import official_ref as ofr
from .kit import DATA_CSV, RESULTS, ROOT, da

OUT = os.path.join(RESULTS, "m5")
P = {k: os.path.join(OUT, v) for k, v in {
    "model": "stage1_model.joblib", "config": "stage1_config.json", "manifest": "model_manifest.json",
    "preds": "predictions.parquet", "daily": "daily_vs_batch.csv", "yearly": "dev_yearly_2015_2024.csv",
    "monthly": "dev_monthly_2015_2024.csv", "readme": "README_frozen.md", "dtm": "p1_day_type_model.joblib"}.items()}
TRAIN = ("2010-01-29", "2014-12-31")
TRAIN_DAYS, TRAIN_ROWS = 1798, 86304
PRED = ("2015-01-01", "2026-10-31")
DEV_END = pd.Timestamp("2024-12-31")
WINDOW_DAYS = 35                     # history kept per call: >= 8 for lag-7, >= 29 for P1's rolling-28 (non-CORE) columns
CUSUM_CONFIG = os.path.join(RESULTS, "m4", "cusum_config.json")
CHECK_DAYS = ["2015-01-01", "2015-07-15", "2016-01-01", "2016-02-29", "2017-12-31", "2018-01-01", "2019-06-30",
              "2020-02-29", "2021-01-01", "2022-05-08", "2023-08-01", "2024-02-29", "2024-12-31", "2025-01-01",
              "2026-03-15", "2026-10-31"]


def ref():
    return json.load(open(ofr.REF_JSON))


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------- P1 feature adapter ----------------
_DTM = {}


def day_type_model(history):
    """P1's day-type model (needed only by the signature of P1's builder): fitted on <= 2013 demand profiles."""
    if "m" not in _DTM:
        gf, _ = ofr.p1()
        if history.index[0] != pd.Timestamp("2010-01-01 00:30"):
            raise ValueError("history must start at 2010-01-01 00:30 to rebuild P1's day-type model")
        pre = history[da.market_day(history.index) <= pd.Timestamp("2013-12-31")]
        _DTM["m"] = gf._fit_day_types(gf._profiles(pre, da.market_day(pre.index), "nsw_demand"), "2013-12-31")
    return _DTM["m"]


def features_for_day(history, weather_next, columns):
    """P1 CORE features of day D+1 from (history up to the end of D, D+1 weather) only."""
    gf, _ = ofr.p1()
    hd = da.market_day(history.index)
    if len(weather_next) != da.N_PERIODS or weather_next.index[0] != history.index[-1] + pd.Timedelta("30min"):
        raise ValueError("weather_next must be the 48 half-hours right after history")
    win = history[hd >= hd.max() - pd.Timedelta(f"{WINDOW_DAYS - 1}D")]
    nxt = pd.DataFrame(np.nan, index=weather_next.index, columns=win.columns)
    nxt[da.WEATHER_COLS] = weather_next[da.WEATHER_COLS]          # D+1 market columns stay NaN: never read
    frame = pd.concat([win, nxt])
    X, _ = gf._construct(frame, da.market_day(frame.index), day_type_model(history))
    out = X.loc[weather_next.index, columns]
    if not np.isfinite(out.to_numpy(dtype=float)).all():
        raise ValueError("NaN/inf in prediction-time features")
    return out


def predict_price(model, tau, history, weather_next):
    """Day-ahead price direction for D+1 (same inputs as the kit's Forecaster.predict).
    Returns a DataFrame indexed like weather_next with p_up (float) and nsw_price_up (int, 1 iff p_up >= tau)."""
    X = features_for_day(history, weather_next, list(model.feature_names_in_))
    p = model.predict_proba(X)[:, 1]
    return pd.DataFrame({"p_up": p, "nsw_price_up": (p >= tau).astype("int64")}, index=weather_next.index)


def batch_features(df):
    """CORE features for every row of the whole file (P1 builder, one pass)."""
    gf, _ = ofr.p1()
    X, _ = gf._construct(df, da.market_day(df.index), day_type_model(df))
    return X[list(gf.CORE)]


# ---------------- training ----------------
def train():
    r = ref()
    gf, _ = ofr.p1()
    data = gf.load_study(DATA_CSV)                                   # rows <= 2015-01-01 00:00 only
    assert data.index.max() == pd.Timestamp("2015-01-01 00:00")
    X, truth = gf.build_features(data)
    cols = r["features"]
    assert cols == list(gf.CORE)
    days = da.market_day(X.index)
    tr = (days >= pd.Timestamp(TRAIN[0])) & (days <= pd.Timestamp(TRAIN[1]))
    Xtr, ytr = X.loc[tr, cols], truth.loc[tr, "nsw_price_up"]
    assert (da.market_day(Xtr.index).nunique(), len(Xtr)) == (TRAIN_DAYS, TRAIN_ROWS)
    assert Xtr.index.is_unique and Xtr.index.is_monotonic_increasing and np.isfinite(Xtr.to_numpy(float)).all()
    t0 = time.perf_counter()
    model = HistGradientBoostingClassifier(**r["params"]).fit(Xtr, ytr)
    secs = time.perf_counter() - t0
    return model, Xtr, ytr, data, secs


def tree_fp(model):
    trees = [hashlib.sha256(p.nodes.tobytes()).hexdigest() for it in model._predictors for p in it]
    return {"n_iter": int(model.n_iter_), "trees_sha256": hashlib.sha256("".join(trees).encode()).hexdigest()}


def batch_predictions(df, model, tau, X=None):
    X = batch_features(df) if X is None else X
    days = da.market_day(X.index)
    sel = (days >= pd.Timestamp(PRED[0])) & (days <= pd.Timestamp(PRED[1]))
    Xp = X.loc[sel, list(model.feature_names_in_)]
    p = model.predict_proba(Xp)[:, 1]
    start = Xp.index - pd.Timedelta("30min")
    return pd.DataFrame({"day": days[sel], "period": (start.hour * 2 + start.minute // 30).astype("int64"),
                         "p_up": p, "pred": (p >= tau).astype("int64")}, index=Xp.index)


def fingerprint(model, Xtr, ytr, preds):
    return {"trees": tree_fp(model), "X_train_sha256": ofr.sha_frame(Xtr), "y_train_sha256": ofr.sha_arr(ytr, "int64"),
            "p_up_sha256": ofr.sha_arr(preds["p_up"], "float64"), "pred_sha256": ofr.sha_arr(preds["pred"], "int64"),
            "params": model.get_params()}


# ---------------- development scores (2015-2024 only) ----------------
def dev_scores(df, preds):
    """Model / b1 / b7 Acc and BalAcc per year and per month, target days 2015-01-01 .. 2024-12-31 only."""
    dev = preds[preds["day"] <= DEV_END]
    assert dev["day"].max() <= DEV_END
    days = sorted(dev["day"].unique())
    y, b1, b7 = [], [], []
    for d in days:
        d = pd.Timestamp(d)
        block = df.loc[d - pd.Timedelta("7D") + pd.Timedelta("30min"): d + pd.Timedelta("1D")]
        h, w = da.make_inputs(block, d)
        y.append(da.targets(block, d)["nsw_price_up"])
        b1.append(da.baseline(h, w, 1)["nsw_price_up"])
        b7.append(da.baseline(h, w, 7)["nsw_price_up"])
    t = pd.DataFrame({"y": pd.concat(y), "b1": pd.concat(b1), "b7": pd.concat(b7)})
    assert t.index.equals(dev.index)
    t["model"], t["day"] = dev["pred"], dev["day"]
    t["year"], t["month"] = t["day"].dt.year, t["day"].dt.to_period("M").astype(str)
    out = {}
    for key in ("year", "month"):
        rows = []
        for k, g in t.groupby(key):
            row = {key: k, "days": g["day"].nunique(), "rows": len(g), "class1_share": float(g["y"].mean() * 100)}
            for m in ("model", "b1", "b7"):
                row[f"{m}_Acc"] = da.accuracy(g["y"], g[m])
                row[f"{m}_BalAcc"] = da.balanced_accuracy(g["y"], g[m])
            rows.append(row)
        out[key] = pd.DataFrame(rows)
    allrow = {"period": "2015-2024", "days": t["day"].nunique(), "rows": len(t)}
    for m in ("model", "b1", "b7"):
        allrow[f"{m}_Acc"], allrow[f"{m}_BalAcc"] = da.accuracy(t["y"], t[m]), da.balanced_accuracy(t["y"], t[m])
    return out["year"], out["month"], allrow


# ---------------- day-by-day evidence ----------------
def daily_vs_batch(df, model, tau, preds, days=CHECK_DAYS):
    rows = []
    for d in days:
        h, w = da.make_inputs(df, d)
        t0 = time.perf_counter()
        o = predict_price(model, tau, h, w)
        secs = time.perf_counter() - t0
        b = preds.loc[w.index]
        rows.append({"day": d, "history_rows": len(h), "seconds": round(secs, 4),
                     "p_up_bitwise_equal": bool(np.array_equal(o["p_up"].to_numpy(), b["p_up"].to_numpy())),
                     "pred_equal": bool(np.array_equal(o["nsw_price_up"].to_numpy(), b["pred"].to_numpy())),
                     "max_abs_diff": float(np.max(np.abs(o["p_up"].to_numpy() - b["p_up"].to_numpy()))),
                     "periods": int(len(o)), "p_up_sha256": ofr.sha_arr(o["p_up"], "float64")[:16]})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fingerprint")
    a = ap.parse_args(argv)
    r = ref()
    model, Xtr, ytr, data, secs = train()
    df = da.load_dataset(DATA_CSV)                 # full file: history for 2015-2026 predictions only
    preds = batch_predictions(df, model, r["tau"])
    fp = fingerprint(model, Xtr, ytr, preds)
    if a.fingerprint:
        json.dump(fp, open(a.fingerprint, "w"), indent=1, default=str)
        print(fp["p_up_sha256"][:16], fp["trees"]["trees_sha256"][:16])
        return
    os.makedirs(OUT, exist_ok=True)
    joblib.dump(model, P["model"])
    joblib.dump(_DTM["m"], P["dtm"])
    preds.to_parquet(P["preds"])
    cus = json.load(open(CUSUM_CONFIG))
    config = {"model": "P1 boost_price, refit on 2010-01-29..2014-12-31 (Stage 1 frozen model)",
              "params_from": "results/price_hgbc/official/boost_price_ref.json (REF) - every HGBC parameter read from there",
              "params": r["params"], "features": r["features"], "tau": r["tau"],
              "train": {"first_day": TRAIN[0], "last_day": TRAIN[1], "days": TRAIN_DAYS, "rows": TRAIN_ROWS,
                        "source": "P1 ablation.features.load_study (rows up to 2015-01-01 00:00)"},
              "feature_code": {"module": "comp9417_project_kit/ablation/features.py", "functions": ["_construct", "_fit_day_types", "CORE"],
                               "p1_commit": r["p1"]["commit"], "day_type_model_fit": "<= 2013-12-31 profiles (P1 default)",
                               "window_days_per_call": WINDOW_DAYS},
              "interface": "predict_price(model, tau, history, weather_next) -> DataFrame[p_up, nsw_price_up] indexed by weather_next.index",
              "predictions": {"file": "results/price_hgbc/m5/predictions.parquet", "first_day": PRED[0], "last_day": PRED[1],
                              "columns": ["day", "period", "p_up", "pred"], "index": "timestamp (interval end)", "labels_stored": False},
              "cusum": {"config": "results/price_hgbc/m4/cusum_config.json", "ref_update": cus["ref_update"],
                        **{k: cus["schemes"]["main"][k] for k in ("q99", "mu0", "sigma0", "k", "h")},
                        "reset_after_retrain": True, "historical_alarm_2014": "2014-09-26 (results/price_hgbc/m4/cusum_path_2014.csv)"}}
    json.dump(config, open(P["config"], "w"), indent=1)
    manifest = {"files": {k: {"path": os.path.relpath(P[k], ROOT), "sha256": sha_file(P[k])} for k in ("model", "config", "preds", "dtm")},
                "train_seconds": round(secs, 3), "trained_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__,
                             "pandas": pd.__version__, "joblib": joblib.__version__}, "platform": platform.platform(),
                "data_csv_sha256": sha_file(DATA_CSV), "ref_sha256": sha_file(ofr.REF_JSON), "p1_commit": r["p1"]["commit"],
                "fingerprint": fp, "pred_rows": int(len(preds)), "pred_days": int(preds["day"].nunique())}
    json.dump(manifest, open(P["manifest"], "w"), indent=1, default=str)
    dv = daily_vs_batch(df, model, r["tau"], preds)
    dv.to_csv(P["daily"], index=False)
    yearly, monthly, allrow = dev_scores(df, preds)
    yearly.to_csv(P["yearly"], index=False, float_format="%.17g")
    monthly.to_csv(P["monthly"], index=False, float_format="%.17g")
    json.dump(allrow, open(os.path.join(OUT, "dev_all_2015_2024.json"), "w"), indent=1)
    pd.set_option("display.width", 250)
    print(json.dumps({k: manifest[k] for k in ("train_seconds", "pred_rows", "pred_days")}))
    print(dv.to_string(index=False))
    print(yearly.round(3).to_string(index=False))
    print(allrow)


if __name__ == "__main__":
    main()
