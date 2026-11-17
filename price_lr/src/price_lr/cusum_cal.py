"""M7 step 1: calibrate the log-loss CUSUM on the 2014 out-of-sample errors of the LR model (design section 3).

Reference predictions: LR (log_interactions31, C=1, no weighting) fitted on 2010-01-29 .. 2013-12-31 and applied
to 2014, i.e. P4's validation_predictions column `log_interactions31_probability`. The refit here must equal it
bit for bit. Only rows up to 2015-01-01 00:00 are read. Everything else is HGBC M4 unchanged (functions imported
from src.price_hgbc.m4 / cusum; tau for the misclassification comparison = frozen tau0 = 0.42).
Usage: python -m src.price_lr.cusum_cal
"""
import hashlib
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from src.price_hgbc import cusum as cs
from src.price_hgbc import m4
from src.price_hgbc import m3_common as c

from . import data as D
from .env import RESULTS, ROOT, VALID_CSV, check_inputs, lu, sha_file

OUT = os.path.join(RESULTS, "cusum")
TAU0 = 0.42
CODE_FILES = ["src/price_lr/cusum_cal.py", "src/price_lr/data.py", "src/price_lr/env.py",
              "src/price_hgbc/cusum.py", "src/price_hgbc/m4.py"]


def reference_2014():
    raw = D.load_raw()
    pre = D.prefix(raw, "2014-12-31")                         # nothing after 2015-01-01 00:00
    f = D.Frame(pre)
    tr, te = f.rows("2010-01-29", "2013-12-31"), f.rows("2014-01-01", "2014-12-31")
    assert int(tr.sum()) == 68784 and int(te.sum()) == 17520
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("error", ConvergenceWarning)
        model = lu.estimator(1.0, None).fit(f.X.loc[tr], f.y.loc[tr])
    p = model.predict_proba(f.X.loc[te])[:, 1]
    v = pd.read_csv(VALID_CSV, index_col="timestamp", parse_dates=True, float_precision="round_trip")
    assert v.index.equals(f.X.index[te]), "P4 validation file covers different rows"
    assert v.index.max() == pd.Timestamp("2015-01-01 00:00")
    p4 = v["log_interactions31_probability"].to_numpy()
    same_y = bool(np.array_equal(f.y.loc[te].to_numpy(), v["truth"].to_numpy()))
    assert same_y, "2014 truth differs from P4"
    # Deviation from design section 3 (recorded in selfcheck/M7.md): the refit is not bitwise equal to P4's file
    # (cross-machine floating point in the lbfgs fit; features and the frozen artifact's predictions ARE bitwise
    # equal). The reference is P4's column, as the design names it; the refit is kept as a check and the whole
    # calibration is repeated on it (run(): `refit_check`) to show the choice does not change any output.
    dp = np.abs(p - p4)
    check = {"bitwise_equal": bool(np.array_equal(p, p4)), "max_abs_diff": float(dp.max()),
             "rows_differing": int((dp > 0).sum()), "rows": int(len(p)),
             "pred_at_tau0_equal": bool(np.array_equal(p >= TAU0, p4 >= TAU0))}
    assert check["max_abs_diff"] < 1e-12 and check["pred_at_tau0_equal"], check
    df14 = pd.read_csv(D.DATA_CSV, index_col="timestamp", parse_dates=True, nrows=((pd.Timestamp('2014-12-31') - D.START).days + 1) * 48)
    assert df14.index.max() == pd.Timestamp("2015-01-01 00:00")
    off = D.official_labels(df14, "2014-01-01", "2014-12-31")
    assert np.array_equal(off.to_numpy(), f.y.loc[te].to_numpy()), "label definition differs from da.targets (2014)"
    return f.X.index[te], f.y.loc[te].to_numpy(), p4, p, check


def calibrate(idx, y, p):
    ll = cs.daily_logloss(y, p, idx)
    mc = cs.daily_misclass(y, (p >= TAU0).astype(int), idx)
    d = pd.DataFrame({"day": ll.index, "logloss": ll.to_numpy(), "misclass": mc.to_numpy()})
    stats = m4.ref_stats_table(d)
    blocks = m4.week_blocks(d["day"])
    arl = m4.arl_table(d, stats, blocks, m4.boot_matrix(len(blocks)))
    return d, stats, arl


def run():
    check_inputs()
    idx, y, p, p_refit, check = reference_2014()
    _, st_r, arl_r = calibrate(idx, y, p_refit)
    d = calibrate(idx, y, p)[0]
    d["weekday"] = pd.DatetimeIndex(d["day"]).day_of_week
    stats = m4.ref_stats_table(d)
    main = stats.set_index("scheme").loc["main"]
    d["logloss_trunc"] = np.minimum(d["logloss"], main["q99"])
    d["truncated"] = d["logloss"] > main["q99"]
    diag = m4.diagnostics(d, main["q99"])
    blocks = m4.week_blocks(d["day"])
    W = m4.boot_matrix(len(blocks))
    arl = m4.arl_table(d, stats, blocks, W)
    h = int(arl[(arl["scheme"] == "main") & arl["selected"]]["h"].iloc[0])
    meets = bool(arl[(arl["scheme"] == "main") & arl["selected"]]["meets_target"].iloc[0])
    det = cs.Cusum(main["mu0"], main["sigma0"], main["q99"], main["k"], h)
    path = cs.run_batch(pd.Series(d["logloss"].to_numpy(), index=d["day"]), det)
    delay = m4.delay_table(stats, d, blocks, W, h)
    path_r = cs.run_batch(pd.Series(calibrate(idx, y, p_refit)[0]["logloss"].to_numpy(), index=d["day"]),
                          cs.Cusum(*st_r.set_index("scheme").loc["main", ["mu0", "sigma0", "q99", "k"]], h))
    num = ["q99", "mu0", "sigma0"]
    fin = stats["scheme"] != "no_trunc"                                  # no_trunc has q99 = +inf
    check.update({"stats_max_abs_diff": float(np.abs(stats.loc[fin, num].to_numpy() - st_r.loc[fin, num].to_numpy()).max()),
                  "no_trunc_mu0_sigma0_max_abs_diff": float(np.abs(stats.loc[~fin, ["mu0", "sigma0"]].to_numpy()
                                                                   - st_r.loc[~fin, ["mu0", "sigma0"]].to_numpy()).max()),
                  "h_same": int(arl_r[(arl_r["scheme"] == "main") & arl_r["selected"]]["h"].iloc[0]) == h,
                  "arl_table_same": bool(arl[["arl0_median", "arl0_q05", "arl0_q95"]].equals(arl_r[["arl0_median", "arl0_q05", "arl0_q95"]])),
                  "alarms_2014_same": bool(path["alarm"].equals(path_r["alarm"]))})
    meta = {"refit_check": check, "blocks": len(blocks), "block_first_day": str(pd.Timestamp(d["day"].iloc[blocks[0][0]]).date()),
            "block_last_day": str(pd.Timestamp(d["day"].iloc[blocks[-1][-1]]).date()),
            "boot_matrix_sha256": hashlib.sha256(np.ascontiguousarray(W).tobytes()).hexdigest(),
            "proba_2014_sha256": c.sha_arr(p, "float64")}
    return d, stats, diag, arl, path, delay, meta, h, meets


def build_config(stats, arl, meta):
    schemes = {}
    for sch, role in (("main", "selected detector (log loss)"), ("misclass", "comparison only (not used)")):
        s = stats.set_index("scheme").loc[sch]
        a = arl[(arl["scheme"] == sch) & arl["selected"]].iloc[0]
        schemes[sch] = {"error": s["error"], "truncate": bool(s["truncate"]), "q99": float(s["q99"]), "mu0": float(s["mu0"]),
                        "sigma0": float(s["sigma0"]), "k": float(s["k"]), "h": int(a["h"]),
                        "arl0_median": m4._num(float(a["arl0_median"])), "arl0_q05": m4._num(float(a["arl0_q05"])),
                        "arl0_q95": m4._num(float(a["arl0_q95"])), "meets_target": bool(a["meets_target"]), "role": role}
    return {
        "model": "LR log_interactions31 (C=1, class_weight=None), group selected price model",
        "error_definitions": {"logloss": "mean over the 48 half-hours of day D of -[y log p + (1-y) log(1-p)], p clipped",
                              "misclass": f"share of day D's 48 half-hours with 1[p >= {TAU0}] != y"},
        "logloss_clip": list(cs.LOGLOSS_CLIP), "k": cs.K_DEFAULT, "h_candidates": list(cs.H_CANDIDATES),
        "arl0_target_days": cs.ARL0_TARGET, "schemes": schemes,
        "timing": "update S_D at the end of day D; after an alarm the retrained model forecasts from D+1",
        "ref_update_options": {"A": "keep the 2014 mu0, sigma0, q99 after every retrain",
                               "B": "re-estimate from the new model's OOS errors (not used for LR; design 9.4)"},
        "ref_update": "A",
        "ref_update_decision": {"choice": "A", "decided_by": "user", "recorded_in": "results/price_lr/strategy_design.md section 10"},
        "bootstrap": {"method": "weekly block bootstrap (full Mon-Sun weeks of 2014)", "block_days": m4.BLOCK,
                      "series_days": m4.N_DAYS_SIM, "replicates": m4.N_REP, "seed": m4.SEED,
                      "arl_per_replicate": "3650 / n_alarms", **{k: v for k, v in meta.items() if k != "refit_check"}},
        "data_period": {"calibration": "2014 out-of-sample predictions of LR fitted 2010-01-29..2013-12-31",
                        "latest_timestamp_read": "2015-01-01 00:00"},
        "ref": {"validation_predictions": "_group_main/logistic_price/results/frozen/validation_predictions.csv.gz",
                "sha256": sha_file(VALID_CSV), "tau0": TAU0, "refit_check": meta["refit_check"]},
        "code_commit": None, "code_commit_note": "workspace is not a git repository; code identified by SHA-256 below",
        "code_sha256": {f: sha_file(os.path.join(ROOT, f)) for f in CODE_FILES},
        "design_sha256": sha_file(os.path.join(RESULTS, "strategy_design.md")),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    d, stats, diag, arl, path, delay, meta, h, meets = run()
    for name, t in (("daily_err_2014.csv", d), ("reference_stats_2014.csv", stats), ("err_diagnostics_2014.csv", diag),
                    ("arl0_by_h.csv", arl), ("cusum_path_2014.csv", path), ("delay_by_shift.csv", delay)):
        t.to_csv(os.path.join(OUT, name), index=False, float_format="%.17g")
    cfg = build_config(stats, arl, meta)
    cs.validate_config(cfg)
    json.dump(cfg, open(os.path.join(OUT, "cusum_config.json"), "w"), indent=1, allow_nan=False)
    json.dump({"reads": [{"file": "course CSV via load_raw/prefix", "max_timestamp": "2015-01-01 00:00"},
                         {"file": os.path.relpath(VALID_CSV, ROOT), "max_timestamp": "2015-01-01 00:00"}],
               "latest_timestamp_read": "2015-01-01 00:00"}, open(os.path.join(OUT, "data_access_log.json"), "w"), indent=1)
    for fn, args, name in ((m4.fig_path, (path, h), "fig_cusum_path_2014.png"), (m4.fig_arl, (arl,), "fig_arl0_by_h.png"),
                           (m4.fig_delay, (delay,), "fig_delay_by_shift.png")):
        fig, ax = fn(*args)
        if name.startswith("fig_cusum"):
            ax.set_title(f"LR: upper CUSUM on 2014 OOS log loss (k = 0.5, h = {h}): {int(path['alarm'].sum())} alarm(s)",
                         color=c.INK, fontsize=11, loc="left")
        fig.savefig(os.path.join(OUT, name), dpi=150, facecolor=c.SURFACE)
    pd.set_option("display.width", 250)
    print(stats.to_string(index=False))
    print(arl.to_string(index=False))
    print("selected h:", h, "meets target:", meets, "| 2014 alarms:", int(path["alarm"].sum()),
          list(path.loc[path["alarm"], "day"].astype(str)))
    print(delay.to_string(index=False))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
