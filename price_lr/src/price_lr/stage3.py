"""Stage 3, step 1 (2026-11-18): first scoring of the LR S0-S3 on 2025-01-01 .. 2026-10-31 and the section 11.4 veto.

Design (frozen 2026-11-17): results/price_lr/strategy_design.md section 11.4 and 11.6.
  * P and R come from stage3_provisional.json (written 11-17 without any 2025-2026 number).
  * Switch to R only if BalAcc(P) < BalAcc(b1) and BalAcc(R) >= BalAcc(b1) on the 669 days (unrounded values).
  * The paired bootstrap here is DESCRIPTIVE only (same settings as section 5); it plays no part in the decision.
  * CUSUM: S0 monitor-only path and the S3 path continue causally from 2015-01-01; 2025-2026 rows are written here.
All outputs go to results/price_lr/stage3/. M7 files are read, never rewritten.
Usage: python -m src.price_lr.stage3
"""
import json
import os

import numpy as np
import pandas as pd

from src.price_hgbc import cusum as cs
from src.price_hgbc import m3_common as c

from . import data as D
from .env import RESULTS, sha_file
from .evaluate import STRATS, WINDOW, scores
from .replay import DEV_END, FROZEN_ID, REPLAY, Context, monitor_s0, run_s3

OUT = os.path.join(RESULTS, "stage3")
HOLD = (DEV_END + pd.Timedelta("1D"), REPLAY[1])          # 2025-01-01 .. 2026-10-31
N_DAYS = 669
PROVISIONAL = os.path.join(RESULTS, "stage3_provisional.json")
PROVISIONAL_SHA = "94bbcde179f834c980f93dd2e4db3a42f45a09b8888504633d8d7a16369a54cc"
COLS = STRATS + ["b1", "b7"]


def truth_table(df_full):
    preds = {}
    for s in STRATS:
        p = pd.read_parquet(os.path.join(RESULTS, f"{s.lower()}_predictions.parquet"))
        preds[s] = p[(p["day"] >= HOLD[0]) & (p["day"] <= HOLD[1])]
    idx = preds["S0"].index
    assert all(v.index.equals(idx) for v in preds.values())
    t = D.truth_and_baselines(df_full, *HOLD)
    assert t.index.equals(idx)
    for s in STRATS:
        t[s] = preds[s]["pred"].to_numpy()
    t["day"] = preds["S0"]["day"].to_numpy()
    assert t["day"].nunique() == N_DAYS and len(t) == N_DAYS * 48
    return t, preds


def full_table(t, log):
    b1 = scores(t["y"], t["b1"])
    s0 = scores(t["y"], t["S0"])
    rows = []
    for s in COLS:
        sc = scores(t["y"], t[s])
        r = {"strategy": s, "days": t["day"].nunique(), "rows": len(t),
             "class1_rows": int(t["y"].sum()), "class0_rows": int((1 - t["y"]).sum()), **sc,
             "BalAcc_minus_b1": sc["BalAcc"] - b1["BalAcc"], "BalAcc_minus_S0": sc["BalAcc"] - s0["BalAcc"],
             "Acc_minus_b1": sc["Acc"] - b1["Acc"]}
        if s in STRATS:
            lg = log[(log["strategy"] == s)]
            live = lg[(pd.to_datetime(lg["effective_from"]) <= HOLD[1]) & (pd.to_datetime(lg["effective_to"]) >= HOLD[0])]
            r["models_deployed"] = ";".join(live["model_id"])
            r["taus_deployed"] = ";".join(f"{x:g}" for x in live["tau"])
            r["retrains_effective_in_2025_2026"] = int((pd.to_datetime(lg["effective_from"]) >= HOLD[0]).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def grouped(t, key):
    k = pd.DatetimeIndex(t["day"])
    k = k.year if key == "year" else k.to_period("M").astype(str)
    out = []
    for kv, g in t.groupby(k):
        row = {key: kv, "days": g["day"].nunique(), "class1_share": float(g["y"].mean() * 100)}
        for s in COLS:
            sc = scores(g["y"], g[s])
            row[f"{s}_Acc"], row[f"{s}_BalAcc"] = sc["Acc"], sc["BalAcc"]
        out.append(row)
    return pd.DataFrame(out)


def boot_descriptive(t, F):
    days = sorted(t["day"].unique())
    B = c.boot_indices(len(days))
    samp = {s: c.boot_balacc(t["y"].to_numpy(), t[s].to_numpy(), t.index, B) for s in STRATS + ["b1"]}
    pairs = [(s, "S0") for s in ("S1", "S2", "S3")] + [("S3", "S1")] + [(s, "b1") for s in STRATS]
    pairs += [(F, s) for s in STRATS if s != F and (F, s) not in pairs]
    rows = []
    for a, b in pairs:
        lo, hi = c.ci(samp[a] - samp[b])
        ba, bb = scores(t["y"], t[a])["BalAcc"], scores(t["y"], t[b])["BalAcc"]
        rows.append({"a": a, "b": b, "BalAcc_a": ba, "BalAcc_b": bb, "diff_pp": ba - bb, "diff_lo": lo, "diff_hi": hi,
                     "ci_excludes_0": bool(lo > 0 or hi < 0), "n_days": len(days), "n_boot": c.N_BOOT,
                     "seed": c.BOOT_SEED, "role": "descriptive only; not used by section 11.4"})
    return pd.DataFrame(rows), c.sha(np.ascontiguousarray(B).tobytes())


def veto(cmp_, st):
    P, R = st["P_provisional"], st["R_backup"]
    b = cmp_.set_index("strategy")["BalAcc"]
    p_below = bool(b[P] < b["b1"])
    r_not_below = bool(b[R] >= b["b1"])
    switch = p_below and r_not_below
    final = R if switch else P
    if switch:
        reason = "P below b1 and R not below b1 -> switch to R"
    elif not p_below:
        reason = "P not below b1 -> keep P"
    else:
        reason = "P below b1 but R also below b1 -> keep P"
    return {"rule": "strategy_design.md section 11.4", "period": [str(HOLD[0].date()), str(HOLD[1].date())],
            "n_days": N_DAYS, "provisional_file_sha256": PROVISIONAL_SHA, "P": P, "R": R,
            "BalAcc_P": float(b[P]), "BalAcc_R": float(b[R]), "BalAcc_b1": float(b["b1"]),
            "P_below_b1": p_below, "R_not_below_b1": r_not_below, "switch_to_R": switch, "reason": reason,
            "final_strategy": final, "final_window": WINDOW[final],
            "comparison": "unrounded values; strict '<' for 'below'",
            "note": "2025-2026 were already partly seen: frozen LR (S0) scores by the group, and "
                    "2025-11-01..2026-10-31 used by P4 for tau selection on 11-16 (design section 0). "
                    "This is the first scoring of S1-S3 under the pre-registered rule, not an unseen holdout."}


def cusum_records(ctx):
    """Recompute S0 monitor and S3 paths from 2015-01-01 (state carried causally) and return the 2025-2026 rows."""
    s0 = pd.read_parquet(os.path.join(RESULTS, "s0_predictions.parquet"))
    det = cs.Cusum.from_config(ctx.cusum_cfg, "main")
    y = ctx.labels.loc[s0.index]
    mon = cs.run_batch(cs.daily_logloss(y.to_numpy(), s0["p_up"].to_numpy(), s0.index), det)
    mon["day"] = pd.to_datetime(mon["day"])
    old = pd.read_csv(os.path.join(RESULTS, "s0_monitor_cusum.csv"), parse_dates=["day"])
    head = mon[mon["day"] <= DEV_END].reset_index(drop=True)
    chk_s0 = {"rows_2015_2024": len(old), "equal_to_m7_file": bool(
        head["day"].equals(old["day"]) and np.array_equal(head["S"].to_numpy(), old["S"].to_numpy())
        and np.array_equal(head["alarm"].to_numpy(), old["alarm"].to_numpy()))}
    # a direct check that monitor_s0 (M7) is the same computation
    m7 = monitor_s0(ctx, s0)
    chk_s0["monitor_s0_reproduced"] = bool(np.array_equal(m7["S"].to_numpy(), old["S"].to_numpy()))

    p3, l3, e3, d3 = run_s3(ctx, mode="batch", save=False)
    st = pd.read_parquet(os.path.join(RESULTS, "s3_predictions.parquet"))
    ev_old = pd.read_csv(os.path.join(RESULTS, "s3_events.csv"))
    dly_old = pd.read_csv(os.path.join(RESULTS, "s3_daily_cusum.csv"))
    d3["day"] = pd.to_datetime(d3["day"])
    dev = d3[d3["day"] <= DEV_END]
    old_dev = dly_old[pd.to_datetime(dly_old["day"]) <= DEV_END]
    chk_s3 = {"predictions_bitwise_equal": bool(p3.index.equals(st.index) and np.array_equal(p3["p_up"].to_numpy(), st["p_up"].to_numpy())
                                                 and (p3["model_id"].to_numpy() == st["model_id"].to_numpy()).all()),
              "events_equal": bool(len(e3) == len(ev_old) and (e3["alarm_day"].astype(str).to_numpy() == ev_old["alarm_day"].astype(str).to_numpy()).all()
                                   and (e3["retrained"].to_numpy() == ev_old["retrained"].to_numpy()).all()),
              "S_2015_2024_equal_to_m7_file": bool(np.array_equal(dev["S"].to_numpy(), old_dev["S"].to_numpy())),
              "alarm_flags_all_days_equal": bool((d3["alarm"].to_numpy() == dly_old["alarm"].to_numpy()).all()),
              "retrains_total": int(len(l3) - 1),
              "retrains_effective_in_2025_2026": int((pd.to_datetime(l3["effective_from"]) >= HOLD[0]).sum()),
              "budget_left_at_2025_01_01": 6 - int((pd.to_datetime(l3.loc[l3["model_id"] != FROZEN_ID, "effective_from"]) <= DEV_END).sum())}
    mon_h = mon[mon["day"] >= HOLD[0]].copy()
    d3_h = d3[d3["day"] >= HOLD[0]].copy()
    e3["alarm_day"] = pd.to_datetime(e3["alarm_day"])
    e3_h = e3[e3["alarm_day"] >= HOLD[0]].copy()
    return mon, mon_h, d3_h, e3_h, chk_s0, chk_s3


def alarms_by_year(mon):
    a = mon[mon["alarm"]]
    return a.groupby(pd.DatetimeIndex(a["day"]).year).size().rename("alarms").rename_axis("year").reset_index()


def main():
    assert sha_file(PROVISIONAL) == PROVISIONAL_SHA, "stage3_provisional.json changed since 11-17"
    st = json.load(open(PROVISIONAL))
    assert st["uses_2025_2026"] is False and (st["P_provisional"], st["R_backup"]) == ("S1", "S2")
    os.makedirs(OUT, exist_ok=True)
    ctx = Context()                                       # input SHAs, label check through 2026-10-31
    df = ctx.df_full.loc[ctx.df_full.index <= HOLD[1] + pd.Timedelta("1D")]
    log = pd.read_csv(os.path.join(RESULTS, "models_log.csv"))
    t, _ = truth_table(df)
    cmp_ = full_table(t, log)
    yearly, monthly = grouped(t, "year"), grouped(t, "month")
    boot, bsha = boot_descriptive(t, st["F"])
    v = veto(cmp_, st)
    mon, mon_h, d3_h, e3_h, chk_s0, chk_s3 = cusum_records(ctx)
    assert chk_s0["equal_to_m7_file"] and chk_s0["monitor_s0_reproduced"]
    assert all(chk_s3[k] for k in ("predictions_bitwise_equal", "events_equal", "S_2015_2024_equal_to_m7_file",
                                   "alarm_flags_all_days_equal"))
    assert chk_s3["retrains_effective_in_2025_2026"] == 0

    w = lambda df_, name: df_.to_csv(os.path.join(OUT, name), index=False, float_format="%.17g")
    w(cmp_, "holdout_2025_2026_comparison.csv")
    w(yearly, "holdout_2025_2026_yearly.csv")
    w(monthly, "holdout_2025_2026_monthly.csv")
    w(boot, "holdout_2025_2026_paired_bootstrap_DESCRIPTIVE.csv")
    w(mon_h, "s0_monitor_cusum_2025_2026.csv")
    w(alarms_by_year(mon), "s0_monitor_alarms_by_year_2015_2026.csv")
    w(d3_h, "s3_daily_cusum_2025_2026.csv")
    w(e3_h, "s3_events_2025_2026.csv")
    json.dump(v, open(os.path.join(OUT, "veto.json"), "w"), indent=1)
    meta = {"period": v["period"], "n_days": N_DAYS, "latest_timestamp_read": str(df.index.max()),
            "boot_index_sha256": bsha, "boot_role": "descriptive only",
            "cusum_checks": {"S0_monitor": chk_s0, "S3": chk_s3}, "inputs_checked": True}
    json.dump(meta, open(os.path.join(OUT, "stage3_meta.json"), "w"), indent=1)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(cmp_[["strategy", "Acc", "BalAcc", "BalAcc_minus_b1", "BalAcc_minus_S0", "models_deployed"]].round(3).to_string(index=False))
    print(yearly[["year", "days", "class1_share"] + [f"{s}_BalAcc" for s in COLS]].round(2).to_string(index=False))
    print(boot[["a", "b", "diff_pp", "diff_lo", "diff_hi"]].round(3).to_string(index=False))
    print(json.dumps(v, indent=1))
    print(json.dumps(meta["cusum_checks"], indent=1))
    print("S0 monitor alarms by year:", alarms_by_year(mon).to_dict("records"))
    print("S3 events 2025-2026:", len(e3_h), e3_h["suppressed_reason"].value_counts().to_dict())


if __name__ == "__main__":
    main()
