"""M7 consistency checks (design section 6): daily interface vs batch, S3 daily-mode replay vs batch replay.

Finding recorded in selfcheck/M7.md: features from P4's daily interface are bitwise equal to the batch features,
and a 48-row predict call on either is bitwise equal; predicting a whole multi-day segment in one call changes
probabilities by <= ~1e-16 (BLAS blocking). S0/S1/S2 store segment-call predictions (as P4's replay does; S0 is
bitwise equal to P4), so for them we require: daily == per-day batch bitwise, and daily vs segment classes equal
with |dp| reported. S3 predicts per day in both modes, so it must match bitwise.
Usage: python -m src.price_lr.consistency
"""
import json
import os

import joblib
import numpy as np
import pandas as pd

from .env import RESULTS, ROOT
from .replay import Context, features_for_day

SAMPLES = {"S1": ["S1_r1_2016-12-31", "S1_r3_2020-12-31", "S1_r4_2022-12-31"],
           "S2": ["S2_r1_2016-12-31", "S2_r2_2018-12-31", "S2_r4_2022-12-31"]}


def segment_days(lg):
    a, b = pd.Timestamp(lg["effective_from"]), pd.Timestamp(lg["effective_to"])
    return [a, a + pd.Timedelta(days=(b - a).days // 2), b]


def main():
    ctx = Context()
    log = pd.read_csv(os.path.join(RESULTS, "models_log.csv")).set_index("model_id")
    out = {"scheduled": [], "s3_daily_vs_batch": None}
    for s, mids in SAMPLES.items():
        pr = pd.read_parquet(os.path.join(RESULTS, f"{s.lower()}_predictions.parquet"))
        for mid in mids:
            lg = log.loc[mid]
            b = joblib.load(os.path.join(ROOT, lg["model_path"]))
            for d in segment_days(lg):
                m = ctx.f.day == d
                pdl = b["pipeline"].predict_proba(features_for_day(ctx, d))[:, 1]
                pbt = b["pipeline"].predict_proba(ctx.f.X.loc[m])[:, 1]
                seg = pr[pr["day"] == d]
                assert (seg["model_id"] == mid).all()
                out["scheduled"].append({"strategy": s, "model_id": mid, "day": str(d.date()),
                                         "daily_eq_perday_batch_bitwise": bool(np.array_equal(pdl, pbt)),
                                         "class_eq_stored": bool(np.array_equal((pdl >= b["threshold"]).astype(int), seg["pred"].to_numpy())),
                                         "max_abs_dp_vs_stored": float(np.abs(pdl - seg["p_up"].to_numpy()).max())})
    dm = os.path.join(RESULTS, "s3_daily_mode_predictions.parquet")
    if os.path.exists(dm):
        a, b = pd.read_parquet(os.path.join(RESULTS, "s3_predictions.parquet")), pd.read_parquet(dm)
        ca = pd.read_csv(os.path.join(RESULTS, "s3_daily_cusum.csv"))
        cb = pd.read_csv(os.path.join(RESULTS, "s3_daily_mode_cusum.csv"))
        out["s3_daily_vs_batch"] = {"rows": len(a), "index_equal": bool(a.index.equals(b.index)),
                                    "p_up_bitwise": bool(np.array_equal(a["p_up"].to_numpy(), b["p_up"].to_numpy())),
                                    "pred_equal": bool(np.array_equal(a["pred"].to_numpy(), b["pred"].to_numpy())),
                                    "model_id_equal": bool((a["model_id"].to_numpy() == b["model_id"].to_numpy()).all()),
                                    "alarm_days_equal": bool(ca["alarm"].equals(cb["alarm"])),
                                    "retrain_days_equal": bool(ca["retrained"].equals(cb["retrained"]))}
    ok = all(r["daily_eq_perday_batch_bitwise"] and r["class_eq_stored"] for r in out["scheduled"])
    out["scheduled_all_ok"] = ok
    out["scheduled_max_abs_dp_vs_stored"] = max(r["max_abs_dp_vs_stored"] for r in out["scheduled"])
    json.dump(out, open(os.path.join(RESULTS, "selfcheck", "consistency.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "scheduled"}, indent=1))
    assert ok and (out["s3_daily_vs_batch"] is None or all(v for k, v in out["s3_daily_vs_batch"].items() if k != "rows"))


if __name__ == "__main__":
    main()
