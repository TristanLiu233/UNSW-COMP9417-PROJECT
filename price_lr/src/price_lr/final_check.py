"""Stage 3 step 2 (2026-11-18): check the final LR artifact (design section 11.5).

  python -m src.price_lr.final_check                       # check results/price_lr/stage3/final_artifact (local run)
  python -m src.price_lr.final_check --reference DIR       # also compare with another artifact dir (e.g. P4's)

Required (stop if any fails): tau 0.49, final rows 293,760, inner rows 276,240, validation rows 17,520,
36 candidates, recipe log_interactions31, C=1, class_weight None, seed 17.
Cross-machine comparison: training.json equal, array max |diff| <= 1e-12, identical classes on the 672 slots of
2026-10-18..10-31 through PriceClassifier.predict (a consistency check, not a score).
Output: results/price_lr/stage3/final_artifact_check.json
"""
import argparse
import json
import os
import platform
import sys

import joblib
import numpy as np
import pandas as pd

from . import data as D
from .env import GROUP, RESULTS, check_inputs, lf, sha_file

ART = os.path.join(RESULTS, "stage3", "final_artifact")
OUT = os.path.join(RESULTS, "stage3", "final_artifact_check.json")
P4_SHA = "7e898b2aebd146b83f40df869a9ffbc728bc447ca22970e872b51b103131dd50"
P4_TAU_SCORE = {"BalAcc": 78.896947, "Accuracy": 78.835616}          # notes/final_training.md (6 decimals)
DAYS = pd.date_range("2026-10-18", "2026-10-31")
ARRAYS = (("scaler.mean_", 0, "mean_"), ("scaler.scale_", 0, "scale_"), ("scaler.var_", 0, "var_"),
          ("lr.coef_", -1, "coef_"), ("lr.intercept_", -1, "intercept_"))


def required(t, b):
    f = t["folds"][0]
    best = max(t["threshold_search"] if "threshold_search" in t else t["search"], key=lambda r: r["BalAcc"])
    checks = {"tau_0.49": t["threshold"] == 0.49 == b["threshold"], "final_rows_293760": t["train_rows"] == 293760 == b["train_rows"],
              "inner_rows_276240": f["train_rows"] == 276240, "validation_rows_17520": f["validation_rows"] == 17520,
              "validation_2025-11-01_2026-10-31": (f["validation_start"], f["validation_end"]) == ("2025-11-01", "2026-10-31"),
              "candidates_36": len(t.get("threshold_search", t.get("search", []))) == 36,
              "recipe": t["recipe"] == b["recipe"] == "log_interactions31", "C_1": t["C"] == 1,
              "class_weight_None": t["class_weight"] is None, "seed_17": t["random_state"] == 17,
              "train_end": t["train_end"] == b["train_end"] == "2026-10-31",
              "tau_score_matches_P4_note_6dp": round(best["BalAcc"], 6) == P4_TAU_SCORE["BalAcc"]
                                                 and round(best["Accuracy"], 6) == P4_TAU_SCORE["Accuracy"]}
    return checks


def predict_672(art, raw):
    sys.path.insert(0, GROUP)
    from logistic_price.classifier import PriceClassifier
    clf = PriceClassifier(art)
    out = []
    for d in DAYS:
        idx = pd.date_range(d + pd.Timedelta(minutes=30), periods=48, freq="30min")
        hist, w = raw.loc[:d], raw.loc[idx, lf.WEATHER]
        p = clf.predict(hist, w)
        assert p.index.equals(idx) and len(p) == 48 and set(p["nsw_price_up"].unique()) <= {0, 1}
        from logistic_price.features import build_features_for_day
        from logistic_price.transforms import transform_features
        x = transform_features(build_features_for_day(hist, w), clf.recipe)[clf.columns]
        out.append(pd.DataFrame({"pred": p["nsw_price_up"], "p_up": clf.pipeline.predict_proba(x)[:, 1]}, index=idx))
    return pd.concat(out)


def compare(a, b, raw):
    ta, tb = (json.load(open(os.path.join(x, "training.json"))) for x in (a, b))
    ba, bb = (joblib.load(os.path.join(x, "logistic.joblib")) for x in (a, b))
    diffs = {name: float(np.max(np.abs(getattr(ba["pipeline"][i], attr) - getattr(bb["pipeline"][i], attr))))
             for name, i, attr in ARRAYS}
    pa, pb = predict_672(a, raw), predict_672(b, raw)
    res = {"training_json_equal": ta == tb,
           "training_json_diff_keys": sorted(k for k in set(ta) | set(tb) if ta.get(k) != tb.get(k)),
           "columns_equal": ba["columns"] == bb["columns"], "n_iter": [int(ba["pipeline"][-1].n_iter_[0]), int(bb["pipeline"][-1].n_iter_[0])],
           "array_max_abs_diff": diffs, "arrays_within_1e-12": all(v <= 1e-12 for v in diffs.values()),
           "arrays_bitwise": all(v == 0 for v in diffs.values()),
           "slots": len(pa), "classes_identical_672": bool((pa["pred"].to_numpy() == pb["pred"].to_numpy()).all()),
           "proba_max_abs_diff_672": float(np.max(np.abs(pa["p_up"].to_numpy() - pb["p_up"].to_numpy())))}
    res["pass"] = res["training_json_equal"] and res["columns_equal"] and res["arrays_within_1e-12"] and res["classes_identical_672"]
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference")
    args = ap.parse_args(argv)
    check_inputs()
    raw = D.load_raw()
    t = json.load(open(os.path.join(ART, "training.json")))
    b = joblib.load(os.path.join(ART, "logistic.joblib"))
    req = required(t, b)
    sha = sha_file(os.path.join(ART, "logistic.joblib"))
    import sklearn, scipy, threadpoolctl
    rec = {"artifact": os.path.relpath(ART, RESULTS), "logistic_joblib_sha256": sha,
           "training_json_sha256": sha_file(os.path.join(ART, "training.json")),
           "required_checks": req, "required_all_pass": all(req.values()),
           "P4_recorded_sha256": P4_SHA, "sha_equal_to_P4": sha == P4_SHA,
           "environment": {"python": platform.python_version(), "platform": platform.platform(), "sklearn": sklearn.__version__,
                           "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                           "joblib": joblib.__version__, "threadpoolctl": threadpoolctl.__version__}}
    p = predict_672(ART, raw)
    rec["predict_672"] = {"slots": len(p), "loads_with_PriceClassifier": True, "valid_output": True,
                          "role": "consistency only, not a score"}
    if args.reference:
        rec["reference_dir"] = args.reference
        rec["reference_sha256"] = sha_file(os.path.join(args.reference, "logistic.joblib"))
        rec["comparison"] = compare(ART, args.reference, raw)
    if os.path.exists(OUT):
        old = json.load(open(OUT))
        rec["history"] = old.get("history", []) + [{k: old[k] for k in old if k != "history"}]
    json.dump(rec, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k != "history"}, indent=1))
    assert rec["required_all_pass"], "required check failed: stop and report"


if __name__ == "__main__":
    main()
