"""Stage 3 step 1 checks (strategy_design.md section 11.4, 11.6). Run: pytest tests/price_lr -q"""
import glob
import json
import os

import pandas as pd

from src.price_lr.env import RESULTS, sha_file
from src.price_lr.stage3 import PROVISIONAL, PROVISIONAL_SHA

R = lambda *p: os.path.join(RESULTS, *p)
HOLD0 = pd.Timestamp("2025-01-01")


def test_provisional_unchanged():
    assert sha_file(PROVISIONAL) == PROVISIONAL_SHA


def test_veto_follows_rule_mechanically():
    v = json.load(open(R("stage3", "veto.json")))
    st = json.load(open(PROVISIONAL))
    assert (v["P"], v["R"]) == (st["P_provisional"], st["R_backup"])
    cmp_ = pd.read_csv(R("stage3", "holdout_2025_2026_comparison.csv")).set_index("strategy")
    assert (cmp_["days"] == 669).all()
    for k, s in (("BalAcc_P", v["P"]), ("BalAcc_R", v["R"]), ("BalAcc_b1", "b1")):
        assert v[k] == cmp_.loc[s, "BalAcc"]
    p_below, r_ok = v["BalAcc_P"] < v["BalAcc_b1"], v["BalAcc_R"] >= v["BalAcc_b1"]
    assert (v["P_below_b1"], v["R_not_below_b1"]) == (p_below, r_ok)
    assert v["final_strategy"] == (v["R"] if (p_below and r_ok) else v["P"])


def test_holdout_scores_only_under_stage3():
    """2025-2026 scores live only in results/price_lr/stage3/ (plus the g_analog *_2015_2025 files and handoff copies);
    every M7 summary still ends at 2024-12-31."""
    for f in glob.glob(R("*.csv")):
        d = pd.read_csv(f, nrows=0).columns
        name = os.path.basename(f)
        if "year" in d:
            assert pd.read_csv(f)["year"].max() <= 2024, name
        if "month" in d:
            assert pd.read_csv(f)["month"].max() <= "2024-12", name
    for f in glob.glob(R("stage3", "*.csv")):
        d = pd.read_csv(f)
        for col in ("day", "alarm_day"):
            if col in d:
                assert pd.to_datetime(d[col]).min() >= HOLD0 or "2015_2026" in f, f


def test_s3_no_retrain_in_2025_2026():
    ev = pd.read_csv(R("stage3", "s3_events_2025_2026.csv"))
    assert not ev["retrained"].any() and set(ev["suppressed_reason"]) == {"budget"}
    log = pd.read_csv(R("models_log.csv"))
    s3 = log[log["strategy"] == "S3"]
    assert (pd.to_datetime(s3["effective_from"]) < HOLD0).all()
    meta = json.load(open(R("stage3", "stage3_meta.json")))
    assert meta["cusum_checks"]["S3"]["retrains_effective_in_2025_2026"] == 0
    assert meta["latest_timestamp_read"] == "2026-11-01 00:00:00"


def test_final_artifact_required_checks():
    import joblib
    rec = json.load(open(R("stage3", "final_artifact_check.json")))
    assert rec["required_all_pass"] and all(rec["required_checks"].values())
    assert sha_file(R("stage3", "final_artifact", "logistic.joblib")) == rec["logistic_joblib_sha256"]
    t = json.load(open(R("stage3", "final_artifact", "training.json")))
    b = joblib.load(R("stage3", "final_artifact", "logistic.joblib"))
    assert t["threshold"] == b["threshold"] == 0.49 and t["train_rows"] == b["train_rows"] == 293760
    v = json.load(open(R("stage3", "veto.json")))
    assert v["final_window"] == "expanding" and t["training_policy"] == "expanding_from_2010-01-29"


def test_g_analog_2025_added_apart():
    new = pd.read_csv(R("g_analog", "g_analog_weeks_2015_2025.csv"), float_precision="round_trip")
    old = pd.read_csv(R("g_analog", "g_analog_weeks.csv"), float_precision="round_trip")
    assert len(new) == 22 and len(old) == 20 and sorted(new["year"].unique()) == list(range(2015, 2026))
    assert new[new["year"] <= 2024].reset_index(drop=True)[old.columns].equals(old)
    g = json.load(open(R("g_analog", "g_analog_summary_2015_2025.json")))
    v = json.load(open(R("stage3", "veto.json")))
    assert g["selected_method"] == v["final_window"]
    sel = new[new["method"] == g["selected_method"]]["G_pp"]
    assert g["selected"]["n11_2015_2025"]["n"] == 11 and g["selected"]["n11_2015_2025"]["median"] == float(sel.median())
    assert g["selected"]["n10_2015_2024"] == json.load(open(R("g_analog", "g_analog_summary.json")))["main"][g["selected_method"]]
    assert g["supplement_37_43"]["n"] == 15 and all(c["bitwise_equal"] for c in g["daily_interface_spot_checks_2025"])
