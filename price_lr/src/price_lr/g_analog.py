"""M7 step 4: historical analog of the live-week gain G = BA_m - BA_b1 (design section 12; report section 5 only).

Main: for Y = 2015..2024 train with the final-training method at cutoff Y-10-31 and predict Y-12-07..Y-12-13
(model age 37-43 days, as for the live week), one 48-row predict call per day as in the package. Both methods: expanding (P4 fit_final) and 3-year sliding
(fit_sliding, start = (Y-3)-11-01). Supplement: 37-43-day windows of models actually deployed in S1-S3.
Not used for selecting any strategy. `main` (M7, 11-17) reads nothing after 2024-12-31 for scoring.
`add_2025` (11-18, after the 2025-2026 opening, section 12.4) adds Y = 2025 and writes the n = 11 version to new
files; the n = 10 files from 11-17 are left as they are.
Usage: python -m src.price_lr.g_analog [add2025]
"""
import json
import os

import numpy as np
import pandas as pd

from . import data as D
from .env import RESULTS, ft
from .evaluate import scores
from .fit_sliding import fit_sliding
from .replay import Context, features_for_day

OUT = os.path.join(RESULTS, "g_analog")
YEARS = range(2015, 2025)
D1 = pd.Timedelta("1D")


def summary(g):
    g = np.asarray(g, dtype=float)
    return {"n": int(len(g)), "median": float(np.median(g)), "q10": float(np.quantile(g, 0.1)),
            "q90": float(np.quantile(g, 0.9)), "min": float(g.min()), "max": float(g.max()),
            "weeks_G_gt_0": int((g > 0).sum())}


def analog_year(ctx, df, Y, rows, checks):
    """Section 12.2-12.3 for one year Y (both training methods); appends to rows / checks."""
    cut = pd.Timestamp(f"{Y}-10-31")
    pre = D.prefix(ctx.raw, cut)
    week = pd.date_range(f"{Y}-12-07", f"{Y}-12-13")
    t = D.truth_and_baselines(df, week[0], week[-1])
    m = ctx.rows(week[0], week[-1])
    assert t.index.equals(ctx.f.X.index[m]) and len(t) == 336
    for method in ("expanding", "sliding"):
        if method == "expanding":
            bundle, audit = ft.fit_final(pre, cut)
            start = D.FIRST_TRAIN_DAY
        else:
            start = pd.Timestamp(f"{Y - 3}-11-01")
            bundle, audit = fit_sliding(pre, cut, start)
        # one predict call per day (48 rows), as the package does in the live week
        p = np.concatenate([bundle["pipeline"].predict_proba(ctx.f.X.loc[ctx.f.day == d])[:, 1] for d in week])
        pred = (p >= bundle["threshold"]).astype(int)
        spot = week[Y % 7]                                   # one day per year through the daily interface
        pd_ = bundle["pipeline"].predict_proba(features_for_day(ctx, spot))[:, 1]
        sm = np.asarray(ctx.f.day[m] == spot)
        checks.append({"year": Y, "method": method, "day": str(spot.date()), "bitwise_equal": bool(np.array_equal(pd_, p[sm]))})
        sm_, sb = scores(t["y"], pred), scores(t["y"], t["b1"])
        ages = (week - cut).days
        rows.append({"year": Y, "method": method, "train_start": str(pd.Timestamp(start).date()), "train_end": str(cut.date()),
                     "train_rows": int(audit["train_rows"]), "tau": float(audit["threshold"]),
                     "week_start": str(week[0].date()), "week_end": str(week[-1].date()),
                     "week_start_weekday": week[0].day_name(), "age_days": f"{ages.min()}-{ages.max()}",
                     "n_slots": len(t), "class1_share": float(t["y"].mean() * 100),
                     "Acc_m": sm_["Acc"], "BA_m": sm_["BalAcc"], "Acc_b1": sb["Acc"], "BA_b1": sb["BalAcc"],
                     "BA_b7": scores(t["y"], t["b7"])["BalAcc"], "G_pp": sm_["BalAcc"] - sb["BalAcc"]})


def main():
    os.makedirs(OUT, exist_ok=True)
    ctx = Context()
    df = ctx.df_full.loc[ctx.df_full.index <= pd.Timestamp("2025-01-01 00:00")]
    rows, checks = [], []
    for Y in YEARS:
        analog_year(ctx, df, Y, rows, checks)
    tab = pd.DataFrame(rows)
    assert all(c["bitwise_equal"] for c in checks), checks
    # supplement: 37-43 days after the cutoff of each deployed model (P4 live_week_framework rules)
    log = pd.read_csv(os.path.join(RESULTS, "models_log.csv"))
    sup, seen = [], set()
    for _, l in log.iterrows():
        key = (l["model_id"] if l["model_id"] == "LR_frozen" else f"{l['strategy']}:{l['model_id']}")
        if key in seen:
            continue
        seen.add(key)
        cut = pd.Timestamp(l["cutoff"])
        w0, w1 = cut + 37 * D1, cut + 43 * D1
        if w1 > pd.Timestamp("2024-12-31") or pd.Timestamp(l["effective_from"]) > w0 or pd.Timestamp(l["effective_to"]) < w1:
            continue
        strat = "S0" if l["model_id"] == "LR_frozen" else l["strategy"]
        pr = pd.read_parquet(os.path.join(RESULTS, f"{strat.lower()}_predictions.parquet"))
        pr = pr[(pr["day"] >= w0) & (pr["day"] <= w1)]
        assert (pr["model_id"] == l["model_id"]).all() and len(pr) == 336
        t = D.truth_and_baselines(df, w0, w1)
        a, b = scores(t["y"], pr["pred"].to_numpy()), scores(t["y"], t["b1"])
        sup.append({"model_id": l["model_id"], "strategy": "all (shared frozen)" if strat == "S0" else strat, "cutoff": str(cut.date()),
                    "window_start": str(w0.date()), "window_end": str(w1.date()), "tau": float(l["tau"]),
                    "BA_m": a["BalAcc"], "BA_b1": b["BalAcc"], "G_pp": a["BalAcc"] - b["BalAcc"]})
    sup = pd.DataFrame(sup)
    summ = {"design": "results/price_lr/strategy_design.md section 12", "years": [min(YEARS), max(YEARS)],
            "Y2025": "added on 11-18 after the 2025-2026 opening (section 12.4)",
            "main": {m: summary(tab.loc[tab["method"] == m, "G_pp"]) for m in ("expanding", "sliding")},
            "BA_b1_by_year": tab[tab["method"] == "expanding"].set_index("year")["BA_b1"].round(3).to_dict(),
            "supplement_37_43": summary(sup["G_pp"]) if len(sup) else None,
            "daily_interface_spot_checks": checks,
            "limitations": ["Dec 7-13 is not Mon-Sun in other years", "n = 10, years not independent",
                            "reanalysis weather in history vs live-week inputs may differ",
                            "analog models follow the final-training method but are not the submitted model",
                            "descriptive historical analog, not a prediction interval or a live score"]}
    tab.to_csv(os.path.join(OUT, "g_analog_weeks.csv"), index=False, float_format="%.17g")
    sup.to_csv(os.path.join(OUT, "g_age37_43_deployed.csv"), index=False, float_format="%.17g")
    json.dump(summ, open(os.path.join(OUT, "g_analog_summary.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    print(tab[["year", "method", "tau", "class1_share", "BA_m", "BA_b1", "G_pp"]].round(2).to_string(index=False))
    print(sup.round(2).to_string(index=False))
    print(json.dumps({k: summ[k] for k in ("main", "supplement_37_43")}, indent=1))


def add_2025():
    """Section 12.4: Y = 2025 added after the 11-18 opening; n = 10 and n = 11 kept apart."""
    import sys
    ctx = Context()
    df = ctx.df_full.loc[ctx.df_full.index <= pd.Timestamp("2026-01-01 00:00")]
    rows, checks = [], []
    for Y in list(YEARS) + [2025]:
        analog_year(ctx, df, Y, rows, checks)
    tab = pd.DataFrame(rows)
    assert all(c["bitwise_equal"] for c in checks), checks
    old = pd.read_csv(os.path.join(OUT, "g_analog_weeks.csv"), float_precision="round_trip")
    new10 = tab[tab["year"] <= 2024].reset_index(drop=True)
    num = [c for c in old.columns if old[c].dtype.kind in "fi"]
    same10 = bool(old[num].equals(new10[num]) and old.drop(columns=num).astype(str).equals(new10[old.columns].drop(columns=num).astype(str)))
    assert same10, "2015-2024 rows differ from the 11-17 file"
    v = json.load(open(os.path.join(RESULTS, "stage3", "veto.json")))
    sel = v["final_window"]
    m10 = json.load(open(os.path.join(OUT, "g_analog_summary.json")))
    y25 = tab[tab["year"] == 2025]
    summ = {"design": "results/price_lr/strategy_design.md section 12.4", "written": "2026-11-18",
            "final_strategy": v["final_strategy"], "selected_method": sel,
            "selected": {"n10_2015_2024": summary(tab.loc[(tab["method"] == sel) & (tab["year"] <= 2024), "G_pp"]),
                         "n11_2015_2025": summary(tab.loc[tab["method"] == sel, "G_pp"])},
            "both_methods": {m: {"n10_2015_2024": summary(tab.loc[(tab["method"] == m) & (tab["year"] <= 2024), "G_pp"]),
                                 "n11_2015_2025": summary(tab.loc[tab["method"] == m, "G_pp"])} for m in ("expanding", "sliding")},
            "Y2025": y25[["method", "train_rows", "tau", "class1_share", "Acc_m", "BA_m", "Acc_b1", "BA_b1", "BA_b7", "G_pp"]].to_dict("records"),
            "BA_b1_by_year": tab[tab["method"] == "expanding"].set_index("year")["BA_b1"].round(3).to_dict(),
            "rows_2015_2024_equal_to_11_17_file": same10,
            "supplement_37_43": m10["supplement_37_43"],
            "supplement_note": "unchanged from 11-17: section 12.5 keeps windows inside 2015-2024 (n = 15)",
            "daily_interface_spot_checks_2025": [c for c in checks if c["year"] == 2025],
            "limitations": m10["limitations"] + [
                "the Y = 2025 week (2025-12-07..13) lies inside 2025-11-01..2026-10-31, which P4 used on 11-16 to select "
                "the final model's tau", "n = 11 replaces 10 in the 'years not independent' caveat"]}
    tab.to_csv(os.path.join(OUT, "g_analog_weeks_2015_2025.csv"), index=False, float_format="%.17g")
    json.dump(summ, open(os.path.join(OUT, "g_analog_summary_2015_2025.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    print(tab[tab["year"] == 2025][["year", "method", "tau", "class1_share", "BA_m", "BA_b1", "G_pp"]].round(3).to_string(index=False))
    print(json.dumps({k: summ[k] for k in ("selected_method", "selected", "both_methods", "rows_2015_2024_equal_to_11_17_file",
                                           "daily_interface_spot_checks_2025")}, indent=1))


if __name__ == "__main__":
    import sys
    add_2025() if sys.argv[1:] == ["add2025"] else main()
