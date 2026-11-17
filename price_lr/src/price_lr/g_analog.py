"""M7 step 4: historical analog of the live-week gain G = BA_m - BA_b1 (design section 12; report section 5 only).

Main: for Y = 2015..2024 train with the final-training method at cutoff Y-10-31 and predict Y-12-07..Y-12-13
(model age 37-43 days, as for the live week), one 48-row predict call per day as in the package. Both methods: expanding (P4 fit_final) and 3-year sliding
(fit_sliding, start = (Y-3)-11-01). Supplement: 37-43-day windows of models actually deployed in S1-S3.
Not used for selecting any strategy. Nothing after 2024-12-31 is read for scoring.
Usage: python -m src.price_lr.g_analog
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


def main():
    os.makedirs(OUT, exist_ok=True)
    ctx = Context()
    df = ctx.df_full.loc[ctx.df_full.index <= pd.Timestamp("2025-01-01 00:00")]
    rows, checks = [], []
    for Y in YEARS:
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


if __name__ == "__main__":
    main()
