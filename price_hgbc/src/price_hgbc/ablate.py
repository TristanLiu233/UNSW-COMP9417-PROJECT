"""M3 leave-one-group-out ablation of the official boost_price (explanation only; never used to re-select).

Every row refits REF's estimator (REF params verbatim) on 2010-01-29 .. 2013-12-31 with one feature group
removed, then scores 2014:
  (a) main result: fixed tau = REF tau (0.45) -> BalAcc, Acc;
  (b) optimistic: tau re-picked ON 2014 with P1's protocol (ablation/tuning.py: thresholds 0.40, 0.45, 0.50,
      0.55, 0.60; highest BalAcc, ties -> closest to 0.5). Uses the evaluation year, hence optimistic.
delta_BalAcc_pp = ablated - full (fixed tau). Paired whole-day bootstrap: n_boot 1000, seed 0, ONE index matrix
shared by all rows. The full_features row must reproduce official/preds_2014.parquet bitwise, otherwise stop.
Usage: python -m src.price_hgbc.ablate [--fingerprint f.json]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import m3_common as c
from . import official_ref as ofr
from .kit import da

P1_THRESHOLDS = (0.40, 0.45, 0.50, 0.55, 0.60)
CSV = os.path.join(c.OUT, "m3_ablation.csv")
JSON = os.path.join(c.OUT, "m3_ablation.json")
PREDS = os.path.join(c.OUT, "m3_ablation_proba_2014.parquet")
FIG = os.path.join(c.OUT, "fig_m3_ablation.png")


def p1_retune(y, proba):
    """P1's threshold rule on the given rows: max BalAcc, ties -> closest to 0.5 (as in ablation/tuning.py)."""
    rows = [(da.balanced_accuracy(y, (proba >= t).astype(int)), t) for t in P1_THRESHOLDS]
    best = min(rows, key=lambda r: (-r[0], abs(round(100 * r[1]) - 50)))
    return best[1], best[0], da.accuracy(y, (proba >= best[1]).astype(int))


def run():
    r = c.ref()
    data, X, truth, cols, tr, va = c.load()
    assert cols == r["features"]
    ytr, yva = truth.loc[tr, "nsw_price_up"], truth.loc[va, "nsw_price_up"].to_numpy()
    idx = X.index[va]
    tau = r["tau"]
    plan = [("full_features", "完整模型", [])] + [(g, zh, gc) for g, (zh, gc) in c.GROUPS.items()]
    probas, rows = {}, []
    for name, zh, drop in plan:
        keep = [x for x in cols if x not in drop]
        model = HistGradientBoostingClassifier(**r["params"]).fit(X.loc[tr, keep], ytr)
        p = model.predict_proba(X.loc[va, keep])[:, 1]
        probas[name] = p
        pred = (p >= tau).astype("int64")
        t2, b2, a2 = p1_retune(yva, p)
        rows.append({"row": name, "group_zh": zh, "dropped": json.dumps(drop), "n_dropped": len(drop),
                     "n_features": len(keep), "tau_fixed": tau, "BalAcc": da.balanced_accuracy(yva, pred),
                     "Acc": da.accuracy(yva, pred),
                     "recall_0": float(np.mean(pred[yva == 0] == 0) * 100), "recall_1": float(np.mean(pred[yva == 1] == 1) * 100),
                     "tau_2014_optimistic": t2, "BalAcc_2014_optimistic": b2, "Acc_2014_optimistic": a2,
                     "proba_sha256": c.sha_arr(p, "float64"), "pred_sha256": c.sha_arr(pred, "int64")})
    # stop if the full model is not the official one
    official = pd.read_parquet(ofr.PREDS)
    if not (np.array_equal(official["proba"].to_numpy(), probas["full_features"]) and official.index.equals(idx)):
        raise RuntimeError("full_features does not reproduce official/preds_2014.parquet bitwise - stop")
    tab = pd.DataFrame(rows)
    full = tab.iloc[0]
    days = sorted(set(da.market_day(idx)))
    B = c.boot_indices(len(days))
    s_full = c.boot_balacc(yva, (probas["full_features"] >= tau).astype(int), idx, B)
    lo, hi, deltas, verdict = [], [], [], []
    for _, row in tab.iterrows():
        d = row["BalAcc"] - full["BalAcc"]
        s = c.boot_balacc(yva, (probas[row["row"]] >= tau).astype(int), idx, B) - s_full
        a, b = c.ci(s)
        deltas.append(d), lo.append(a), hi.append(b)
        if row["row"] == "full_features":
            verdict.append("reference")
        elif a <= 0 <= b:
            verdict.append("未检测到显著贡献 (no significant contribution detected)")
        elif b < 0:
            verdict.append("removing it lowers BalAcc (group contributes)")
        else:
            verdict.append("removing it raises BalAcc")
    tab["delta_BalAcc_pp"], tab["delta_lo"], tab["delta_hi"], tab["verdict"] = deltas, lo, hi, verdict
    tab["delta_BalAcc_optimistic_pp"] = tab["BalAcc_2014_optimistic"] - full["BalAcc_2014_optimistic"]
    meta = {"n_boot": c.N_BOOT, "seed": c.BOOT_SEED, "n_days": len(days),
            "boot_index_sha256": c.sha(np.ascontiguousarray(B).tobytes()),
            "train": r["train"], "valid": r["valid"], "params": r["params"], "tau": tau,
            "p1_thresholds_for_optimistic": list(P1_THRESHOLDS),
            "full_equals_official_parquet": True, "note": "explanatory only - not used to re-select the model"}
    return tab, meta, pd.DataFrame(probas, index=idx)


def plot(tab):
    plt = c.mpl()
    t = tab[tab["row"] != "full_features"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 3.8), facecolor=c.SURFACE)
    c.style(ax)
    ax.grid(axis="x", color=c.GRID, linewidth=0.8)
    y = list(range(len(t)))
    ax.hlines(y, t["delta_lo"], t["delta_hi"], color=c.GRAY, linewidth=2)
    ax.scatter(t["delta_BalAcc_pp"], y, s=44, color=c.ORANGE, zorder=3, edgecolors=c.SURFACE, linewidths=1.5)
    ax.axvline(0, color=c.INK2, linewidth=1)
    for yi, v in zip(y, t["delta_BalAcc_pp"]):
        ax.annotate(f"{v:+.2f}", (v, yi), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, color=c.INK2)
    ax.set_yticks(y, [f"drop {g}" for g in t["row"]], color=c.INK2)
    ax.set_xlabel("BalAcc change vs full model (pp), τ = 0.45; 95% paired day-bootstrap CI", color=c.INK2)
    ax.set_title("Leave-one-group-out ablation, 2014", color=c.INK, fontsize=11, loc="left")
    fig.tight_layout()
    return fig, ax


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fingerprint")
    a = ap.parse_args(argv)
    tab, meta, probas = run()
    if a.fingerprint:
        json.dump({"table": tab.to_dict(orient="list"), "meta": meta}, open(a.fingerprint, "w"), indent=1, default=str)
        return
    os.makedirs(c.OUT, exist_ok=True)
    tab.to_csv(CSV, index=False, float_format="%.10g")
    json.dump(meta, open(JSON, "w"), indent=1, default=str)
    probas.to_parquet(PREDS)
    fig, _ = plot(pd.read_csv(CSV))
    fig.savefig(FIG, dpi=150, facecolor=c.SURFACE)
    print(tab[["row", "n_dropped", "BalAcc", "Acc", "delta_BalAcc_pp", "delta_lo", "delta_hi",
               "tau_2014_optimistic", "BalAcc_2014_optimistic", "verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
