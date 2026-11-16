"""M3 permutation importance of the official boost_price on 2014 (no refitting; explanation only).

Scorer: pred = 1[predict_proba[:, 1] >= REF tau (0.45)], scored with da.balanced_accuracy (never sklearn's 0.5 cut).
  * per column: sklearn.inspection.permutation_importance(n_repeats=10, random_state=0) with that scorer;
  * per group: the same RNG scheme as sklearn (one seed drawn from RandomState(0); one row-index array shuffled
    in place per repeat) but the SAME row permutation is applied to every column of the group. On single-column
    groups this reproduces sklearn's per-column numbers exactly (tested).
importance = baseline score - permuted score (pp of BalAcc).
Usage: python -m src.price_hgbc.m3_permutation
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from . import m3_common as c
from . import official_ref as ofr
from .kit import da

N_REPEATS, RANDOM_STATE = 10, 0
COL_CSV = os.path.join(c.OUT, "m3_permutation_columns.csv")
GRP_CSV = os.path.join(c.OUT, "m3_permutation_groups.csv")
JSON = os.path.join(c.OUT, "m3_permutation.json")
FIG_COL = os.path.join(c.OUT, "fig_m3_permutation_columns.png")
FIG_GRP = os.path.join(c.OUT, "fig_m3_permutation_groups.png")


def make_scorer(tau):
    def scorer(estimator, X, y):
        return da.balanced_accuracy(y, (estimator.predict_proba(X)[:, 1] >= tau).astype(int))
    return scorer


def group_permutation(model, X, y, scorer, groups, n_repeats=N_REPEATS, random_state=RANDOM_STATE):
    """{group: array(n_repeats) of permuted scores}; same row permutation for all columns of a group."""
    seed = np.random.RandomState(random_state).randint(np.iinfo(np.int32).max + 1)   # as sklearn
    out = {}
    for g, cols in groups.items():
        rng = np.random.RandomState(seed)
        Xp = X.copy()
        idx = np.arange(len(Xp))
        scores = []
        for _ in range(n_repeats):
            rng.shuffle(idx)
            for col in cols:
                v = Xp[col].iloc[idx]
                v.index = Xp.index
                Xp[col] = v
            scores.append(scorer(model, Xp, y))
        out[g] = np.array(scores)
    return out


def run():
    r = c.ref()
    _, X, truth, cols, _, va = c.load()
    Xva, yva = X.loc[va, cols], truth.loc[va, "nsw_price_up"].to_numpy()
    model = joblib.load(ofr.MODEL)
    scorer = make_scorer(r["tau"])
    base = scorer(model, Xva, yva)
    assert base == r["metrics_2014"]["BalAcc"], (base, r["metrics_2014"]["BalAcc"])
    pi = permutation_importance(model, Xva, yva, scoring=scorer, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    fg = c.feature_groups_table(r).set_index("feature")
    col = pd.DataFrame({"feature": cols, "group": [fg.loc[x, "group"] for x in cols],
                        "importance_mean_pp": pi.importances_mean, "importance_std_pp": pi.importances_std,
                        "single_column_note": [fg.loc[x, "single_column_note"] for x in cols]})
    for k in range(N_REPEATS):
        col[f"rep{k}"] = pi.importances[:, k]
    col["interpret_alone"] = col["single_column_note"] == ""
    col = col.sort_values("importance_mean_pp", ascending=False).reset_index(drop=True)
    col.insert(0, "rank", range(1, len(col) + 1))
    gp = group_permutation(model, Xva, yva, scorer, {g: v[1] for g, v in c.GROUPS.items()})
    grp = pd.DataFrame({"group": list(gp), "group_zh": [c.GROUPS[g][0] for g in gp],
                        "n_columns": [len(c.GROUPS[g][1]) for g in gp],
                        "importance_mean_pp": [base - s.mean() for s in gp.values()],
                        "importance_std_pp": [(base - s).std() for s in gp.values()]})
    for k in range(N_REPEATS):
        grp[f"rep{k}"] = [base - s[k] for s in gp.values()]
    grp = grp.sort_values("importance_mean_pp", ascending=False).reset_index(drop=True)
    grp.insert(0, "rank", range(1, len(grp) + 1))
    meta = {"baseline_BalAcc": base, "REF_BalAcc": r["metrics_2014"]["BalAcc"], "tau": r["tau"],
            "n_repeats": N_REPEATS, "random_state": RANDOM_STATE, "rows": int(len(yva)), "refit": False,
            "model_file_sha256": ofr.sha_file(ofr.MODEL), "note": "explanatory only - not used to re-select the model"}
    return col, grp, meta


def plot(df, label_col, title, xlabel, colour_flag=None):
    plt = c.mpl()
    t = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(t) + 1.4), facecolor=c.SURFACE)
    c.style(ax)
    ax.grid(axis="x", color=c.GRID, linewidth=0.8)
    y = list(range(len(t)))
    colours = [c.BLUE if (colour_flag is None or f) else c.GRAY
               for f in (t[colour_flag] if colour_flag else [True] * len(t))]
    ax.barh(y, t["importance_mean_pp"], height=0.6, color=colours, xerr=t["importance_std_pp"],
            error_kw={"ecolor": c.INK2, "elinewidth": 1})
    for yi, v, sd in zip(y, t["importance_mean_pp"], t["importance_std_pp"]):
        ax.annotate(f"{v:.2f}", (max(v, 0) + sd, yi), xytext=(5, 0), textcoords="offset points", va="center",
                    fontsize=8, color=c.INK2)
    ax.set_yticks(y, list(t[label_col]), color=c.INK2)
    ax.axvline(0, color=c.INK2, linewidth=1)
    ax.set_xlabel(xlabel, color=c.INK2)
    ax.set_title(title, color=c.INK, fontsize=11, loc="left")
    fig.tight_layout()
    return fig, ax


def plot_columns(col):
    return plot(col, "feature", "Permutation importance by column, 2014 (grey: do not read alone)",
                "BalAcc drop when the column is permuted (pp), mean ± sd of 10 repeats, τ = 0.45", "interpret_alone")


def plot_groups(grp):
    return plot(grp, "group", "Permutation importance by feature group, 2014",
                "BalAcc drop when the group is permuted jointly (pp), mean ± sd of 10 repeats, τ = 0.45")


def main():
    col, grp, meta = run()
    os.makedirs(c.OUT, exist_ok=True)
    col.to_csv(COL_CSV, index=False, float_format="%.10g")
    grp.to_csv(GRP_CSV, index=False, float_format="%.10g")
    json.dump(meta, open(JSON, "w"), indent=1)
    fig, _ = plot_columns(pd.read_csv(COL_CSV))
    fig.savefig(FIG_COL, dpi=150, facecolor=c.SURFACE)
    fig, _ = plot_groups(pd.read_csv(GRP_CSV))
    fig.savefig(FIG_GRP, dpi=150, facecolor=c.SURFACE)
    print(meta)
    print(col[["rank", "feature", "group", "importance_mean_pp", "importance_std_pp", "interpret_alone"]].to_string(index=False))
    print(grp[["rank", "group", "n_columns", "importance_mean_pp", "importance_std_pp"]].to_string(index=False))


if __name__ == "__main__":
    main()
