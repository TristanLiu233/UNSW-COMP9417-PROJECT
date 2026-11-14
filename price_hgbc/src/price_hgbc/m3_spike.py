"""M3 price-spike groups (explanation only). Group of target day T = D+1 by the maximum NSW price of day D:
  low  : max <= 300        mid : 300 < max <= 1000        high : max > 1000   ($/MWh; boundaries fixed)
(a) 2014 only: official HGBC vs b1 (da.baseline, same rows), days, rows, true / predicted class-1 share,
    Acc, BalAcc (kit convention: mean recall over classes present), confusion matrix, per-class recall.
(b) label-only (m3_spike_label_only.csv): all complete target days 2010-01-02 .. 2014-12-31 (2010-01-01 has no
    day D in the file), days and true class-1 share per group. Data stop at 2015-01-01 00:00 (P1 load_study).
Usage: python -m src.price_hgbc.m3_spike
"""
import json
import os

import numpy as np
import pandas as pd

from . import m3_common as c
from . import official_ref as ofr
from .kit import da

BINS = [-np.inf, 300.0, 1000.0, np.inf]
LABELS = ["<=300", "(300,1000]", ">1000"]
MIN_DAYS = 30
CSV_2014 = os.path.join(c.OUT, "m3_spike_2014.csv")
CSV_LABEL = os.path.join(c.OUT, "m3_spike_label_only.csv")
JSON = os.path.join(c.OUT, "m3_spike.json")
FIG_BA = os.path.join(c.OUT, "fig_m3_spike_balacc.png")
FIG_SHARE = os.path.join(c.OUT, "fig_m3_spike_class1_share.png")


def spike_group_of_target_days(data, target_days):
    """Spike group of each target day T from max NSW price of day D = T - 1 (pd.cut, right-closed)."""
    dmax = data["nsw_price"].groupby(da.market_day(data.index)).max()
    prev = dmax.reindex(pd.DatetimeIndex(target_days) - pd.Timedelta("1D"))
    assert prev.notna().all()
    return pd.Series(pd.cut(prev.to_numpy(), BINS, labels=LABELS, right=True).astype(str),
                     index=pd.DatetimeIndex(target_days)), pd.Series(prev.to_numpy(), index=pd.DatetimeIndex(target_days))


def b1_2014(data, days):
    out = []
    for d in days:
        h, w = da.make_inputs(data, d)
        out.append(da.baseline(h, w, 1)["nsw_price_up"])
    return pd.concat(out)


def block(y, pred):
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    rec = {k: (float(np.mean(pred[y == k] == k) * 100) if np.any(y == k) else np.nan) for k in (0, 1)}
    return {"Acc": da.accuracy(y, pred), "BalAcc": da.balanced_accuracy(y, pred), "recall_0": rec[0], "recall_1": rec[1],
            "tn": int(((y == 0) & (pred == 0)).sum()), "fp": int(((y == 0) & (pred == 1)).sum()),
            "fn": int(((y == 1) & (pred == 0)).sum()), "tp": int(((y == 1) & (pred == 1)).sum()),
            "classes_present": int(len(np.unique(y))), "pred_class1_share": float(pred.mean() * 100)}


def run():
    r = c.ref()
    data, X, truth, cols, tr, va = c.load()
    official = pd.read_parquet(ofr.PREDS)
    idx = official.index
    days14 = sorted(set(da.market_day(idx)))
    grp14, dmax14 = spike_group_of_target_days(data, days14)
    row_group = grp14.reindex(da.market_day(idx)).to_numpy()
    b1 = b1_2014(data, days14)
    assert b1.index.equals(idx)
    y, p_m, p_b = official["y"].to_numpy(), official["pred"].to_numpy(), b1.to_numpy()
    rows = []
    for g in LABELS + ["all"]:
        m = np.ones(len(y), bool) if g == "all" else row_group == g
        n_days = len(days14) if g == "all" else int((grp14 == g).sum())
        hg, bb = block(y[m], p_m[m]), block(y[m], p_b[m])
        rows.append({"group": g, "days": n_days, "rows": int(m.sum()), "true_class1_share": float(y[m].mean() * 100),
                     "classes_present": hg["classes_present"],
                     "HGBC_pred_class1_share": hg["pred_class1_share"], "b1_pred_class1_share": bb["pred_class1_share"],
                     **{f"HGBC_{k}": v for k, v in hg.items() if k not in ("classes_present", "pred_class1_share")},
                     **{f"b1_{k}": v for k, v in bb.items() if k not in ("classes_present", "pred_class1_share")},
                     "warn_lt_30_days": bool(g != "all" and n_days < MIN_DAYS)})
    t14 = pd.DataFrame(rows)
    # (b) label only, 2010-2014
    all_days = [d for d in da.complete_days(data) if pd.Timestamp("2010-01-02") <= d <= pd.Timestamp("2014-12-31")]
    grp_all, dmax_all = spike_group_of_target_days(data, all_days)
    lab = ofr.official_labels(data, all_days)
    lab_day = lab.groupby(np.asarray(da.market_day(lab.index))).mean() * 100
    lrows = []
    for g in LABELS + ["all"]:
        dd = grp_all.index if g == "all" else grp_all.index[grp_all == g]
        lrows.append({"group": g, "days": int(len(dd)), "rows": int(len(dd) * 48),
                      "true_class1_share": float(lab_day.loc[dd].mean()) if len(dd) else np.nan,
                      "first_day": str(min(dd).date()) if len(dd) else "", "last_day": str(max(dd).date()) if len(dd) else "",
                      "warn_lt_30_days": bool(g != "all" and len(dd) < MIN_DAYS)})
    tl = pd.DataFrame(lrows)
    high = tl.set_index("group").loc[">1000"]
    rest = lab_day.loc[grp_all.index[grp_all != ">1000"]].mean()
    meta = {"boundaries": "<=300, (300,1000], >1000 $/MWh on max NSW price of day D = target - 1",
            "label_only_period": [str(all_days[0].date()), str(all_days[-1].date())],
            "excluded_label_only": ["2010-01-01 (no day D in the file)"],
            "claim_9_7pct": {"source": "00_overview/explore_output.txt §6b, computed over 2010-2026 (192 days)",
                             "this_check_period": "2010-01-02 .. 2014-12-31 (2015+ not read in M3)",
                             "high_group_days": int(high["days"]), "high_group_class1_share": float(high["true_class1_share"]),
                             "other_days_class1_share": float(rest), "matches_9_7_at_0_1pp": bool(abs(high["true_class1_share"] - 9.7) < 0.05)},
            "kit_convention": "BalAcc = mean recall over the classes present; with one class it is that class's recall",
            "note": "explanatory only - not used to re-select the model"}
    return t14, tl, meta, {"grp14": grp14, "dmax14": dmax14, "grp_all": grp_all}


def plot_balacc(t14):
    plt = c.mpl()
    t = t14[t14["group"] != "all"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.5, 4), facecolor=c.SURFACE)
    c.style(ax)
    ax.grid(axis="y", color=c.GRID, linewidth=0.8)
    x = np.arange(len(t))
    w = 0.36
    for off, col, lab, key in ((-w / 2, c.BLUE, "HGBC (boost_price)", "HGBC_BalAcc"), (w / 2, c.ORANGE, "b1", "b1_BalAcc")):
        ax.bar(x + off, t[key], width=w - 0.04, color=col, label=lab)
        for xi, v in zip(x + off, t[key]):
            ax.annotate(f"{v:.1f}", (xi, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=c.INK2)
    ax.set_xticks(x, [f"{g}\n({d} day{'s' if d != 1 else ''}{', WARN <30' if d < MIN_DAYS else ''})"
                      for g, d in zip(t["group"], t["days"])], color=c.INK2)
    ax.set_ylabel("BalAcc (%)", color=c.INK2)
    ax.set_xlabel("Max NSW price on day D ($/MWh)", color=c.INK2)
    ax.set_title("2014: HGBC vs b1 by previous-day price spike", color=c.INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=c.INK2)
    fig.tight_layout()
    return fig, ax


def plot_share(t14, tl):
    plt = c.mpl()
    a = t14[t14["group"] != "all"].reset_index(drop=True)
    b = tl[tl["group"] != "all"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.5, 4), facecolor=c.SURFACE)
    c.style(ax)
    ax.grid(axis="y", color=c.GRID, linewidth=0.8)
    x = np.arange(len(a))
    w = 0.36
    for off, col, lab, df in ((-w / 2, c.BLUE, "2014", a), (w / 2, c.ORANGE, "2010–2014 (label only)", b)):
        ax.bar(x + off, df["true_class1_share"], width=w - 0.04, color=col, label=lab)
        for xi, v in zip(x + off, df["true_class1_share"]):
            ax.annotate(f"{v:.1f}", (xi, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=c.INK2)
    ax.set_xticks(x, [f"{g}\n2014: {d1} d · 2010–14: {d2} d" for g, d1, d2 in zip(a["group"], a["days"], b["days"])],
                  color=c.INK2, fontsize=8)
    ax.set_ylabel("True class-1 share of half-hours (%)", color=c.INK2)
    ax.set_xlabel("Max NSW price on day D ($/MWh)", color=c.INK2)
    ax.set_title("Share of 'price up' half-hours after a spike day", color=c.INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=c.INK2)
    fig.tight_layout()
    return fig, ax


def main():
    t14, tl, meta, _ = run()
    os.makedirs(c.OUT, exist_ok=True)
    t14.to_csv(CSV_2014, index=False, float_format="%.10g")
    tl.to_csv(CSV_LABEL, index=False, float_format="%.10g")
    json.dump(meta, open(JSON, "w"), indent=1)
    t14r, tlr = pd.read_csv(CSV_2014), pd.read_csv(CSV_LABEL)
    fig, _ = plot_balacc(t14r)
    fig.savefig(FIG_BA, dpi=150, facecolor=c.SURFACE)
    fig, _ = plot_share(t14r, tlr)
    fig.savefig(FIG_SHARE, dpi=150, facecolor=c.SURFACE)
    pd.set_option("display.width", 250)
    print(t14.to_string(index=False))
    print(tl.to_string(index=False))
    print(json.dumps(meta["claim_9_7pct"], indent=1))


if __name__ == "__main__":
    main()
