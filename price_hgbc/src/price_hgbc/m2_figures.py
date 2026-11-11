"""M2 figures from the saved M2 outputs (no re-fitting). Usage: python -m src.price_hgbc.m2_figures"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .m2 import OUT_DIR, paths  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a3a29c"   # validated pair (light surface)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_grid(P):
    r = pd.read_csv(P["ranked"]).sort_values("val_BalAcc")
    g = pd.read_csv(P["grid"]).set_index("config_id")
    best = json.load(open(P["best"]))["best_config_id"]
    ref = g.loc["M1_ref", "val_BalAcc"]
    fig, ax = plt.subplots(figsize=(7.5, 9), facecolor=SURFACE)
    _style(ax)
    y = range(len(r))
    colors = [ORANGE if c == best else GRAY for c in r["config_id"]]
    ax.scatter(r["val_BalAcc"], list(y), s=36, c=colors, zorder=3, edgecolors=SURFACE, linewidths=1.5)
    ax.axvline(ref, color=BLUE, linewidth=2)
    ax.axvline(ref + 0.5, color=BLUE, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(ref, len(r) + 0.3, " M1_ref", color=INK2, fontsize=9, va="bottom")
    ax.text(ref + 0.5, len(r) + 0.3, "M1_ref + 0.5 pp ", color=INK2, fontsize=9, va="bottom", ha="right")
    ax.set_yticks(list(y), r["config_id"], fontsize=7, color=INK2)
    ax.set_xlabel("2013 pooled-fold BalAcc (%) at each config's tuned τ", color=INK2)
    ax.set_title("M2 grid: 48 HGBC configs on the M1 features (best in orange)", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_m2_grid_2013.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_tau(P):
    c = pd.read_csv(P["curves"])
    best = json.load(open(P["best"]))["best_config_id"]
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for cid, col, lab in ((best, ORANGE, f"M2 best ({best.split('_', 1)[0]})"), ("M1_ref", BLUE, "M1_ref")):
        cc = c[c["config_id"] == cid]
        ax.plot(cc["tau"], cc["BalAcc"], color=col, linewidth=2, label=lab)
        i = cc["BalAcc"].idxmax()
        ax.scatter([cc.loc[i, "tau"]], [cc.loc[i, "BalAcc"]], s=64, color=col, edgecolors=SURFACE, linewidths=2, zorder=4)
        ax.annotate(f"{lab}: τ={cc.loc[i, 'tau']:.2f}", (cc.loc[i, "tau"], cc.loc[i, "BalAcc"]),
                    xytext=(8, 6), textcoords="offset points", color=INK2, fontsize=9)
    ax.set_xlabel("τ (predict 1 iff p ≥ τ)", color=INK2)
    ax.set_ylabel("2013 BalAcc (%)", color=INK2)
    ax.set_title("τ search on 2013 (41 points, 0.30–0.70)", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_m2_tau_2013.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_boot(P):
    b = pd.read_csv(P["boot"])
    b = b[(b["b"] == "M1_ref") | (b["part"] == "2a")].copy()
    b["label"] = [f"{r.part}: {r.a.split('_lr')[0] if r.a.startswith('g') else r.a} − {r.b}" for r in b.itertuples()]
    b = b.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.4), facecolor=SURFACE)
    _style(ax)
    y = range(len(b))
    ax.hlines(list(y), b["diff_lo"], b["diff_hi"], color=GRAY, linewidth=2)
    ax.scatter(b["diff_pp"], list(y), s=40, color=[ORANGE if p != "2014 confirmation" else BLUE for p in b["part"]],
               zorder=3, edgecolors=SURFACE, linewidths=1.5)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.axvline(0.5, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(0.5, -0.9, " 0.5 pp adoption bar", color=INK2, fontsize=9, va="center")
    ax.set_ylim(-1.3, len(b) - 0.5)
    ax.set_yticks(list(y), b["label"], fontsize=8, color=INK2)
    ax.set_xlabel("BalAcc difference (pp), paired day bootstrap 95 % CI, n=1000, seed 0", color=INK2, fontsize=9)
    ax.set_title("Gain over M1 (orange: 2013 decision, blue: 2014 check)", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_m2_bootstrap.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    P = paths()
    fig_grid(P)
    fig_tau(P)
    fig_boot(P)
    print("saved figures to", OUT_DIR)


if __name__ == "__main__":
    main()
