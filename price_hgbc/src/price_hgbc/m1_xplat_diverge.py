"""Locate the first divergence between two dumped tuned_C models (from m1_xplat_trees.py).

    python -m src.price_hgbc.m1_xplat_diverge macos-arm64 linux-amd64
Writes xplat/diverge_<A>__<B>.json. Layers compared, in order:
  1 bin edges  2 baseline  3 tree count  4 per-tree structure/values  5 first differing node (field-level)
  6 probabilities (max / median |diff|, rows affected)  7 labels at tau = 0.45.
If trees differ, `split_candidates` recomputes, in THIS environment, every split candidate at the
first differing node (using this env's own first k trees, which are identical to the other env's
by construction) and reports where the other env's choice ranks and by how much gain it lost.
That measures closeness of the decision; it is not by itself proof of why the other env chose it.
"""
import json
import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from .kit import RESULTS  # noqa: E402

XDIR = os.path.join(RESULTS, "selfcheck", "xplat")
STRUCT = ["feature_idx", "bin_threshold", "num_threshold", "left", "right", "is_leaf", "missing_go_to_left"]
NUMERIC = ["value", "gain", "count", "depth"]


def load(env):
    z = np.load(os.path.join(XDIR, f"trees_{env}.npz"), allow_pickle=False)
    return {k: z[k] for k in z.files}


def tree(d, i):
    a, b = d["offsets"][i], d["offsets"][i + 1]
    return {f[5:]: d[f][a:b] for f in d if f.startswith("node_")}


def first_node_diff(ta, tb):
    n = min(len(ta["value"]), len(tb["value"]))
    for j in range(n):
        diff = [f for f in STRUCT + NUMERIC if f in ta and not np.array_equal(ta[f][j], tb[f][j])]
        if diff:
            return j, diff
    return (n, ["n_nodes"]) if len(ta["value"]) != len(tb["value"]) else (None, [])


def _f(x):
    x = np.asarray(x).item()
    return float(x) if isinstance(x, float) else int(x)


def compare(env_a, env_b):
    A, B = load(env_a), load(env_b)
    out = {"a": env_a, "b": env_b}
    out["bin_edges_identical"] = bool(np.array_equal(A["bin_lengths"], B["bin_lengths"])
                                      and np.array_equal(A["bin_edges"], B["bin_edges"]))
    out["baseline_identical"] = bool(np.array_equal(A["baseline"], B["baseline"]))
    na, nb = len(A["offsets"]) - 1, len(B["offsets"]) - 1
    out["n_trees"] = [int(na), int(nb)]
    first = None
    for i in range(min(na, nb)):
        j, fields = first_node_diff(tree(A, i), tree(B, i))
        if j is not None:
            first = (i, j, fields)
            break
    if first is None:
        out["trees_identical"] = True
    else:
        i, j, fields = first
        ta, tb = tree(A, i), tree(B, i)
        out.update(trees_identical=False, first_tree=int(i), first_node=int(j), differing_fields=fields,
                   kind=("split (feature/threshold/structure)" if set(fields) & set(STRUCT)
                         else "leaf/node numeric values only"),
                   node_a={f: _f(ta[f][j]) for f in STRUCT + NUMERIC if f in ta and j < len(ta[f])},
                   node_b={f: _f(tb[f][j]) for f in STRUCT + NUMERIC if f in tb and j < len(tb[f])},
                   identical_trees_before=int(i))
        # parent of node j (same in both, since all earlier nodes are identical)
        parent = [p for p in range(j) if ta["left"][p] == j or ta["right"][p] == j]
        out["parent_node"] = int(parent[0]) if parent else None
        if not ta["is_leaf"][j] and j < len(tb["is_leaf"]) and not tb["is_leaf"][j]:
            out["value_diff_ulps_at_node"] = None
        elif ta["is_leaf"][j] and tb["is_leaf"][j]:
            out["value_diff_ulps_at_node"] = int(abs(np.float64(ta["value"][j]).view(np.int64)
                                                     - np.float64(tb["value"][j]).view(np.int64)))
    pa, pb = A["proba"], B["proba"]
    d = np.abs(pa - pb)
    la, lb = pa >= 0.45, pb >= 0.45
    out["proba"] = {"identical": bool(np.array_equal(pa, pb)), "max_abs_diff": float(d.max()),
                    "median_abs_diff": float(np.median(d)), "rows_differing": int((d > 0).sum()),
                    "median_abs_diff_among_differing": float(np.median(d[d > 0])) if (d > 0).any() else 0.0,
                    "labels_differing_at_0.45": int((la != lb).sum()),
                    "flips_a1_b0": int((la & ~lb).sum()), "flips_a0_b1": int((~la & lb).sum())}
    return out


def split_candidates(env_local, env_other, report):
    """Recompute all candidate splits at the first differing node, in the local environment."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from .m1_xplat_trees import DATA_CSV, KIT, da, gf
    import pandas as pd
    if report.get("trees_identical", True) or report["parent_node"] is None:
        return None
    k, j = report["first_tree"], report["first_node"]
    data = gf.load_study(DATA_CSV)
    X, truth = gf.build_features(data)
    days = da.market_day(data.index)
    tr = (days >= "2010-01-29") & (days < "2014-01-01")
    s = pd.read_csv(os.path.join(KIT, "ablation", "results", "scores.csv"))
    cols = json.loads(s[(s.model == "boost_price") & (s.config == "tuned_C")]["columns"].iloc[0])
    Xtr, y = X.loc[tr, cols], truth.loc[tr, "nsw_price_up"].to_numpy().astype(np.float64)
    params = dict(loss="log_loss", max_leaf_nodes=15, learning_rate=0.1, early_stopping=False, random_state=17)
    if k > 0:
        mk = HistGradientBoostingClassifier(max_iter=k, **params).fit(Xtr, y)
        raw = mk._raw_predict(Xtr).ravel()
        binner = mk._bin_mapper
    else:
        mk = HistGradientBoostingClassifier(max_iter=1, **params).fit(Xtr, y)
        raw = np.full(len(y), mk._baseline_prediction.ravel()[0])
        binner = mk._bin_mapper
    p = 1.0 / (1.0 + np.exp(-raw))
    g = (p - y).astype(np.float32).astype(np.float64)
    h = (p * (1 - p)).astype(np.float32).astype(np.float64)
    Xb = binner.transform(Xtr)
    A = load(env_local)
    t = tree(A, k)
    # samples reaching node j: follow the (identical) path from the root
    path, node = [], j
    while node != 0:
        par = int(np.flatnonzero((t["left"] == node) | (t["right"] == node))[0])
        path.append((par, t["left"][par] == node))
        node = par
    mask = np.ones(len(y), bool)
    for par, went_left in reversed(path):
        f, b = int(t["feature_idx"][par]), int(t["bin_threshold"][par])
        mask &= (Xb[:, f] <= b) if went_left else (Xb[:, f] > b)
    G, H, n = g[mask].sum(), h[mask].sum(), int(mask.sum())
    cands = []
    for f in range(Xb.shape[1]):
        xb = Xb[mask, f]
        gs = np.bincount(xb, weights=g[mask], minlength=256)
        hs = np.bincount(xb, weights=h[mask], minlength=256)
        cs = np.bincount(xb, minlength=256)
        GL, HL, CL = np.cumsum(gs)[:-1], np.cumsum(hs)[:-1], np.cumsum(cs)[:-1]
        ok = (CL >= 20) & (n - CL >= 20) & (HL >= 1e-3) & (H - HL >= 1e-3)
        gain = GL ** 2 / HL + (G - GL) ** 2 / (H - HL) - G ** 2 / H
        for b in np.flatnonzero(ok):
            cands.append((float(gain[b]), f, int(b)))
    cands.sort(reverse=True)
    na, nb = report["node_a"], report["node_b"]

    def rank(fe, bt):
        for r, (gn, f, b) in enumerate(cands):
            if f == fe and b == bt:
                return r, gn
        return None, None
    ra, ga = rank(na.get("feature_idx"), na.get("bin_threshold"))
    rb, gb = rank(nb.get("feature_idx"), nb.get("bin_threshold"))
    return {"note": f"recomputed in {env_local}; numpy float64 sums of float32 g/h (approximates sklearn's histograms)",
            "node_samples": n, "top5": [{"gain": c[0], "feature": cols[c[1]], "feature_idx": c[1], "bin": c[2]}
                                        for c in cands[:5]],
            "choice_a": {"feature": cols[na["feature_idx"]] if not na.get("is_leaf") else "leaf",
                         "bin": na.get("bin_threshold"), "rank": ra, "gain": ga},
            "choice_b": {"feature": cols[nb["feature_idx"]] if not nb.get("is_leaf") else "leaf",
                         "bin": nb.get("bin_threshold"), "rank": rb, "gain": gb},
            "relative_gain_gap": (abs(ga - gb) / abs(ga)) if (ga and gb) else None}


if __name__ == "__main__":
    a, b = sys.argv[1], sys.argv[2]
    rep = compare(a, b)
    if not rep.get("trees_identical", True):
        try:
            rep["split_candidates"] = split_candidates(a, b, rep)
        except Exception as exc:  # noqa: BLE001
            rep["split_candidates"] = {"error": repr(exc)}
    path = os.path.join(XDIR, f"diverge_{a}__{b}.json")
    json.dump(rep, open(path, "w"), indent=1, default=str)
    print(json.dumps(rep, indent=1, default=str))
