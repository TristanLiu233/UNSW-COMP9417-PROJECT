"""Build the three-environment comparison from results/price_hgbc/selfcheck/xplat/*.json.

Usage: python -m src.price_hgbc.m1_xplat_compare
Writes xplat/xplat_comparison.csv, xplat/xplat_trees.csv and xplat/xplat_checks.json.
"""
import glob
import json
import os

import pandas as pd

from .kit import RESULTS

XDIR = os.path.join(RESULTS, "selfcheck", "xplat")
ENVS = ["macos-arm64", "linux-arm64", "linux-amd64"]


def load():
    runs = {}
    for p in sorted(glob.glob(os.path.join(XDIR, "xplat_*_run*.json"))):
        d = json.load(open(p))
        runs[(d["env_label"], d["run"])] = d
    return runs


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, [k for k in x if x[k] != y[k]]
    return None, []


def build():
    runs = load()
    envs = [e for e in ENVS if (e, 1) in runs]
    ref = runs[(envs[0], 1)]
    checks = {"envs": envs, "runs": sorted(f"{e}#{k}" for e, k in runs)}
    # A. inputs and params identical everywhere
    checks["inputs_identical"] = all(runs[k]["inputs"] == ref["inputs"] for k in runs)
    checks["params_identical"] = all(
        [c["params"] for c in runs[k]["configs"]] == [c["params"] for c in ref["configs"]] for k in runs)
    checks["kernels_by_env"] = {e: runs[(e, 1)]["kernels"] for e in envs}
    checks["rs0_equals_rs17"] = {f"{e}#{k}": runs[(e, k)]["tuned_C_random_state_0"] for e, k in runs}
    # B. within-environment repeatability
    checks["repeat_identical"] = {}
    for e in envs:
        if (e, 2) in runs:
            a, b = runs[(e, 1)]["configs"], runs[(e, 2)]["configs"]
            checks["repeat_identical"][e] = all(
                x["pred_sha256"] == y["pred_sha256"] and x["trees"] == y["trees"] for x, y in zip(a, b))
    rows, trows = [], []
    for i, c in enumerate(ref["configs"]):
        price = c["model"] == "boost_price"
        row = {"model": c["model"], "config": c["config"], "params": f"leaves={int(c['param'])}"
               + (f", tau={c['threshold']}" if price else "")}
        metric = "BalAcc" if price else "MAPE"
        row[f"original_{metric}"] = c[f"saved_{metric}"]
        for e in envs:
            x = runs[(e, 1)]["configs"][i]
            row[f"{e}_{metric}"] = x[metric]
            row[f"{e}_mismatch"] = x["mismatch_vs_saved"]
            row[f"{e}_pred"] = x["pred_sha256"][:10]
        row["pred_hash_same_all_envs"] = len({runs[(e, 1)]["configs"][i]["pred_sha256"] for e in envs}) == 1
        rows.append(row)
        for j, e in enumerate(envs):
            for f in envs[j + 1:]:
                it, kinds = first_diff(runs[(e, 1)]["configs"][i]["trees"], runs[(f, 1)]["configs"][i]["trees"])
                trows.append({"model": c["model"], "config": c["config"], "pair": f"{e} vs {f}",
                              "trees_identical": it is None, "first_diff_iteration": it, "fields": ",".join(kinds),
                              "bins_identical": runs[(e, 1)]["configs"][i]["bin_thresholds_sha256"]
                              == runs[(f, 1)]["configs"][i]["bin_thresholds_sha256"]})
    cmp_ = pd.DataFrame(rows)
    trees = pd.DataFrame(trows)
    cmp_.to_csv(os.path.join(XDIR, "xplat_comparison.csv"), index=False)
    trees.to_csv(os.path.join(XDIR, "xplat_trees.csv"), index=False)
    checks["env_summary"] = {e: {k: runs[(e, 1)]["env"].get(k) for k in ("os", "machine", "cpu", "python_compiler")}
                             | {"versions": runs[(e, 1)]["env"]["versions"],
                                "wheel_tags": runs[(e, 1)]["env"]["wheel_tags"],
                                "blas": (runs[(e, 1)]["env"]["numpy_build"] or {}).get("Build Dependencies", {}).get("blas", {}).get("name")
                                if isinstance(runs[(e, 1)]["env"]["numpy_build"], dict) else None,
                                "threadpools": [t.get("internal_api") for t in runs[(e, 1)]["env"]["threadpools"]],
                                "simd_flags": runs[(e, 1)]["env"].get("cpu_flags_simd")} for e in envs}
    checks["tuned_C_canonical_trees"] = canonical_tuned_c(envs)
    json.dump(checks, open(os.path.join(XDIR, "xplat_checks.json"), "w"), indent=1, default=str)
    return cmp_, trees, checks


CANON = ["feature_idx", "num_threshold", "left", "right", "is_leaf", "missing_go_to_left", "value"]


def canonical_tuned_c(envs):
    """Normalised tree identity for tuned_C from the full dumps (trees_<env>.npz).

    The raw `bin_threshold` index is excluded: it depends on how many distinct values a feature has
    on that platform (an extra distinct value shifts all later indices) while the real split
    (`num_threshold`) is unchanged. Compared: split feature, real threshold, topology, missing-value
    direction and node/leaf values, per tree, plus a hash over all 100 trees.
    """
    import hashlib
    import numpy as np
    out = {}
    for e in envs:
        p = os.path.join(XDIR, f"trees_{e}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        off = z["offsets"]
        per_tree = []
        for t in range(len(off) - 1):
            h = hashlib.sha256()
            for f in CANON:
                h.update(np.ascontiguousarray(z[f"node_{f}"][off[t]:off[t + 1]]).tobytes())
            per_tree.append(h.hexdigest()[:16])
        out[e] = {"n_trees": len(per_tree), "per_tree": per_tree,
                  "model_hash": hashlib.sha256("".join(per_tree).encode()).hexdigest(),
                  "raw_index_hash": hashlib.sha256(np.ascontiguousarray(z["node_bin_threshold"]).tobytes()).hexdigest()[:16]}
    names = list(out)
    out["pairs"] = {f"{a} vs {b}": {"canonical_identical": out[a]["model_hash"] == out[b]["model_hash"],
                                    "bin_index_identical": out[a]["raw_index_hash"] == out[b]["raw_index_hash"],
                                    "first_canonical_diff_tree": next((i for i, (x, y) in enumerate(
                                        zip(out[a]["per_tree"], out[b]["per_tree"])) if x != y), None)}
                    for i, a in enumerate(names) for b in names[i + 1:]}
    return out


if __name__ == "__main__":
    c, t, k = build()
    pd.set_option("display.width", 250)
    print(c.to_string(index=False))
    print(t.to_string(index=False))
    print(json.dumps({x: k[x] for x in ("envs", "inputs_identical", "params_identical", "repeat_identical")}, indent=1))
