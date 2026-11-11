"""Environment + model fingerprint for the M1 reproduction diagnosis (boost_price, tuned_C).

Self-contained on purpose: it needs only the course env and a kit checkout (it uses the group's
own ablation/ code for features and the model), so the same file can be copied to the machine that
produced ablation/results and run there unchanged:

    python env_fingerprint.py --kit path/to/comp9417_project_kit --out fp_<machine>.json
    python env_fingerprint.py --compare fp_a.json fp_b.json

What it records
  env      Python build/compiler, OS, CPU, numpy/pandas/scipy/sklearn/threadpoolctl versions, wheel tags
           (= which compiled build), numpy build config (compilers, BLAS/LAPACK, SIMD), OpenMP/BLAS
           runtime (threadpoolctl), thread-related environment variables.
  data     SHA-256 of the whole CSV and of the prefix the ablation read (group method, run.json).
  features SHA-256 of the 18-column train/valid matrices (values+dtypes+names) and of the labels.
  kernels  SHA-256 of numeric-kernel probes on fixed inputs (np.exp/log1p, scipy expit, sklearn's
           HalfBinomialLoss gradient/hessian, float32 summation) -> isolates libm/compiler effects.
  model    get_params(), bin thresholds hash, baseline prediction, per-tree structure and value hashes,
           probability hash, 0/1 hash, BalAcc/Acc, mismatches vs saved tuned_C.
  saved    SHA-256 of the group's saved result files (must never change).
The kit is only read (bytecode writing disabled before importing ablation.*).
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys

sys.dont_write_bytecode = True

GROUP_BALACC = 79.29988384512427
COLS = ["slot_sin", "slot_cos", "season_sin", "season_cos", "weekend",
        "weekday_0", "weekday_1", "weekday_2", "weekday_3", "weekday_4", "weekday_5", "weekday_6",
        "nsw_demand_lag1", "nsw_price_lag1", "nsw_demand_lag7", "nsw_price_lag7",
        "prev_price_mean", "lag1_price_gap"]
SAVED_FILES = ["ablation/results/predictions/boost_price.csv.gz", "ablation/results/selection.csv",
               "ablation/results/scores.csv", "ablation/results/tuning.csv", "ablation/results/run.json"]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha_frame(df):
    import numpy as np
    h = hashlib.sha256()
    h.update(json.dumps([[c, str(t)] for c, t in df.dtypes.items()]).encode())
    h.update(np.ascontiguousarray(df.to_numpy(dtype="float64")).tobytes())
    return h.hexdigest()


def _cpu():
    try:
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True).stdout.strip()
        if sys.platform.startswith("linux"):
            for line in open("/proc/cpuinfo"):
                if line.lower().startswith(("model name", "hardware", "cpu model")):
                    return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor()


def _wheel_tags(dist):
    import importlib.metadata as md
    try:
        text = md.distribution(dist).read_text("WHEEL") or ""
        return [ln.split(":", 1)[1].strip() for ln in text.splitlines() if ln.startswith("Tag:")]
    except Exception:  # noqa: BLE001
        return None


def _dist_version(dist):
    import importlib.metadata as md
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def _pd_option(name):
    import pandas as pd
    try:
        return pd.get_option(name)
    except Exception:  # noqa: BLE001
        return None


def env_info():
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    import threadpoolctl
    try:
        np_cfg = np.show_config(mode="dicts")
        np_cfg = {k: np_cfg.get(k) for k in ("Compilers", "Build Dependencies", "Machine Information",
                                             "SIMD Extensions")}
    except Exception as exc:  # noqa: BLE001
        np_cfg = repr(exc)
    from sklearn.utils._openmp_helpers import _openmp_parallelism_enabled
    return {
        "python": sys.version, "python_compiler": platform.python_compiler(),
        "python_build": platform.python_build(), "implementation": platform.python_implementation(),
        "executable": sys.executable, "os": platform.platform(), "system": platform.system(),
        "machine": platform.machine(), "cpu": _cpu(), "cpu_count": os.cpu_count(),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                     "sklearn": sklearn.__version__, "threadpoolctl": threadpoolctl.__version__},
        "wheel_tags": {d: _wheel_tags(d) for d in ("numpy", "scipy", "scikit-learn", "pandas")},
        "numpy_build": np_cfg,
        "sklearn_openmp_enabled": bool(_openmp_parallelism_enabled()),
        # optional accelerators that silently change pandas reductions if present
        "optional_pandas_accelerators": {d: _dist_version(d) for d in ("bottleneck", "numexpr", "numba")},
        "pandas_use_bottleneck": _pd_option("compute.use_bottleneck"),
        "threadpools": [{k: i.get(k) for k in ("user_api", "internal_api", "prefix", "version",
                                                "num_threads", "threading_layer", "architecture")}
                        for i in threadpoolctl.threadpool_info()],
        "env_vars": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "KMP_AFFINITY", "OMP_PROC_BIND")},
    }


def kernel_probes():
    """Hashes of elementwise / reduction kernels on fixed inputs (no model involved)."""
    import numpy as np
    from scipy.special import expit
    from sklearn._loss.loss import HalfBinomialLoss
    rng = np.random.default_rng(12345)
    raw = rng.normal(0, 3, 200_000)
    y = (rng.random(200_000) < 0.5).astype(np.float64)
    g = np.empty_like(raw, dtype=np.float32)
    h = np.empty_like(raw, dtype=np.float32)
    HalfBinomialLoss().gradient_hessian(y_true=y, raw_prediction=raw, gradient_out=g, hessian_out=h,
                                        n_threads=1)
    f32 = rng.normal(0, 1, 1_000_003).astype(np.float32)
    return {"np_exp": sha(np.exp(raw).tobytes()), "np_log1p": sha(np.log1p(np.abs(raw)).tobytes()),
            "scipy_expit": sha(expit(raw).tobytes()),
            "sk_binomial_grad_f32": sha(g.tobytes()), "sk_binomial_hess_f32": sha(h.tobytes()),
            "np_sum_f32": float(np.sum(f32)), "np_sum_f32_hex": float(np.sum(f32)).hex()}


def model_fingerprint(kit, threads=1, random_state=17):
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    if kit not in sys.path:
        sys.path.insert(0, kit)
    from ablation import features as gf
    from harness import dayahead as da

    csv = os.path.join(kit, "data", "nem_weather_halfhourly_2010_2026.csv")
    data = gf.load_study(csv)
    with open(csv, "rb") as fh:  # group method (ablation/run.py): header + len(data) lines
        prefix = hashlib.sha256()
        for _ in range(len(data) + 1):
            prefix.update(fh.readline())
    X, truth = gf.build_features(data)
    days = da.market_day(data.index)
    tr = (days >= "2010-01-29") & (days < "2014-01-01")
    va = days.year == 2014
    Xtr, Xva = X.loc[tr, COLS], X.loc[va, COLS]
    ytr, yva = truth.loc[tr, "nsw_price_up"], truth.loc[va, "nsw_price_up"].to_numpy()
    params = dict(loss="log_loss", max_leaf_nodes=15, max_iter=100, learning_rate=0.1,
                  early_stopping=False, random_state=random_state)
    with threadpool_limits(limits=threads):
        m = HistGradientBoostingClassifier(**params).fit(Xtr, ytr)
        p = m.predict_proba(Xva)[:, 1]
    pred = (p >= 0.45).astype(int)
    trees = []
    for it in m._predictors:
        nodes = it[0].nodes
        struct = nodes[["feature_idx", "bin_threshold", "left", "right", "is_leaf"]].tobytes()
        trees.append({"structure": sha(struct)[:16], "values": sha(nodes["value"].tobytes())[:16],
                      "gains": sha(nodes["gain"].tobytes())[:16], "full": sha(nodes.tobytes())[:16]})
    saved = pd.read_csv(os.path.join(kit, "ablation", "results", "predictions", "boost_price.csv.gz"),
                        index_col="timestamp")["tuned_C"].to_numpy()
    return {
        "data": {"csv_sha256": sha_file(csv), "prefix_sha256": prefix.hexdigest(),
                 "prefix_matches_run_json": prefix.hexdigest() == json.load(
                     open(os.path.join(kit, "ablation", "results", "run.json")))["csv_prefix_sha256"]},
        "features": {"train_rows": int(tr.sum()), "valid_rows": int(va.sum()), "columns": COLS,
                     "X_train_sha256": sha_frame(Xtr), "X_valid_sha256": sha_frame(Xva),
                     "y_train_sha256": sha(ytr.to_numpy().astype("int64").tobytes()),
                     "y_valid_sha256": sha(yva.astype("int64").tobytes()),
                     "train_index_monotonic": bool(Xtr.index.is_monotonic_increasing)},
        "model": {"threads": threads,
                  "params": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else repr(v))
                             for k, v in m.get_params().items()},
                  "is_categorical_": None if m.is_categorical_ is None else m.is_categorical_.tolist(),
                  "n_iter_": int(m.n_iter_),
                  "bin_thresholds_sha256": sha(b"".join(np.asarray(t).tobytes()
                                                        for t in m._bin_mapper.bin_thresholds_)),
                  "baseline_prediction": float(np.ravel(m._baseline_prediction)[0]).hex(),
                  "trees": trees,
                  "proba_sha256": sha(p.astype("float64").tobytes()),
                  "pred_sha256": sha(pred.astype("int64").tobytes()),
                  "BalAcc": da.balanced_accuracy(yva, pred), "Acc": da.accuracy(yva, pred),
                  "BalAcc_minus_group": da.balanced_accuracy(yva, pred) - GROUP_BALACC,
                  "mismatch_vs_saved_tuned_C": int((saved != pred).sum())},
        "saved_files_sha256": {f: sha_file(os.path.join(kit, f)) for f in SAVED_FILES},
    }


def fingerprint(kit, threads=1):
    out = {"env": env_info(), "kernels": kernel_probes()}
    out.update(model_fingerprint(kit, threads=threads, random_state=17))
    out["model_random_state_0"] = {k: v for k, v in model_fingerprint(kit, threads, 0)["model"].items()
                                   if k in ("proba_sha256", "pred_sha256", "BalAcc")}
    return out


def compare(a, b):
    """Human-readable diff of two fingerprints; returns list of differing keys."""
    A, B = json.load(open(a)), json.load(open(b))
    diffs = []

    def walk(x, y, path):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                walk(x.get(k), y.get(k), f"{path}.{k}" if path else k)
        elif path.endswith("trees") and isinstance(x, list) and isinstance(y, list):
            for i, (tx, ty) in enumerate(zip(x, y)):
                if tx != ty:
                    kinds = [k for k in tx if tx[k] != ty.get(k)]
                    diffs.append(f"{path}: first differing tree = iteration {i} (fields {kinds})")
                    return
        elif x != y:
            diffs.append(f"{path}: {json.dumps(x)[:120]}  !=  {json.dumps(y)[:120]}")
    walk(A, B, "")
    return diffs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kit")
    ap.add_argument("--out")
    ap.add_argument("--threads", type=int, default=1, help="OpenMP threads for the fit (group used 1)")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    if args.compare:
        d = compare(*args.compare)
        print("\n".join(d) if d else "fingerprints identical")
        return
    fp = fingerprint(os.path.abspath(args.kit), threads=args.threads)
    text = json.dumps(fp, indent=1, default=str)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
    m = fp["model"]
    print(f"{fp['env']['os']} | {fp['env']['machine']} | {fp['env']['cpu']} | py {platform.python_version()}")
    print(f"prefix matches run.json: {fp['data']['prefix_matches_run_json']} | "
          f"BalAcc {m['BalAcc']:.5f} (group {GROUP_BALACC:.5f}) | mismatches vs saved {m['mismatch_vs_saved_tuned_C']}")


if __name__ == "__main__":
    main()
