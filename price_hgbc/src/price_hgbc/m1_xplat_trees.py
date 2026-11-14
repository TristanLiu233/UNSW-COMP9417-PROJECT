"""Dump the complete fitted tuned_C boost_price model (all nodes of all trees, bin edges, baseline,
2014 probabilities) for cross-environment divergence analysis.

    python -m src.price_hgbc.m1_xplat_trees --env linux-amd64
Writes results/price_hgbc/selfcheck/xplat/trees_<env>.npz. Fitted with the group's own predict_raw
(captured estimator), exactly as in m1_xplat.py. Kit read-only, no bytecode written.
"""
import argparse
import json
import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .kit import DATA_CSV, KIT, RESULTS  # noqa: E402

if KIT not in sys.path:
    sys.path.insert(0, KIT)
from ablation import features as gf  # noqa: E402
from ablation import models as gm  # noqa: E402
from harness import dayahead as da  # noqa: E402

XDIR = os.path.join(RESULTS, "selfcheck", "xplat")
_cap = []


class _CapC(gm.HistGradientBoostingClassifier):
    def fit(self, *a, **k):
        _cap.append(self)
        return super().fit(*a, **k)


def dump(env_label, config="tuned_C"):
    gm.HistGradientBoostingClassifier = _CapC
    data = gf.load_study(DATA_CSV)
    X, truth = gf.build_features(data)
    days = da.market_day(data.index)
    tr, va = (days >= "2010-01-29") & (days < "2014-01-01"), days.year == 2014
    s = pd.read_csv(os.path.join(KIT, "ablation", "results", "scores.csv"))
    r = s[(s.model == "boost_price") & (s.config == config)].iloc[0]
    cols = json.loads(r["columns"])
    raw = gm.predict_raw("boost_price", r["parameter"], X.loc[tr, cols], truth.loc[tr, "nsw_price_up"],
                         X.loc[va, cols])
    m = _cap[-1]
    nodes = [it[0].nodes for it in m._predictors]
    offsets = np.cumsum([0] + [len(n) for n in nodes])
    allnodes = np.concatenate(nodes)
    fields = {f"node_{f}": allnodes[f] for f in allnodes.dtype.names}
    bins = m._bin_mapper.bin_thresholds_
    os.makedirs(XDIR, exist_ok=True)
    out = os.path.join(XDIR, f"trees_{env_label}.npz")
    np.savez_compressed(out, offsets=offsets, proba=np.asarray(raw, dtype="float64"),
                        baseline=np.ravel(m._baseline_prediction).astype("float64"),
                        bin_lengths=np.array([len(b) for b in bins]),
                        bin_edges=np.concatenate([np.asarray(b, dtype="float64") for b in bins]),
                        columns=np.array(cols), **fields)
    print("wrote", out, "trees", len(nodes), "nodes", len(allnodes))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--config", default="tuned_C")
    a = ap.parse_args()
    dump(a.env, a.config)
