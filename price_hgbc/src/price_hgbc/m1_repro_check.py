"""M1 reproduction probe for the group's boost_price (18 cols, 15 leaves, tau 0.45).

Prints the platform, this environment's 2014 BalAcc/Acc, and the row mismatches against the
group's saved predictions (ablation/results/predictions/boost_price.csv.gz, column tuned_C).
Run on any machine with the course env:  python -m src.price_hgbc.m1_repro_check
"""
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from . import features_simple as fs
from .kit import DATA_CSV, KIT, da

GROUP_BALACC, GROUP_ACC = 79.29988384512427, 79.21803652968038


def main(random_state=0):
    df = da.load_dataset(DATA_CSV)
    X = fs.build_features(df)
    Xtr, ytr = fs.design(df, "2010-01-29", "2013-12-31", X)
    Xva, yva = fs.design(df, "2014-01-01", "2014-12-31", X)
    model = HistGradientBoostingClassifier(loss="log_loss", max_leaf_nodes=15, max_iter=100,
                                           learning_rate=0.1, early_stopping=False,
                                           random_state=random_state).fit(Xtr, ytr)
    pred = (model.predict_proba(Xva)[:, 1] >= 0.45).astype(int)
    saved = pd.read_csv(f"{KIT}/ablation/results/predictions/boost_price.csv.gz",
                        parse_dates=["timestamp"], index_col="timestamp")["tuned_C"]
    assert saved.index.equals(Xva.index)
    bal, acc = da.balanced_accuracy(yva, pred), da.accuracy(yva, pred)
    print("platform:", platform.platform(), platform.machine(), "python", platform.python_version(),
          "sklearn", sklearn.__version__, "numpy", np.__version__)
    print(f"BalAcc {bal:.5f} (group {GROUP_BALACC:.5f}, diff {bal - GROUP_BALACC:+.5f})  "
          f"Acc {acc:.5f} (group {GROUP_ACC:.5f})  row mismatches vs saved: {int((saved.to_numpy() != pred).sum())}")
    return bal, acc


if __name__ == "__main__":
    main()
