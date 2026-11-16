"""M2.5 step 1: move (never delete) the M1 / M2 outputs to results/price_hgbc/_archive_m1m2/.

Each file is hashed (SHA-256) before the move and again at its new path; the move is refused if a
destination already exists. Writes _archive_m1m2/MANIFEST.csv (old path, new path, bytes, sha before,
sha after). Stage reports and pytest logs in selfcheck/ (M1.md, M2.md, M*_pytest_*.txt) stay where they
are as the audit trail; only generated outputs move.
Usage: python -m src.price_hgbc.archive_m1m2
"""
import hashlib
import os

import pandas as pd

from .kit import RESULTS

ARCHIVE = os.path.join(RESULTS, "_archive_m1m2")
TOP_LEVEL = ["m1.csv", "m1", "m2", "m2_PR_draft.md", "m2_best_config.json", "m2_bootstrap.csv",
             "m2_feature_groups.csv", "m2_grid.csv", "m2_top5.csv"]
SELFCHECK = ["selfcheck/m1_repeat_run1.json", "selfcheck/m1_repeat_run2.json", "selfcheck/m1_sensitivity_rs1.json"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def files_to_move():
    out = []
    for rel in TOP_LEVEL + SELFCHECK:
        p = os.path.join(RESULTS, rel)
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                out += [os.path.relpath(os.path.join(root, n), RESULTS) for n in sorted(names)]
        elif os.path.exists(p):
            out.append(rel)
        else:
            raise FileNotFoundError(p)
    return sorted(out)


def main():
    rels = files_to_move()
    rows = []
    for rel in rels:
        src, dst = os.path.join(RESULTS, rel), os.path.join(ARCHIVE, rel)
        if os.path.exists(dst):
            raise FileExistsError(dst)
        rows.append({"old_path": f"results/price_hgbc/{rel}", "new_path": f"results/price_hgbc/_archive_m1m2/{rel}",
                     "bytes": os.path.getsize(src), "sha256_before": sha(src)})
    for rel in rels:
        dst = os.path.join(ARCHIVE, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.rename(os.path.join(RESULTS, rel), dst)           # same file system: a move, not a copy
    for r, rel in zip(rows, rels):
        r["sha256_after"] = sha(os.path.join(ARCHIVE, rel))
    for d in ("m1", "m2"):                                    # remove the now-empty source folders only
        p = os.path.join(RESULTS, d)
        if os.path.isdir(p) and not os.listdir(p):
            os.rmdir(p)
    man = pd.DataFrame(rows)
    man["match"] = man["sha256_before"] == man["sha256_after"]
    man.to_csv(os.path.join(ARCHIVE, "MANIFEST.csv"), index=False)
    print(man[["old_path", "bytes", "match"]].to_string(index=False))
    assert man["match"].all()


if __name__ == "__main__":
    main()
