"""Read-only access to the group's LR code (worktree `_group_main/`, main c23c959) and the course data.

`_group_main/src` is a regular package named `src`, which would shadow this workspace's `src` namespace.
Instead of putting it on sys.path we register only `src.common` from it under its own name, then put
`_group_main` on sys.path for `logistic_price` (a name this workspace does not use).
"""
import hashlib
import importlib.util
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GROUP = os.path.join(ROOT, "_group_main")
GROUP_COMMIT = "c23c95956eea95dd56e311ccb22a25fdbf2b5457"
RESULTS = os.path.join(ROOT, "results", "price_lr")

# SHA-256 prefixes recorded in strategy_design.md section 1 (checked on every run)
EXPECTED_SHA16 = {
    "src/common/features.py": "49a0b02495e60466",
    "logistic_price/features.py": "010c622f7a2573fe",
    "logistic_price/transforms.py": "621968c17eab4d65",
    "logistic_price/tuning.py": "e5abf4ff5a85020e",
    "logistic_price/final_training.py": "6f9c7c493559706b",
    "logistic_price/classifier.py": "96843515dc5ccae6",
    "logistic_price/handoff/selected_price_model.json": "7211f41d5d5ffe10",
    "logistic_price/results/frozen/artifacts/log_interactions31/logistic.joblib": "7b9afa822b1e0417",
    "logistic_price/results/frozen/replay_predictions.csv.gz": "a7b6d6c37a43107a",
    "logistic_price/results/frozen/validation_predictions.csv.gz": "bdeee696dc2ea8d1",
}
DATA_SHA16 = "c13711002906140e"


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _register_group():
    if "logistic_price" in sys.modules:
        return
    common_dir = os.path.join(GROUP, "src", "common")
    spec = importlib.util.spec_from_file_location("src.common", os.path.join(common_dir, "__init__.py"),
                                                  submodule_search_locations=[common_dir])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["src.common"] = mod
    spec.loader.exec_module(mod)
    if GROUP not in sys.path:
        sys.path.insert(0, GROUP)


_register_group()

from src.price_hgbc.kit import DATA_CSV, da  # noqa: E402  (official harness, same file as main's harness)
from logistic_price import final_training as ft  # noqa: E402
from logistic_price import features as lf  # noqa: E402
from logistic_price import transforms as lt  # noqa: E402
from logistic_price import tuning as lu  # noqa: E402
from src.common import features as common_features  # noqa: E402

assert os.path.realpath(common_features.__file__) == os.path.realpath(os.path.join(GROUP, "src", "common", "features.py"))

FROZEN_DIR = os.path.join(GROUP, "logistic_price", "results", "frozen")
FROZEN_ARTIFACT = os.path.join(FROZEN_DIR, "artifacts", "log_interactions31", "logistic.joblib")
REPLAY_CSV = os.path.join(FROZEN_DIR, "replay_predictions.csv.gz")
VALID_CSV = os.path.join(FROZEN_DIR, "validation_predictions.csv.gz")
RECIPE = "log_interactions31"


def check_inputs(full=True):
    """Assert the group snapshot and data are the ones the design was written against."""
    import subprocess
    head = subprocess.run(["git", "-C", GROUP, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert head == GROUP_COMMIT, f"_group_main is at {head}, expected {GROUP_COMMIT}"
    got = {p: sha_file(os.path.join(GROUP, p))[:16] for p in EXPECTED_SHA16}
    bad = {p: (got[p], e) for p, e in EXPECTED_SHA16.items() if got[p] != e}
    assert not bad, f"group files changed: {bad}"
    if full:
        assert sha_file(DATA_CSV)[:16] == DATA_SHA16, "course CSV changed"
    return {"group_commit": head, **{p: got[p] for p in got}, "data_csv": DATA_SHA16}
