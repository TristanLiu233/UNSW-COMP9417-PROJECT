"""Paths to the (read-only) project kit and access to its official harness/dayahead.py."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# All price_hgbc results were produced against the group repo's `baseline` branch (commit 116e296) checked out as
# <ROOT>/comp9417_project_kit. `main` has newer ablation/ code, so there is deliberately no fallback to the repo root:
# see the README of this workstream ("Setup") for the one-line `git worktree` command.
KIT = os.path.join(ROOT, "comp9417_project_kit")
if not os.path.isfile(os.path.join(KIT, "harness", "dayahead.py")):
    raise RuntimeError(f"price_hgbc needs the kit at {KIT} (the group repo's `baseline` branch, commit 116e296). "
                       f"From {ROOT} run: git worktree add comp9417_project_kit baseline  "
                       "(and put nem_weather_halfhourly_2010_2026.csv in comp9417_project_kit/data/).")
DATA_CSV = os.path.join(KIT, "data", "nem_weather_halfhourly_2010_2026.csv")
RESULTS = os.path.join(ROOT, "results", "price_hgbc")
ARCHIVE = os.path.join(RESULTS, "_archive_m1m2")   # M1 / M2 outputs (M2.5: non-formal sensitivity analyses)

_HARNESS = os.path.join(KIT, "harness")
if _HARNESS not in sys.path:
    sys.path.insert(0, _HARNESS)

import dayahead as da  # noqa: E402  (official day boundaries, targets, baselines, metrics)

__all__ = ["ROOT", "KIT", "DATA_CSV", "RESULTS", "ARCHIVE", "da"]
