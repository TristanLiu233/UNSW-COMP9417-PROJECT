"""M0 self-checks."""
import importlib.metadata as md
import os
import re
import sys

import pandas as pd
import pytest

from conftest import KIT

REQ = os.path.join(KIT, "requirements.txt")


def _pinned():
    out = []
    for line in open(REQ, encoding="utf-8"):
        line = line.split("#")[0].strip()
        m = re.fullmatch(r"([A-Za-z0-9_.-]+)==(\S+)", line)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


# ---- M0.1 environment ----
def test_python_is_311():
    assert sys.version_info[:2] == (3, 11), sys.version


def test_requirements_parsed():
    assert len(_pinned()) == 10


@pytest.mark.parametrize("name,want", _pinned())
def test_package_version_pinned(name, want):
    assert md.version(name) == want


# ---- M0.2 example package + check_submission (parses the saved log) ----
ROOT = os.path.dirname(KIT)
M0_CHECK = os.path.join(ROOT, "results", "price_hgbc", "m0_check.txt")


def _check_log():
    assert os.path.exists(M0_CHECK), "run M0.2 first"
    return open(M0_CHECK, encoding="utf-8").read().splitlines()


def test_m0_check_exit_codes_zero():
    codes = [ln for ln in _check_log() if ln.startswith("[exit code:")]
    assert codes == ["[exit code: 0]", "[exit code: 0]"]  # train.py, check_submission


def test_m0_check_14_ok_0_fail():
    lines = _check_log()
    ok = [ln for ln in lines if re.fullmatch(r"\d{4}-\d{2}-\d{2}  ok    .*", ln)]
    assert len(ok) == 14
    assert not any("FAIL" in ln for ln in lines)
    assert "All days ran successfully." in lines


def test_m0_check_kit_untouched():
    lines = _check_log()
    assert any(ln.endswith("UNCHANGED") for ln in lines)
    assert any(ln.endswith("IDENTICAL") for ln in lines)


# ---- M0.3 baselines on 2014 (recomputed from dayahead.py, ~7 s) ----
@pytest.fixture(scope="module")
def raw():
    from src.price_hgbc.kit import DATA_CSV, da
    return da.load_dataset(DATA_CSV)


@pytest.fixture(scope="module")
def baseline_tab(raw):
    from src.price_hgbc.m0_baselines import baseline_scores
    return baseline_scores(raw, 2014)


@pytest.mark.parametrize("name,expected", [("b1", 73.99), ("b7", 74.28)])
def test_baseline_balacc_2014(baseline_tab, name, expected):
    assert abs(baseline_tab.loc[name, "BalAcc"] - expected) <= 0.01


@pytest.mark.parametrize("name", ["b1", "b7"])
def test_baseline_matches_sklearn(baseline_tab, name):
    r = baseline_tab.loc[name]
    assert r["BalAcc"] == pytest.approx(r["BalAcc_sklearn"], abs=1e-9)
    assert r["Acc"] == pytest.approx(r["Acc_sklearn"], abs=1e-9)


def test_baseline_covers_all_2014(baseline_tab):
    assert (baseline_tab["n_days"] == 365).all()
    assert (baseline_tab["n_halfhours"] == 365 * 48).all()


# ---- M0.4 common.py ----
@pytest.fixture(scope="module")
def data(raw):
    from src.price_hgbc.common import add_day_period
    return add_day_period(raw)


def test_day_matches_dayahead(data):
    from src.price_hgbc.kit import da
    assert (pd.DatetimeIndex(data["day"]) == da.market_day(data.index)).all()


def test_each_complete_day_has_48_unique_periods(data):
    from src.price_hgbc.kit import da
    complete = set(da.complete_days(data))
    sub = data[data["day"].isin(complete)]
    g = sub.groupby("day")["period"]
    assert (g.size() == 48).all()
    assert (g.nunique() == 48).all()
    assert (g.min() == 0).all() and (g.max() == 47).all()


def test_period_consistent_with_offset(data):
    off = (data.index - pd.DatetimeIndex(data["day"])) / pd.Timedelta("30min") - 1
    assert (off.to_numpy() == data["period"].to_numpy()).all()


def test_midnight_row_belongs_to_previous_day(data):
    row = data.loc[pd.Timestamp("2014-03-02 00:00")]
    assert row["day"] == pd.Timestamp("2014-03-01")
    assert row["period"] == 47
    first = data.loc[pd.Timestamp("2014-03-02 00:30")]
    assert first["day"] == pd.Timestamp("2014-03-02") and first["period"] == 0


def test_no_incomplete_days(data):
    from src.price_hgbc.common import incomplete_days
    bad = incomplete_days(data)
    assert bad == [], f"incomplete days: {bad}"
    assert data["day"].nunique() == 6148


def test_index_sorted_unique(data):
    assert data.index.is_monotonic_increasing and data.index.is_unique


def test_year_splits_chronological_and_disjoint(data):
    from src.price_hgbc.common import stage_splits
    s = stage_splits(data)
    tr, va, sim = s["train"], s["valid"], s["sim"]
    assert len(tr) + len(va) + len(sim) == len(data)
    assert tr.index.max() < va.index.min() and va.index.max() < sim.index.min()
    assert set(pd.DatetimeIndex(tr["day"]).year) == {2010, 2011, 2012, 2013}
    assert set(pd.DatetimeIndex(va["day"]).year) == {2014}
    assert len(va) == 365 * 48
    # split is by market day: the 2014-01-01 00:00 row belongs to 2013-12-31 -> train
    assert pd.Timestamp("2014-01-01 00:00") in tr.index
    assert pd.Timestamp("2015-01-01 00:00") in va.index


# ---- M0 acceptance: starter notebook runs in .venv (~5-10 s, launches a kernel) ----
def test_starter_notebook_runs_in_venv():
    from src.price_hgbc.m0_run_notebook import run
    s = run()
    assert s["n_error_outputs"] == 0
    p = s["probe"]
    assert p is not None, "probe cell produced no output"
    assert os.path.realpath(p["executable"]) == os.path.realpath(sys.executable)
    assert os.path.join(".venv", "bin") in p["executable"]
    assert p["python"].startswith("3.11.")
    assert (p["numpy"], p["pandas"], p["sklearn"]) == (md.version("numpy"), md.version("pandas"),
                                                         md.version("scikit-learn"))


# ---- M0.2 (slow): really re-run check_submission on the kit's example_package ----
def _kit_hashes():
    import hashlib
    out = {}
    for base, dirs, files in os.walk(KIT):
        dirs[:] = [d for d in dirs if d not in (".git", "data")]
        for f in files:
            p = os.path.join(base, f)
            out[os.path.relpath(p, KIT)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return out


@pytest.mark.slow
def test_check_submission_rerun_live():
    import subprocess
    before = _kit_hashes()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "check_submission.py", "example_package"], cwd=KIT,
                       capture_output=True, text=True, env=env, timeout=600)
    lines = (r.stdout + r.stderr).splitlines()
    assert r.returncode == 0, r.stdout + r.stderr
    ok = [ln for ln in lines if re.fullmatch(r"\d{4}-\d{2}-\d{2}  ok    .*", ln)]
    assert len(ok) == 14
    assert not any("FAIL" in ln for ln in lines)
    assert "All days ran successfully." in lines
    assert _kit_hashes() == before, "check_submission changed files in the kit"
