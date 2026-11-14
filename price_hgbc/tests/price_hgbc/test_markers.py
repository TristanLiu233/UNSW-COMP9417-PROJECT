"""Test-management check: the `historical_repro` marker selects exactly the authorised tests.

Guards against other (failing) tests being moved out of the routine runs by mislabelling.
"""
import os
import subprocess
import sys

from conftest import ROOT

# 1) the 79.29988 gate; 2) cross-platform input identity (user decision 2026-11-11, M1 resume);
# 3) cross-platform tree-threshold detail (user decision 2026-11-11, M1 resume, 2nd).
# Each entry was authorised individually; nothing else may carry the marker.
ALLOWED_HISTORICAL = {
    "tests/price_hgbc/test_m1_gate.py::test_m1_reproduces_group_balacc_within_0_01",
    "tests/price_hgbc/test_m1_xplat.py::test_inputs_identical_across_all_envs_and_runs",
    "tests/price_hgbc/test_m1_xplat.py::test_tuned_c_macos_vs_linux_differ_only_by_8_one_ulp_thresholds",
}


def _collect(marker_expr):
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/price_hgbc", "--collect-only", "-q",
                        "-p", "no:cacheprovider", "-m", marker_expr],
                       cwd=ROOT, capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert r.returncode in (0, 5), r.stdout[-2000:] + r.stderr[-2000:]
    return {ln.strip() for ln in r.stdout.splitlines() if "::" in ln}


def test_historical_repro_marker_selects_only_authorised_tests():
    assert _collect("historical_repro") == ALLOWED_HISTORICAL


def test_routine_runs_exclude_only_authorised_tests():
    everything = _collect("not historical_repro or historical_repro")
    routine = _collect("not historical_repro")
    assert everything - routine == ALLOWED_HISTORICAL


# ---- M2.5: archive marker (M1 / M2 tests) ----
ARCHIVED_MODULES = {"tests/price_hgbc/test_m1.py", "tests/price_hgbc/test_m2.py"}


def _collect_default():
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/price_hgbc", "--collect-only", "-q",
                        "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert r.returncode in (0, 5), r.stdout[-2000:] + r.stderr[-2000:]
    return {ln.strip() for ln in r.stdout.splitlines() if "::" in ln}


def test_archive_marker_selects_exactly_the_m1_m2_modules():
    archived = _collect("archive")
    assert archived and {n.split("::")[0] for n in archived} == ARCHIVED_MODULES
    everything = _collect("not historical_repro or historical_repro")
    in_modules = {n for n in everything if n.split("::")[0] in ARCHIVED_MODULES}
    assert archived == in_modules                     # every test in those modules, nothing else


def test_default_run_excludes_archive_only():
    everything = _collect("not historical_repro or historical_repro")
    default = _collect_default()
    assert everything - default == _collect("archive")
