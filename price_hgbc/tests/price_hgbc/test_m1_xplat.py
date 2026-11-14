"""M1 cross-platform experiment: checks on the recorded artifacts (+ opt-in Docker re-run).

The artifacts come from src/price_hgbc/m1_xplat.py (macOS) and src/price_hgbc/run_xplat_docker.sh
(Linux containers). The 79.300 gate itself stays in test_m1_gate.py and is not touched here.
"""
import json
import os
import re
import subprocess

import pytest

from conftest import KIT, ROOT

XDIR = os.path.join(ROOT, "results", "price_hgbc", "selfcheck", "xplat")
ENVS = {"macos-arm64": ("Darwin", "arm64"), "linux-arm64": ("Linux", "aarch64"),
        "linux-amd64": ("Linux", "x86_64")}
PINNED = [ln.strip() for ln in open(os.path.join(KIT, "requirements.txt"))
          if re.fullmatch(r"[A-Za-z0-9_.-]+==\S+", ln.strip())]


def _run(env, k):
    return json.load(open(os.path.join(XDIR, f"xplat_{env}_run{k}.json")))


@pytest.mark.parametrize("env", ENVS)
@pytest.mark.parametrize("k", [1, 2])
def test_run_artifact_present(env, k):
    assert os.path.exists(os.path.join(XDIR, f"xplat_{env}_run{k}.json"))


@pytest.mark.parametrize("env", ENVS)
def test_environment_is_what_it_claims(env):
    e = _run(env, 1)["env"]
    system, machine = ENVS[env]
    assert e["system"] == system and e["machine"] == machine
    assert e["versions"]["numpy"] == "2.4.6" and e["versions"]["pandas"] == "3.1.0"
    assert e["versions"]["scipy"] == "1.17.1" and e["versions"]["sklearn"] == "1.9.1"
    assert e["versions"]["threadpoolctl"] == "3.8.0"
    if env.startswith("linux"):
        assert e["python"].startswith("3.11.13")


@pytest.mark.parametrize("env", ["linux-arm64", "linux-amd64"])
def test_container_installed_exactly_the_pins(env):
    freeze = set(open(os.path.join(XDIR, f"pip_freeze_{env}.txt")).read().split())
    lower = {x.lower() for x in freeze}
    for pin in PINNED:
        assert pin.lower() in lower, pin


# 2026-11-11, user-authorised (M1 resume): cross-platform input identity of the historical diagnosis
# artifacts is a historical-reproduction check, not a current-environment check. Assertion unchanged.
@pytest.mark.historical_repro
def test_inputs_identical_across_all_envs_and_runs():
    ref = _run("macos-arm64", 1)["inputs"]
    for env in ENVS:
        for k in (1, 2):
            assert _run(env, k)["inputs"] == ref, (env, k)


def test_hgb_params_identical_across_envs():
    ref = [c["params"] for c in _run("macos-arm64", 1)["configs"]]
    for env in ENVS:
        assert [c["params"] for c in _run(env, 1)["configs"]] == ref
    for c in _run("macos-arm64", 1)["configs"]:
        assert c["params"]["early_stopping"] is False and c["params"]["max_iter"] == 100


@pytest.mark.parametrize("env", ENVS)
def test_two_repetitions_bitwise_identical(env):
    a, b = _run(env, 1)["configs"], _run(env, 2)["configs"]
    for x, y in zip(a, b):
        assert (x["config"], x["pred_sha256"], x["trees"]) == (y["config"], y["pred_sha256"], y["trees"])


@pytest.mark.parametrize("env", ENVS)
def test_random_state_0_equals_17(env):
    r = _run(env, 1)["tuned_C_random_state_0"]
    assert r["raw_equal_rs17"] and r["trees_equal_rs17"]


def test_macos_matches_earlier_evidence():
    got = {(c["model"], c["config"]): c["mismatch_vs_saved"] for c in _run("macos-arm64", 1)["configs"]}
    assert got[("boost_price", "tuned_C")] == 157 and got[("boost_price", "tuned_CR")] == 0
    assert got[("boost_price", "C")] == 478 and got[("boost_demand", "drop_F_qld_price")] == 0


def test_comparison_table_consistent_with_runs():
    import pandas as pd
    cmp_ = pd.read_csv(os.path.join(XDIR, "xplat_comparison.csv"))
    for env in ENVS:
        for c in _run(env, 1)["configs"]:
            row = cmp_[(cmp_.model == c["model"]) & (cmp_.config == c["config"])].iloc[0]
            assert row[f"{env}_mismatch"] == c["mismatch_vs_saved"]


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("RUN_DOCKER_XPLAT") != "1",
                    reason="opt-in: needs Docker Desktop running; set RUN_DOCKER_XPLAT=1")
@pytest.mark.parametrize("plat", ["arm64", "amd64"])
def test_docker_rerun(plat):
    r = subprocess.run(["bash", os.path.join(ROOT, "src", "price_hgbc", "run_xplat_docker.sh"), plat],
                       capture_output=True, text=True, timeout=7200)
    assert r.returncode == 0, r.stderr[-2000:]


# ---- finer-grained facts (added after the runs; the strict all-inputs test above is kept as is) ----
def test_data_and_labels_identical_across_all_envs():
    keys = ("prefix_rows", "train_rows", "valid_rows", "X_used_cols", "y_price_sha256", "y_demand_sha256")
    ref = {k: _run("macos-arm64", 1)["inputs"][k] for k in keys}
    for env in ENVS:
        for k in (1, 2):
            assert {x: _run(env, k)["inputs"][x] for x in keys} == ref, (env, k)


def test_linux_arm64_and_amd64_bit_identical():
    a, b = _run("linux-arm64", 1), _run("linux-amd64", 1)
    assert a["inputs"] == b["inputs"] and a["kernels"] == b["kernels"]
    for x, y in zip(a["configs"], b["configs"]):
        assert (x["raw_sha256"], x["trees"], x["bin_thresholds_sha256"]) == \
               (y["raw_sha256"], y["trees"], y["bin_thresholds_sha256"])


def test_prediction_labels_identical_across_all_envs():
    ref = _run("macos-arm64", 1)["configs"]
    for env in ENVS:
        for x, y in zip(_run(env, 1)["configs"], ref):
            assert (x["config"], x["pred_sha256"], x["mismatch_vs_saved"]) == (y["config"], y["pred_sha256"], y["mismatch_vs_saved"])


def test_no_env_reproduces_historical_tuned_c():
    for env in ENVS:
        c = next(x for x in _run(env, 1)["configs"] if x["model"] == "boost_price" and x["config"] == "tuned_C")
        assert c["mismatch_vs_saved"] == 157


# 2026-11-11, user-authorised (M1 resume, 2nd decision): reads only the saved cross-platform tree
# artifacts (xplat/trees_*.npz) of the historical diagnosis. Assertions unchanged.
@pytest.mark.historical_repro
def test_tuned_c_macos_vs_linux_differ_only_by_8_one_ulp_thresholds():
    import numpy as np
    A = np.load(os.path.join(XDIR, "trees_macos-arm64.npz"))
    for other in ("linux-arm64", "linux-amd64"):
        B = np.load(os.path.join(XDIR, f"trees_{other}.npz"))
        assert np.array_equal(A["offsets"], B["offsets"]) and len(A["offsets"]) - 1 == 100
        for f in ("feature_idx", "left", "right", "is_leaf", "missing_go_to_left", "value", "gain", "count"):
            assert np.array_equal(A["node_" + f], B["node_" + f]), f
        a, b = A["node_num_threshold"], B["node_num_threshold"]
        d = np.flatnonzero(a != b)
        assert len(d) == 8
        assert np.abs(a[d].view(np.int64) - b[d].view(np.int64)).max() == 1
        assert set(A["columns"][A["node_feature_idx"][d]]) <= {"season_sin", "season_cos"}
        pa, pb = A["proba"], B["proba"]
        assert np.abs(pa - pb).max() <= 2.3e-16
        assert ((pa >= 0.45) != (pb >= 0.45)).sum() == 0


def test_docker_state_restored():
    before = set(open(os.path.join(XDIR, "docker_images_before.txt")).read().split("\n")) - {""}
    after = set(open(os.path.join(XDIR, "docker_images_after.txt")).read().split("\n")) - {""}
    vb = set(open(os.path.join(XDIR, "docker_volumes_before.txt")).read().split()) - {""}
    va = set(open(os.path.join(XDIR, "docker_volumes_after.txt")).read().split()) - {""}
    assert before <= after and vb <= va
    assert "Stopping Docker Desktop" in open(os.path.join(XDIR, "docker_ops.log")).read().split("## ")[-1]
