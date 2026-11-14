import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KIT = os.path.join(ROOT, "comp9417_project_kit")     # `baseline` branch (116e296) checkout; see the workstream README
for p in (ROOT, os.path.join(KIT, "harness")):
    if p not in sys.path:
        sys.path.insert(0, p)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: takes more than 30 seconds")
    # User-authorised exception (2026-11-11): bit-exact reproduction of the group's historical
    # HGBC predictions is no longer a gate. Only the test(s) carrying this marker are excluded
    # from routine runs (-m "not historical_repro"); their assertions are unchanged and they
    # can still be run on their own with -m "historical_repro".
    config.addinivalue_line("markers", "historical_repro: bit-exact reproduction of the group's "
                            "historical result (diagnostic only; failure = accepted historical difference)")
    # M2.5 (user decision 2026-11-11): M1 / M2 are non-formal sensitivity analyses. Their tests carry
    # `archive` and are left out of a plain `pytest` run; any explicit -m expression decides for itself
    # (e.g. -m "not archive and not historical_repro", or -m archive to run them).
    config.addinivalue_line("markers", "archive: M1/M2 sensitivity-analysis checks (outputs in "
                            "results/price_hgbc/_archive_m1m2); excluded from the default run")


def pytest_collection_modifyitems(config, items):
    if config.option.markexpr:          # an explicit -m expression is respected as given
        return
    keep, drop = [], []
    for item in items:
        (drop if item.get_closest_marker("archive") else keep).append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep
