# Archive: M1 and M2 (non-formal sensitivity analyses since M2.5)

Moved here on 2026-11-11 (M2.5, user decision) by `python -m src.price_hgbc.archive_m1m2`: moved, not copied or
deleted. `MANIFEST.csv` lists every file's old path, new path, size and SHA-256 before and after the move
(21 files, all equal).

* The formal price-direction reference is now P1's `boost_price`: `results/price_hgbc/official/`.
* M1 (`m1.csv`, `m1/`) and M2 (`m2_*`, `m2/`) remain as sensitivity analyses only.
* Their stage reports and pytest logs stay in `results/price_hgbc/selfcheck/` (`M1.md`, `M2.md`, `M1_pytest_*`,
  `M2_pytest_*`) as the audit trail. Paths quoted there refer to the pre-archive locations; map them with `MANIFEST.csv`.
* Tests: `tests/price_hgbc/test_m1.py` and `test_m2.py` carry the `archive` marker. A plain `pytest` skips them.
  Run them with `pytest tests/price_hgbc -m archive`; they read and write this folder.
* `src/price_hgbc/m2.py` now writes here. `src/price_hgbc/m1_baseline.py` is left byte-identical (its hash is
  checked by the archived M2 tests), so its full run (`python -m src.price_hgbc.m1_baseline` without `--out`)
  would still write to the old `results/price_hgbc/m1*` paths. Do not run it that way.
