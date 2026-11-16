"""M1 reproduction gate (user rule): |BalAcc - 79.29988| <= 0.01 for the group's boost_price.

Expected to FAIL until the M1 blocker is resolved; it is intentionally not marked xfail.
2026-11-11, user-authorised: this test is tagged `historical_repro` and is excluded from routine
runs (-m "not historical_repro"). The assertion is unchanged; run it alone with
`pytest tests/price_hgbc -m historical_repro`, and record its failure as an accepted historical
difference in the self-check report.
"""
import pytest

from src.price_hgbc.m1_repro_check import GROUP_BALACC, main


@pytest.mark.historical_repro
def test_m1_reproduces_group_balacc_within_0_01():
    bal, _ = main(random_state=0)
    assert abs(bal - GROUP_BALACC) <= 0.01
