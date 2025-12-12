<<<<<<< HEAD
import os
import json
import pathlib
import pytest

from funds import FundManager


TEST_STATE = pathlib.Path('tests/tmp_funds_state.json')


def teardown_module(module):
    try:
        if TEST_STATE.exists():
            TEST_STATE.unlink()
    except Exception:
        pass


def test_reserve_confirm_release_cycle():
    # start fresh
    if TEST_STATE.exists():
        TEST_STATE.unlink()
    fm = FundManager(initial_fund=1000, state_file=str(TEST_STATE))
    assert fm.available_fund() == pytest.approx(1000.0)

    ok = fm.reserve(300)
    assert ok is True
    # after reserve available should decrease by reserved amount
    assert fm.available_fund() == pytest.approx(700.0)

    # confirm should clear reservation
    fm.confirm(300)
    assert fm._reserved == pytest.approx(0.0)
    # available_fund returns to pre-reserve in this implementation
    assert fm.available_fund() == pytest.approx(1000.0)

    # release on nothing should be safe
    fm.release(100)
    assert fm._reserved == pytest.approx(0.0)


def test_place_order_and_add_funds_and_persistence():
    if TEST_STATE.exists():
        TEST_STATE.unlink()
    fm = FundManager(initial_fund=500, state_file=str(TEST_STATE))
    assert fm.available_fund() == pytest.approx(500.0)


    ok = fm.place_order(200)
    assert ok is True
    # place_order now immediately deducts from available funds
    assert fm.available_fund() == pytest.approx(300.0)

    fm.add_funds(250)
    assert fm.available_fund() == pytest.approx(550.0)

    # persistence: new manager should load same values
    fm2 = FundManager(initial_fund=0, state_file=str(TEST_STATE))
    # available_fund of fm2 should equal persisted available
    assert fm2.available_fund() == pytest.approx(fm.available_fund())
=======
import os
import json
import pathlib
import pytest

from funds import FundManager


TEST_STATE = pathlib.Path('tests/tmp_funds_state.json')


def teardown_module(module):
    try:
        if TEST_STATE.exists():
            TEST_STATE.unlink()
    except Exception:
        pass


def test_reserve_confirm_release_cycle():
    # start fresh
    if TEST_STATE.exists():
        TEST_STATE.unlink()
    fm = FundManager(initial_fund=1000, state_file=str(TEST_STATE))
    assert fm.available_fund() == pytest.approx(1000.0)

    ok = fm.reserve(300)
    assert ok is True
    # after reserve available should decrease by reserved amount
    assert fm.available_fund() == pytest.approx(700.0)

    # confirm should clear reservation
    fm.confirm(300)
    assert fm._reserved == pytest.approx(0.0)
    # available_fund returns to pre-reserve in this implementation
    assert fm.available_fund() == pytest.approx(1000.0)

    # release on nothing should be safe
    fm.release(100)
    assert fm._reserved == pytest.approx(0.0)


def test_place_order_and_add_funds_and_persistence():
    if TEST_STATE.exists():
        TEST_STATE.unlink()
    fm = FundManager(initial_fund=500, state_file=str(TEST_STATE))
    assert fm.available_fund() == pytest.approx(500.0)


    ok = fm.place_order(200)
    assert ok is True
    # place_order now immediately deducts from available funds
    assert fm.available_fund() == pytest.approx(300.0)

    fm.add_funds(250)
    assert fm.available_fund() == pytest.approx(550.0)

    # persistence: new manager should load same values
    fm2 = FundManager(initial_fund=0, state_file=str(TEST_STATE))
    # available_fund of fm2 should equal persisted available
    assert fm2.available_fund() == pytest.approx(fm.available_fund())
>>>>>>> 74f1ab306ca4f7cbafdafeccf820148ccd40d52d
