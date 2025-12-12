<<<<<<< HEAD
import sys
from pathlib import Path

from funds import FundManager


TEST_STATE = Path('tests/tmp_funds_state.json')


def cleanup():
    try:
        if TEST_STATE.exists():
            TEST_STATE.unlink()
    except Exception:
        pass


def assert_almost(a, b, tol=1e-6):
    if abs(a - b) > tol:
        raise AssertionError(f"Assertion failed: {a} !~ {b}")


def test_reserve_confirm_release_cycle():
    cleanup()
    fm = FundManager(initial_fund=1000, state_file=str(TEST_STATE))
    assert_almost(fm.available_fund(), 1000.0)

    ok = fm.reserve(300)
    if not ok:
        raise AssertionError("reserve failed")
    assert_almost(fm.available_fund(), 700.0)

    fm.confirm(300)
    assert_almost(fm._reserved, 0.0)
    assert_almost(fm.available_fund(), 1000.0)

    fm.release(100)
    assert_almost(fm._reserved, 0.0)


def test_place_order_and_add_funds_and_persistence():
    cleanup()
    fm = FundManager(initial_fund=500, state_file=str(TEST_STATE))
    assert_almost(fm.available_fund(), 500.0)

    ok = fm.place_order(200)
    if not ok:
        raise AssertionError("place_order returned False")
    # immediate deduction expected
    assert_almost(fm.available_fund(), 300.0)

    fm.add_funds(250)
    assert_almost(fm.available_fund(), 550.0)

    fm2 = FundManager(initial_fund=0, state_file=str(TEST_STATE))
    assert_almost(fm2.available_fund(), fm.available_fund())


def main():
    tests = [
        test_reserve_confirm_release_cycle,
        test_place_order_and_add_funds_and_persistence,
    ]
    ok = True
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except Exception as e:
            ok = False
            print(f"FAIL: {t.__name__} -> {e}")

    cleanup()
    if not ok:
        print("Some tests failed")
        sys.exit(2)
    print("All tests passed")


if __name__ == '__main__':
    main()
=======
import sys
from pathlib import Path

from funds import FundManager


TEST_STATE = Path('tests/tmp_funds_state.json')


def cleanup():
    try:
        if TEST_STATE.exists():
            TEST_STATE.unlink()
    except Exception:
        pass


def assert_almost(a, b, tol=1e-6):
    if abs(a - b) > tol:
        raise AssertionError(f"Assertion failed: {a} !~ {b}")


def test_reserve_confirm_release_cycle():
    cleanup()
    fm = FundManager(initial_fund=1000, state_file=str(TEST_STATE))
    assert_almost(fm.available_fund(), 1000.0)

    ok = fm.reserve(300)
    if not ok:
        raise AssertionError("reserve failed")
    assert_almost(fm.available_fund(), 700.0)

    fm.confirm(300)
    assert_almost(fm._reserved, 0.0)
    assert_almost(fm.available_fund(), 1000.0)

    fm.release(100)
    assert_almost(fm._reserved, 0.0)


def test_place_order_and_add_funds_and_persistence():
    cleanup()
    fm = FundManager(initial_fund=500, state_file=str(TEST_STATE))
    assert_almost(fm.available_fund(), 500.0)

    ok = fm.place_order(200)
    if not ok:
        raise AssertionError("place_order returned False")
    # immediate deduction expected
    assert_almost(fm.available_fund(), 300.0)

    fm.add_funds(250)
    assert_almost(fm.available_fund(), 550.0)

    fm2 = FundManager(initial_fund=0, state_file=str(TEST_STATE))
    assert_almost(fm2.available_fund(), fm.available_fund())


def main():
    tests = [
        test_reserve_confirm_release_cycle,
        test_place_order_and_add_funds_and_persistence,
    ]
    ok = True
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except Exception as e:
            ok = False
            print(f"FAIL: {t.__name__} -> {e}")

    cleanup()
    if not ok:
        print("Some tests failed")
        sys.exit(2)
    print("All tests passed")


if __name__ == '__main__':
    main()
>>>>>>> 74f1ab306ca4f7cbafdafeccf820148ccd40d52d
