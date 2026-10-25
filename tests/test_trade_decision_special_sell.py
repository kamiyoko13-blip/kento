import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ninibo1127", ROOT / "ninibo1127.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_ten_percent_rise_sells_configured_share_of_position(monkeypatch):
    monkeypatch.setenv("TRADE_RATIO", "0.90")
    monkeypatch.setenv("SELL_ON_RISE_PCT", "10")
    monkeypatch.setenv("DISABLE_SELLS", "0")

    result = mod.trade_decision(
        current_price=14300000,
        btc_balance=0.0018,
        buy_btc=None,
        last_buy_price=13000000,
        rsi=50,
        bb_lower=None,
        last_sell_price=None,
        ema_trend="down",
    )
    assert result["action"] == "sell"
    assert result["amount"] == pytest.approx(0.00162)


def test_no_sell_when_balance_is_zero(monkeypatch):
    monkeypatch.setenv("TARGET_SELL_BTC", "0.0011")
    monkeypatch.setenv("SELL_ON_RISE_PCT", "10")
    monkeypatch.setenv("DISABLE_SELLS", "0")

    result = mod.trade_decision(
        current_price=14300000,
        btc_balance=0.0,
        buy_btc=None,
        last_buy_price=13000000,
        rsi=80,
        bb_lower=None,
        last_sell_price=None,
        ema_trend="down",
    )

    assert result["action"] == "hold"
    assert result["amount"] == 0.0


def test_trade_ratio_can_be_overridden_from_env(monkeypatch):
    monkeypatch.setenv("TRADE_RATIO", "0.75")
    monkeypatch.setenv("SELL_ON_RISE_PCT", "10")
    monkeypatch.setenv("DISABLE_SELLS", "0")

    result = mod.trade_decision(
        current_price=14300000,
        btc_balance=0.0018,
        buy_btc=None,
        last_buy_price=13000000,
        rsi=50,
        bb_lower=None,
        last_sell_price=None,
        ema_trend="down",
    )

    assert result["action"] == "sell"
    assert result["amount"] == pytest.approx(0.00135)


def test_rebuy_is_triggered_at_last_sell_price_without_drop_percentage(monkeypatch):
    monkeypatch.setenv("BUY_ON_DROP_PCT", "0.50")
    monkeypatch.setenv("BUY_BUDGET_JPY", "20000")
    monkeypatch.setenv("ESTIMATED_FEE_RATE", "0.0012")
    monkeypatch.setenv("MIN_ORDER_BTC", "0.0011")

    result = mod.trade_decision(
        current_price=12_300_000,
        btc_balance=0,
        last_sell_price=12_300_000,
        jpy_balance=17_000,
    )

    assert result["action"] == "buy"
    assert result["reason"] == "自動買い: 直近売値以下に下落"
    assert result["amount"] == pytest.approx(0.00138045)
    assert result["amount"] * result["price"] * 1.0012 <= 17_000


def test_rebuy_waits_while_price_is_above_last_sell(monkeypatch):
    monkeypatch.setenv("BUY_BUDGET_JPY", "20000")
    monkeypatch.setenv("MIN_ORDER_BTC", "0.0011")

    result = mod.trade_decision(
        current_price=12_600_000,
        btc_balance=0,
        last_sell_price=12_300_000,
        jpy_balance=17_000,
    )

    assert result["action"] == "hold"


def test_rebuy_is_skipped_when_last_sell_price_is_unknown():
    result = mod.trade_decision(
        current_price=12_000_000,
        btc_balance=0,
        rsi=40,
        jpy_balance=17_000,
    )

    assert result["action"] == "hold"


def test_buy_amount_is_capped_by_budget_and_available_balance(monkeypatch):
    monkeypatch.setenv("BUY_BUDGET_JPY", "20000")
    monkeypatch.setenv("ESTIMATED_FEE_RATE", "0.0012")
    monkeypatch.setenv("MIN_ORDER_BTC", "0.0011")

    amount = mod.calculate_buy_amount(12_300_000, 17_000)

    assert amount == pytest.approx(0.00138045)
    assert amount * 12_300_000 * 1.0012 <= 17_000
    assert amount * 12_300_000 * 1.0012 <= 20_000
    assert mod.calculate_buy_amount(12_300_000, 30_000) == pytest.approx(0.00162406)


def test_buy_amount_is_zero_when_balance_cannot_cover_minimum_order(monkeypatch):
    monkeypatch.setenv("BUY_BUDGET_JPY", "20000")
    monkeypatch.setenv("MIN_ORDER_BTC", "0.0011")
    assert mod.calculate_buy_amount(12_300_000, 10_000) == 0.0
