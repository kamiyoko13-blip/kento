import importlib.util
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ninibo1127", ROOT / "ninibo1127.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_sell_decision_sells_entire_position_once():
    os.environ["TARGET_SELL_BTC"] = "0.0011"
    os.environ["SPECIAL_SELL_BTC"] = "0.0007"
    os.environ["SPECIAL_SELL_TARGET_PRICE"] = "14300000"
    os.environ["DISABLE_SELLS"] = "0"
    os.environ["PROFIT_TAKE_PCT"] = "10.0"

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
    assert result["amount"] == 0.00162


def test_no_repeated_sell_when_balance_is_zero():
    os.environ["TARGET_SELL_BTC"] = "0.0011"
    os.environ["SPECIAL_SELL_BTC"] = "0.0007"
    os.environ["SPECIAL_SELL_TARGET_PRICE"] = "14300000"
    os.environ["DISABLE_SELLS"] = "0"

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


def test_trade_ratio_can_be_overridden_from_env():
    os.environ["TRADE_RATIO"] = "0.75"
    os.environ["PROFIT_TAKE_PCT"] = "10.0"

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
    assert result["amount"] == 0.00135
