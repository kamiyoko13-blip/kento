import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ninibo1127", ROOT / "ninibo1127.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_exchange_uses_canonical_bitbank_credentials(monkeypatch):
    monkeypatch.setenv("DRY_RUN", "0")
    monkeypatch.setenv("BITBANK_API_KEY", "canonical-key")
    monkeypatch.setenv("BITBANK_API_SECRET", "canonical-secret")
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("SECRET_KEY", raising=False)
    captured = {}
    monkeypatch.setattr(mod.ccxt, "bitbank", lambda config: captured.update(config) or captured)

    mod.create_bitbank_exchange()

    assert captured["apiKey"] == "canonical-key"
    assert captured["secret"] == "canonical-secret"


def test_exchange_accepts_legacy_credential_names(monkeypatch):
    monkeypatch.setenv("DRY_RUN", "0")
    monkeypatch.delenv("BITBANK_API_KEY", raising=False)
    monkeypatch.delenv("BITBANK_API_SECRET", raising=False)
    monkeypatch.setenv("API_KEY", "legacy-key")
    monkeypatch.setenv("SECRET_KEY", "legacy-secret")
    captured = {}
    monkeypatch.setattr(mod.ccxt, "bitbank", lambda config: captured.update(config) or captured)

    mod.create_bitbank_exchange()

    assert captured["apiKey"] == "legacy-key"
    assert captured["secret"] == "legacy-secret"


def test_live_exchange_fails_clearly_without_credentials(monkeypatch):
    monkeypatch.setenv("DRY_RUN", "0")
    for name in ("BITBANK_API_KEY", "BITBANK_API_SECRET", "API_KEY", "SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError, match="Bitbank API credentials are missing"):
        mod.create_bitbank_exchange()


def test_dry_run_exchange_can_initialize_without_credentials(monkeypatch):
    monkeypatch.setenv("DRY_RUN", "1")
    for name in ("BITBANK_API_KEY", "BITBANK_API_SECRET", "API_KEY", "SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    captured = {}
    monkeypatch.setattr(mod.ccxt, "bitbank", lambda config: captured.update(config) or captured)

    mod.create_bitbank_exchange()

    assert captured["apiKey"] == ""
    assert captured["secret"] == ""


def test_dry_run_order_does_not_call_exchange(monkeypatch):
    class Exchange:
        def create_limit_buy_order(self, *args):
            raise AssertionError("live order API must not be called in DRY_RUN")

    monkeypatch.setenv("DRY_RUN", "1")
    monkeypatch.setattr(mod, "log_order", lambda *args: None)

    result = mod.execute_order(Exchange(), "BTC/JPY", "buy", 0.001, 10000000)

    assert result["id"] == "dry_order"


def test_live_order_uses_exchange_api(monkeypatch):
    class Exchange:
        def create_limit_buy_order(self, pair, amount, price):
            return {"id": "order-1", "pair": pair, "amount": amount, "price": price}

    monkeypatch.setenv("DRY_RUN", "0")
    result = mod.execute_order(Exchange(), "BTC/JPY", "buy", 0.001, 10000000)

    assert result["id"] == "order-1"
