# DRY_RUN and test guide

Use `DRY_RUN` to exercise the bot without submitting orders. Install the project
dependencies first:

```powershell
python -m pip install -r requirements.txt
```

## Safe import check

This check imports the bot module but does not start the bot loop or send orders:

```powershell
$env:DRY_RUN = '1'
python run_dry.py
```

## Run the bot in simulation mode

This starts the continuous bot loop. It can fetch public market data, but orders
are simulated while `DRY_RUN=1`. Stop it with Ctrl+C:

```powershell
$env:DRY_RUN = '1'
python ninibo1127.py
```

To clear the setting afterwards:

```powershell
Remove-Item Env:DRY_RUN
```

The exchange API credentials are not required for DRY_RUN. For live trading,
set `BITBANK_API_KEY` and `BITBANK_API_SECRET` in the local `.env` file. The
legacy names `API_KEY` and `SECRET_KEY` are also accepted. Never commit real
credentials.

Useful settings include:

- `DRY_RUN`: Set to `1`, `true`, `yes`, or `on` to simulate orders.
- `DRY_RUN_PRICE`: Price used when a simulated order has no explicit price.
- `MIN_ORDER_BTC`: Minimum BTC amount used by the bot.
- `BUY_BUDGET_JPY`: Maximum yen budget per buy order (default `20000`). The
  order size is also capped by the exchange's currently free JPY balance,
  estimated fees, and `MIN_ORDER_BTC`.
- `DRY_RUN_JPY_BALANCE`: Simulated free JPY balance for sizing DRY_RUN buys
  (default `17000`).
- `TARGET_SELL_BTC`: BTC amount targeted for a qualifying sell.
- `SELL_ON_RISE_PCT`: Profit-rise threshold for the automatic sell rule.
- `DISABLE_SELLS`: Set to `1` to suppress sell signals.

When no BTC is held, the bot's re-buy rule uses the latest recorded sell price:
it can submit a limit buy when the current price is at or below that sell price.
There is no extra percentage-drop threshold. If the latest sell price cannot be
confirmed from the exchange or local trade history, the bot skips buying.

Before live trading, verify the exchange's current order limits and test the
strategy in DRY_RUN. DRY_RUN prevents order submission; it does not guarantee
that all other external services or market-data requests are disabled.
