import traceback
try:
    import ccxt
    print("ccxt version:", getattr(ccxt, "__version__", "unknown"))
    ex = None
    # try some likely exchange classes
    for name in ("bitbank","bitflyer","binance","zaif","bitstamp"):
        if hasattr(ccxt, name):
            ex = getattr(ccxt, name)
            print("found exchange class in ccxt:", name)
            break
    if ex is None:
        print("No supported exchange class found in ccxt.")
    else:
        e = ex()
        print("exchange id:", getattr(e, "id", "unknown"))
        print("has fetchOHLCV:", e.has.get("fetchOHLCV"))
        try:
            # try load_markets to populate e.markets
            if not getattr(e, "markets", None):
                e.load_markets()
            print("markets loaded, count:", len(getattr(e, "markets", {})))
        except Exception as me:
            print("load_markets failed:", repr(me))
        # pick a symbol to try fetch_ohlcv
        symbol = None
        if getattr(e, "markets", None):
            keys = list(e.markets.keys())
            # prefer BTC/JPY if present
            if "BTC/JPY" in e.markets:
                symbol = "BTC/JPY"
            elif "BTC/USDT" in e.markets:
                symbol = "BTC/USDT"
            elif keys:
                symbol = keys[0]
        if not symbol:
            print("No symbol available in markets; will try 'BTC/JPY' directly")
            symbol = "BTC/JPY"
        try:
            print("trying fetch_ohlcv for", symbol)
            o = e.fetch_ohlcv(symbol, timeframe="1d", limit=5)
            print("ohlcv sample len:", len(o))
        except Exception as fe:
            print("fetch_ohlcv failed:", repr(fe))
except Exception:
<<<<<<< HEAD
    traceback.print_exc()
=======
    traceback.print_exc()
>>>>>>> 74f1ab306ca4f7cbafdafeccf820148ccd40d52d
