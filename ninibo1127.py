from colorama import init as colorama_init
colorama_init(autoreset=True)
# --- シグナル生成のダミー関数 ---
def generate_signals(df):
    # 本番用: RSIとトレンドを考慮したシグナル生成
    rsi = None
    closes = None
    if isinstance(df, dict):
        rsi_list = df.get('rsi_list')
        closes = df.get('closes')
        if rsi_list and len(rsi_list) > 0:
            rsi = rsi_list[-1]
    elif hasattr(df, 'columns') and 'rsi' in df.columns:
        rsi = df['rsi'].iloc[-1]
        closes = df['close'].tolist() if 'close' in df.columns else None

    # トレンド判定（例: 直近5本の終値で上昇/下降判定）
    trend = None
    if closes and len(closes) >= 5:
        if closes[-1] > closes[-5]:
            trend = 'up'
        elif closes[-1] < closes[-5]:
            trend = 'down'
        else:
            trend = 'side'

    # シグナル判定
    if rsi is not None:
        if rsi < 30 and trend == 'up':
            return 'buy', f'RSI={int(rsi)} & 上昇トレンド'
        elif rsi > 70 and trend == 'down':
            return 'sell_all', f'RSI={int(rsi)} & 下降トレンド'
        elif rsi < 30:
            return 'hold', f'RSI={int(rsi)} だがトレンド弱い'
        elif rsi > 70:
            return 'hold', f'RSI={int(rsi)} だがトレンド弱い'
        else:
            return 'hold', f'RSI={int(rsi)} 通常'
    return 'hold', 'no signal'
# --- OHLCV取得のラッパー関数 ---
def get_ohlcv(exchange, symbol, timeframe='5m', limit=300):
    return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
# --- DRY_RUN用のデフォルト価格定数 ---
DRY_RUN_PRICE = 13000000  # 例: 1300万円（2026年1月相場に合わせて調整）

import os
import datetime
import pytz
JST = pytz.timezone('Asia/Tokyo')

# --- 表形式でデータを表示するユーティリティ ---
def print_table(rows, headers=None, color=None):
    from tabulate import tabulate
    from colorama import Fore, Style
    table = tabulate(rows, headers=headers, tablefmt="simple")
    if color:
        print(color + table + Style.RESET_ALL)
    else:
        print(table)
# --- 5分足OHLCVを1時間足にリサンプリング ---
def resample_ohlcv_5m_to_1h(ohlcv_5m):
    import pandas as pd
    if not ohlcv_5m or len(ohlcv_5m) < 12:
        return []
    df = pd.DataFrame(ohlcv_5m, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("datetime", inplace=True)
    agg_dict = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "timestamp": "first"}
    ohlcv_1h = df.resample("1H").agg(agg_dict)
    ohlcv_1h = ohlcv_1h.dropna().reset_index(drop=True)
    result = []
    for _, row in ohlcv_1h.iterrows():
        result.append([
            int(row["timestamp"]),
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            float(row["volume"])
        ])
    return result
# --- last_buy_priceをstateにセットするヘルパー ---
def set_last_buy_price(state, price):
    state["last_buy_price"] = price
# 必要な外部ライブラリのimport
import pandas as pd
import matplotlib.pyplot as plt
# === ログ出力の本物実装 ===
import logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# 必要な外部ライブラリのimport
import pandas as pd
import matplotlib.pyplot as plt
# === ログ出力の本物実装 ===
def log_error(*args, **kwargs):
    logging.error(' '.join(str(a) for a in args))

def log_info(*args, **kwargs):
    logging.info(' '.join(str(a) for a in args))

def log_warn(*args, **kwargs):
    logging.warning(' '.join(str(a) for a in args))

def log_debug(*args, **kwargs):
    logging.debug(' '.join(str(a) for a in args))
def get_latest_price(exchange, pair='BTC/JPY'):
    # 最新価格取得。失敗時は0.0
    try:
        ticker = exchange.fetch_ticker(pair)
        return float(ticker.get('last', 0))
    except Exception:
        return 0.0
import numpy as np
import talib
# --- EMAの傾き計算関数 ---
def calc_ema_slope(closes, period=20, span=3):
    """
    closes: 終値リスト（最新が最後）
    period: EMAの期間
    span: 何本前と比較するか
    return: (最新EMA - span本前EMA) / span
    """
    if len(closes) < period + span:
        import time
        # --- 必要な定数・変数を関数内で初期化 ---
        PAIR = 'BTC/JPY'
        PROFIT_TAKE_PCT = 10.0
        BUY_MORE_PCT = 10.0
        MIN_ORDER_BTC = 0.0027
        positions_file = 'positions_state.json'
        state = {'last_buy_price': None}
        positions = []
        updated_positions = []
        last_buy_price = None
        # 本番運用: exchangeがNoneなら即エラー
        if exchange is None:
            raise RuntimeError("本番APIインスタンスが作成できませんでした。APIキー・シークレット・.env設定を再確認してください。")
        trade_log_file = 'trade_history.json'

        def log_trade(action, price, amount, status, reason=None):
            import json, datetime
            # 約定（filled/closed）のみ記録
            if status not in ("closed", "filled"):
                return
            log_entry = {
                'action': action,
                'price': price,
                'amount': amount,
                'status': status,
                'reason': reason,
                'timestamp': datetime.datetime.now().isoformat()
            }
            try:
                if os.path.exists(trade_log_file):
                    with open(trade_log_file, 'r', encoding='utf-8') as f:
                        logs = json.load(f)
                else:
                    logs = []
                logs.append(log_entry)
                with open(trade_log_file, 'w', encoding='utf-8') as f:
                    json.dump(logs, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[ERROR] 取引履歴保存エラー: {e}")
# --- 1時間足のRSIリスト取得関数 ---
def get_rsi_1h_series(exchange, pair='BTC/JPY', limit=300, ohlcv_data=None):
    # 1時間足のOHLCVデータからRSIリストを返す
    import os
    import time
    import datetime
    import numpy as np
    import talib

    try:
        # 取得本数を30本に増やす（RSI計算のため最低15本以上推奨、余裕を持たせる）
        limit = 30
        if ohlcv_data is not None:
            ohlcv = ohlcv_data
        else:
            # まずAPIで1時間足を取得
            ohlcv = exchange.fetch_ohlcv(pair, timeframe='1h', limit=limit)
        # 取得失敗や本数不足なら5分足から1時間足を自作
        if not ohlcv or len(ohlcv) < 15:
            print(f"[WARN] 1h足データ不足({len(ohlcv) if ohlcv else 0}本)、5分足から合成します")
            ohlcv_5m = exchange.fetch_ohlcv(pair, timeframe='5m', limit=limit*12)
            ohlcv = resample_ohlcv_5m_to_1h(ohlcv_5m)
        if not ohlcv or len(ohlcv) < 15:
            print(f"[ERROR] 1h足データ合成失敗: {ohlcv}")
            return []
        # NoneやNaNを除去
        ohlcv = [c for c in ohlcv if c and all(pd.notna(x) for x in c)]
        if len(ohlcv) < 15:
            print(f"[ERROR] 1h足データに有効な足が15本未満: {len(ohlcv)}本")
            return []
        closes = [float(c[4]) for c in ohlcv]
        # import pandas as pd はグローバルで実施済み
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        def calc_rsi(series, period=14):
            delta = series.diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            ma_up = up.rolling(window=period, min_periods=period).mean()
            ma_down = down.rolling(window=period, min_periods=period).mean()
            rsi = 100 - (100 / (1 + ma_up / ma_down))
            rsi = rsi.clip(lower=0, upper=100)
            return rsi.astype(float)
        rsi_series = calc_rsi(df['close'], period=14)
        return rsi_series.tolist()
    except Exception as e:
        print(f"[ERROR] get_rsi_1h_series失敗: {e}")
        return []
print("ninibo1127.py:", __file__)
# --- bitbank用ccxtラッパー ---

import os
import logging
logger = logging.getLogger("ninibo_logger")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
import ccxt
# 表示用ライブラリ
from tabulate import tabulate
from colorama import Fore, Style, init as colorama_init
colorama_init(autoreset=True)

# --- インジケータ計算関数（RSI・終値リストなど） ---



def resample_ohlcv_5m_to_15m(ohlcv_5m):
    # 5分足OHLCVを15分足にリサンプリング
    if not ohlcv_5m or len(ohlcv_5m) < 3:
        return []
    import pandas as pd
    df = pd.DataFrame(ohlcv_5m, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("datetime", inplace=True)
    agg_dict = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "timestamp": "first"}
    ohlcv_15m = df.resample("15T").agg(agg_dict)
    ohlcv_15m = ohlcv_15m.dropna().reset_index(drop=True)
    result = []
    for _, row in ohlcv_15m.iterrows():
        result.append([
            int(row["timestamp"]),
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            float(row["volume"])
        ])
    return result

def create_bitbank_exchange():
    api_key = os.getenv('BITBANK_API_KEY')
    api_secret = os.getenv('BITBANK_API_SECRET')
    exchange = ccxt.bitbank({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
    })
    return exchange


# --- メインループ（Botの実行部分） ---
def compute_indicators(exchange, pair='BTC/JPY', timeframe='5m', limit=300, ohlcv_data=None):
    # Calculate RSI and close list from OHLCV data
    try:
        if ohlcv_data is not None:
            ohlcv = ohlcv_data
        else:
            ohlcv = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
        if not ohlcv or len(ohlcv) < 15:
            return {}
        closes = [float(c[4]) for c in ohlcv]
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        # RSI計算
        def calc_rsi(series, period=14):
            delta = series.diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            ma_up = up.rolling(window=period, min_periods=period).mean()
            ma_down = down.rolling(window=period, min_periods=period).mean()
            rsi = 100 - (100 / (1 + ma_up / ma_down))
            # 念のため0～100の範囲にクリップ
            rsi = rsi.clip(lower=0, upper=100)
            return rsi.astype(float)
        rsi_series = calc_rsi(df['close'], period=14)
        rsi_list = rsi_series.tolist()
        rsi_14 = rsi_list[-1] if len(rsi_list) >= 1 else None
        return {
            'rsi_14': rsi_14,
            'rsi_list': rsi_list,
            'closes': closes
        }
    except Exception as e:
        print(f"[ERROR] compute_indicators失敗: {e}")
        return {}
        return {
            'total': balance.get('total', {}),
            'free': balance.get('free', {}),
            'used': balance.get('used', {})
        }
    except Exception as e:
        print(f"[ERROR] 残高取得エラー: {e}")
        return {'total': {}, 'free': {}, 'used': {}}
def trade_decision(*args, **kwargs):
    # 本番用のロジックに統一（例: 直近のRSIやBB下限などで判定）
    current_price = kwargs.get('current_price') if 'current_price' in kwargs else args[0] if len(args) > 0 else None
    btc_balance = kwargs.get('btc_balance', 0.0027)
    min_order_btc = kwargs.get('min_order_btc', 0.0027)
    last_buy_price = kwargs.get('last_buy_price')
    rsi = kwargs.get('rsi')
    bb_lower = kwargs.get('bb_lower')
    # 主要なトレード判断材料を表で表示
    print_table([
        ["現在価格", current_price],
        ["BTC残高", btc_balance],
        ["直近買値", last_buy_price],
        ["RSI", rsi],
        ["BB下限", bb_lower]
    ], headers=["項目", "値"], color=Fore.GREEN)

    # --- 5分足RSIが一度でも30未満になったらフラグを立て、35以上40以下に戻ったら買い、買ったらフラグを下ろす ---
    global rsi_buy_flag, prev_rsi_value
    try:
        rsi_buy_flag
    except NameError:
        rsi_buy_flag = False
    try:
        prev_rsi_value
    except NameError:
        prev_rsi_value = None

    print(f"[DEBUG] trade_decision: rsi_buy_flag={rsi_buy_flag}, prev_rsi_value={prev_rsi_value}, rsi={rsi}, btc_balance={btc_balance}, min_order_btc={min_order_btc}, last_buy_price={last_buy_price}")

    # RSIが30未満ならフラグを立てる（何度も立て直してもOK）
    if rsi is not None and rsi < 30:
        print(f"[DEBUG] RSI < 30: フラグセット rsi={rsi}")
        rsi_buy_flag = True
        prev_rsi_value = rsi
        return {'action': 'hold', 'amount': 0.0, 'price': current_price, 'buy_condition': False, 'sell_condition': False}

    # フラグが立っていて、RSIが35以上40以下で買い（BTC保有量に関係なく、JPY残高のみ考慮）
    if rsi_buy_flag and rsi is not None and 35 <= rsi <= 40 and prev_rsi_value is not None:
        # --- 日本円残高チェック ---
        free_jpy = None
        if exchange is not None:
            try:
                balance = exchange.fetch_balance()
                free_jpy = balance.get('free', {}).get('JPY', None)
            except Exception as e:
                print(f"[ERROR] JPY残高取得エラー: {e}")
        # 注文に必要な日本円を計算（現在価格×注文量）
        jpy_needed = current_price * min_order_btc if current_price and min_order_btc else 0
        if free_jpy is not None:
            print(f"[DEBUG] JPY free残高={free_jpy}, 必要JPY={jpy_needed}")
        if free_jpy is not None and free_jpy > (jpy_needed + 1000):
            print(f"[DEBUG] BUY条件成立: rsi={rsi}, prev_rsi_value={prev_rsi_value}, JPY残高OK")
            rsi_buy_flag = False  # フラグリセット
            prev_rsi_value = None
            return {'action': 'buy', 'amount': min_order_btc, 'price': current_price, 'buy_condition': True}
        else:
            print(f"[DEBUG] JPY残高不足: free={free_jpy}, 必要={jpy_needed + 1000}")
            # 買い条件は満たすが残高不足
            return {'action': 'hold', 'amount': 0.0, 'price': current_price, 'buy_condition': False, 'sell_condition': False, 'reason': 'JPY残高不足'}

    # 条件を満たさない場合はhold
    print(f"[DEBUG] HOLD: 条件未達 rsi={rsi}, rsi_buy_flag={rsi_buy_flag}, prev_rsi_value={prev_rsi_value}")
    return {'action': 'hold', 'amount': 0.0, 'price': current_price, 'buy_condition': False, 'sell_condition': False}
def execute_order(*args, **kwargs):
    # 本番API呼び出し（必ず指値注文のみ対応）
    exchange = args[0] if len(args) > 0 else kwargs.get('exchange')
    pair = args[1] if len(args) > 1 else kwargs.get('pair', 'BTC/JPY')
    side = args[2] if len(args) > 2 else kwargs.get('side', 'buy')
    amount = args[3] if len(args) > 3 else kwargs.get('amount')
    price = args[4] if len(args) > 4 else kwargs.get('price')
    try:
        if side == 'buy':
            order = exchange.create_order(pair, 'limit', 'buy', amount, price)
        elif side == 'sell':
            order = exchange.create_order(pair, 'limit', 'sell', amount, price)
        else:
            log_error(f"無効な注文タイプです: {side}")
            return None
        # 表形式で表示
        order_disp = [
            ["注文ID", order.get('id', order.get('order_id', 'N/A'))],
            ["通貨ペア", order.get('symbol', pair)],
            ["注文タイプ", order.get('type', 'limit')],
            ["売買", order.get('side', side)],
            ["価格", order.get('price', price)],
            ["数量", order.get('amount', amount)],
            ["ステータス", order.get('status', 'N/A')],
        ]
        print_table(order_disp, headers=["項目", "値"], color=Fore.CYAN)
        return order
    except Exception as e:
        print(Fore.RED + f"[ERROR] 注文APIエラー: {e}" + Style.RESET_ALL)
        return {'error': str(e)}
def sell_all_positions(*args, **kwargs):
    # 本番API呼び出し
    positions = args[0] if len(args) > 0 else kwargs.get('positions', [])
    exchange = args[1] if len(args) > 1 else kwargs.get('exchange')
    pair = args[2] if len(args) > 2 else kwargs.get('pair', 'BTC/JPY')
    results = []
    total_btc = sum([pos.get('amount', 0) for pos in positions])
    sell_amount = round(total_btc * 0.85, 8)
    if total_btc > 0 and sell_amount > 0:
        try:
            current_price = get_latest_price(exchange, pair)
            if not current_price or current_price <= 0:
                print(Fore.RED + f"[ERROR] 指値売却のための現在価格取得失敗" + Style.RESET_ALL)
                results.append({'amount': sell_amount, 'error': 'Price fetch failed', 'status': 'error'})
                return results
            order = execute_order(exchange, pair, 'sell', sell_amount, current_price)
            order_disp = [
                ["注文ID", order.get('id', order.get('order_id', 'N/A'))],
                ["通貨ペア", order.get('symbol', pair)],
                ["注文タイプ", order.get('type', 'limit')],
                ["売買", order.get('side', 'sell')],
                ["価格", order.get('price', current_price)],
                ["数量", order.get('amount', sell_amount)],
                ["ステータス", order.get('status', 'N/A')],
            ]
            print_table(order_disp, headers=["項目", "値"], color=Fore.MAGENTA)
            with open("sell_log.txt", "a", encoding="utf-8") as logf:
                print(f"[DEBUG] 売却APIレスポンス: {order}", file=logf)
            results.append({'amount': sell_amount, 'order': order, 'status': 'sold'})
        except Exception as e:
            print(Fore.RED + f"[ERROR] 売却APIエラー: {e}" + Style.RESET_ALL)
            with open("sell_log.txt", "a", encoding="utf-8") as logf:
                print(f"[ERROR] 売却APIエラー: {e}", file=logf)
            results.append({'amount': sell_amount, 'error': str(e), 'status': 'error'})
    else:
        print(Fore.RED + f"[ERROR] 売却可能なBTCが不足しています（保有: {total_btc} BTC）" + Style.RESET_ALL)
        results.append({'amount': sell_amount, 'error': 'Insufficient BTC', 'status': 'error'})
    return results
def get_last_buy_price(state):
    return state.get('last_buy_price', None)

# 未定義関数のダミー定義（なければ）
def get_open_orders(exchange, pair='BTC/JPY', limit=50):
    try:
        orders = exchange.fetch_orders(pair, limit=limit)
        result = []
        for order in orders:
            try:
                result.append({
                    'id': order.get('id'),
                    'symbol': order.get('symbol'),
                    'type': order.get('type'),
                    'side': order.get('side'),
                    'price': float(order.get('price', 0)) if order.get('price') else None,
                    'amount': float(order.get('amount', 0)),
                    'filled': float(order.get('filled', 0)),
                    'remaining': float(order.get('remaining', 0)),
                    'cost': float(order.get('cost', 0)) if order.get('cost') else None,
                    'status': order.get('status'),
                    'timestamp': order.get('timestamp'),
                    'datetime': order.get('datetime')
                })
            except Exception:
                continue
        return result
    except Exception as e:
        print(f"[ERROR] 注文履歴取得エラー: {e}")
        return []
def cancel_order(exchange, order_id, pair='BTC/JPY'):
    try:
        result = exchange.cancel_order(order_id, pair)
        print(f"[DEBUG] 注文キャンセルAPIレスポンス: {result}")
        return result
    except Exception as e:
        print(f"[ERROR] 注文キャンセルAPIエラー: {e}")
        return None
def run_bot(exchange, fund_manager, dry_run=False, smtp_host=None, smtp_port=None, smtp_user=None, smtp_password=None, to_email=None):
    import numpy as np
    import talib
    from colorama import Fore, Style
    # --- closes_5m, ohlcv_15mを必ず定義（未定義参照エラー防止） ---
    closes_5m = []
    ohlcv_15m = []
    try:
        ohlcv_5m = exchange.fetch_ohlcv('BTC/JPY', timeframe='5m', limit=686)
        closes_5m = [float(r[4]) for r in ohlcv_5m if r and len(r) >= 5 and r[4] is not None]
    except Exception as e:
        print(f"[ERROR] 5分足データ取得エラー: {e}")

    # --- EMA系出力 ---
    ema488_5m = ema494_5m = ema405_5m = ema660_5m = None
    price_now = closes_5m[-1] if closes_5m else None
    if closes_5m and len(closes_5m) >= 488:
        ema488_5m = float(talib.EMA(np.array(closes_5m), timeperiod=488)[-1])
        print(f"[DEBUG] 5m足EMA488: {ema488_5m}")
    if closes_5m and len(closes_5m) >= 494:
        ema494_5m = float(talib.EMA(np.array(closes_5m), timeperiod=494)[-1])
    if closes_5m and len(closes_5m) >= 405:
        ema405_5m = float(talib.EMA(np.array(closes_5m), timeperiod=405)[-1])
    if closes_5m and len(closes_5m) >= 660:
        ema660_5m = float(talib.EMA(np.array(closes_5m), timeperiod=660)[-1])
        print(f"[DEBUG] 価格: {price_now}, EMA660: {ema660_5m}")
    print_table([
        ["5m足EMA488", ema488_5m],
        ["5m足EMA494 (pritry赤波線)", ema494_5m],
        ["5m足EMA405", ema405_5m],
        ["5m足EMA660 (赤波線)", ema660_5m],
        ["5m足現在価格", price_now]
    ], headers=["項目", "値"], color=Fore.MAGENTA)

    # --- 1時間足OHLCVデータ取得 ---
    print("[DEBUG] run_bot: 1時間足OHLCVデータ取得開始")
    try:
        ohlcv_1h = exchange.fetch_ohlcv('BTC/JPY', timeframe='1h', limit=300)
        print(f"[DEBUG] ohlcv_1h: {ohlcv_1h[:3]} ... total={len(ohlcv_1h)}")
        if not ohlcv_1h or len(ohlcv_1h) < 15:
            print(f"[WARN] 1時間足OHLCVデータが{len(ohlcv_1h) if ohlcv_1h else 0}本しかありません。5分足から合成します")
            ohlcv_5m = exchange.fetch_ohlcv('BTC/JPY', timeframe='5m', limit=300)
            ohlcv_1h = resample_ohlcv_5m_to_1h(ohlcv_5m)
            print(f"[DEBUG] 合成後の1時間足: {ohlcv_1h[:3]} ... total={len(ohlcv_1h)}")
        # --- 1時間足EMA100傾き率メインの買い判定ロジック ---
        closes_1h = [float(r[4]) for r in ohlcv_1h if r and len(r) >= 5 and r[4] is not None]
        can_buy = False
        if len(closes_1h) >= 103:
            import numpy as np
            import talib
            price_1h = closes_1h[-1]
            ema100_1h = float(talib.EMA(np.array(closes_1h), timeperiod=100)[-1])
            ema100_prev = float(talib.EMA(np.array(closes_1h[:-3]), timeperiod=100)[-1])
            ema100_slope = (ema100_1h - ema100_prev) / 3 / ema100_1h
            # ボリンジャーバンド
            sma_1h = float(talib.SMA(np.array(closes_1h), timeperiod=20)[-1])
            std_1h = float(np.std(closes_1h[-20:]))
            bb_upper_1h = sma_1h + 2 * std_1h
            bb_lower_1h = sma_1h - 2 * std_1h
            dist_to_bb_upper = bb_upper_1h - price_1h
            dist_to_bb_lower = price_1h - bb_lower_1h
            rsi_1h = float(talib.RSI(np.array(closes_1h), timeperiod=14)[-1]) if len(closes_1h) >= 15 else None
            print(f"[DEBUG] 1h足EMA100={ema100_1h}, EMA100傾き率={ema100_slope:.5f}, 現在価格={price_1h}")
            print(f"[DEBUG] 1h足BB上限={bb_upper_1h}, BB下限={bb_lower_1h}, BB上まで={dist_to_bb_upper}, BB下まで={dist_to_bb_lower}")
            if rsi_1h is not None:
                print(f"[DEBUG] 1h足RSI={rsi_1h}")
            # --- EMA100傾き率で判定 ---
            if ema100_slope < -0.001:
                print("[INFO] 1時間足: EMA100傾き率<-0.1%で買い禁止")
                can_buy = False
            elif price_1h < ema100_1h:
                # BB下限タッチ+RSI回復など厳しめ条件
                bb_touch = price_1h <= bb_lower_1h * 1.01  # BB下限付近
                rsi_recover = rsi_1h is not None and rsi_1h > 35
                if bb_touch and rsi_recover:
                    print("[INFO] 1時間足: EMA100下＆BB下限タッチ+RSI回復で買い可")
                    try:
                        # --- 5分足テーブル ---
                        print("[DEBUG] run_bot: get_ohlcv呼び出し前")
                        df = get_ohlcv(exchange, 'BTC/JPY', timeframe='5m', limit=300)
                        print(f"[DEBUG] 取得した5分足データ: ... total={len(df) if df is not None else 0}")
                        indicators_5m = compute_indicators(exchange, pair='BTC/JPY', timeframe='5m', limit=300, ohlcv_data=df)
                        ema100_slope_5m = None
                        rsi_5m = None
                        closes_5m = []
                        if indicators_5m:
                            import math
                            closes_5m = indicators_5m.get('closes', [])
                            rsi_5m = indicators_5m.get('rsi_14')
                            closes_5m_valid = [v for v in closes_5m if v is not None and not (isinstance(v, float) and math.isnan(v))]
                            rsi_5m_disp = int(rsi_5m) if rsi_5m is not None and not (isinstance(rsi_5m, float) and math.isnan(rsi_5m)) else "データ不足"
                            print_table([
                                ["5m足RSI", rsi_5m_disp],
                                ["5m足closes本数", len(closes_5m_valid)],
                                ["5m足closesサンプル", closes_5m_valid[:5] if closes_5m_valid else []]
                            ], headers=["項目", "値"], color=Fore.CYAN)

                            # --- 5分足EMA100傾き率計算と厳格な買い禁止判定 ---
                            if len(closes_5m) >= 103:
                                import numpy as np
                                import talib
                                ema100_now = float(talib.EMA(np.array(closes_5m), timeperiod=100)[-1])
                                ema100_prev = float(talib.EMA(np.array(closes_5m[:-3]), timeperiod=100)[-1])
                                ema100_slope_5m = (ema100_now - ema100_prev) / 3 / ema100_now  # 割合で傾き率
                                print(f"[DEBUG] 5m足EMA100傾き率: {ema100_slope_5m:.5f}")
                                if ema100_slope_5m < -0.0015:
                                    print(Fore.MAGENTA + f"[INFO] 5分足EMA100傾き率が-0.15%未満（{ema100_slope_5m*100:.3f}%）のため買い禁止" + Style.RESET_ALL)
                                    # ここで以降の買い判定ロジックをスキップする場合はreturnやフラグで制御可能
                                    # 例: return などで即時抜ける場合
                                    return
                        else:
                            print(f"[DEBUG] 5分足インジケータ: 取得失敗または空")
                    except Exception as e:
                        print(f"[ERROR] run_bot例外: {e}")
                        import traceback
                        traceback.print_exc()
        closes_15m = [float(r[4]) for r in ohlcv_15m if r and len(r) >= 5 and r[4] is not None]
        price_now_15m = closes_15m[-1] if closes_15m else None
        ema20_15m = talib.EMA(np.array(closes_15m), timeperiod=20)[-1] if len(closes_15m) >= 20 else None
        ema30_15m = talib.EMA(np.array(closes_15m), timeperiod=30)[-1] if len(closes_15m) >= 30 else None
        ema20_slope_15m = (ema20_15m - talib.EMA(np.array(closes_15m[:-3]), timeperiod=20)[-1]) / 3 if len(closes_15m) >= 23 else None
        ema30_slope_15m = (ema30_15m - talib.EMA(np.array(closes_15m[:-3]), timeperiod=30)[-1]) / 3 if len(closes_15m) >= 33 else None
        # 5分足データから15分相当のEMA（5分足EMA60=15分足EMA20, EMA90=15分足EMA30）
        ohlcv_5m = exchange.fetch_ohlcv('BTC/JPY', timeframe='5m', limit=180)
        closes_5m = [float(r[4]) for r in ohlcv_5m if r and len(r) >= 5 and r[4] is not None]
        ema60_5m = talib.EMA(np.array(closes_5m), timeperiod=60)[-1] if len(closes_5m) >= 60 else None
        ema90_5m = talib.EMA(np.array(closes_5m), timeperiod=90)[-1] if len(closes_5m) >= 90 else None
        ema60_slope_5m = (ema60_5m - talib.EMA(np.array(closes_5m[:-3]), timeperiod=60)[-1]) / 3 if len(closes_5m) >= 63 else None
        ema90_slope_5m = (ema90_5m - talib.EMA(np.array(closes_5m[:-3]), timeperiod=90)[-1]) / 3 if len(closes_5m) >= 93 else None
        print_table([
            ["15m足EMA20", ema20_15m],
            ["15m足EMA20傾き", ema20_slope_15m],
            ["15m足EMA30", ema30_15m],
            ["15m足EMA30傾き", ema30_slope_15m],
            ["15m足現在価格", price_now_15m],
            ["5m足EMA60(15m相当)", ema60_5m],
            ["5m足EMA60傾き", ema60_slope_5m],
            ["5m足EMA90(15m相当)", ema90_5m],
            ["5m足EMA90傾き", ema90_slope_5m]
        ], headers=["項目", "値"], color=Fore.YELLOW)
    except Exception as e:
        print(f"[ERROR] run_bot: 15分足EMA判定エラー: {e}")
    try:
        # --- 5分足テーブル ---
        print("[DEBUG] run_bot: get_ohlcv呼び出し前")
        df = get_ohlcv(exchange, 'BTC/JPY', timeframe='5m', limit=300)
        print(f"[DEBUG] 取得した5分足データ: ... total={len(df) if df is not None else 0}")
        indicators_5m = compute_indicators(exchange, pair='BTC/JPY', timeframe='5m', limit=300, ohlcv_data=df)
        ema100_slope_5m = None
        rsi_5m = None
        closes_5m = []
        if indicators_5m:
            import math
            closes_5m = indicators_5m.get('closes', [])
            rsi_5m = indicators_5m.get('rsi_14')
            closes_5m_valid = [v for v in closes_5m if v is not None and not (isinstance(v, float) and math.isnan(v))]
            rsi_5m_disp = int(rsi_5m) if rsi_5m is not None and not (isinstance(rsi_5m, float) and math.isnan(rsi_5m)) else "データ不足"
            print_table([
                ["5m足RSI", rsi_5m_disp],
                ["5m足closes本数", len(closes_5m_valid)],
                ["5m足closesサンプル", closes_5m_valid[:5] if closes_5m_valid else []]
            ], headers=["項目", "値"], color=Fore.CYAN)

            # --- 5分足EMA100傾き率計算と厳格な買い禁止判定 ---
            buy_candidate = False
            if len(closes_5m) >= 103:
                import numpy as np
                import talib
                ema100_now = float(talib.EMA(np.array(closes_5m), timeperiod=100)[-1])
                ema100_prev = float(talib.EMA(np.array(closes_5m[:-3]), timeperiod=100)[-1])
                ema100_slope_5m = (ema100_now - ema100_prev) / 3 / ema100_now  # 割合で傾き率
                print(f"[DEBUG] 5m足EMA100傾き率: {ema100_slope_5m:.5f}")
                price_now_5m = closes_5m[-1] if closes_5m else None
                ema20_5m = float(talib.EMA(np.array(closes_5m), timeperiod=20)[-1]) if len(closes_5m) >= 20 else None
                ema30_5m = float(talib.EMA(np.array(closes_5m), timeperiod=30)[-1]) if len(closes_5m) >= 30 else None
                # BB下限
                sma_5m = float(talib.SMA(np.array(closes_5m), timeperiod=20)[-1]) if len(closes_5m) >= 20 else None
                std_5m = float(np.std(closes_5m[-20:])) if len(closes_5m) >= 20 else None
                bb_lower_5m = sma_5m - 2 * std_5m if sma_5m is not None and std_5m is not None else None
                bb_touch = price_now_5m is not None and bb_lower_5m is not None and price_now_5m <= bb_lower_5m * 1.01
                rsi_entry = rsi_5m is not None and 35 <= rsi_5m <= 45

                # 1. EMA100傾きが下なら買い禁止
                if ema100_slope_5m < 0:
                    print(Fore.MAGENTA + f"[INFO] 5分足EMA100傾き率がマイナス（{ema100_slope_5m*100:.3f}%）のため買い禁止" + Style.RESET_ALL)
                    return
                # 2. 価格がEMA100の下でもおしめ条件なら買い候補
                if price_now_5m is not None and price_now_5m < ema100_now:
                    if ((ema20_5m is not None and abs(price_now_5m - ema20_5m) < ema20_5m*0.01) or
                        (ema30_5m is not None and abs(price_now_5m - ema30_5m) < ema30_5m*0.01)) and bb_touch and rsi_entry:
                        buy_candidate = True
                        print(Fore.GREEN + f"[INFO] 押し目条件成立: EMA20/30, BB下限, RSI35-45" + Style.RESET_ALL)
                # 3. 価格がEMA100以上なら通常買い候補
                elif price_now_5m is not None and price_now_5m >= ema100_now:
                    ohlcv_15m = []  # 例外時も未定義参照を防ぐため先に初期化
                    try:
                        # --- 15分足テーブル ---
                        ohlcv_15m = exchange.fetch_ohlcv('BTC/JPY', timeframe='15m', limit=50)
                        total_15m = len(ohlcv_15m) if ohlcv_15m is not None else 0
                        print(f"[DEBUG] 取得した15分足データ: ... total={total_15m}")
                        if not ohlcv_15m or total_15m == 0:
                            print("[WARN] 15分足データが取得できていません")
                            rsi_15m_disp2 = "データ不足"
                            closes_15m_sample = "データ不足"
                            closes_15m_valid = []
                        else:
                            indicators_15m = compute_indicators(exchange, pair='BTC/JPY', timeframe='15m', limit=50, ohlcv_data=ohlcv_15m)
                            import math
                            rsi_15m = indicators_15m.get('rsi_14') if indicators_15m else None
                            closes_15m = indicators_15m.get('closes', []) if indicators_15m else []
                            closes_15m_valid = [v for v in closes_15m if v is not None and not (isinstance(v, float) and math.isnan(v))]
                            if len(closes_15m_valid) >= 14 and rsi_15m is not None and not (isinstance(rsi_15m, float) and math.isnan(rsi_15m)):
                                rsi_15m_disp2 = int(rsi_15m)
                                closes_15m_sample = closes_15m_valid[:5] if closes_15m_valid else []
                            else:
                                print(f"[ERROR] 15分足データが15本未満です。RSI計算不可")
                                rsi_15m_disp2 = "データ不足"
                                closes_15m_sample = "データ不足"
                        print_table([
                            ["15m足RSI", rsi_15m_disp2],
                            ["15m足closes本数", len(closes_15m_valid)],
                            ["15m足closesサンプル", closes_15m_sample]
                        ], headers=["項目", "値"], color=Fore.GREEN)
                    except Exception as e:
                        print(f"[ERROR] run_bot: 15分足EMA判定エラー: {e}")
                        # 例外時もohlcv_15mを必ず初期化し、以降の参照でエラーにならないようにする
                        ohlcv_15m = []
                        closes_15m_valid = []
                        closes_15m_sample = "データ不足"
                        rsi_15m_disp2 = "データ不足"
                        try:
                            print_table([
                                ["15m足RSI", rsi_15m_disp2],
                                ["15m足closes本数", len(closes_15m_valid)],
                                ["15m足closesサンプル", closes_15m_sample]
                            ], headers=["項目", "値"], color=Fore.GREEN)
                        except Exception as e2:
                            print(f"[ERROR] 15分足テーブル出力エラー: {e2}")
    except Exception as e:
        print(f"[ERROR] run_bot例外: {e}")
        import traceback
        traceback.print_exc()

        # --- 15分足テーブル ---
        ohlcv_15m = exchange.fetch_ohlcv('BTC/JPY', timeframe='15m', limit=50)
        total_15m = len(ohlcv_15m) if ohlcv_15m is not None else 0
        print(f"[DEBUG] 取得した15分足データ: ... total={total_15m}")
        if not ohlcv_15m or total_15m == 0:
            print("[WARN] 15分足データが取得できていません")
            rsi_15m_disp2 = "データ不足"
            closes_15m_sample = "データ不足"
            closes_15m_valid = []
        else:
            indicators_15m = compute_indicators(exchange, pair='BTC/JPY', timeframe='15m', limit=50, ohlcv_data=ohlcv_15m)
            import math
            rsi_15m = indicators_15m.get('rsi_14') if indicators_15m else None
            closes_15m = indicators_15m.get('closes', []) if indicators_15m else []
            closes_15m_valid = [v for v in closes_15m if v is not None and not (isinstance(v, float) and math.isnan(v))]
            if len(closes_15m_valid) >= 14 and rsi_15m is not None and not (isinstance(rsi_15m, float) and math.isnan(rsi_15m)):
                rsi_15m_disp2 = int(rsi_15m)
                closes_15m_sample = closes_15m_valid[:5] if closes_15m_valid else []
            else:
                print(f"[ERROR] 15分足データが15本未満です。RSI計算不可")
                rsi_15m_disp2 = "データ不足"
                closes_15m_sample = "データ不足"
        print_table([
            ["15m足RSI", rsi_15m_disp2],
            ["15m足closes本数", len(closes_15m_valid)],
            ["15m足closesサンプル", closes_15m_sample]
        ], headers=["項目", "値"], color=Fore.GREEN)

    try:
        # --- シグナル判定 ---
        print("[DEBUG] run_bot: generate_signals呼び出し前")
        ohlcv_1h = exchange.fetch_ohlcv('BTC/JPY', timeframe='1h', limit=300)
        if not ohlcv_1h or len(ohlcv_1h) < 15:
            print(f"[WARN] 1時間足OHLCVデータが{len(ohlcv_1h) if ohlcv_1h else 0}本しかありません。5分足から合成します")
            ohlcv_5m = exchange.fetch_ohlcv('BTC/JPY', timeframe='5m', limit=300)
            ohlcv_1h = resample_ohlcv_5m_to_1h(ohlcv_5m)
            print(f"[DEBUG] 合成後の1時間足: {ohlcv_1h[:3]} ... total={len(ohlcv_1h)}")
        # シグナル生成
        df_1h = {'closes': [float(r[4]) for r in ohlcv_1h if r and len(r) >= 5 and r[4] is not None]}
        signal, message = generate_signals(df_1h)
        print(f"[DEBUG] run_bot: シグナル={signal}, メッセージ={message}")
        sell_signal = signal == 'sell_all'
        print(f"[INFO] シグナル: {signal}, 理由: {message}")

        # --- 買いシグナル時にメール通知 ---
        if signal == 'buy':
            send_notification(
                smtp_host, smtp_port, smtp_user, smtp_password, to_email,
                "【BTC自動売買】買いシグナル発生", f"買い時です！理由: {message}"
            )
            # 日本円残高取得
            free_jpy = None
            try:
                balance = exchange.fetch_balance()
                free_jpy = balance.get('free', {}).get('JPY', None)
            except Exception as e:
                print(f"[ERROR] JPY残高取得エラー: {e}")
            # 85%でBTC購入
            if free_jpy and price_now:
                btc_amount = round((free_jpy * 0.85) / price_now, 8)
                order_result = execute_order(exchange, 'BTC/JPY', 'buy', btc_amount, price_now)
                if order_result and not order_result.get('error'):
                    send_notification(
                        smtp_host, smtp_port, smtp_user, smtp_password, to_email,
                        "【BTC自動売買】買いました", f"BTCを購入しました。注文詳細: {order_result}"
                    )
        # --- 売りシグナル時はBTC全量売却 ---
        if signal == 'sell_all':
            # BTC残高取得
            btc_balance = None
            try:
                balance = exchange.fetch_balance()
                btc_balance = balance.get('free', {}).get('BTC', None)
            except Exception as e:
                print(f"[ERROR] BTC残高取得エラー: {e}")
            if btc_balance and price_now:
                order_result = execute_order(exchange, 'BTC/JPY', 'sell', btc_balance, price_now)
                if order_result and not order_result.get('error'):
                    send_notification(
                        smtp_host, smtp_port, smtp_user, smtp_password, to_email,
                        "【BTC自動売買】売りました", f"BTCを全量売却しました。注文詳細: {order_result}"
                    )
    except Exception as e:
        print(f"[ERROR] run_bot例外: {e}")
        import traceback
        traceback.print_exc()


import smtplib
from email.mime.text import MIMEText
from email.utils import formatdate

def send_notification(smtp_host, smtp_port, smtp_user, smtp_password, to_email, subject, message):
    """SMTPサーバーを使ってメール通知を送信する関数。"""
    try:
        msg = MIMEText(message, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = to_email
        msg["Date"] = formatdate(localtime=True)
        with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, [to_email], msg.as_string())
        print(f"[INFO] 通知メール送信完了: {subject}")
    except Exception as e:
        print(f"[ERROR] 通知メール送信失敗: {e}")
        
    # --- 5分足インジケータ取得・表示 ---
    try:
        # 5分足OHLCVデータ取得
        ohlcv_5m = exchange.fetch_ohlcv('BTC/JPY', timeframe='5m', limit=50)
        print(f"[DEBUG] 取得した5分足データ: ... total={len(ohlcv_5m) if ohlcv_5m is not None else 0}")
        indicators_5m_overview = compute_indicators(exchange, pair='BTC/JPY', timeframe='5m', limit=50, ohlcv_data=ohlcv_5m)
        if indicators_5m_overview:
            print(f"[DEBUG] 5分足インジケータ: rsi_14={indicators_5m_overview.get('rsi_14')}, closes本数={len(indicators_5m_overview.get('closes', []))}")
        else:
            print(f"[DEBUG] 5分足インジケータ: 取得失敗または空")
        rsi_5m_list = indicators_5m_overview.get('rsi_list', [])
        closes_5m = indicators_5m_overview.get('closes', [])
        rsi_5m = rsi_5m_list[-1] if rsi_5m_list else None
        closes_sample_5m = closes_5m[:5] if len(closes_5m) > 5 else closes_5m
        # RSI値がNone, NaN, 0の場合は"データ不足"や"計算不能"と表示
        if rsi_5m is None or rsi_5m != rsi_5m:
            rsi_5m_disp = "データ不足"
        elif rsi_5m == 0:
            rsi_5m_disp = "計算不能"
        else:
            rsi_5m_disp = int(rsi_5m)
        print_table([
            ["5m足RSI", rsi_5m_disp],
            ["5m足closes本数", len(closes_5m)],
            ["5m足closesサンプル", closes_sample_5m]
        ], headers=["項目", "値"], color=Fore.GREEN)
    except Exception as e:
        print(f"[ERROR] 5分足インジケータ取得エラー: {e}")
        print(f"[DEBUG] except到達: {e}")
    # --- 5分足インジケータ取得・表示 ---
    try:
        # 5分足OHLCVデータ取得
        ohlcv_5m = exchange.fetch_ohlcv('BTC/JPY', timeframe='5m', limit=50)
        indicators_5m_overview = compute_indicators(exchange, pair='BTC/JPY', timeframe='5m', limit=50, ohlcv_data=ohlcv_5m)
        rsi_5m_list = indicators_5m_overview.get('rsi_list', [])
        closes_5m = indicators_5m_overview.get('closes', [])
        rsi_5m = rsi_5m_list[-1] if rsi_5m_list else None
        closes_sample_5m = closes_5m[:5] if len(closes_5m) > 5 else closes_5m
        # RSI値がNone, NaN, 0の場合は"データ不足"や"計算不能"と表示
        if rsi_5m is None or rsi_5m != rsi_5m:
            rsi_5m_disp = "データ不足"
        elif rsi_5m == 0:
            rsi_5m_disp = "計算不能"
        else:
            rsi_5m_disp = int(rsi_5m)
        print_table([
            ["5m足RSI", rsi_5m_disp],
            ["5m足closes本数", len(closes_5m)],
            ["5m足closesサンプル", closes_sample_5m]
        ], headers=["項目", "値"], color=Fore.CYAN)
    except Exception as e:
        print(f"[ERROR] 5分足インジケータ取得エラー: {e}")
        print(f"[DEBUG] except到達: {e}")

    print("\n[INFO] --- 売買判定タイムフレームの説明 ---")
    print("[INFO] 1時間足: RSI35～40で反発時のみ売買判定を有効化")
    print("[INFO] 15分足: 買い時はRSI45以上でフィルタ、売り時はRSI60～65で警戒")
    print("[INFO] 5分足: RSI30割れ→35～38復帰で最終的な売買判定（エントリー/イグジット）")
    # --- 1時間足RSI反発フラグ ---
    rsi_1h_list = None
    rsi_1h_rebound = False
    try:
        rsi_1h_list = get_rsi_1h_series(exchange, pair='BTC/JPY')
        if rsi_1h_list and len(rsi_1h_list) >= 2:
            prev_rsi_1h = rsi_1h_list[-2]
            latest_rsi_1h = rsi_1h_list[-1]
            if 35 <= latest_rsi_1h <= 40 and latest_rsi_1h > prev_rsi_1h:
                rsi_1h_rebound = True
        # テーブル形式で1h足RSI情報を表示
        import math
        closes_1h_valid = [v for v in rsi_1h_list if v is not None and not (isinstance(v, float) and math.isnan(v))]
        rsi_1h_disp = int(latest_rsi_1h) if 'latest_rsi_1h' in locals() and latest_rsi_1h is not None and not (isinstance(latest_rsi_1h, float) and math.isnan(latest_rsi_1h)) else "データ不足"
        print_table([
            ["1h足RSI", rsi_1h_disp],
            ["1h足closes本数", len(closes_1h_valid)],
            ["1h足closesサンプル", closes_1h_valid[:5] if closes_1h_valid else []]
        ], headers=["項目", "値"], color=Fore.GREEN)
    except Exception as e:
        print(f"[ERROR] 1h足RSI反発判定エラー: {e}")

    # --- 1時間足インジケータ取得 ---
    try:
        indicators_1h = compute_indicators(exchange, pair='BTC/JPY', timeframe='1h', limit=50)
    except Exception as e:
        indicators_1h = None
        print(f"[ERROR] 1時間足インジケータ取得エラー: {e}")

    # --- 15分足インジケータ取得 ---
    try:
        ohlcv_15m = exchange.fetch_ohlcv('BTC/JPY', timeframe='15m', limit=300)
        total_15m = len(ohlcv_15m) if ohlcv_15m is not None else 0
        print(f"[DEBUG] 取得した15分足データ: ... total={total_15m}")
        if not ohlcv_15m or total_15m == 0:
            print("[WARN] 15分足データが取得できていません")
        else:
            print(f"[DEBUG] 15分足データ先頭: {ohlcv_15m[:2]}")
            print(f"[DEBUG] 15分足データ末尾: {ohlcv_15m[-2:]}")
        indicators_15m = compute_indicators(exchange, pair='BTC/JPY', timeframe='15m', limit=50, ohlcv_data=ohlcv_15m)
        import math
        rsi_15m = indicators_15m.get('rsi_14') if indicators_15m else None
        closes_15m = indicators_15m.get('closes', []) if indicators_15m else []
        closes_15m_valid = [v for v in closes_15m if v is not None and not (isinstance(v, float) and math.isnan(v))]
        rsi_15m_disp = int(rsi_15m) if rsi_15m is not None and not (isinstance(rsi_15m, float) and math.isnan(rsi_15m)) else "データ不足"
        if len(closes_15m_valid) >= 14 and rsi_15m is not None and not (isinstance(rsi_15m, float) and math.isnan(rsi_15m)):
            rsi_15m_disp2 = int(rsi_15m)
            closes_15m_sample = closes_15m_valid[:5] if closes_15m_valid else []
        else:
            print(f"[ERROR] 15分足データが15本未満です。RSI計算不可")
            rsi_15m_disp2 = "データ不足"
            closes_15m_sample = "データ不足"
        print_table([
            ["15m足RSI", rsi_15m_disp2],
            ["15m足closes本数", len(closes_15m_valid)],
            ["15m足closesサンプル", closes_15m_sample]
        ], headers=["項目", "値"], color=Fore.GREEN)
        # 15分足OHLCVデータを表形式で5本分表示（デバッグ出力付き）
        if ohlcv_15m and len(ohlcv_15m) > 0:
            ohlcv_headers = ["timestamp", "open", "high", "low", "close", "volume"]
            ohlcv_rows = ohlcv_15m[-5:] if len(ohlcv_15m) >= 5 else ohlcv_15m
            print(f"[DEBUG] 15分足OHLCVデータ（直近5本）: {ohlcv_rows}")
            print_table(ohlcv_rows, headers=ohlcv_headers, color=Fore.YELLOW)
        else:
            print("[WARN] 15分足OHLCVデータが空です")
        if not indicators_15m:
            print(f"[DEBUG] 15分足インジケータ: 取得失敗または空")
    except Exception as e:
        print(f"[ERROR] 15分足インジケータ取得エラー: {e}")
        # 例外時も必ずテーブルを出す
        print_table([
            ["15m足RSI", "データ不足"],
            ["15m足closes本数", "データ不足"],
            ["15m足closesサンプル", "データ不足"]
        ], headers=["項目", "値"], color=Fore.GREEN)

    except Exception as e:
        print(f"[ERROR] 5分足判定エラー: {e}")


    import time
    # --- 必要な定数・変数を関数内で初期化 ---
    PAIR = 'BTC/JPY'
    PROFIT_TAKE_PCT = 10.0
    BUY_MORE_PCT = 10.0
    MIN_ORDER_BTC = 0.0027
    positions_file = 'positions_state.json'
    state = {'last_buy_price': None}
    positions = []
    updated_positions = []
    last_buy_price = None
    # 本番運用: exchangeがNoneなら即エラー
    if exchange is None:
        raise RuntimeError("本番APIインスタンスが作成できませんでした。APIキー・シークレット・.env設定を再確認してください。")
    trade_log_file = 'trade_history.json'
    def log_trade(action, price, amount, status, reason=None):
        import json, datetime
        # 約定（filled/closed）のみ記録
        if status not in ("closed", "filled"):
            return
        try:
            # ログエントリを作成
            log_entry = {
                'action': action,
                'price': price,
                'amount': amount,
                'status': status,
                'reason': reason,
                'timestamp': datetime.datetime.now().isoformat()
            }
            # 既存のログファイルがあれば読み込み、なければ空リスト
            if os.path.exists(trade_log_file):
                with open(trade_log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            else:
                logs = []
            logs.append(log_entry)
            with open(trade_log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ERROR] 取引履歴保存エラー: {e}")

    import hashlib
    notify_state_file = 'notify_state.json'
    def should_notify_once(key_name):
        import json, time
        now = time.time()
        state_file = 'notify_once_state.json'
        state = {}
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
            except Exception:
                state = {}
        last = state.get(key_name, {'time': 0})
        # 1回だけ通知（ポジションが変化するまで）
        if now - last['time'] > 60*60*24:  # 24時間以上経過でリセット（安全策）
            state[key_name] = {'time': 0}
        if last['time'] == 0:
            state[key_name] = {'time': now}
            try:
                with open(state_file, 'w', encoding='utf-8') as f:
                    json.dump(state, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            return True
        return False

    buy_signal = False
    sell_signal = False
    while True:
        sell_signal = False
        buy_signal = False
        sell_signal = False
        # --- 1時間足逆張り禁止ロジックの閾値定義 ---
        RSI_H1_FORBIDDEN = 40
        RSI_H1_EXTREME = 30
        # --- 強いトレンド判定（例: 短期SMA > 長期SMA で強い上昇トレンド） ---
        # ※トレンド判定は従来通り、1時間足RSI判定はget_rsi_1h_seriesに統一
        indicators_trend = compute_indicators_long(exchange, pair='BTC/JPY', timeframe='1h', limit=50)
        sma_short = indicators_trend.get('sma_short_50', 0)
        sma_long = indicators_trend.get('sma_long_200', 0)
        is_strong_trend = sma_short > sma_long if sma_short and sma_long else False
        # 1時間足で逆張り可否を判定（get_rsi_1h_seriesの結果を使う）
        rsi_1h_list = get_rsi_1h_series(exchange, pair='BTC/JPY')
        rsi_1h = rsi_1h_list[-1] if rsi_1h_list and len(rsi_1h_list) > 0 else None
        can_counter_trade = False
        prev_rsi_1h = None
        latest_rsi_1h = None
        # --- 5分足・15分足RSIを毎ループ表示 ---
        try:
            indicators_5m = compute_indicators(exchange, pair='BTC/JPY', timeframe='5m', limit=50)
            rsi_5m_list = indicators_5m.get('rsi_list', [])
            rsi_5m = rsi_5m_list[-1] if rsi_5m_list else None
            print_table([
                ["5m足RSI", rsi_5m if rsi_5m is not None else "データ不足"]
            ], headers=["項目", "値"], color=Fore.CYAN)
        except Exception as e:
            print(f"[ERROR] 5分足RSI表示エラー: {e}")

        try:
            indicators_15m = compute_indicators(exchange, pair='BTC/JPY', timeframe='15m', limit=50)
            rsi_15m = indicators_15m.get('rsi_14')
            print_table([
                ["15m足RSI", rsi_15m if rsi_15m is not None else "データ不足"]
            ], headers=["項目", "値"], color=Fore.MAGENTA)
            try:
                indicators_1h = compute_indicators(exchange, pair='BTC/JPY', timeframe='1h', limit=50)
                print(f"[DEBUG] indicators_1h: {indicators_1h}")  # 追加: 1時間足インジケータの中身
                rsi_1h = indicators_1h.get('rsi_14')
                print(f"[DEBUG] rsi_1h: {rsi_1h}")  # 追加: 1時間足RSIの値
                print_table([
                    ["1h足RSI", rsi_1h if rsi_1h is not None else "データ不足"]
                ], headers=["項目", "値"], color=Fore.YELLOW)
            except Exception as e:
                print(f"[ERROR] 1時間足RSI表示エラー: {e}")
        except Exception as e:
            print(f"[ERROR] 15分足RSI表示エラー: {e}")
        if not rsi_1h_list or len(rsi_1h_list) < 2:
            print(f"[WARN] 1時間足のRSIリストが取得できません: {rsi_1h_list}", flush=True)
            # 1時間足が取得できない場合でも5分足シグナルを優先
            can_counter_trade = True
        else:
            prev_rsi_1h = rsi_1h_list[-2]
            latest_rsi_1h = rsi_1h_list[-1]
            # NoneやNaNが混じる場合も5分足シグナルを優先
            if prev_rsi_1h is None or latest_rsi_1h is None:
                print(f"[WARN] 1時間足RSIにNoneが含まれています: prev={prev_rsi_1h}, latest={latest_rsi_1h}")
                can_counter_trade = True
            elif (isinstance(prev_rsi_1h, float) and (prev_rsi_1h != prev_rsi_1h)) or (isinstance(latest_rsi_1h, float) and (latest_rsi_1h != latest_rsi_1h)):
                print(f"[WARN] 1時間足RSIにNaNが含まれています: prev={prev_rsi_1h}, latest={latest_rsi_1h}")
                can_counter_trade = True
            else:
                # 1時間足RSIが横ばい・反転兆候（レンジ・逆張り向き）
                if (latest_rsi_1h <= 40 and latest_rsi_1h > prev_rsi_1h) or (latest_rsi_1h >= 60 and latest_rsi_1h < prev_rsi_1h):
                    can_counter_trade = True
        if not can_counter_trade:
            print(f"[INFO] 1時間足で逆張り不可: RSI(前): {prev_rsi_1h}, RSI(最新): {latest_rsi_1h}", flush=True)
            time.sleep(30)
            try:
                ohlcv_15m = exchange.fetch_ohlcv('BTC/JPY', timeframe='15m', limit=300)
                total_15m = len(ohlcv_15m) if ohlcv_15m is not None else 0
                print(f"[DEBUG] 取得した15分足データ: ... total={total_15m}")
                if not ohlcv_15m or total_15m < 15:
                    print("[WARN] 15分足データが15本未満です。5分足から合成します")
                    ohlcv_5m = exchange.fetch_ohlcv('BTC/JPY', timeframe='5m', limit=300)
                    ohlcv_15m = resample_ohlcv_5m_to_15m(ohlcv_5m)
                    total_15m = len(ohlcv_15m) if ohlcv_15m is not None else 0
                    print(f"[DEBUG] 合成後の15分足: {ohlcv_15m[:3]} ... total={total_15m}")
                if not ohlcv_15m or total_15m == 0:
                    rsi_15m_disp2 = "データ不足"
                    closes_15m_sample = "データ不足"
                    closes_15m_valid = []
                else:
                    indicators_15m = compute_indicators(exchange, pair='BTC/JPY', timeframe='15m', limit=50, ohlcv_data=ohlcv_15m)
                    import math
                    rsi_15m = indicators_15m.get('rsi_14') if indicators_15m else None
                    closes_15m = indicators_15m.get('closes', []) if indicators_15m else []
                    closes_15m_valid = [v for v in closes_15m if v is not None and not (isinstance(v, float) and math.isnan(v))]
                    if len(closes_15m_valid) >= 14 and rsi_15m is not None and not (isinstance(rsi_15m, float) and math.isnan(rsi_15m)):
                        rsi_15m_disp2 = int(rsi_15m)
                        closes_15m_sample = closes_15m_valid[:5] if closes_15m_valid else []
                    else:
                        print(f"[ERROR] 15分足データが15本未満です。RSI計算不可")
                        rsi_15m_disp2 = "データ不足"
                        closes_15m_sample = "データ不足"
                # --- RSI反発ロジック ---
                rsi_list = []
            except Exception:
                bb_lower = None
                bb_upper = None
            # --- 5分足RSIリスト生成 ---
            rsi_list = []
            if indicators and 'rsi_list' in indicators:
                rsi_list = indicators['rsi_list']
            elif indicators and 'rsi_14_list' in indicators:
                rsi_list = indicators['rsi_14_list']

            rsi_buy_signal = False
            prev_rsi = None
            latest_rsi = None
            if rsi_list and len(rsi_list) >= 2:
                prev_rsi = rsi_list[-2]
                latest_rsi = rsi_list[-1]
                print(f"[DEBUG] 5m足RSI反発判定: prev_rsi={prev_rsi}, latest_rsi={latest_rsi}")
                if 25 <= prev_rsi <= 28 and latest_rsi > prev_rsi:
                    rsi_buy_signal = True
            else:
                print(f"[DEBUG] 5m足RSIリスト条件未達: rsi_list={rsi_list}")
            if rsi_buy_signal:
                print(f"[DEBUG] rsi_buy_signal=True: 買い注文ロジックに進みます")
                try:
                    smtp_host = os.getenv('SMTP_HOST')
                    smtp_port = int(os.getenv('SMTP_PORT', '465'))
                    smtp_user = os.getenv('SMTP_USER')
                    smtp_password = os.getenv('SMTP_PASS')
                    to_email = os.getenv('TO_EMAIL')
                    if smtp_host and to_email:
                        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        subject = f"【買いシグナル】RSI反発+BB-2σ下限 {now}"
                        message = f"【シグナル】\n時刻: {now}\n現在価格: {current_price} 円\nRSI(前): {prev_rsi}\nRSI(最新): {latest_rsi}\n条件揃いで買い推奨。"
                        send_notification(smtp_host, smtp_port, smtp_user, smtp_password, to_email, subject, message)
                except Exception as e:
                    print(f"⚠️ 買いシグナルメール送信エラー: {e}")

            # --- 5分足トレーリングストップ売り判定 ---
            trailing_start_pct = 0.05  # +5%でトレーリング開始
            trailing_width_pct = 0.05  # 直近高値から-5%で売り
            stop_loss_pct = -0.06      # -6%で損切り
            entry_price = None
            if positions and len(positions) > 0:
                entry_price = float(positions[0].get('price', 0))
            trailing_high = None
            try:
                closes_5m = [float(r[4]) for r in exchange.fetch_ohlcv('BTC/JPY', timeframe='5m', limit=300)]
                trailing_high = max(closes_5m) if closes_5m else None
            except Exception:
                trailing_high = None
            trailing_sell = False
            trailing_reason = ""
            # --- 価格・インジケータ取得 ---
            current_price = get_latest_price(exchange, PAIR)
            indicators = compute_indicators(exchange, pair=PAIR, timeframe='5m', limit=1000)
            rsi = indicators.get('rsi_14') if indicators else None
            bb_upper = None
            try:
                closes = [float(pos['price']) for pos in positions] if positions else [current_price]
                import numpy as np
                period = 14
                if len(closes) >= period:
                    sma = np.mean(closes[-period:])
                    std = np.std(closes[-period:])
                    bb_upper = sma + 2 * std
            except Exception:
                bb_upper = None

            if entry_price and trailing_high:
                if current_price >= entry_price * (1 + trailing_start_pct):
                    trigger_price = trailing_high * (1 - trailing_width_pct)
                    if current_price <= trigger_price:
                        trailing_sell = True
                        trailing_reason = f"トレーリングストップ発動: 直近高値({trailing_high})から-4%({trigger_price})割れ"
            stop_loss = False
            stop_loss_reason = ""
            if entry_price and current_price <= entry_price * (1 + stop_loss_pct):
                stop_loss = True
                stop_loss_reason = f"損切り発動: 買値({entry_price})から-6%({entry_price * (1 + stop_loss_pct)})割れ"
            if trailing_sell or stop_loss or ((rsi is not None and 65 <= rsi <= 75) and (bb_upper is not None and current_price >= bb_upper)):
                try:
                    smtp_host = os.getenv('SMTP_HOST')
                    smtp_port = int(os.getenv('SMTP_PORT', '465'))
                    smtp_user = os.getenv('SMTP_USER')
                    smtp_password = os.getenv('SMTP_PASS')
                    to_email = os.getenv('TO_EMAIL')
                    if smtp_host and to_email:
                        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        if trailing_sell:
                            subject = f"【売りシグナル】トレーリングストップ発動 {now}"
                            message = f"【トレーリングストップ】\n時刻: {now}\n現在価格: {current_price} 円\n直近高値: {trailing_high}\nトリガー価格: {trigger_price}\n買値: {entry_price}\n{trailing_reason}"
                        elif stop_loss:
                            subject = f"【売りシグナル】損切り発動 {now}"
                            message = f"【損切り】\n時刻: {now}\n現在価格: {current_price} 円\n買値: {entry_price}\n{stop_loss_reason}"
                        else:
                            subject = f"【売りシグナル】RSI65-75+BB+2σ上限 {now}"
                            message = f"【シグナル】\n時刻: {now}\n現在価格: {current_price} 円\nRSI: {rsi}\nBB上限: {bb_upper}\n条件揃いで売り推奨。"
                        send_notification(smtp_host, smtp_port, smtp_user, smtp_password, to_email, subject, message)
                except Exception as e:
                    print(f"⚠️ 売りシグナルメール送信エラー: {e}")
        time.sleep(30)

        # --- positions_state.jsonの再読込・保存・売買判定・注文・通知ロジックをここで統合 ---
        try:
            import json
            # positions_state.json再読込
            try:
                with open(positions_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    positions = loaded.get('positions', [])
                    last_buy_price = loaded.get('last_buy_price')
                elif isinstance(loaded, list):
                    positions = loaded
                    last_buy_price = positions[-1]['price'] if positions else None
                set_last_buy_price(state, last_buy_price)
            except Exception:
                positions = []
                set_last_buy_price(state, None)

            # --- API残高取得とDEBUG出力 ---
            try:
                api_balance = get_account_balance(exchange)
                btc_api_balance = api_balance.get('free', {}).get('BTC', 0)
                use_btc = btc_api_balance * 0.8
            except Exception:
                btc_api_balance = 0
                use_btc = 0

            # 現在価格取得
            current_price = get_latest_price(exchange, PAIR)
            btc_balance = sum([float(pos.get('amount', 0)) for pos in positions])
            import builtins
            builtins.positions = positions

            # --- インジケータ取得 ---
            indicators = compute_indicators(exchange, pair=PAIR, timeframe='5m', limit=1000)
            rsi = indicators.get('rsi_14')
            bb_lower = None
            try:
                closes = [float(pos['price']) for pos in positions] if positions else [current_price]
                import numpy as np
                period = 14
                if len(closes) >= period:
                    sma = np.mean(closes[-period:])
                    std = np.std(closes[-period:])
                    bb_lower = sma - 2 * std
            except Exception:
                bb_lower = None

            td = trade_decision(current_price, btc_balance, MIN_ORDER_BTC, last_buy_price, rsi, bb_lower)

            # --- メール通知ロジック ---
            smtp_host = os.getenv('SMTP_HOST')
            smtp_port = int(os.getenv('SMTP_PORT', '465'))
            smtp_user = os.getenv('SMTP_USER')
            smtp_password = os.getenv('SMTP_PASS')
            to_email = os.getenv('TO_EMAIL')
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            reason = ""
            import logging
            if td.get('action') == 'sell':
                reason = "利確条件成立（買値より10%以上上昇）"
                try:
                    sell_results = sell_all_positions(positions, exchange, PAIR)
                    for res in sell_results:
                        order = res.get('order')
                        if order and order.get('status') in ('closed', 'filled'):
                            positions = []
                            set_last_buy_price(state, None)
                            save_data = {
                                "positions": positions,
                                "last_buy_price": get_last_buy_price(state)
                            }
                            with open(positions_file, 'w', encoding='utf-8') as f:
                                json.dump(save_data, f, ensure_ascii=False, indent=2)
                            log_trade('sell', order.get('price'), order.get('amount'), order.get('status'), reason)
                            if smtp_host and to_email:
                                subject = f"【売り約定】BTC売却 {now}"
                                message = f"【売り約定】\n時刻: {now}\n価格: {order.get('price')} 円\n数量: {order.get('amount')} BTC\n注文ID: {order.get('id', order.get('order_id', 'N/A'))}\n根拠: {reason}"
                                send_notification(smtp_host, smtp_port, smtp_user, smtp_password, to_email, subject, message)
                        else:
                            print("[INFO] 売り注文は未約定またはエラーのため通知/ログしません")
                except Exception as e:
                    print(f"[ERROR] 売却処理例外: {e}", flush=True)
            elif td.get('action') == 'buy':
                if rsi is not None and rsi <= 35:
                    reason = f"RSIが35以下（現在値: {rsi}）"
                else:
                    reason = "買い条件成立"
                buy_cost = current_price * MIN_ORDER_BTC
                can_buy = (
                    not positions and
                    fund_manager.available_fund() - buy_cost >= 1000 and
                    (get_dry_run_flag() or last_buy_price is None or current_price < last_buy_price * 0.98)
                )
                if can_buy:
                    notify_key = "buy"
                    if should_notify_once(notify_key):
                        print(f"[INFO] buy判定: 資金・閾値チェックを通過して注文を出します")
                        if fund_manager.place_order(buy_cost):
                            order = execute_order(exchange, PAIR, 'buy', MIN_ORDER_BTC, current_price)
                            if order and order.get('status') in ('closed', 'filled'):
                                positions.append({'price': order.get('price', current_price), 'amount': order.get('amount', MIN_ORDER_BTC), 'timestamp': time.time()})
                                set_last_buy_price(state, order.get('price', current_price))
                                try:
                                    save_data = {
                                        "positions": positions,
                                        "last_buy_price": get_last_buy_price(state)
                                    }
                                    with open(positions_file, 'w', encoding='utf-8') as f:
                                        json.dump(save_data, f, ensure_ascii=False, indent=2)
                                except Exception as e:
                                    print(f"ポジション保存エラー: {e}")
                                if smtp_host and to_email:
                                    subject = f"【買い約定】BTC購入 {now}"
                                    message = f"【買い約定】\n時刻: {now}\n価格: {order.get('price')} 円\n数量: {order.get('amount')} BTC\n注文ID: {order.get('id', order.get('order_id', 'N/A'))}\n根拠: {reason}"
                                    send_notification(smtp_host, smtp_port, smtp_user, smtp_password, email_to, subject, message)
                                log_trade('buy', order.get('price'), order.get('amount'), order.get('status'), reason)
                            else:
                                print("[INFO] 買い注文は未約定またはエラーのため通知/ログしません")
            # --- ポジションが空のときだけ買い判定 ---
            if not positions:
                # ポジションが空ならbuy通知キーをリセット
                state_file = 'notify_once_state.json'
                if os.path.exists(state_file):
                    try:
                        import json
                        with open(state_file, 'r', encoding='utf-8') as f:
                            state_notify = json.load(f)
                        state_notify['buy'] = {'time': 0}
                        with open(state_file, 'w', encoding='utf-8') as f:
                            json.dump(state_notify, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
                prev_high = current_price
                buy_threshold = prev_high * 0.9
                buy_cost = current_price * MIN_ORDER_BTC
                if current_price <= buy_threshold and fund_manager.available_fund() - buy_cost >= 1000:
                    if fund_manager.place_order(buy_cost):
                        order = execute_order(exchange, PAIR, 'buy', MIN_ORDER_BTC, current_price)
                        updated_positions.append({'price': current_price, 'amount': MIN_ORDER_BTC, 'timestamp': time.time()})
                        set_last_buy_price(state, current_price)
                        try:
                            if smtp_host and email_to:
                                subject = f"BTC購入通知 {now}"
                                message = f"【BTC購入】\n時刻: {now}\n数量: {MIN_ORDER_BTC} BTC\n価格: {current_price} 円"
                                send_notification(smtp_host, smtp_port, smtp_user, smtp_password, email_to, subject, message)
                        except Exception as e:
                            print(f"⚠️ 購入通知メール送信エラー: {e}")
                        # ポジション情報とlast_buy_priceの保存
                        try:
                            save_data = {
                                "positions": updated_positions,
                                "last_buy_price": get_last_buy_price(state)
                            }
                            with open(positions_file, 'w', encoding='utf-8') as f:
                                json.dump(save_data, f, ensure_ascii=False, indent=2)
                        except Exception as e:
                            print(f"ポジション保存エラー: {e}")
            # ポジションがなくなったらsell通知キーもリセット
            if positions == []:
                state_file = 'notify_once_state.json'
                if os.path.exists(state_file):
                    try:
                        import json
                        with open(state_file, 'r', encoding='utf-8') as f:
                            state_notify = json.load(f)
                        state_notify['sell'] = {'time': 0}
                        with open(state_file, 'w', encoding='utf-8') as f:
                            json.dump(state_notify, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
        except Exception as e:
            print(f"[ERROR] run_botメインループ例外: {e}", flush=True)

        # --- 買い注文の未約定時リトライ ---
        limit_factor = 1.002  # デフォルト値を設定
        for _ in range(10):  # 最大10回リトライ（最大20分）
            time.sleep(30)  # 30秒待機
            open_orders = get_open_orders(exchange, PAIR)
            if not open_orders:
                print("[INFO] 買い注文が約定しました")
                break
            for o in open_orders:
                cancel_order(exchange, o['id'], PAIR)
            limit_price = get_latest_price(exchange, PAIR) * limit_factor
            order = execute_order(exchange, PAIR, 'buy', MIN_ORDER_BTC, limit_price)
            print(f"[INFO] 再指値買い注文発行: {limit_price}")

        # --- 売り判定（強いトレンド時も売りは実行） ---
        if sell_signal and positions:
            # 深夜判定関数
            def is_night():
                import datetime
                now = datetime.datetime.now().hour
                return now < 6 or now >= 23
            limit_factor = 1.004 if is_night() else 1.002  # 深夜は0.4%上、それ以外は0.2%上
            email_to = os.getenv('TO_EMAIL')
            if smtp_host and email_to:
                subject = f"【買い約定】BTC購入 {now}"
                message = f"【買い約定】\n時刻: {now}\n価格: {order.get('price')} 円\n数量: {order.get('amount')} BTC\n注文ID: {order.get('id', order.get('order_id', 'N/A'))}\n根拠: {reason}"
                send_notification(smtp_host, smtp_port, smtp_user, smtp_password, email_to, subject, message)
            if smtp_host and email_to:
                subject = f"BTC購入通知 {now}"
                message = f"【BTC購入】\n時刻: {now}\n数量: {MIN_ORDER_BTC} BTC\n価格: {current_price} 円"
                send_notification(smtp_host, smtp_port, smtp_user, smtp_password, email_to, subject, message)
            limit_price = current_price * limit_factor
            order = execute_order(exchange, PAIR, 'sell', MIN_ORDER_BTC, limit_price)
            print(f"[INFO] 指値売り注文発行: {limit_price}")
            for _ in range(10):
                time.sleep(30)
                open_orders = get_open_orders(exchange, PAIR)
                if not open_orders:
                    print("[INFO] 売り注文が約定しました")
                    break
                for o in open_orders:
                    cancel_order(exchange, o['id'], PAIR)
                limit_price = get_latest_price(exchange, PAIR) * limit_factor
                order = execute_order(exchange, PAIR, 'sell', MIN_ORDER_BTC, limit_price)
                print(f"[INFO] 再指値売り注文発行: {limit_price}")
        elif buy_signal and not positions and is_strong_trend:
            # --- forbidden判定（RSI・EMA傾き・RSI下降） ---
            forbidden = False
            rsi_H1_is_falling = False
            if rsi_1h is not None and prev_rsi_1h is not None:
                rsi_H1_is_falling = rsi_1h < prev_rsi_1h
            if (
                ema20_slope is not None and rsi_1h is not None and rsi_H1_is_falling and
                ema20_slope < -0.25 and rsi_1h < RSI_H1_FORBIDDEN
            ):
                forbidden = True
            # 例外：極端な売られ過ぎは逆張り許可
            if rsi_1h is not None and rsi_1h < RSI_H1_EXTREME:
                forbidden = False
                        # --- 1時間足終値リストを取得 ---
            closes_h1 = indicators_1h.get('closes', [])
            # --- rsi_1h, prev_rsi_1h, ema20_slopeを安全に定義 ---
            rsi_1h = None
            prev_rsi_1h = None
            ema20_slope = None
            if 'indicators_1h' in locals() and indicators_1h:
                if 'rsi_list' in indicators_1h and len(indicators_1h['rsi_list']) >= 2:
                    prev_rsi_1h = indicators_1h['rsi_list'][-2]
                    rsi_1h = indicators_1h['rsi_list'][-1]
                if 'closes' in indicators_1h and indicators_1h['closes'] and len(indicators_1h['closes']) >= 23:
                    ema20_slope = calc_ema_slope(indicators_1h['closes'], period=20, span=3)
            # --- scoreを動的に計算する例 ---
            score = 0
            # 5分足RSIが35~38なら+1
            if rsi_5m is not None and 35 <= rsi_5m <= 38:
                score += 1
            # 1時間足RSIが35~40で反発なら+1
            if rsi_1h is not None and 35 <= rsi_1h <= 40 and rsi_1h > prev_rsi_1h:
                score += 1
            # BB下限付近なら+1
            if bb_lower is not None and current_price <= bb_lower:
                score += 1
            # 強いトレンドなら+1
            if is_strong_trend:
                score += 1
            # forbiddenならscore-1
            forbidden = False  # forbidden未定義エラー対策
            if forbidden:
                score -= 1
            # --- ログ出力追加 ---
            try:
                logger.info(
                    f"H! forbidden=({forbidden}), "
                    f"score=({score}), "
                    f"rsi5m=({rsi_5m:.1f}), rsi1h=({rsi_1h:.1f}), bb_lower=({bb_lower}), price=({current_price})"
                )
            except Exception as e:
                print(f"[WARN] logger.info出力失敗: {e}")
            print("[INFO] 強いトレンド中のため逆張り買いを回避 (score計算済み)")

def compute_indicators_long(exchange, pair='BTC/JPY', timeframe='1h', limit=1000):
    # Fetch OHLCV and compute a set of indicators. Returns dict of values (may contain None).
    import time
    max_retries = 5
    backoff_base = 3
    for attempt in range(1, max_retries + 1):
        try:
            raw = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
            print(f"[DEBUG] fetch_ohlcv: timeframe={timeframe}, limit={limit}, 実際取得件数={len(raw)}")
            break
        except Exception as e:
            wait_sec = backoff_base * attempt
            print(f"[RETRY] fetch_ohlcv失敗 (attempt {attempt}): {e} {wait_sec}s後リトライ")
            if attempt == max_retries:
                # エラーログ保存
                with open('fetch_ohlcv_error.log', 'a', encoding='utf-8') as logf:
                    logf.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{pair} {timeframe}] fetch_ohlcv失敗: {e}\n")
                # 通知（約定通知と同じグローバル関数を利用）
                try:
                    send_notification('','', '', '', '', f"fetch_ohlcv失敗", f"{pair} {timeframe}でデータ取得に連続失敗: {e}")
                except Exception:
                    pass
                raw = []
            else:
                time.sleep(wait_sec)
    else:
        raw = []
    indicators = {}
    # prepare lists
    closes = [float(r[4]) for r in raw if r and len(r) >= 5 and r[4] is not None]
    # 足データの本数と一部サンプルを表で表示
    closes_sample = closes[:5] if len(closes) > 5 else closes
    print_table([
        [f"{timeframe}足closes本数", len(closes)],
        [f"closesサンプル", closes_sample]
    ], headers=["項目", "値"], color=Fore.BLUE)
    highs = [float(r[2]) for r in raw if r and len(r) >= 3 and r[2] is not None]
    lows = [float(r[3]) for r in raw if r and len(r) >= 4 and r[3] is not None]

    indicators['latest_close'] = closes[-1] if closes else None
    indicators['sma_short_50'] = compute_sma_from_list(closes, 50)
    indicators['sma_long_200'] = compute_sma_from_list(closes, 200)
    indicators['ema_12'] = compute_ema(closes, 12)
    indicators['ema_26'] = compute_ema(closes, 26)
    indicators['atr_14'] = compute_atr(raw, period=14)
    # RSIリスト（反発判定用）
    if len(closes) >= 14:
        # RSI計算前のclosesサンプルを表で表示
        print_table([
            ["RSI計算前closes本数", len(closes)],
            ["closesサンプル", closes[:5] if len(closes) > 5 else closes]
        ], headers=["項目", "値"], color=Fore.BLUE)
        rsi_list = []
        for i in range(len(closes)):
            if i+1 >= 14:
                rsi_val = compute_rsi(closes[i+1-14:i+1], period=14, exchange=exchange, pair=pair)
                rsi_list.append(rsi_val)
            else:
                rsi_list.append(None)
        # RSIリストの一部を表で表示
        print_table([
            ["RSIリスト本数", len(rsi_list)],
            ["RSIリストサンプル", rsi_list[-5:] if len(rsi_list) > 5 else rsi_list]
        ], headers=["項目", "値"], color=Fore.BLUE)
        indicators['rsi_list'] = rsi_list
        indicators['rsi_14'] = rsi_list[-1] if rsi_list else None
    else:
        print(f"[DEBUG] closesが14未満のためRSI計算不可: closes({len(closes)})")
        indicators['rsi_list'] = None
        indicators['rsi_14'] = None
    # recent high over 20 periods
    try:
        indicators['recent_high_20'] = max(highs[-20:]) if highs and len(highs) >= 1 else None
    except Exception:
        indicators['recent_high_20'] = None
    return indicators

# --- 未定義関数・変数のダミー実装 ---
def compute_sma_from_list(closes, period):
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period

def compute_ema(closes, period):
    if len(closes) < period:
        return None
    k = 2 / (period + 1)
    ema = closes[0]
    for price in closes[1:]:
        ema = price * k + ema * (1 - k)
    return ema

def compute_atr(raw, period=14):
    return 0.0


# --- RSI計算関数（引数名をvaluesに統一）---
def compute_rsi(values, period=14, exchange=None, pair='BTC/JPY', days=30):
    # Compute RSI from list of close prices. Returns float or None.
    try:
        if values is None:
            return None
        vals = [float(v) for v in values if v is not None]
        if len(vals) < period + 1:
            return None
        if exchange is not None:
            df = get_ohlcv(exchange, pair, timeframe='1d', limit=max(10, days + 5))
            if df is None or len(df) == 0:
                return None
            closes = [float(v) for v in df['close'] if v is not None]
            # ここで必要ならclosesを使った追加処理を記述
        # If no exchange, just compute RSI from values
        deltas = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception as e:
        try:
            log_warn(f"⚠️ RSI計算に失敗: {e}")
        except Exception:
            pass
        return None


def get_latest_price(exchange, pair='BTC/JPY'):
    import time
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            ticker = exchange.fetch_ticker(pair)
            if 'last' in ticker:
                return float(ticker['last'])
            else:
                print(f"[RETRY] 価格情報に'last'がありません (attempt {attempt})")
        except Exception as e:
            print(f"[RETRY] 価格取得失敗 (attempt {attempt}): {e}")
        time.sleep(30)
    print("[ERROR] 価格取得に3回失敗しました。Noneを返します。")
    return None

# --- 売買判定ロジック ---
def trade_decision(current_price, btc_balance=0.0027, buy_btc=None, last_buy_price=None, rsi=None, bb_lower=None):
    # current_price: 現在のBTC/JPY価格
    # btc_balance: 現在のBTC総保有量
    # buy_btc: 売買対象のBTC量(全保有BTCの80%を推奨)
    # last_buy_price: 直近の買値(売却判定に使用)
    # rsi: 最新のRSI値
    # bb_lower: ボリンジャーバンド下限
    # デバッグ用: 各値を出力
    print(f"[DEBUG] trade_decision: current_price={current_price}, btc_balance={btc_balance}, last_buy_price={last_buy_price}, rsi={rsi}, bb_lower={bb_lower}")
    # ポジションの最初の買値を基準にする
    # 買い時のBTC量は全保有BTCの80%を使う
    if buy_btc is None:
        buy_btc = round(btc_balance * 0.8, 8) if btc_balance > 0 else 0.002
    first_buy_price = None
    try:
        # グローバルなpositions変数があれば参照
        import builtins
        positions = getattr(builtins, 'positions', None)
        if positions and isinstance(positions, list) and len(positions) > 0:
            first_buy_price = float(positions[0].get('price', 0))
    except Exception:
        pass
    # fallback: last_buy_price
    if first_buy_price is None:
        first_buy_price = last_buy_price
    # 利確（+10%）条件を削除。売りはトレーリングストップ・損切りのみ。
    # 買い判定: BTC未保有、RSI<=35
    if btc_balance == 0 and (rsi is not None and rsi <= 35):
        return {'action': 'buy', 'amount': buy_btc, 'price': current_price, 'buy_condition': True}
    # 何もしない
    return {'action': 'hold', 'amount': 0.0, 'price': current_price, 'buy_condition': False, 'sell_condition': False}

# --- BTC残高を売買結果で更新する ---
def sell_all_positions(positions, exchange, pair):
    # 保有BTCの80%を売却する
    # positions: 保有ポジションリスト
    # exchange: ccxtの取引所オブジェクト
    # pair: 通貨ペア(例: 'BTC/JPY')
    results = []
    total_btc = sum([pos.get('amount', 0) for pos in positions])
    sell_amount = round(total_btc * 0.8, 8)
    if total_btc > 0 and sell_amount > 0:
        try:
            order = exchange.create_market_sell_order(pair, sell_amount)
            print(f"[DEBUG] 売却APIレスポンス: {order}", flush=True)
            with open("sell_log.txt", "a", encoding="utf-8") as logf:
                print(f"[DEBUG] 売却APIレスポンス: {order}", file=logf)
            results.append({'amount': sell_amount, 'order': order, 'status': 'sold'})
        except Exception as e:
            print(f"[ERROR] 売却APIエラー: {e}", flush=True)
            with open("sell_log.txt", "a", encoding="utf-8") as logf:
                print(f"[ERROR] 売却APIエラー: {e}", file=logf)
            results.append({'amount': sell_amount, 'error': str(e), 'status': 'error'})
    else:
        print(f"[ERROR] 売却可能なBTCが不足しています（保有: {total_btc} BTC）", flush=True)
        results.append({'amount': sell_amount, 'error': 'Insufficient BTC', 'status': 'error'})
    return results
def update_btc_balance(btc_balance, trade_result):
    # btc_balance: 現在のBTC残高
    # trade_result: nの戻り値(dict)
    action = trade_result.get('action')
    amount = trade_result.get('amount', 0.0)
    if action == 'sell':
        btc_balance -= amount
    elif action == 'buy':
        btc_balance += amount
    return btc_balance
import os
from dotenv import load_dotenv

import os
from dotenv import load_dotenv
from pathlib import Path
# --- .envの読み込みは最上部で必ず実行 ---
dotenv_path = os.getenv('DOTENV_PATH')
if not dotenv_path:
    # スクリプトと同じディレクトリを優先
    dotenv_path = str(Path(__file__).parent / '.env')
    if not os.path.exists(dotenv_path):
        # カレントディレクトリも試す
        dotenv_path = '.env'
load_dotenv(dotenv_path=dotenv_path)
def get_dry_run_flag():
    return str(os.getenv('DRY_RUN', '')).lower() in ('1', 'true', 'yes', 'on')
import logging




if __name__ == "__main__":
    import time
    print("[DEBUG] ファイル先頭到達")
    print("mainブロック実行開始")
    print("[INFO] .envからBITBANK_API_KEY/BITBANK_API_SECRETを自動ロードします")
    try:
        print("[DEBUG] Exchange初期化直前")
        exchange = create_bitbank_exchange()  # bitbank用ccxtラッパーを必ず利用
        print("[DEBUG] FundManager初期化直前")
        fund_manager = None  # 必要ならFundManagerクラスをここで初期化
        while True:
            print("[DEBUG] run_bot呼び出し直前")
            run_bot(exchange, fund_manager)
            time.sleep(30)
        print("[DEBUG] mainブロックtry直後（run_bot呼び出し後）")
    except Exception as e:
        print(f"[ERROR] mainブロック例外: {e}")
        print("[INFO] .envファイルにBITBANK_API_KEY/BITBANK_API_SECRETを設定してください")
        import traceback
        traceback.print_exc()


# --- メインループ（Botの実行部分） ---
    order = None
import json
from typing import Optional
from pathlib import Path
def _make_internal_fund_manager_class():
    class FundManagerStub:
        def __init__(self, initial_fund: float = 0.0, state_file: Optional[str] = None):
            import threading
            self._lock = threading.Lock()
            self._state_file = Path(state_file) if state_file else None
            self._available = float(initial_fund or 0.0)
            self._reserved = 0.0
            try:
                if self._state_file and self._state_file.exists():
                    raw = json.loads(self._state_file.read_text(encoding='utf-8'))
                    self._available = float(raw.get('available', self._available))
                    self._reserved = float(raw.get('reserved', 0.0))
            except Exception as e:
                pass
        def get_positions_reserved(self, positions_file='positions_state.json'):
            try:
                pf = Path(positions_file)
                if not pf.exists():
                    return 0.0
                with pf.open(encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict) and 'positions' in data:
                    data = data['positions']
                if not isinstance(data, list):
                    return 0.0
                total = 0.0
                for pos in data:
                    price = float(pos.get('price', 0))
                    amount = float(pos.get('amount', 0))
                    total += price * amount
                return total
            except Exception as e:
                try:
                    print(f"⚠️ positions_state.json予約額取得エラー: {e}")
                except Exception:
                    pass
                return 0.0
        def _persist(self):
            if not self._state_file:
                return
            try:
                obj = {'available': float(self._available), 'reserved': float(self._reserved)}
                self._state_file.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
            except Exception as e:
                print(f"⚠️ FundManagerStub保存エラー: {e}")

        def available_fund(self) -> float:
            try:
                return float(self._available)
            except Exception as e:
                print(f"⚠️ available_fund取得エラー: {e}")
                return 0.0 

        def place_order(self, cost: float) -> bool:
            try:
                c = float(cost)
            except Exception as e:
                print(f"⚠️ place_order変換エラー: {e}")
                return False
            with self._lock:
                if self._available < c:
                    return False
                self._available = float(self._available) - c
                self._persist()
            return True

        def add_funds(self, amount: float) -> None:
            try:
                a = float(amount)
            except Exception as e:
                print(f"⚠️ add_funds変換エラー: {e}")
                return
            with self._lock:
                self._available = float(self._available) + a
                self._persist()

        def reserve(self, cost: float) -> bool:
            try:
                c = float(cost)
            except Exception as e:
                print(f"⚠️ reserve変換エラー: {e}")
                return False
            with self._lock:
                reserved_from_positions = self.get_positions_reserved()
                # DRY_RUN時のみ予約額のデバッグ出力を行う
                try:
                    if str(os.getenv('DRY_RUN', '')).lower() in ('1', 'true', 'yes', 'on'):
                        print(f"予約フェーズ: 予約額（JPY）={reserved_from_positions}")
                except Exception:
                    pass
                if self._available - c < 1000:
                    print(f"[WARN] 注文後に1000円未満となるため予約不可: 残高={self._available}, コスト={c}")
                    return False
                self._available = float(self._available) - c
                self._reserved = float(self._reserved) + c
                self._persist()
                return True

        def confirm(self, cost: float) -> None:
            try:
                c = float(cost)
            except Exception as e:
                print(f"⚠️ confirm変換エラー: {e}")
                return
            with self._lock:
                self._reserved = max(0.0, float(self._reserved) - c)
                self._persist()

        def release(self, cost: float) -> None:
            try:
                c = float(cost)
            except Exception as e:
                print(f"⚠️ release変換エラー: {e}")
                return
            with self._lock:
                self._reserved = max(0.0, float(self._reserved) - c)
                self._available = float(self._available) + c
                self._persist()
    return FundManagerStub

def _make_internal_fund_manager_class():
    class FundManagerStub:
        def __init__(self, initial_fund: float = 0.0, state_file: Optional[str] = None):
            import threading
            self._lock = threading.Lock()
            self._state_file = Path(state_file) if state_file else None
            self._available = float(initial_fund or 0.0)
            self._reserved = 0.0
            try:
                if self._state_file and self._state_file.exists():
                    raw = json.loads(self._state_file.read_text(encoding='utf-8'))
                    self._available = float(raw.get('available', self._available))
                    self._reserved = float(raw.get('reserved', 0.0))
            except Exception as e:
                pass
        def get_positions_reserved(self, positions_file='positions_state.json'):
            try:
                pf = Path(positions_file)
                if not pf.exists():
                    return 0.0
                with pf.open(encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict) and 'positions' in data:
                    data = data['positions']
                if not isinstance(data, list):
                    return 0.0
                total = 0.0
                for pos in data:
                    price = float(pos.get('price', 0))
                    amount = float(pos.get('amount', 0))
                    total += price * amount
                return total
            except Exception as e:
                try:
                    print(f"⚠️ positions_state.json予約額取得エラー: {e}")
                except Exception:
                    pass
                return 0.0
        def _persist(self):
            if not self._state_file:
                return
            try:
                obj = {'available': float(self._available), 'reserved': float(self._reserved)}
                self._state_file.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
            except Exception as e:
                print(f"⚠️ FundManagerStub保存エラー: {e}")

        def available_fund(self) -> float:
            try:
                return float(self._available)
            except Exception as e:
                print(f"⚠️ available_fund取得エラー: {e}")
                return 0.0

        def place_order(self, cost: float) -> bool:
            try:
                c = float(cost)
            except Exception:
                return False
            with self._lock:
                if self._available < c:
                    return False
                self._available = float(self._available) - c
                self._persist()
            return True

        def add_funds(self, amount: float) -> None:
            try:
                a = float(amount)
            except Exception:
                return
            with self._lock:
                self._available = float(self._available) + a
                self._persist()

        def reserve(self, cost: float) -> bool:
            try:
                c = float(cost)
            except Exception:
                return False
            with self._lock:
                reserved_from_positions = self.get_positions_reserved()
                # DRY_RUN時のみ予約額のデバッグ出力を行う
                try:
                    if str(os.getenv('DRY_RUN', '')).lower() in ('1', 'true', 'yes', 'on'):
                        print(f"予約フェーズ: 予約額（JPY）={reserved_from_positions}")
                except Exception:
                    pass
                if self._available - c < 1000:
                    print(f"[WARN] 注文後に1000円未満となるため予約不可: 残高={self._available}, コスト={c}")
                    return False
                self._available = float(self._available) - c
                self._reserved = float(self._reserved) + c
                self._persist()
                return True

        def confirm(self, cost: float) -> None:
            try:
                c = float(cost)
            except Exception:
                return
            with self._lock:
                self._reserved = max(0.0, float(self._reserved) - c)
                self._persist()

        def release(self, cost: float) -> None:
            try:
                c = float(cost)
            except Exception:
                return
            with self._lock:
                self._reserved = max(0.0, float(self._reserved) - c)
                self._available = float(self._available) + c
                self._persist()
    return FundManagerStub

_InternalFundManager = _make_internal_fund_manager_class()
try:
    from funds import FundManager as _ImportedFundManager  # type: ignore
    required = ('available_fund', 'place_order', 'add_funds')
    if all(hasattr(_ImportedFundManager, name) for name in required):
        FundManager = _ImportedFundManager
    else:
        FundManager = _InternalFundManager
except Exception as e:
    print(f"[INFO] FundManager外部import失敗: {e}")
    FundManager = _InternalFundManager

class FundAdapter:
    def __init__(self, fund_manager=None, initial_fund: float = 0.0, dry_run: bool = False):
        import threading
        self._fund = fund_manager
        self._dry_run = bool(dry_run)
        self._local_total = float(initial_fund or 0.0)
        self._local_used = 0.0
        self._lock = threading.Lock()

    def available_fund(self) -> float:
        if self._fund is not None and not self._dry_run and hasattr(self._fund, 'available_fund'):
            try:
                return float(self._fund.available_fund())
            except Exception:
                pass
        with self._lock:
            return float(self._local_total) - float(self._local_used)

    def reserve(self, cost: float) -> bool:
        try:
            c = float(cost)
        except Exception:
            return False
        with self._lock:
            # 注文後に1000円以上残る場合のみ許可
            if self.available_fund() - c < 1000:
                return False
            self._local_used += c
            return True

    def place_order(self, cost: float) -> bool:
        return self.reserve(cost)

    def add_funds(self, amount: float) -> None:
        try:
            a = float(amount)
        except Exception:
            return
        with self._lock:
            self._local_total += a

    def confirm(self, cost: float) -> None:
        try:
            c = float(cost)
        except Exception:
            return
        with self._lock:
            self._local_used = max(0.0, self._local_used - c)

    def release(self, cost: float) -> None:
        try:
            c = float(cost)
        except Exception:
            return
        with self._lock:
            self._local_used = max(0.0, self._local_used - c)
            self._local_total += c

def _adapt_fund_manager_instance(fm):
    try:
        dry_run_env = str(os.getenv('DRY_RUN', '')).lower() in ('1', 'true', 'yes', 'on')
    except Exception:
        dry_run_env = False
    if fm is not None and all(hasattr(fm, name) for name in ('reserve', 'confirm', 'release', 'available_fund')):
        return fm
    return FundAdapter(fund_manager=fm, initial_fund=fm.fund if fm and hasattr(fm, 'fund') else 0.0)
import logging

# データ取得間隔（秒）
interval_seconds = 300
# --- ロギング関数の再定義 ---
def log_debug(*args, **kwargs):
    msg = ' '.join(str(a) for a in args)
    try:
        logging.getLogger().debug(msg)
    except Exception:
        pass
    try:
        print(msg, **kwargs)
    except Exception:
        print(msg)

def log_error(*args, **kwargs):
    msg = ' '.join(str(a) for a in args)
    try:
        logging.getLogger().error(msg)
    except Exception:
        pass
    try:
        print(msg, **kwargs)
    except Exception:
        print(msg)
def log_info(*args, **kwargs):
    msg = ' '.join(str(a) for a in args)
    try:
        logging.getLogger().info(msg)
    except Exception:
        pass
    try:
        print(msg, **kwargs)
    except Exception:
        print(msg)

def log_warn(*args, **kwargs):
    msg = ' '.join(str(a) for a in args)
    try:
        logging.getLogger().warning(msg)
    except Exception:
        pass
    try:
        print(msg, **kwargs)
    except Exception:
        print(msg)


# === DI対応版のエントリーポイント ===

# --- 未定義グローバル変数・定数・関数のダミー定義・import ---
import os

# Add FileLock import for file locking
try:
    from filelock import FileLock
except ImportError:
    FileLock = None  # Fallback if filelock is not installed
env_paths = ['.env']


# --- ロギング関数の再定義 ---
import datetime


# --- .envの読み込みは最上部で必ず実行 ---
import os
from dotenv import load_dotenv
from pathlib import Path
dotenv_path = os.getenv('DOTENV_PATH') or str(Path(__file__).parent / '.env')
load_dotenv(dotenv_path=dotenv_path)
def get_dry_run_flag():
    return str(os.getenv('DRY_RUN', '')).lower() in ('1', 'true', 'yes', 'on')
import logging
import hashlib
import requests

# config.jsonからAPIキー・シークレットを読み込む
def load_api_keys(config_path='config.json'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config['api_key'], config['api_secret']

# bitbankの残高取得APIを呼び出す
def get_bitbank_balance(api_key, api_secret):
    # 本番用API呼び出し例（未使用なら削除可）
    try:
        import hmac
        import time
        url = 'https://api.bitbank.cc/v1/user/assets'
        nonce = str(int(time.time() * 1000))
        payload = ''
        message = nonce + api_key + payload
        sign = hmac.new(api_secret.encode('utf-8'), message.encode('utf-8'), hashlib.sha256).hexdigest()
        headers = {
            'ACCESS-KEY': api_key,
            'ACCESS-NONCE': nonce,
            'ACCESS-SIGNATURE': sign,
            'Content-Type': 'application/json'
        }
        response = requests.get(url, headers=headers)
        return response.json()
    except Exception as e:
        print(f"[ERROR] bitbank残高API呼び出し失敗: {e}")
        return {}

# --- connect_to_bitbank: Bitbank用の簡易接続関数（未定義エラー対策のダミー実装） ---
    # connect_to_bitbankのダミー実装（未使用なら削除可）
    api_key = os.getenv("API_KEY")
    secret_key = os.getenv("SECRET_KEY")
    try:
        import ccxt
        return ccxt.bitbank({
            'apiKey': api_key or "",
            'secret': secret_key or "",
        })
    except Exception as e:
        print(f"[ERROR] ccxt.bitbank初期化失敗: {e}")
        return None

# --- 残高取得ユーティリティ ---
from typing import Dict, Any

def get_account_balance(exchange) -> Dict[str, Dict[str, Any]]:
    # 残高情報を取得
    # Returns: dict { 'total': {...}, 'free': {...}, 'used': {...} }
    try:
        if str(os.getenv('DRY_RUN', '0')).lower() in ('1', 'true', 'yes', 'on'):
            return {
                'total': {'JPY': 100000.0, 'BTC': 0.0},
                'free': {'JPY': 100000.0, 'BTC': 0.0},
                'used': {'JPY': 0.0, 'BTC': 0.0}
            }
        balance = exchange.fetch_balance()
        return {
            'total': balance.get('total', {}),
            'free': balance.get('free', {}),
            'used': balance.get('used', {})
        }
    except Exception as e:
        print(f"[ERROR] 残高取得エラー: {e}")
        return {'total': {}, 'free': {}, 'used': {}}


def get_open_orders(exchange, pair='BTC/JPY', limit=50):
    # Get active open orders (unfilled orders). Returns a list of order info dicts.
    try:
        orders = exchange.fetch_orders(pair, limit=limit)
        result = []
        for order in orders:
            try:
                result.append({
                    'id': order.get('id'),
                    'symbol': order.get('symbol'),
                    'type': order.get('type'),
                    'side': order.get('side'),
                    'price': float(order.get('price', 0)) if order.get('price') else None,
                    'amount': float(order.get('amount', 0)),
                    'filled': float(order.get('filled', 0)),
                    'remaining': float(order.get('remaining', 0)),
                    'cost': float(order.get('cost', 0)) if order.get('cost') else None,
                    'status': order.get('status'),
                    'timestamp': order.get('timestamp'),
                    'datetime': order.get('datetime')
                })
            except Exception:
                continue
        return result
    except Exception as e:
        try:
            log_error(f"❌ 注文履歴取得エラー: {e}")
        except Exception:
            pass
        return []


def cancel_order(exchange, order_id, pair='BTC/JPY'):
    # Cancel the order with the specified order ID. Returns dict or None.
    try:
        if str(os.getenv('DRY_RUN', '0')).lower() in ('1', 'true', 'yes', 'on'):
            print(f"🔧 DRY_RUN: 注文キャンセル（ID: {order_id}）はシミュレーションです")
            return {'id': order_id, 'status': 'canceled'}
        result = exchange.cancel_order(order_id, pair)
        try:
            log_info(f"✅ 注文キャンセル成功: ID={order_id}")
        except Exception:
            pass
        return result
    except Exception as e:
        try:
            log_error(f"❌ 注文キャンセルエラー: {e}")
        except Exception:
            pass
        return None


def get_my_trades(exchange, pair='BTC/JPY', limit=100):
    # Get your trade history (private API).
    try:
        if str(os.getenv('DRY_RUN', '0')).lower() in ('1', 'true', 'yes', 'on'):
            return []
        trades = exchange.fetch_my_trades(pair, limit=limit)
        result = []
        for trade in trades:
            try:
                result.append({
                    'id': trade.get('id'),
                    'order': trade.get('order'),
                    'symbol': trade.get('symbol'),
                    'type': trade.get('type'),
                    'side': trade.get('side'),
                    'price': float(trade.get('price', 0)),
                    'amount': float(trade.get('amount', 0)),
                    'cost': float(trade.get('cost', 0)),
                    'fee': trade.get('fee'),
                    'timestamp': trade.get('timestamp'),
                    'datetime': trade.get('datetime')
                })
            except Exception:
                continue
        return result
    except Exception as e:
        try:
            log_error(f"❌ 約定履歴取得エラー: {e}")
        except Exception:
            pass
        return []


def get_deposit_address(exchange, currency='BTC'):
    # Get deposit address for withdrawal.
    try:
        if str(os.getenv('DRY_RUN', '0')).lower() in ('1', 'true', 'yes', 'on'):
            return {'address': 'dry_run_address', 'tag': None, 'currency': currency}
        address_info = exchange.fetch_deposit_address(currency)
        return {
            'address': address_info.get('address'),
            'tag': address_info.get('tag'),
            'currency': address_info.get('currency'),
            'network': address_info.get('network')
        }
    except Exception as e:
        try:
            log_error(f"❌ デポジットアドレス取得エラー: {e}")
        except Exception:
            pass
        return {}


def request_withdrawal(exchange, currency, amount, address, tag=None):
    # Request withdrawal
    try:
        if str(os.getenv('DRY_RUN', '0')).lower() in ('1', 'true', 'yes', 'on'):
            # DRY_RUN: Withdrawal request simulation
            print(f"🔧 DRY_RUN: Withdrawal request simulation ({amount} {currency} → {address})")
            return {'id': 'dry_withdraw_id', 'currency': currency, 'amount': amount}
        params = {}
        if tag:
            params['tag'] = tag
        result = exchange.withdraw(currency, amount, address, params=params)
        try:
            # Withdrawal request succeeded
            log_info(f"✅ Withdrawal request succeeded: {amount} {currency}")
        except Exception:
            pass
        return result
    except Exception as e:
        try:
            # Withdrawal request error
            log_error(f"❌ Withdrawal request error: {e}")
        except Exception:
            pass
        return None


def compute_dynamic_threshold(exchange, pair='BTC/JPY', days=30, buffer_jpy=500, buffer_pct=0.01):
    # Compute dynamic threshold from past OHLCV data
    try:
        def get_ohlcv(exchange, pair='BTC/JPY', timeframe='1d', limit=100):
            try:
                raw = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
                if not raw or len(raw) == 0:
                    return None
                df = pd.DataFrame(raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df
            except Exception as e:
                print(f"⚠️ OHLCV取得エラー: {e}")
                return None

        df = get_ohlcv(exchange, pair, timeframe='1d', limit=max(10, days + 5))
        if df is None or len(df) == 0:
            return None, None, None
        closes = []
        try:
            closes = [float(v) for v in df['close'] if v is not None]
        except Exception:
            for i in range(len(df)):
                try:
                    closes.append(float(df.iloc[i]['close']))
                except Exception:
                    pass
        if not closes:
            return None, None, None
        min_close = min(closes)
        max_close = max(closes)
        try:
            ratio = float(os.environ.get('DYN_THRESHOLD_RATIO', 1.0))
        except Exception:
            ratio = 1.0
        if ratio and float(ratio) > 0:
            threshold = float(min_close) + (float(max_close) - float(min_close)) * float(ratio)
        elif buffer_jpy and float(buffer_jpy) > 0:
            threshold = float(min_close) + float(buffer_jpy)
        else:
            threshold = float(min_close) * (1.0 + float(buffer_pct))
        return float(threshold), float(min_close), float(max_close)
    except Exception as e:
        try:
            log_warn(f"⚠️ dynamic threshold computation failed: {e}")
        except Exception:
            pass
        return None, None, None


def get_ohlcv(exchange, pair='BTC/JPY', timeframe='1d', limit=100):
    # Fetch OHLCV data and return as DataFrame
    try:
        raw = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
        if not raw or len(raw) == 0:
            return None
        df = pd.DataFrame(raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"⚠️ OHLCV取得エラー: {e}")
        return None

def compute_sma_from_ohlcv(exchange, pair='BTC/JPY', days=30):
    # Calculate simple moving average (SMA) from daily OHLCV. Return None on failure.
    df = get_ohlcv(exchange, pair, timeframe='1d', limit=max(10, days + 5))
    if df is None or len(df) < days:
        return None
    vals = []
    try:
        for i in range(-days, 0):
            vals.append(float(df['close'].iloc[i]))
    except Exception:
        pass
    return sum(vals) / len(vals) if vals else None


def get_recent_high(exchange, pair='BTC/JPY', days=30):
    # Return max high value in last N days. Return None on failure.
    df = get_ohlcv(exchange, pair, timeframe='1d', limit=max(10, days + 5))
    if df is None or len(df) == 0:
        return None
    try:
        highs = [float(v) for v in df['high'] if v is not None]
    except Exception:
        highs = []
    return max(highs) if highs else None


def compute_ema(values, period):
    # Exponential moving average for last `period` values. Returns None if insufficient data.
    try:
        vals = [float(v) for v in values if v is not None]
        if len(vals) < period or period <= 0:
            return None
        k = 2.0 / (period + 1)
        # start with SMA for first EMA
        ema = sum(vals[-period:]) / float(period)
        for price in vals[-period + 1:]:
            ema = price * k + ema * (1 - k)
        return ema
    except Exception:
        return None


def compute_atr(ohlcv_rows, period=14):
    # Compute ATR (Average True Range) from OHLCV rows (list of [ts, o, h, l, c, v] or DataFrame-like). Returns ATR float or None.
    try:
        # normalize input into list of tuples (o,h,l,c)
        rows = []
        if ohlcv_rows is None:
            return None
        # If it's a DataFrame-like object
        if hasattr(ohlcv_rows, 'iloc'):
            for i in range(len(ohlcv_rows)):
                try:
                    row = ohlcv_rows.iloc[i]
                    rows.append((float(row['open']), float(row['high']), float(row['low']), float(row['close'])))
                except Exception:
                    pass
        else:
            for r in ohlcv_rows:
                try:
                    # r may be [ts,o,h,l,c,v]
                    if len(r) >= 5:
                        # r[1]=open, r[2]=high, r[3]=low, r[4]=close
                        rows.append((float(r[1]), float(r[2]), float(r[3]), float(r[4])))
                except Exception:
                    pass

        if len(rows) < period + 1:
            return None

        trs = []
        for i in range(1, len(rows)):
            prev_close = rows[i - 1][3]
            high = rows[i][1]
            low = rows[i][2]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)

        if len(trs) < period:
            return None
        # simple moving average of last `period` TRs
        return sum(trs[-period:]) / float(period)
    except Exception:
        return None


def compute_rsi(values, period=14, exchange=None, pair='BTC/JPY', days=30):
    # Compute RSI from list of close prices. Returns float or None.
    try:
        if values is None:
            return None
        vals = [float(v) for v in values if v is not None]
        if len(vals) < period + 1:
            return None
        if exchange is not None:
            df = get_ohlcv(exchange, pair, timeframe='1d', limit=max(10, days + 5))
            if df is None or len(df) == 0:
                return None
            closes = [float(v) for v in df['close'] if v is not None]
            # ここで必要ならclosesを使った追加処理を記述
        # If no exchange, just compute RSI from values
        deltas = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception as e:
        try:
            log_warn(f"⚠️ RSI計算に失敗: {e}")
        except Exception:
            pass
        return None


def get_recent_trades(exchange, pair='BTC/JPY', limit=100):
    # Get recent trade history as a list of dicts.
    try:
        trades = exchange.fetch_trades(pair, limit=limit)
        result = []
        for trade in trades:
            try:
                result.append({
                    'timestamp': trade.get('timestamp'),
                    'datetime': trade.get('datetime'),
                    'price': float(trade.get('price', 0)),
                    'amount': float(trade.get('amount', 0)),
                    'side': trade.get('side', 'unknown')
                })
            except Exception:
                continue
        return result
    except Exception as e:
        try:
            log_warn(f"Failed to fetch trade history: {e}")
        except Exception:
            pass
        return []


def analyze_orderbook_pressure(orderbook_data):
    # Analyze buy/sell pressure from order book.
    try:
        bids = orderbook_data.get('bids', [])
        asks = orderbook_data.get('asks', [])
        
        buy_volume = sum(float(bid[1]) for bid in bids if len(bid) >= 2)
        sell_volume = sum(float(ask[1]) for ask in asks if len(ask) >= 2)
        
        ratio = None
        signal = 'NEUTRAL'
        
        if sell_volume > 0:
            ratio = buy_volume / sell_volume
            if ratio > 1.2:
                signal = 'BULLISH'  # 買い圧力が強い
            elif ratio < 0.8:
                signal = 'BEARISH'  # 売り圧力が強い
        
        return {
            'buy_pressure': buy_volume,
            'sell_pressure': sell_volume,
            'pressure_ratio': ratio,
            'signal': signal
        }
    except Exception:
        return {
            'buy_pressure': 0,
            'sell_pressure': 0,
            'pressure_ratio': None,
            'signal': 'NEUTRAL'
        }



def compute_sma_from_list(values, period):
    # Compute simple moving average from a list
    if not values or len(values) < period:
        return None
    return sum(values[-period:]) / period


def write_indicators_csv(indicators: dict, pair: str, signal: str = 'NONE', csv_path='indicators.csv'):
    # Append indicators as a CSV row. Creates header if file does not exist.
    try:
        import csv
        file_exists = os.path.exists(csv_path)
        with open(csv_path, 'a', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            if not file_exists:
                writer.writerow(['timestamp', 'pair', 'price', 'sma_short_50', 'sma_long_200', 'ema_12', 'ema_26', 'atr_14', 'rsi_14', 'recent_high_20', 'signal'])
            ts = datetime.datetime.now(JST).isoformat()
            # 必要ならここでlogger.infoを呼ぶ（値は関数引数で渡すこと）
            logger.info(
                f"CSV: ts={ts}, pair={pair}, price={indicators.get('price')}, "
                f"sma_short_50={indicators.get('sma_short_50')}, sma_long_200={indicators.get('sma_long_200')}, "
                f"ema_12={indicators.get('ema_12')}, ema_26={indicators.get('ema_26')}, atr_14={indicators.get('atr_14')}, "
                f"rsi_14={indicators.get('rsi_14')}, recent_high_20={indicators.get('recent_high_20')}, signal={signal}"
            )
            writer.writerow([
                ts,
                pair,
                indicators.get('price'),
                indicators.get('sma_short_50'),
                indicators.get('sma_long_200'),
                indicators.get('ema_12'),
                indicators.get('ema_26'),
                indicators.get('atr_14'),
                indicators.get('rsi_14'),
                indicators.get('recent_high_20'),
                signal
            ])
    except Exception:
        # never raise from logging function
        pass


# -----------------------------
# ヘルパー: 手数料考慮の数量計算
# -----------------------------
import math

def round_down_qty(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    factor = 1.0 / step
    return math.floor(qty * factor) / factor


def compute_qty_for_budget_with_fee(reserved_jpy: float, price_jpy: float,
                                    min_btc: float = 0.0001, step: float = 0.0001,
                                    available_jpy: float = 0.0, balance_buffer: float = 0.0):
    # Return (qty, cost_jpy, fee_jpy) for given budget and price.
    try:
        fee_rate = float(os.getenv('FEE_RATE', '0.001'))
    except Exception:
        fee_rate = 0.001
    try:
        fee_fixed = float(os.getenv('FEE_FIXED_JPY', '0.0'))
    except Exception:
        fee_fixed = 0.0

    if price_jpy <= 0 or reserved_jpy <= 0:
        return 0.0, 0.0, 0.0

    max_allowed_jpy = reserved_jpy
    if available_jpy is not None:
        try:
            max_allowed_jpy = min(max_allowed_jpy, float(available_jpy) - float(balance_buffer))
        except Exception:
            max_allowed_jpy = reserved_jpy

    if max_allowed_jpy <= 0:
        return 0.0, 0.0, 0.0

    # 手数料込みで概算できる最大qty
    approx_qty = max_allowed_jpy / (price_jpy * (1.0 + fee_rate))
    qty = round_down_qty(approx_qty, step)

    # 最小数量を満たしているか
    if qty < min_btc:
        # AUTO_RESIZE を許可していれば一段階だけ増やして試す
        if os.getenv('AUTO_RESIZE', '0') == '1':
            try:
                max_mult = float(os.getenv('AUTO_RESIZE_MAX_MULTIPLIER', '1.5'))
            except Exception:
                max_mult = 1.5
            # 再計算
            approx_qty = (max_allowed_jpy * max_mult) / (price_jpy * (1.0 + fee_rate))
            qty = round_down_qty(approx_qty, step)
            if qty < min_btc:
                return 0.0, 0.0, 0.0
        else:
            return 0.0, 0.0, 0.0

    # コストと手数料を計算
    cost_jpy = qty * price_jpy
    fee_jpy = cost_jpy * fee_rate + fee_fixed
    return qty, cost_jpy, fee_jpy



def get_last_buy_time(state):
    return state.get("last_buy_time")


def set_last_buy_time(state, ts=None):
    import time
    state["last_buy_time"] = ts or int(time.time())


def record_position(state, side, price, qty):
    import time
    print("DEBUG: record_position called", side, price, qty)
    state.setdefault("positions", [])
    state["positions"].append({
        "side": side,
        "price": float(price),
        "qty": float(qty),
        "time": int(time.time())
    })
    if len(state["positions"]) > 50:
        state["positions"] = state["positions"][-50:]
    print(f"DEBUG: record_position saving state with positions={state['positions']}")
    # save_state(state)  # 削除済み
    print("DEBUG: record_position finished")


def is_slippage_too_large(reference_price, latest_price):
    print("DEBUG: save_state called")
    try:
        if reference_price is None or latest_price is None:
            return False
        reference_price = float(reference_price)
        latest_price = float(latest_price)
        if reference_price == 0:
            return False
        delta_pct = abs((latest_price - reference_price) / reference_price) * 100.0
        return delta_pct > 5.0  # MAX_SLIPPAGE_PCTの代用値
    except Exception:
        return False

# === 3. 売買シグナルの判定（MA 25/75/200 + 買い増しロジック） ===
def generate_signals(df):
    # テスト用: DRY_RUNかつFORCE_SELL_SIGNAL環境変数が有効なら必ず売りシグナル
    if str(os.getenv('DRY_RUN', '0')).lower() in ('1', 'true', 'yes', 'on') and str(os.getenv('FORCE_SELL_SIGNAL', '0')).lower() in ('1', 'true', 'yes', 'on'):
        return 'sell_all', '【テスト】FORCE_SELL_SIGNALにより強制売りシグナル発生'
    # Generate buy/sell signals from price data.
    # データ数が200本必要
    if df is None or len(df) < 200:
        # エラーメッセージを改善
        try:
            log_warn(f"⚠️ データが不足しています。最低200本必要ですが、{len(df) if df is not None else 0}本しかありません。")
        except Exception:
            log_warn(f"⚠️ データが不足しています。最低200本必要ですが、{len(df) if df is not None else 0}本しかありません。")
        return None

    # 短期25、中期75、長期200を追加
    df['short_mavg'] = df['close'].rolling(window=25).mean()
    df['mid_mavg'] = df['close'].rolling(window=75).mean()
    df['long_mavg'] = df['close'].rolling(window=200).mean()

    # RSI計算（14期間）
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    latest_data = df.iloc[-1]
    previous_data = df.iloc[-2]

    signal = None
    message = None

    # --- RSI単独による売買判定（35以下で買い、70以上で売り）---
    if latest_data['rsi'] <= 35:
        signal = 'buy_entry'
        message = f"✅ RSI買い注文: RSI={latest_data['rsi']:.2f} (35以下)"
        return signal, message
    elif latest_data['rsi'] >= 70:
        signal = 'sell_all'
        message = f"❌ RSI売り注文: RSI={latest_data['rsi']:.2f} (70以上)"
        return signal, message


    # 従来のトレンドフィルターも残す
    is_uptrend = latest_data['mid_mavg'] > latest_data['long_mavg']
    mid_mavg_is_rising = latest_data['mid_mavg'] > previous_data['mid_mavg']

    if (previous_data['short_mavg'] <= previous_data['mid_mavg'] and
        latest_data['short_mavg'] > latest_data['mid_mavg'] and
        is_uptrend and mid_mavg_is_rising):
        signal = 'buy_entry'
        message = "✅ 新規エントリーシグナル (GC 25/75、トレンド確認) が発生しました。"
        return signal, message
    elif latest_data['close'] > latest_data['short_mavg'] and is_uptrend:
        signal = 'buy_add'
        message =  "📈 買い増しシグナル (押し目買い) が発生しました。"
    elif not is_uptrend or latest_data['mid_mavg'] < previous_data['mid_mavg']:
        signal = 'sell_all'
        message = "❌ 全決済シグナル (長期トレンド終了/反転) が発生しました。"
    return signal, message


# === 4. 注文の整形 ===

def log_order(action, pair, amount, price=None):
    # Format order log message
    msg = f"{action}注文: {amount:.4f} {pair.split('/')[0]} {'@ ' + str(price) if price else '（成行）'}"
    try:
        logging.getLogger().info(msg)
    except Exception:
        pass
    try:
        print(msg)
    except Exception:
        print(msg)
    return msg

# === 5. 注文の実行 ===


# --- 注文実行ユーティリティ ---
def execute_order(exchange, pair, order_type, amount, price=None):
    # Place order on Bitbank (ccxt)
    try:
        order = None

        # DRY_RUN の場合は実際の注文 API 呼び出しを行わず、シミュレーションを返す
        if str(os.getenv('DRY_RUN', '0')).lower() in ('1', 'true', 'yes', 'on'):
            # 価格が未指定なら DRY_RUN_PRICE を使う
            try:
                p = float(price) if price is not None else float(os.getenv('DRY_RUN_PRICE', str(DRY_RUN_PRICE)))
            except Exception:
                p = float(DRY_RUN_PRICE)
            simulated_cost = None
            try:
                simulated_cost = float(amount) * p
            except Exception:
                simulated_cost = None

            action_label = "💰 (DRY) 買い" if order_type == 'buy' else "💸 (DRY) 売り"
            log_order(action_label, pair, amount, price)
            simulated = {'id': 'dry_order', 'amount': amount, 'cost': simulated_cost}
            try:
                log_info("ℹ️ DRY_RUN: 注文は実行されませんでした（シミュレーション）")
            except Exception:
                pass
            return simulated

        if order_type == 'buy':
            if price:
                return execute_order(exchange, pair, 'buy', amount, price)
            else:
                log_info("⚠️ 価格未指定のため買い注文をスキップ")
                return None
        elif order_type == 'sell':
            if price:
                return execute_order(exchange, pair, 'sell', amount, price)
            else:
                log_info("⚠️ 価格未指定のため売り注文をスキップ")
                return None
        else:
            log_error(f"無効な注文タイプです: {order_type}")
            return None

        if order and isinstance(order, dict) and 'id' in order:
            log_info("注文成功:", order.get('id'))
            print(f"[ORDER SUCCESS] APIレスポンス: {order}")
            return order
        else:
            log_error("注文に失敗しました:", order)
            print(f"[ORDER FAIL] APIレスポンス: {order}")
            return None

    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            log_error(f"❌ 注文実行中にエラーが発生しました: {e}")
        except Exception:
            pass
        return None

# === 6. メインループ（Botの実行部分） ===
# Small helper: in DRY_RUN or when AUTO_FIX_FUNDS is enabled, ensure FundManager has a reasonable balance
def _ensure_fund_manager_has_funds(fm, initial_amount=None):
    pass  # No longer needed in BTC-only mode
    # Auto-fix funds is now opt-in via AUTO_FIX_FUNDS. This avoids silently
    # modifying funds during regular DRY_RUNs — operator must explicitly enable it.
    try:
        auto_fix = str(os.getenv('AUTO_FIX_FUNDS', '')).lower() in ('1', 'true', 'yes', 'on')
    except Exception:
        auto_fix = False

    if not auto_fix:
        # 不要なtmp_path関連の処理を削除
        return

    # DEBUG: run_bot entry
    try:
        DRY_RUN = os.getenv("DRY_RUN", "0").lower() in ["1", "true", "yes", "on"]
        log_debug(f"DEBUG: run_bot start - DRY_RUN={DRY_RUN}")
    except Exception:
        log_debug("DEBUG: run_bot start (print failed)")

    # 実行時チェック: 必要な環境変数は dry_run のときは緩和する
    env_dry_run = str(os.getenv("DRY_RUN", "")).lower() in ["1", "true", "yes", "on"]
    if not env_dry_run:
        # 実運用時に必須の環境変数
        required_env_vars = ["SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS", "TO_EMAIL", "API_KEY", "SECRET_KEY"]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"以下の環境変数が .env に設定されていません: {', '.join(missing_vars)}")
    else:
        # DRY_RUN の場合は外部依存を必須にしない
        required_env_vars = []

    # Exchange/FundManager の準備
    # exchangeのグローバル初期化・利用は削除（mainブロックでのみ初期化・run_botに渡す）

def run_bot(exchange, fund_manager, dry_run=False):
    # 板情報取得
    try:
        orderbook = exchange.fetch_order_book('BTC/JPY')
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        current_price = get_latest_price(exchange, 'BTC/JPY')
    except Exception as e:
        print(f"⚠️ 板情報取得・判定エラー: {e}")
        return "板情報取得エラー"

    PAIR = 'BTC/JPY'
    PROFIT_TAKE_PCT = 10.0
    BUY_MORE_PCT = 10.0
    MIN_ORDER_BTC = 0.001

    import json
    import time
    positions_file = 'positions_state.json'  # ファイル名を必ず固定
    # ポジション情報とlast_buy_priceの読み込み
    state = {}
    positions = []
    last_buy_price = None
    if os.path.exists(positions_file):
        try:
            with open(positions_file, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                positions = loaded.get('positions', [])
                last_buy_price = loaded.get('last_buy_price')
            elif isinstance(loaded, list):
                positions = loaded
                last_buy_price = positions[-1]['price'] if positions else None
            set_last_buy_price(state, last_buy_price)
        except Exception as e:
            print(f"[ERROR] positionsファイル初期読込例外: {e}")
            positions = []
            set_last_buy_price(state, None)
    else:
        positions = []
        set_last_buy_price(state, None)
    try:
        while True:
            # positions_state.json再読込
            try:
                with open(positions_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    positions = loaded.get('positions', [])
                    last_buy_price = loaded.get('last_buy_price')
                elif isinstance(loaded, list):
                    positions = loaded
                    last_buy_price = positions[-1]['price'] if positions else None
                set_last_buy_price(state, last_buy_price)
                logging.info(f"positions読み込み直後: {positions}")
                print(f"[DEBUG] positions読み込み直後: {positions}", flush=True)
                print(f"[DEBUG] last_buy_price: {last_buy_price}", flush=True)
                # ここでAPI残高取得
                api_balance = get_account_balance(exchange)
                # BTC残高は0.002のみ扱う
                btc_api_balance = 0.002
                # 現在価格取得
                current_price = get_latest_price(exchange, 'BTC/JPY')
                logging.info(f"現在価格: {current_price}")
                print(f"[DEBUG] 現在価格: {current_price}", flush=True)
                # --- 5分足EMA100傾き率による買い禁止判定 ---
                indicators_5m = compute_indicators(exchange, pair='BTC/JPY', timeframe='5m', limit=300)
                ema100_slope_5m = None
                closes_5m = indicators_5m.get('closes', []) if indicators_5m else []
                if closes_5m and len(closes_5m) >= 103:
                    import numpy as np
                    import talib
                    ema100_now = float(talib.EMA(np.array(closes_5m), timeperiod=100)[-1])
                    ema100_prev = float(talib.EMA(np.array(closes_5m[:-3]), timeperiod=100)[-1])
                    ema100_slope_5m = (ema100_now - ema100_prev) / 3 / ema100_now
                    print(f"[DEBUG] 5m足EMA100傾き率: {ema100_slope_5m:.5f}")
                    if ema100_slope_5m < -0.0015:
                        print(Fore.RED + f"[INFO] 5分足EMA100傾き率が-0.15%未満（{ema100_slope_5m*100:.3f}%）のため買い禁止" + Style.RESET_ALL)
                        # 買い判定をスキップ
                        continue

                # --- 5分足RSIで最終エントリー判定 ---
                indicators_5m = compute_indicators(exchange, pair='BTC/JPY', timeframe='5m', limit=300)
                rsi_5m = indicators_5m.get('rsi_14') if indicators_5m else None
                closes_5m = indicators_5m.get('closes', []) if indicators_5m else []
                ema20_5m = None
                ema100_5m = None
                ema100_slope_5m = None
                if closes_5m and len(closes_5m) >= 103:
                    import numpy as np
                    import talib
                    ema20_5m = float(talib.EMA(np.array(closes_5m), timeperiod=20)[-1])
                    ema100_5m = float(talib.EMA(np.array(closes_5m), timeperiod=100)[-1])
                    ema100_prev = float(talib.EMA(np.array(closes_5m[:-3]), timeperiod=100)[-1])
                    ema100_slope_5m = (ema100_5m - ema100_prev) / 3 / ema100_5m
                print(f"[DEBUG] 5分足RSI: {rsi_5m}, EMA20: {ema20_5m}, EMA100: {ema100_5m}, EMA100傾き率: {ema100_slope_5m}")
                # 売買判定
                btc_balance = sum([float(pos.get('amount', 0)) for pos in positions])
                # 5分足RSIが35～45ゾーン かつ 価格>EMA20 かつ EMA100傾き率>-0.15% でbuy
                buy_signal = False
                if btc_balance == 0 and rsi_5m is not None and 35 <= rsi_5m <= 45:
                    if closes_5m and len(closes_5m) > 0 and ema20_5m is not None and ema100_slope_5m is not None:
                        price_now = closes_5m[-1]
                        if price_now > ema20_5m and ema100_slope_5m > -0.0015:
                            buy_signal = True
                if buy_signal:
                    td = {'action': 'buy', 'amount': MIN_ORDER_BTC, 'price': current_price, 'buy_condition': True}
                else:
                    td = {'action': 'hold', 'amount': 0.0, 'price': current_price, 'buy_condition': False}
                print(f"[DEBUG] trade_decision result: {td}", flush=True)
                if td.get('action') == 'sell':
                    print("[DEBUG] 売り判定: 条件成立", flush=True)
                    # 売り注文を実行
                    try:
                        # ポジション全て売却
                        sell_results = sell_all_positions(positions, exchange, PAIR)
                        print(f"[DEBUG] 売却結果: {sell_results}", flush=True)
                        # ポジション情報をクリア
                        positions = []
                        set_last_buy_price(state, None)
                        # 保存
                        save_data = {
                            "positions": positions,
                            "last_buy_price": get_last_buy_price(state)
                        }
                        with open(positions_file, 'w', encoding='utf-8') as f:
                            json.dump(save_data, f, ensure_ascii=False, indent=2)
                        print("[DEBUG] 売却後、ポジション情報クリア・保存", flush=True)
                    except Exception as e:
                        print(f"[ERROR] 売却処理例外: {e}", flush=True)
                elif td.get('action') == 'buy':
                    print("[DEBUG] 買い判定: 5分足RSI<=35でエントリー", flush=True)
                    # ここで買い注文ロジックを実行（既存のbuyロジックを流用）
                    # ...既存のbuyロジック...
                else:
                    print("[DEBUG] 買い判定: 条件不成立（5分足RSIフィルタ）", flush=True)
            except Exception as e:
                logging.error(f"positionsファイル読み込み例外: {e}")
                print(f"[ERROR] positionsファイル読み込み例外: {e}", flush=True)
            logging.info("ループ末尾: sleep前")
            print("[DEBUG] ループ末尾: sleep前", flush=True)
            time.sleep(1)
            logging.info("ループ突入")
            print("[DEBUG] ループ突入", flush=True)
    except Exception as e:
        logging.error(f"メインループ内で例外発生: {e}")
        print(f"[ERROR] メインループ内で例外発生: {e}", flush=True)
        import traceback
        traceback.print_exc()

    # ポジションが空のときだけ買い判定
    updated_positions = []
    if not updated_positions:
        prev_high = max([float(pos['price']) for pos in positions]) if positions else current_price
        buy_threshold = prev_high * 0.9
        buy_cost = current_price * MIN_ORDER_BTC
        if current_price <= buy_threshold and fund_manager.available_fund() - buy_cost >= 1000:
            if fund_manager.place_order(buy_cost):
                order = execute_order(exchange, PAIR, 'buy', MIN_ORDER_BTC, current_price)

                updated_positions.append({'price': current_price, 'amount': MIN_ORDER_BTC, 'timestamp': time.time()})
                # last_buy_priceを更新
                set_last_buy_price(state, current_price)
                try:
                    smtp_host = os.getenv('SMTP_HOST')
                    smtp_port = int(os.getenv('SMTP_PORT', '465'))
                    smtp_user = os.getenv('SMTP_USER')
                    smtp_password = os.getenv('SMTP_PASS')
                    email_to = os.getenv('TO_EMAIL')
                    if smtp_host and email_to:
                        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        subject = f"BTC購入通知 {now}"
                        message = f"【BTC購入】\n時刻: {now}\n数量: {MIN_ORDER_BTC} BTC\n価格: {current_price} 円"
                        send_notification(smtp_host, smtp_port, smtp_user, smtp_password, email_to, subject, message)
                except Exception as e:
                    print(f"⚠️ 購入通知メール送信エラー: {e}")

    # ポジション情報とlast_buy_priceの保存
    try:
        # last_buy_priceも一緒に保存
        save_obj = updated_positions.copy()
        # last_buy_priceを別ファイルやpositions_state.jsonに保存したい場合は下記のように拡張可能
        # ここではpositions_state.jsonにlast_buy_priceも含めて保存（リスト＋値の混在を避ける場合はdict化推奨）
        save_data = {
            "positions": updated_positions,
            "last_buy_price": get_last_buy_price(state)
        }
        with open(positions_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"ポジション保存エラー: {e}")

    import time
    time.sleep(30)  # 30秒待機
    return "run_bot executed"

def sell_all_positions(positions, exchange=None, pair='BTC/JPY'):
    results = []
    for pos in positions:
        qty = pos.get("qty")
        if qty is None:
            qty = pos.get("amount")
        if qty is None:
            logging.warning(f"ポジションに数量(qty/amount)がありません: {pos}")
            print(f"[WARN] ポジションに数量(qty/amount)がありません: {pos}", flush=True)
            result = {"price": pos.get("price"), "qty": None, "status": "error", "error": "No qty/amount in position"}
            results.append(result)
            continue
        try:
            # 実際のAPI呼び出し（成行売り注文）
            order = exchange.create_order(pair, 'market', 'sell', qty)
            result = {"price": pos.get("price"), "qty": qty, "status": "sold", "order_id": order.get("id")}
        except Exception as e:
            result = {"price": pos.get("price"), "qty": qty, "status": "error", "error": str(e)}
        results.append(result)
    return results

def get_latest_price(exchange, pair='BTC/JPY'):
    # 最新価格取得。失敗時は0.0
    try:
        ticker = exchange.fetch_ticker(pair)
        return float(ticker.get('last', 0))
    except Exception:
        return 0.0
# --- 汎用RSI横ばい判定関数 ---
def rsi_flat(rsi_series, lookback=3, tolerance=1.0):
    # RSI横ばい判定
    if not rsi_series or len(rsi_series) < lookback:
        return False
    window = rsi_series[-lookback:]
    return max(window) - min(window) <= tolerance