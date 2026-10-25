# --- グローバル定数・変数定義 ---
import os

PAIR = 'BTC/JPY'
MIN_ORDER_BTC = float(os.getenv('MIN_ORDER_BTC', '0.0011'))
positions_file = 'positions_state.json'
state = {}
# --- 必要な標準ライブラリのimport ---

def cleanup_trade_history(days=7):
    """trade_history.jsonからdays日以上前の履歴を削除する"""
    import json, datetime
    trade_log_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'trade_history.json'))
    if not os.path.exists(trade_log_file):
        return
    try:
        with open(trade_log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
        now = datetime.datetime.now()
        cutoff = now - datetime.timedelta(days=days)
        def is_recent(entry):
            ts = entry.get('timestamp')
            if not ts:
                return True  # タイムスタンプがなければ消さない
            try:
                dt = datetime.datetime.fromisoformat(ts)
                return dt >= cutoff
            except Exception:
                return True
        new_logs = [e for e in logs if is_recent(e)]
        if len(new_logs) < len(logs):
            with open(trade_log_file, 'w', encoding='utf-8') as f:
                json.dump(new_logs, f, ensure_ascii=False, indent=2)
            print(f"[INFO] trade_history.json: {len(logs)-len(new_logs)}件の古い履歴を削除しました")
    except Exception as e:
        print(f"[ERROR] trade_history.jsonクリーンアップ失敗: {e}")
# --- ターミナルで指標やシグナルを色付きで表示するユーティリティ ---
from colorama import Fore, Style
import numpy as np
from collections import deque
import time
import ccxt
import json
import pandas as pd


def should_send_notification(key_name, cooldown_seconds=None, state_file='notify_state.json'):
    if cooldown_seconds is None:
        cooldown_seconds = int(os.getenv('NOTIFY_COOLDOWN_SEC', '3600'))

    now = time.time()
    state = {}
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
        except Exception:
            state = {}

    last_sent = float(state.get(key_name, {}).get('time', 0) or 0)
    if now - last_sent < cooldown_seconds:
        return False

    state[key_name] = {'time': now}
    try:
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return True

def colorize_value(value, kind='rsi'):
    """
    指標値に応じて色を返す（RSI, BB, EMA, シグナル）
    kind: 'rsi', 'bb', 'ema', 'signal'
    """
    if kind == 'rsi':
        if value is None:
            return f"{Style.DIM}--{Style.RESET_ALL}"
        if value < 30:
            return f"{Fore.CYAN}{value:.2f}{Style.RESET_ALL}"
        elif value > 70:
            return f"{Fore.MAGENTA}{value:.2f}{Style.RESET_ALL}"
        else:
            return f"{Fore.GREEN}{value:.2f}{Style.RESET_ALL}"
    elif kind == 'signal':
        if value == 'buy':
            return f"{Fore.CYAN}BUY{Style.RESET_ALL}"
        elif value == 'sell_all':
            return f"{Fore.MAGENTA}SELL{Style.RESET_ALL}"
        elif value == 'hold':
            return f"{Fore.YELLOW}HOLD{Style.RESET_ALL}"
        else:
            return f"{Style.DIM}{value}{Style.RESET_ALL}"
    elif kind == 'bb':
        # BBは上下で色分け
        return f"{Fore.BLUE}{value:.2f}{Style.RESET_ALL}"
    elif kind == 'ema':
        return f"{Fore.LIGHTYELLOW_EX}{value:.2f}{Style.RESET_ALL}"
    else:
        return str(value)

def print_colored_indicators(timeframe, rsi, ema, bb_upper, bb_middle, bb_lower, signal=None):
    # タイムフレームごとに色を変える
    if timeframe == '1h':
        tf_color = Fore.GREEN
    elif timeframe == '15m':
        tf_color = Fore.BLUE
    elif timeframe == '5m':
        tf_color = Fore.YELLOW
    else:
        tf_color = Style.RESET_ALL

    def safe_colorize(val, kind):
        if val is None:
            return f"{Fore.LIGHTBLACK_EX}データ不足{Style.RESET_ALL}"
        try:
            return colorize_value(val, kind)
        except Exception:
            return f"{Fore.LIGHTBLACK_EX}ERR{Style.RESET_ALL}"

    # 売買判断は常に黄色
    signal_str = f"{Fore.YELLOW}{signal.upper() if signal else ''}{Style.RESET_ALL}" if signal else ''
    print(f"{tf_color}[{timeframe}]{Style.RESET_ALL} "
          f"RSI: {safe_colorize(rsi, 'rsi')}  "
          f"EMA: {safe_colorize(ema, 'ema')}  "
          f"BB: {safe_colorize(bb_lower, 'bb')} - {safe_colorize(bb_middle, 'bb')} - {safe_colorize(bb_upper, 'bb')}  "
          f"SIGNAL: {signal_str}")
# --- 1分足OHLCVからテクニカル指標を計算する関数群 ---
def calc_rsi_from_ohlcv(ohlcv_df, period=14):
    """DataFrameからRSIを計算し、最新値を返す"""
    import numpy as np
    if ohlcv_df is None or len(ohlcv_df) < period:
        return None
    close = ohlcv_df['close']
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window=period, min_periods=period).mean()
    ma_down = down.rolling(window=period, min_periods=period).mean()
    rsi = 100 - (100 / (1 + ma_up / ma_down))
    return rsi.iloc[-1] if not rsi.isnull().all() else None

def calc_ema_from_ohlcv(ohlcv_df, period=20):
    """DataFrameからEMAを計算し、最新値を返す"""
    if ohlcv_df is None or len(ohlcv_df) < period:
        return None
    ema = ohlcv_df['close'].ewm(span=period, adjust=False).mean()
    return ema.iloc[-1] if not ema.isnull().all() else None

def calc_bb_from_ohlcv(ohlcv_df, period=20, num_std=2):
    """DataFrameからボリンジャーバンド（上・中・下）を計算し、最新値を返す"""
    if ohlcv_df is None or len(ohlcv_df) < period:
        return None, None, None
    close = ohlcv_df['close']
    ma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return upper.iloc[-1], ma.iloc[-1], lower.iloc[-1]
# --- .envファイルから環境変数を自動ロード ---
from dotenv import load_dotenv
load_dotenv()
# --- bitbank用ccxtインスタンス生成 ---
import ccxt
import json
def create_bitbank_exchange():
    api_key = (os.getenv('BITBANK_API_KEY') or os.getenv('API_KEY') or '').strip()
    api_secret = (os.getenv('BITBANK_API_SECRET') or os.getenv('SECRET_KEY') or '').strip()
    if not get_dry_run_flag() and (not api_key or not api_secret):
        raise ValueError(
            "Bitbank API credentials are missing. Set BITBANK_API_KEY and "
            "BITBANK_API_SECRET (or the legacy API_KEY and SECRET_KEY) in .env."
        )
    exchange = ccxt.bitbank({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
    })
    return exchange


def _normalize_trade_action(entry):
    action = entry.get('action')
    if action in ('buy', 'sell'):
        return action
    side = str(entry.get('side', '')).lower()
    if side in ('buy', 'sell'):
        return side
    return None


def _trade_time_key(entry):
    ts = entry.get('timestamp')
    if isinstance(ts, (int, float)):
        # ccxtのtimestampは通常ms
        return float(ts) / (1000.0 if ts > 10**11 else 1.0)
    if isinstance(ts, str):
        # 数字文字列 (秒/ミリ秒) を優先処理
        try:
            v = float(ts)
            return v / (1000.0 if v > 10**11 else 1.0)
        except Exception:
            pass
        # ISO8601文字列
        try:
            return datetime.datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
        except Exception:
            pass
    dt = entry.get('datetime')
    if isinstance(dt, str):
        try:
            return datetime.datetime.fromisoformat(dt.replace('Z', '+00:00')).timestamp()
        except Exception:
            pass
    return 0.0


def sync_trade_history_with_exchange(exchange, pair='BTC/JPY', limit=200):
    """bitbankの約定履歴を取得し、trade_history.jsonへ重複なしで統合する。"""
    trade_log_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'trade_history.json'))
    existing = []
    if os.path.exists(trade_log_file):
        try:
            with open(trade_log_file, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                existing = loaded
        except Exception:
            existing = []

    merged = {}

    def _key(row):
        rid = row.get('id')
        if rid:
            return f"id:{rid}"
        return "|".join([
            str(row.get('timestamp') or row.get('datetime') or ''),
            str(row.get('action') or row.get('side') or ''),
            str(row.get('price') or ''),
            str(row.get('amount') or ''),
        ])

    for row in existing:
        if isinstance(row, dict):
            merged[_key(row)] = row

    try:
        trades = exchange.fetch_my_trades(pair, limit=limit)
    except Exception as e:
        print(f"[WARN] 取引所の約定履歴取得失敗。ローカル履歴を使用します: {e}")
        return sorted(merged.values(), key=_trade_time_key)
    for t in trades:
        row = {
            'datetime': t.get('datetime'),
            'timestamp': t.get('timestamp') or t.get('datetime'),
            'symbol': t.get('symbol'),
            'side': t.get('side'),
            'action': t.get('side'),
            'price': t.get('price'),
            'amount': t.get('amount'),
            'cost': t.get('cost'),
            'fee': t.get('fee'),
            'order': t.get('order'),
            'id': t.get('id'),
            'status': t.get('status', 'filled'),
            'info': t.get('info'),
        }
        merged[_key(row)] = row

    def _sort_ts(row):
        return _trade_time_key(row)

    logs = sorted(merged.values(), key=_sort_ts)
    with open(trade_log_file, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    return logs

# --- 取引所APIから売買履歴を取得してtrade_history.jsonに記録 ---
def fetch_and_save_trade_history(exchange, pair='BTC/JPY', limit=100):
    try:
        trades = exchange.fetch_my_trades(pair, limit=limit)
        trade_log_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'trade_history.json'))
        logs = []
        if os.path.exists(trade_log_file):
            with open(trade_log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        for t in trades:
            log_entry = {
                'datetime': t.get('datetime'),
                'symbol': t.get('symbol'),
                'side': t.get('side'),
                'price': t.get('price'),
                'amount': t.get('amount'),
                'cost': t.get('cost'),
                'fee': t.get('fee'),
                'order': t.get('order'),
                'id': t.get('id'),
                'status': t.get('status', 'filled'),
                'info': t.get('info'),
            }
            logs.append(log_entry)
        with open(trade_log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        print(f"[INFO] {len(trades)}件の履歴をtrade_history.jsonに記録しました")
    except Exception as e:
        print(f"[ERROR] 売買履歴取得・記録エラー: {e}")
# --- 取引所APIから売買履歴を取得してtrade_history.jsonに記録する ---
import ccxt
import json
def fetch_and_save_trade_history(exchange, pair='BTC/JPY', limit=100):
    try:
        trades = exchange.fetch_my_trades(pair, limit=limit)
        trade_log_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'trade_history.json'))
        logs = []
        if os.path.exists(trade_log_file):
            with open(trade_log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        for t in trades:
            log_entry = {
                'datetime': t.get('datetime'),
                'symbol': t.get('symbol'),
                'side': t.get('side'),
                'price': t.get('price'),
                'amount': t.get('amount'),
                'cost': t.get('cost'),
                'fee': t.get('fee'),
                'order': t.get('order'),
                'id': t.get('id'),
                'status': t.get('status', 'filled'),
                'info': t.get('info'),
            }
            logs.append(log_entry)
        with open(trade_log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        print(f"[INFO] {len(trades)}件の履歴をtrade_history.jsonに記録しました")
    except Exception as e:
        print(f"[ERROR] 売買履歴取得・記録エラー: {e}")
# --- WebSocketで板情報・出来高を取得する実用サンプル ---
import threading
try:
    import websocket
except Exception:
    websocket = None
import json

orderbook_data = None  # 板情報
volume_data = None     # 出来高

def on_ws_message_depth(ws, message):
    global orderbook_data, volume_data
    try:
        # Socket.IO '40'（connect）受信時に購読リクエストを送信
        if message == '40':
            subscribe_msg = '42["subscribe", {"channel": "depth_btc_jpy"}]'
            ws.send(subscribe_msg)
            print("[WebSocket] depth_btc_jpy購読リクエスト送信")
            return
        msg = message
        if len(msg) > 1 and msg[0].isdigit() and msg[1] == '{':
            msg = msg[1:]
        if msg.startswith('42'):
            try:
                arr = json.loads(msg[2:])
                if isinstance(arr, list) and len(arr) == 2 and arr[0] == 'depth_btc_jpy':
                    data = arr[1]
                    orderbook_data = data
                    print(f"[板情報] {orderbook_data}")
                    # 板情報から出来高推定（例: ask/bidの合計量）
                    volume_data = sum([float(v[1]) for v in data.get('asks', [])]) + sum([float(v[1]) for v in data.get('bids', [])])
                    print(f"[推定出来高] {volume_data}")
                return
            except Exception as e:
                print(f"[WebSocket ERROR] depthパース失敗: {e}")
                return
        try:
            data = json.loads(msg)
            if isinstance(data, dict) and data.get('channel') == 'depth_btc_jpy':
                orderbook_data = data['data']
                print(f"[板情報] {orderbook_data}")
        except Exception as e:
            print(f"[WebSocket ERROR] JSONパース失敗: {e}")
    except Exception as e:
        print(f"[WebSocket ERROR] {e}")

def start_bitbank_ws_depth():
    if websocket is None:
        print("[WARN] websocket-clientが未インストールのためWebSocketを起動できません")
        return
    # ループ開始時に古い履歴をクリーンアップ
    cleanup_trade_history(days=7)
    while True:
        try:
            ws_url = "wss://stream.bitbank.cc/socket.io/?EIO=3&transport=websocket"
            ws = websocket.WebSocketApp(
                ws_url,
                on_message=on_ws_message_depth
            )
            ws.run_forever()
        except Exception as e:
            print(f"[ERROR] WebSocket再接続エラー: {e}")
        print("[INFO] 5秒後にWebSocket再接続します")
        import time
        time.sleep(5)
# --- WebSocketでbitbankの現在価格を取得 ---
import threading
try:
    import websocket
except Exception:
    websocket = None
import json

latest_ws_price = None  # グローバルで最新価格を保持
# --- 1分足OHLCVビルダーをグローバルで用意 ---
ohlcv_builder = None

def on_ws_message(ws, message):
    global latest_ws_price, ohlcv_builder
    try:
        # --- 価格受信時にOHLCVビルダーへ反映 ---
        # print(f"[WebSocket RAW] {repr(message)}")
        # Socket.IO '40'（connect）受信時に購読リクエストを送信
        if message == '40':
            subscribe_msg = '42["subscribe", {"channel": "ticker_btc_jpy"}]'
            ws.send(subscribe_msg)
            print("[WebSocket] ticker_btc_jpy購読リクエスト送信")
            return
        # Socket.IOプロトコル: 先頭が数字+JSONの場合の前処理
        msg = message
        if len(msg) > 1 and msg[0].isdigit() and msg[1] == '{':
            msg = msg[1:]
        # 先頭が'42'（Socket.IOのeventメッセージ）なら配列形式
        if msg.startswith('42'):
            try:
                arr = json.loads(msg[2:])
                # print(f"[WebSocket PARSED] {repr(arr)}")
                # ['channel', {...}] の形式
                if isinstance(arr, list) and len(arr) == 2 and arr[0] == 'ticker_btc_jpy':
                    data = arr[1]
                    latest_ws_price = float(data['last'])
                    # --- ここでOHLCVビルダーに価格を渡す ---
                    if ohlcv_builder is not None:
                        ohlcv_builder.update(latest_ws_price)
                    print(f"[WebSocket] 現在価格: {latest_ws_price}")
                return
            except Exception as e:
                print(f"[WebSocket ERROR] 42パース失敗: {e}")
                return
        # 通常のJSONとしてパース
        try:
            data = json.loads(msg)
            print(f"[WebSocket PARSED] {repr(data)}")
            if isinstance(data, dict) and data.get('channel') == 'ticker_btc_jpy':
                latest_ws_price = float(data['data']['last'])
                if ohlcv_builder is not None:
                    ohlcv_builder.update(latest_ws_price)
                print(f"[WebSocket] 現在価格: {latest_ws_price}")
            else:
                print(f"[WebSocket INFO] 受信データはdict型でない、またはchannelが一致しません: {type(data)}")
        except Exception as e:
            print(f"[WebSocket ERROR] JSONパース失敗: {e}")
    except Exception as e:
        print(f"[WebSocket ERROR] {e}")


def start_bitbank_ws():
    global ohlcv_builder
    # --- OHLCVビルダーを初期化 ---
    if websocket is None:
        print("[WARN] websocket-clientが未インストールのためWebSocketを起動できません")
        return
    if ohlcv_builder is None:
        ohlcv_builder = RealtimeOHLCVBuilder(interval_sec=60, max_bars=500)
    while True:
        try:
            ws_url = "wss://stream.bitbank.cc/socket.io/?EIO=3&transport=websocket"
            ws = websocket.WebSocketApp(
                ws_url,
                on_message=on_ws_message
            )
            ws.run_forever()
        except Exception as e:
            print(f"[ERROR] WebSocket再接続エラー: {e}")
        print("[INFO] 5秒後にWebSocket再接続します")
        time.sleep(5)

def get_account_balance(exchange):
    """
    ccxtのfetch_balanceをラップし、{'total': {...}, 'free': {...}, 'used': {...}}形式で返す。
    失敗時は空dictを返す。
    """
    try:
        balance = exchange.fetch_balance()
        # ccxt標準の形式をそのまま返す
        return {
            'total': balance.get('total', {}),
            'free': balance.get('free', {}),
            'used': balance.get('used', {})
        }
    except Exception as e:
        import logging
        logging.error(f"get_account_balance失敗: {e}")
        return {'total': {}, 'free': {}, 'used': {}}

from colorama import init as colorama_init
colorama_init(autoreset=True)
# --- シグナル生成のダミー関数 ---
def generate_signals(df):
    # 本番用: RSIとトレンドを考慮したシグナル生成
    rsi = None
    closes = None
    if isinstance(df, dict) and 'rsi_list' in df and 'closes' in df:
        rsi = df['rsi_list'][-1] if df['rsi_list'] else None
        closes = df['closes']
    elif hasattr(df, 'columns') and 'rsi' in df.columns and 'close' in df.columns:
        rsi = df['rsi'].iloc[-1] if not df['rsi'].isnull().all() else None
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


import datetime
import pytz
JST = pytz.timezone('Asia/Tokyo')

# --- 表形式でデータを表示するユーティリティ ---
# --- 5分足OHLCVを1時間足にリサンプリング ---
def resample_ohlcv_5m_to_1h(ohlcv_5m):
    import pandas as pd
    if not ohlcv_5m or len(ohlcv_5m) < 12:
        return []
    df = pd.DataFrame(ohlcv_5m, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("datetime", inplace=True)
    agg_dict = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "timestamp": "first"}
    ohlcv_1h = df.resample("1h").agg(agg_dict)  # pandas推奨の小文字'h'に修正
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
import threading
import time

# --- WebSocketの最新価格から1分足OHLCVを自動生成するクラス ---
class RealtimeOHLCVBuilder:
    """
    WebSocketの最新価格から1分足OHLCVを自動生成し、DataFrameに蓄積するクラス
    """
    def __init__(self, interval_sec=60, max_bars=500):
        self.interval_sec = interval_sec  # 1分足なら60
        self.max_bars = max_bars
        self.ohlcv = []  # [timestamp, open, high, low, close, volume]
        self.current_bar = None
        self.last_bar_time = None
        self.lock = threading.Lock()

    def update(self, price, volume=0.0):
        now = int(time.time())
        bar_time = now - (now % self.interval_sec)
        with self.lock:
            if self.last_bar_time != bar_time:
                # 新しいバー開始
                if self.current_bar:
                    self.ohlcv.append(self.current_bar)
                    if len(self.ohlcv) > self.max_bars:
                        self.ohlcv.pop(0)
                self.current_bar = [bar_time * 1000, price, price, price, price, volume]
                self.last_bar_time = bar_time
            else:
                # 既存バー更新
                self.current_bar[2] = max(self.current_bar[2], price)  # high
                self.current_bar[3] = min(self.current_bar[3], price)  # low
                self.current_bar[4] = price  # close
                self.current_bar[5] += volume  # volume加算

    def get_ohlcv(self):
        with self.lock:
            bars = list(self.ohlcv)
            if self.current_bar:
                bars.append(self.current_bar)
            return bars

    def to_dataframe(self):
        import pandas as pd
        bars = self.get_ohlcv()
        if not bars:
            return pd.DataFrame()
        df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("datetime", inplace=True)
        return df
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
    if period <= 0 or span <= 0 or len(closes) < period + span:
        return None

    values = np.asarray(closes, dtype=float)
    ema_now = float(talib.EMA(values, timeperiod=period)[-1])
    ema_previous = float(talib.EMA(values[:-span], timeperiod=period)[-1])
    if not np.isfinite(ema_now) or not np.isfinite(ema_previous):
        return None
    return (ema_now - ema_previous) / span
# --- 1時間足のRSIリスト取得関数 ---
def get_rsi_1h_series(exchange, pair='BTC/JPY', limit=300, ohlcv_data=None):
    # 1時間足のOHLCVデータからRSIリストを返す
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

def run_bot(exchange, fund_manager, dry_run=False):
    import builtins
    # --- trade_history.jsonの直近履歴を読み込む ---
    import json
    trade_log_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'trade_history.json'))
    logs = []
    if os.path.exists(trade_log_file):
        try:
            with open(trade_log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        except Exception as e:
            print(f"[WARN] trade_history.jsonの読み込み失敗: {e}")
    else:
        print("[INFO] trade_history.jsonが存在しません")
    builtins.logs = logs
    # --- 1h, 15m, 5m足のテクニカル指標を色付きで表示 ---
    try:
        import pandas as pd
        print("[INFO] 1h足データ表示は省略しました")

        # 15分足
        ohlcv_15m = exchange.fetch_ohlcv('BTC/JPY', timeframe='15m', limit=30)
        df_15m = pd.DataFrame(ohlcv_15m, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df_15m["datetime"] = pd.to_datetime(df_15m["timestamp"], unit="ms")
        df_15m.set_index("datetime", inplace=True)
        rsi_15m = calc_rsi_from_ohlcv(df_15m, period=14)
        ema_15m = calc_ema_from_ohlcv(df_15m, period=20)
        bb_upper_15m, bb_middle_15m, bb_lower_15m = calc_bb_from_ohlcv(df_15m, period=20, num_std=2)
        print_colored_indicators('15m', rsi_15m, ema_15m, bb_upper_15m, bb_middle_15m, bb_lower_15m)

        # 5分足
        ohlcv_5m = exchange.fetch_ohlcv('BTC/JPY', timeframe='5m', limit=30)
        df_5m = pd.DataFrame(ohlcv_5m, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df_5m["datetime"] = pd.to_datetime(df_5m["timestamp"], unit="ms")
        df_5m.set_index("datetime", inplace=True)
        rsi_5m = calc_rsi_from_ohlcv(df_5m, period=14)
        ema_5m = calc_ema_from_ohlcv(df_5m, period=20)
        bb_upper_5m, bb_middle_5m, bb_lower_5m = calc_bb_from_ohlcv(df_5m, period=20, num_std=2)
        print_colored_indicators('5m', rsi_5m, ema_5m, bb_upper_5m, bb_middle_5m, bb_lower_5m)
    except Exception as e:
        print(f"[ERROR] テクニカル指標表示エラー: {e}")
    # --- リアルタイム価格の履歴を保持するリスト ---
    price_history = deque(maxlen=300)
    RSI_PERIOD = 14  # 標準的なRSI期間に戻す
    rsi_buy_flag = False
    PAIR = 'BTC/JPY'
    PAIR = 'BTC/JPY'
    MIN_ORDER_BTC = float(os.getenv('MIN_ORDER_BTC', '0.0011'))
    positions_file = 'positions_state.json'
    state = {}
    positions = []
    last_buy_price = None
    # 初期化時に直近売値を指定値でセット
    last_sell_price = 10151701

    def get_last_buy_price(state):
        return state.get('last_buy_price') if isinstance(state, dict) else None
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
        except Exception:
            positions = []
    else:
        positions = []
    last_buy_time = None
    COOLDOWN_SECONDS = 60 * 10

    def has_position():
        try:
            # 取引履歴のpositionsではなく、取引所から同期した実残高を優先する。
            return float(btc_balance or 0.0) >= (MIN_ORDER_BTC * 0.5)
        except Exception:
            return False

    while True:
        print("\n==================== run_bot ループ ====================")
        current_price = None
        btc_balance = 0.0
        jpy_balance = None
        try:
            print(f"[DEBUG] ticker取得: PAIR={PAIR}")
            ticker = exchange.fetch_ticker(PAIR)
            print(f"[DEBUG] ticker内容: last={ticker.get('last')}, high={ticker.get('high')}, low={ticker.get('low')}, bid={ticker.get('bid')}, ask={ticker.get('ask')}")
            current_price = ticker['last'] if 'last' in ticker else None
            print(f"[DEBUG] 現在価格: {current_price}")
            if current_price:
                price_history.append(current_price)
                print(f"[DEBUG] price_history追加: 最新={current_price} / 履歴長={len(price_history)}")

            # メール判定の前に実保有量を同期する
            if not dry_run:
                balance = exchange.fetch_balance()
                total = balance.get('total', {}) if isinstance(balance, dict) else {}
                free = balance.get('free', {}) if isinstance(balance, dict) else {}
                btc_total = total.get('BTC') if isinstance(total, dict) else None
                btc_free = free.get('BTC') if isinstance(free, dict) else None
                jpy_total = total.get('JPY') if isinstance(total, dict) else None
                jpy_free = free.get('JPY') if isinstance(free, dict) else None
                btc_balance = float(btc_total if btc_total is not None else (btc_free or 0.0))
                jpy_balance = float(jpy_free if jpy_free is not None else (jpy_total or 0.0))
                print(f"[DEBUG] BTC残高同期: {btc_balance}")
                print(f"[DEBUG] 利用可能JPY残高同期: {jpy_balance:.0f}円")
            else:
                jpy_balance = float(os.getenv('DRY_RUN_JPY_BALANCE', '17000'))

            
        except Exception as e:
            print(f"[ERROR] run_bot: ticker取得例外: {e}")

        # --- 自作RSI計算（WebSocket価格履歴から） ---
        rsi = None
        if len(price_history) >= RSI_PERIOD:
            closes = np.array(price_history)[-RSI_PERIOD:]
            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

        # --- 直近売値・買値の取得 ---
        last_buy_price = None
        last_sell_price = None
        try:
            if not dry_run:
                logs = sync_trade_history_with_exchange(exchange, pair=PAIR, limit=200)
            # 順序に依存せず、時刻が最も新しいbuy/sellを採用
            buy_entries = [e for e in logs if _normalize_trade_action(e) == 'buy' and e.get('price') is not None]
            sell_entries = [e for e in logs if _normalize_trade_action(e) == 'sell' and e.get('price') is not None]
            if buy_entries:
                latest_buy = max(buy_entries, key=_trade_time_key)
                last_buy_price = float(latest_buy.get('price', 0))
            if sell_entries:
                latest_sell = max(sell_entries, key=_trade_time_key)
                last_sell_price = float(latest_sell.get('price', 0))

            # 実保有がゼロ以下なら、買値基準の下落アラートを停止（誤通知防止）
            if btc_balance <= 0.0:
                last_buy_price = None
            print(f"[DEBUG] last_buy_price={last_buy_price}, last_sell_price={last_sell_price}")
        except Exception as e:
            print(f"[WARN] 直近売買価格取得エラー: {e}")

        # --- EMA傾きによるトレンド判定 ---
        ema_trend = None
        try:
            closes = list(price_history)
            if len(closes) >= 23:
                import talib
                ema20_now = float(talib.EMA(np.array(closes), timeperiod=20)[-1])
                ema20_prev = float(talib.EMA(np.array(closes[:-3]), timeperiod=20)[-1])
                ema20_slope = (ema20_now - ema20_prev) / 3
                if ema20_slope > 0:
                    ema_trend = 'up'
                elif ema20_slope < 0:
                    ema_trend = 'down'
                else:
                    ema_trend = 'side'
        except Exception as e:
            print(f"[WARN] EMAトレンド判定エラー: {e}")

        # --- trade_decision呼び出し ---
        td = trade_decision(
            current_price,
            btc_balance=btc_balance,
            buy_btc=None,
            last_buy_price=last_buy_price,
            rsi=rsi,
            bb_lower=None,
            last_sell_price=last_sell_price,
            ema_trend=ema_trend,
            jpy_balance=jpy_balance,
        )

        # --- 売り判定: 直近の買値より高い場合のみ売り注文を許可 ---
        from colorama import Fore, Style
        action = td.get('action')
        amount = td.get('amount')
        price = td.get('price')

        # --- ポジション管理に基づく売買制御 ---
        # ポジションが無い場合のみ買い注文を許可

        if action == 'buy':
            if has_position():
                print(f"{Fore.YELLOW}【買い禁止】既にポジション保有中のため新規買い注文は出しません{Style.RESET_ALL}")
                action = 'hold'
            elif last_sell_price is None:
                print(f"{Fore.YELLOW}【買い見送り】直近売値を確認できないため、買い注文は出しません{Style.RESET_ALL}")
                action = 'hold'
            elif price is not None and price > last_sell_price:
                print(f"{Fore.YELLOW}【買い禁止】直近売値({last_sell_price:.0f}円)より高いため購入しません（現値={price:.0f}円）{Style.RESET_ALL}")
                action = 'hold'
            else:
                try:
                    if dry_run:
                        print(f"[DRY_RUN] 買い注文をシミュレーション: {amount} BTC @ {price} JPY")
                        order = {'id': 'dry_run', 'status': 'dry_run', 'amount': amount, 'price': price}
                    else:
                        order = exchange.create_limit_buy_order(PAIR, amount, price)
                    print(f"[ORDER] 指値買い注文発注: {order}")
                    # 約定時のみtrade_history.jsonに記録
                    if order and order.get('status') in ('closed', 'filled'):
                        log_trade('buy', order.get('price'), order.get('amount'), order.get('status'), reason='自動売買')
                except Exception as e:
                    print(f"[ERROR] 指値買い注文失敗: {e}")

        # ポジションがある場合のみ売り注文を許可

        elif action == 'sell':
            if not has_position():
                print(f"{Fore.YELLOW}【売り禁止】ポジション未保有のため売却注文は出しません{Style.RESET_ALL}")
                action = 'hold'
            elif last_buy_price is not None and price is not None and price <= last_buy_price:
                print(f"{Fore.YELLOW}【売り禁止】買値({last_buy_price:.0f}円)以下のため売却しません（現値={price:.0f}円）{Style.RESET_ALL}")
                action = 'hold'
            else:
                try:
                    if dry_run:
                        print(f"[DRY_RUN] 売り注文をシミュレーション: {amount} BTC @ {price} JPY")
                        order = {'id': 'dry_run', 'status': 'dry_run', 'amount': amount, 'price': price}
                    else:
                        order = exchange.create_limit_sell_order(PAIR, amount, price)
                    print(f"[ORDER] 指値売り注文発注: {order}")
                    # 約定時のみtrade_history.jsonに記録
                    if order and order.get('status') in ('closed', 'filled'):
                        log_trade('sell', order.get('price'), order.get('amount'), order.get('status'), reason='自動売買')
                except Exception as e:
                    print(f"[ERROR] 指値売り注文失敗: {e}")

        # アクションごとの表示
        if action == 'buy':
            print(f"{Fore.CYAN}【買いシグナル】{price:.0f}円で{amount}BTCを購入します！{Style.RESET_ALL}")
        elif action == 'sell':
            print(f"{Fore.MAGENTA}【売りシグナル】{price:.0f}円で{amount}BTCを売却します！{Style.RESET_ALL}")
        elif action == 'hold':
            print(f"{Fore.YELLOW}【ホールド】売買なし（様子見中）{Style.RESET_ALL}")
        else:
            print(f"{Fore.LIGHTBLACK_EX}【シグナル】{action}（詳細: {td}）{Style.RESET_ALL}")
        print(f"[INFO] trade_decision結果: {td}")

        # --- 実際の売買処理はここに（省略/既存ロジックと統合可） ---
        
        time.sleep(30)

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
        
# --- trade_history.jsonに注文履歴を記録するグローバル関数 ---
def log_trade(action, price, amount, status, reason=None):
    """Append a completed trade to trade_history.json."""
    import json
    import datetime
    import os

    if status not in ("closed", "filled"):
        return

    trade_log_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "trade_history.json")
    )
    log_entry = {
        "action": action,
        "price": price,
        "amount": amount,
        "status": status,
        "reason": reason,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    try:
        if os.path.exists(trade_log_file):
            with open(trade_log_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []
        if not isinstance(logs, list):
            logs = []
        logs.append(log_entry)
        with open(trade_log_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[ERROR] 取引履歴保存エラー: {e}")


        # --- 買い注文の未約定時リトライ ---
        # 注文発行はtrade_decisionに一元化したため、未約定リトライロジックは削除

        # --- 売り判定（強いトレンド時も売りは実行） ---
        # 注文発行はtrade_decisionに一元化したため、ここでの売り注文・リトライロジックは削除
        # 未定義変数・判定ロジック削除

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
    # テーブル出力削除
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
        rsi_list = []
        for i in range(len(closes)):
            if i+1 >= 14:
                rsi_val = compute_rsi(closes[i+1-14:i+1], period=14, exchange=exchange, pair=pair)
                rsi_list.append(rsi_val)
            else:
                rsi_list.append(None)
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
def trade_decision(current_price, btc_balance=0.0027, buy_btc=None, last_buy_price=None, rsi=None, bb_lower=None, last_sell_price=None, ema_trend=None, jpy_balance=None, rsi_history=None, pair='BTC/JPY'):
    # --- 予想外の大幅変動を検知しメール通知 ---
    try:
        smtp_host = os.getenv('SMTP_HOST')
        smtp_port = int(os.getenv('SMTP_PORT', '465'))
        smtp_user = os.getenv('SMTP_USER')
        smtp_password = os.getenv('SMTP_PASS')
        to_email = os.getenv('TO_EMAIL')
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # 最新の取引価格を必ず取得
        latest_price = current_price
        # --- 価格取得失敗時はOHLCV終値で補完 ---
        if (current_price is None or (isinstance(current_price, (float, int)) and current_price <= 0)) and 'exchange' in globals():
            try:
                ohlcv = globals()['exchange'].fetch_ohlcv(pair, timeframe='1m', limit=1)
                if ohlcv and len(ohlcv) > 0 and len(ohlcv[-1]) >= 5:
                    latest_price = float(ohlcv[-1][4])
                    print(f"[INFO] current_priceが取得できなかったため、1分足終値({latest_price})で補完")
            except Exception as e:
                print(f"[WARN] OHLCVからの価格補完も失敗: {e}")
        # 通常の価格取得
        if 'exchange' in globals() and callable(globals().get('get_latest_price', None)):
            try:
                tmp_price = get_latest_price(globals()['exchange'], pair)
                if tmp_price is not None:
                    latest_price = tmp_price
            except Exception:
                pass
        has_btc_position = False
        try:
            has_btc_position = float(btc_balance or 0.0) > 0.0
        except Exception:
            has_btc_position = False

        # 直近買値から-5%以上下落（保有中のみ通知）
        if has_btc_position and last_buy_price is not None and latest_price is not None and latest_price < last_buy_price * 0.95:
            if smtp_host and to_email and should_send_notification('price_drop_5pct'):
                subject = f"【警告】価格急落通知 {now}"
                message = f"現在価格({latest_price})が直近買値({last_buy_price})から5%以上下落しました。相場急変にご注意ください。"
                send_notification(smtp_host, smtp_port, smtp_user, smtp_password, to_email, subject, message)
        # 直近売値から+5%以上上昇
        if last_sell_price is not None and latest_price is not None and latest_price > last_sell_price * 1.05:
            if smtp_host and to_email and should_send_notification('price_surge_5pct'):
                subject = f"【警告】価格急騰通知 {now}"
                message = f"現在価格({latest_price})が直近売値({last_sell_price})から5%以上上昇しました。相場急変にご注意ください。"
                send_notification(smtp_host, smtp_port, smtp_user, smtp_password, to_email, subject, message)
        # 以降の判定もlatest_priceを使う
        current_price = latest_price
    except Exception as e:
        print(f"[WARN] 急変動通知メール送信失敗: {e}")

    # current_price: 現在のBTC/JPY価格
    # btc_balance: 現在のBTC総保有量
    # buy_btc: 売買対象のBTC量(全保有BTCの85%を推奨)
    # last_buy_price: 直近の買値(売却判定に使用)
    # rsi: 最新のRSI値
    # bb_lower: ボリンジャーバンド下限
    # last_sell_price: 直近の売値
    # ema_trend: EMAやトレンド判定（例: 'up', 'down', 'side'）
    print(f"[DEBUG] trade_decision: current_price={str(current_price)}, btc_balance={str(btc_balance)}, last_buy_price={str(last_buy_price)}, last_sell_price={str(last_sell_price)}, rsi={str(rsi)}, bb_lower={str(bb_lower)}, ema_trend={str(ema_trend)}")
    knife_rebound_pct = float(os.getenv('KNIFE_REBOUND_PCT', '0.005') or 0.005)
    deep_drop_reentry_pct = float(os.getenv('DEEP_DROP_REENTRY_PCT', '0.08') or 0.08)
    rebound_entry_pct = float(os.getenv('REBOUND_ENTRY_PCT', '0.005') or 0.005)
    profit_take_pct = float(os.getenv('SELL_ON_RISE_PCT', '10') or 10)
    rebound_rsi_min = float(os.getenv('REBOUND_RSI_MIN', '35') or 35)
    rebound_rsi_max = float(os.getenv('REBOUND_RSI_MAX', '62') or 62)
    trade_ratio = max(0.0, min(1.0, float(os.getenv('TRADE_RATIO', '0.90') or 0.90)))
    btc_balance_f = float(btc_balance or 0.0)

    if btc_balance_f > 0.0 and current_price is not None and last_buy_price is not None:
        current_price_f = float(current_price)
        last_buy_price_f = float(last_buy_price)
        take_profit_threshold = last_buy_price_f * (1.0 + profit_take_pct / 100.0)
        if current_price_f >= take_profit_threshold - 1e-6:
            sell_amount = round(btc_balance_f * trade_ratio, 8)
            if sell_amount > 0.0:
                return {
                    'action': 'sell',
                    'amount': sell_amount,
                    'price': current_price_f,
                    'sell_condition': True,
                    'reason': f"利益確定：買値{last_buy_price_f:.0f}円を{profit_take_pct:.1f}%超え（{trade_ratio:.0%}売却）"
                }

    # A flat account only re-enters against a verified last sell price.
    if btc_balance_f <= 0.0 and last_sell_price is None:
        print("[INFO] 買い見送り: 直近売値が確認できないため")
        return {'action': 'hold', 'amount': 0.0, 'price': current_price, 'buy_condition': False, 'sell_condition': False}

    if btc_balance_f <= 0.0 and current_price is None:
        print("[INFO] 買い見送り: 現在価格が取得できないため")
        return {'action': 'hold', 'amount': 0.0, 'price': current_price, 'buy_condition': False, 'sell_condition': False}

    if btc_balance_f <= 0.0 and float(current_price) > float(last_sell_price):
        print(f"[INFO] 買い禁止: 現在価格が直近売値({last_sell_price})以上のため")
        return {'action': 'hold', 'amount': 0.0, 'price': current_price, 'buy_condition': False, 'sell_condition': False}

    # Re-enter at or below the latest sell price, with yen budget and balance caps.
    if btc_balance_f <= 0.0:
        buy_amount = calculate_buy_amount(current_price, jpy_balance)
        if buy_amount > 0.0:
            return {
                'action': 'buy',
                'amount': buy_amount,
                'price': float(current_price),
                'buy_condition': True,
                'reason': '自動買い: 直近売値以下に下落'
            }
        print("[INFO] 買い見送り: 利用可能JPY残高または注文予算が最小注文額に不足")
        return {'action': 'hold', 'amount': 0.0, 'price': current_price, 'buy_condition': False, 'sell_condition': False}

    buy_btc = calculate_buy_amount(current_price, jpy_balance) if buy_btc is None else min(
        max(0.0, float(buy_btc)),
        calculate_buy_amount(current_price, jpy_balance),
    )

    # --- 落ちたナイフ回避: 直近安値から一定率反発していない場合は買い禁止 ---
    recent_lows = None
    try:
        import builtins
        closes = getattr(builtins, 'price_history', None)
        if closes and isinstance(closes, (list, tuple)) and len(closes) >= 20:
            recent_lows = min(closes[-20:])
    except Exception:
        pass
    if recent_lows is not None and current_price < recent_lows * (1.0 + max(0.0, knife_rebound_pct)):
        print(f"[INFO] 買い禁止: 直近安値({recent_lows})から{knife_rebound_pct*100:.2f}%以上反発していない (現値={current_price})")
        return {'action': 'hold', 'amount': 0.0, 'price': current_price, 'buy_condition': False, 'sell_condition': False}

    # --- 急落後の小反発でのみ再エントリー ---
    if last_sell_price is not None and recent_lows is not None and last_sell_price > 0 and recent_lows > 0:
        deep_drop_price = last_sell_price * (1.0 - max(0.0, deep_drop_reentry_pct))
        rebound_price = recent_lows * (1.0 + max(0.0, rebound_entry_pct))
        if current_price > deep_drop_price:
            print(f"[INFO] 買い禁止: 急落不足 (sell={last_sell_price}, 要件<= {deep_drop_price}, 現値={current_price})")
            return {'action': 'hold', 'amount': 0.0, 'price': current_price, 'buy_condition': False, 'sell_condition': False}
        if current_price < rebound_price:
            print(f"[INFO] 買い禁止: 反発不足 (low={recent_lows}, 要件>= {rebound_price}, 現値={current_price})")
            return {'action': 'hold', 'amount': 0.0, 'price': current_price, 'buy_condition': False, 'sell_condition': False}

    if rsi is None or rsi < rebound_rsi_min or rsi > rebound_rsi_max:
        print(f"[INFO] 買い禁止: 急落後リバウンドRSI条件未達 rsi={rsi}")
        return {'action': 'hold', 'amount': 0.0, 'price': current_price, 'buy_condition': False, 'sell_condition': False}

    if buy_btc <= 0.0:
        print("[INFO] 買い見送り: 利用可能JPY残高または注文予算が最小注文額に不足")
        return {'action': 'hold', 'amount': 0.0, 'price': current_price, 'buy_condition': False, 'sell_condition': False}

    # 実注文と約定通知は呼び出し側で一元化する。
    return {'action': 'buy', 'amount': buy_btc, 'price': current_price, 'buy_condition': True}


def calculate_buy_amount(current_price, jpy_balance):
    """Calculate a BTC order size under both the per-order budget and free balance."""
    import math

    if current_price is None or jpy_balance is None:
        return 0.0

    price = float(current_price)
    available_jpy = float(jpy_balance)
    budget_jpy = float(os.getenv('BUY_BUDGET_JPY', '20000'))
    fee_rate = float(os.getenv('ESTIMATED_FEE_RATE', os.getenv('FEE_RATE', '0.0012')))
    min_order_btc = float(os.getenv('MIN_ORDER_BTC', '0.0011'))
    values = (price, available_jpy, budget_jpy, fee_rate, min_order_btc)
    if (
        not all(math.isfinite(value) for value in values)
        or price <= 0
        or available_jpy <= 0
        or budget_jpy <= 0
        or fee_rate < 0
        or fee_rate >= 1
        or min_order_btc < 0
    ):
        return 0.0

    spendable_jpy = min(available_jpy, budget_jpy)
    amount = math.floor((spendable_jpy / (price * (1.0 + fee_rate))) * 1e8) / 1e8
    return amount if amount >= min_order_btc else 0.0


# --- BTC残高を売買結果で更新する ---
def sell_all_positions(positions, exchange, pair, sell_amount=None):
    # 判定結果の数量を売却する
    # positions: 保有ポジションリスト
    # exchange: ccxtの取引所オブジェクト
    # pair: 通貨ペア(例: 'BTC/JPY')
    results = []
    total_btc = sum([pos.get('amount', 0) for pos in positions])
    if sell_amount is None:
        sell_amount = total_btc
    sell_amount = min(max(float(sell_amount or 0.0), 0.0), total_btc)
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
from dotenv import load_dotenv

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




# --- メインループ（Botの実行部分） ---
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

# Add FileLock import for file locking
try:
    from filelock import FileLock
except ImportError:
    FileLock = None  # Fallback if filelock is not installed
env_paths = ['.env']


# --- ロギング関数の再定義 ---
import datetime


# --- .envの読み込みは最上部で必ず実行 ---
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
    api_key = os.getenv('BITBANK_API_KEY') or os.getenv('API_KEY')
    api_secret = os.getenv('BITBANK_API_SECRET') or os.getenv('SECRET_KEY')
    if api_key and api_secret:
        return api_key, api_secret
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return api_key or config['api_key'], api_secret or config['api_secret']

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

def connect_to_bitbank():
    return create_bitbank_exchange()

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
                order = exchange.create_limit_buy_order(pair, amount, price)
            else:
                log_info("⚠️ 価格未指定のため買い注文をスキップ")
                return None
        elif order_type == 'sell':
            if price:
                order = exchange.create_limit_sell_order(pair, amount, price)
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
        required_env_vars = ["SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS", "TO_EMAIL"]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if not (os.getenv("BITBANK_API_KEY") or os.getenv("API_KEY")):
            missing_vars.append("BITBANK_API_KEY (or API_KEY)")
        if not (os.getenv("BITBANK_API_SECRET") or os.getenv("SECRET_KEY")):
            missing_vars.append("BITBANK_API_SECRET (or SECRET_KEY)")
        if missing_vars:
            raise ValueError(f"以下の環境変数が .env に設定されていません: {', '.join(missing_vars)}")
    else:
        # DRY_RUN の場合は外部依存を必須にしない
        required_env_vars = []

    # Exchange/FundManager の準備
    # exchangeのグローバル初期化・利用は削除（mainブロックでのみ初期化・run_botに渡す）

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


def main():
    """Initialize dependencies and start the single canonical bot loop."""
    print("[INFO] .envからBITBANK_API_KEY/BITBANK_API_SECRETを自動ロードします")
    try:
        exchange = create_bitbank_exchange()
        run_bot(exchange, fund_manager=None)
    except Exception as e:
        print(f"[ERROR] mainブロック例外: {e}")
        print("[INFO] .envファイルにBITBANK_API_KEY/BITBANK_API_SECRETを設定してください")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()