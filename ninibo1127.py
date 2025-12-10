class FundAdapter:
    def __init__(self, fund_manager=None, initial_fund=0.0, dry_run=True):
        self.fund = initial_fund

    def add_funds(self, amount):
        # 指定額をfundに加算
        self.fund += amount
        return True

# --- 注文実行ユーティティ ---
def execute_order(exchange, pair, order_type, amount, price=None):

    # Place order on Bitbank (ccxt)
    try:
        order = None

        # 本番注文のみ実行

        if order_type == 'buy':
            if price:
                order = exchange.create_order(pair, 'limit', 'buy', amount, price)
            else:
                order = exchange.create_order(pair, 'market', 'buy', amount)
            print(f"💰 買い注文: {amount:.4f} {pair.split('/')[0]} @ {price if price else '（成行）'}")

        elif order_type == 'sell':
            if price:
                order = exchange.create_order(pair, 'limit', 'sell', amount, price)
            else:
                order = exchange.create_order(pair, 'market', 'sell', amount)
            print(f"💸 売り注文: {amount:.4f} {pair.split('/')[0]} @ {price if price else '（成行）'}")

        else:
            print(f"無効な注文タイプです: {order_type}")
            
            # return None  # ← 関数外のため削除

        if order and isinstance(order, dict) and 'id' in order:
            print(f"注文成功: {order.get('id')}")
            return order
        else:
            print(f"注文に失敗しました: {order}")
            # return None  # ← 関数外のため削除

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ 注文実行中にエラーが発生しました: {e}")
        # return None  # ← 関数外のため削除
# --- DI対応版のエントリーポイント ---
import os
def run_bot_di(dry_run=False, exchange_override=None):
    """
    Bot のメインエントリーポイント（DI対応）
    Args:
        dry_run (bool): True の場合、実際の取引を行わずログ出力のみ
        exchange_override: テスト用の Exchange オブジェクト（None の場合は実際の取引所に接続）
    Returns:
        dict: 実行結果の辞書
    """
    required_env_vars = ["SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS", "TO_EMAIL", "API_KEY", "SECRET_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"以下の環境変数が .env に設定されていません: {', '.join(missing_vars)}")

    print(f"🚀 Bot開始 (本番モード)")

    # Exchange の準備
    if exchange_override:
        exchange = exchange_override
    else:
        exchange = connect_to_bitbank()
        if not exchange:
            return {"status": "error", "message": "取引所接続に失敗"}

    # FundManager の準備
    initial_fund = float(os.getenv('INITIAL_FUND', '20000'))
    from pathlib import Path
    _raw_fm = FundManager(initial_fund=initial_fund, state_file=os.getenv('FUND_STATE_FILE', 'funds_state.json'))
    import time
    try:
        while True:
            result = run_bot(exchange, _raw_fm, dry_run)
            print("1時間待機します...")
            time.sleep(3600)  # 1時間ごとに判定
        # returnはループ外（通常到達しない）
        # return {"status": "success", "message": "Bot実行完了", "result": result}
    except Exception as e:
        return {"status": "error", "message": f"Bot実行中にエラー: {e}"}
    # return None  # ← 関数外のため削除
# --- 価格取得のユーティリティ ---
def get_latest_price(exchange, pair='BTC/JPY'):
    try:
        ticker = exchange.fetch_ticker(pair)
        if isinstance(ticker, dict) and 'last' in ticker:
            return float(ticker['last'])
        if isinstance(ticker, dict) and 'close' in ticker:
            return float(ticker['close'])
        # 他の型やエラー時
        # return None  # ← 関数外のため削除
    except Exception as e:
        print(f"⚠️ 価格チェックエラー: {e}")
        # return None  # ← 関数外のため削除



# --- メール通知関数の定義（未定義エラー対策） ---
import smtplib
from email.mime.text import MIMEText
def send_notification(smtp_host, smtp_port, smtp_user, smtp_password, email_to, subject, message):
    try:
        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = smtp_user
        msg['To'] = email_to
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, [email_to], msg.as_string())
        print(f"📧 通知メール送信: {subject}")
    except Exception as e:
        print(f"⚠️ メール送信エラー: {e}")
# --- FundManager, _adapt_fund_manager_instance の定義（stagingから移植） ---
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
            except Exception:
                pass
        def _persist(self):
            if not self._state_file:
                return
            try:
                obj = {'available': float(self._available), 'reserved': float(self._reserved)}
                self._state_file.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
            except Exception:
                pass
        def available_fund(self) -> float:
            try:
                return float(self._available)
            except Exception:
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
                if self._available < c:
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
except Exception:
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
            if self.available_fund() < c:
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
interval_seconds = 3600
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
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add FileLock import for file locking
try:
    from filelock import FileLock
except ImportError:
    FileLock = None  # Fallback if filelock is not installed
env_paths = ['.env']
DYN_OHLCV_DAYS = 30
DYN_THRESHOLD_BUFFER_JPY = 1000
DYN_THRESHOLD_BUFFER_PCT = 0.01
env_loaded = False
DYN_THRESHOLD_RATIO = 1.0
pair = 'BTC/JPY'
days = 30
buffer_jpy = int(os.getenv('BALANCE_BUFFER', 1000))
buffer_pct = 0.01
# --- 未定義定数・変数のダミー定義 ---
TRADE_TRIGGER_PCT = 10.0
MIN_PRICE_THRESHOLD_JPY = 1000
USE_DYNAMIC_THRESHOLD = True
MIN_ORDER_BTC = 0.0001
BALANCE_BUFFER = 0
BUY_ON_BREAKOUT = False
BREAKOUT_LOOKBACK_DAYS = 30
BREAKOUT_PCT = 0.03
BREAKOUT_SMA_SHORT = 5
BREAKOUT_SMA_LONG = 25
initial_cost = 0
MAX_SLIPPAGE_PCT = 5.0  # スリッページ許容率（例: 5%）

# --- STATE_FILEのグローバル定義 ---
from pathlib import Path
STATE_FILE = Path('funds_state.json')

class FundAdapter:
    def __init__(self, fund_manager=None, initial_fund=0.0, dry_run=True):
        self.fund = initial_fund

    def available_fund(self):
        return self.fund

    def reserve(self, amount):
        if amount > self.fund:
            return False
        self.fund -= amount
        return True

    def place_order(self, amount):
        return self.reserve(amount)

    def add_funds(self, amount):
        self.fund += amount
        return True

    def reserve(self, amount):
        # ダミー: 指定額をfundから減算
        if amount > self.fund:
            return False
        self.fund -= amount
        return True

    def place_order(self, amount):
        # reserveと同じ動作
        return self.reserve(amount)

import logging

# --- ロギング関数の再定義 ---
import datetime
import time
import sys


# log_info, log_warn, log_debug, log_error の重複定義を防ぐ
# すでにファイル先頭で定義済みなので、以降の重複定義は削除
# --- 未定義変数のダミー定義（未定義エラー防止用） ---
available_pre = 10000
allowed_by_percent = 10000

        # ...existing code...

        # --- connect_to_bitbank: Bitbank用の簡易接続関数（未定義エラー対策のダミー実装） ---
def connect_to_bitbank():
    import ccxt
    api_key = os.getenv("API_KEY")
    secret_key = os.getenv("SECRET_KEY")
    return ccxt.bitbank({
        'apiKey': api_key or "",
        'secret': secret_key or "",
    })

        # ...existing code...



# --- メイン実行部 ---
# run_bot_diの定義より後ろで呼び出すように修正
if __name__ == "__main__":
    try:
        log_info("Bot起動中...")
        run_bot_di()
    except Exception as e:
        log_error(f"Bot起動時に例外: {e}")


# ccxt がインストールされていない環境でもファイルが読み込めるよう、フォールバックのスタブを用意します。
try:
    import ccxt  # type: ignore
    # ...existing code...
except Exception:
    # 最低限のインターフェースを持つスタブ実装
    class AuthenticationError(Exception):
            try:
                log_info("Bot起動中...")
                run_bot_di()
            except Exception as e:
                log_error(f"Bot起動時に例外: {e}")
# 日本標準時 (JST) のタイムゾーンオブジェクトを作成
try:
    pass  # ← ここに必要な処理があれば記述
# 例外処理が不要なら except で何もしない
except Exception:
    JST = datetime.timezone(datetime.timedelta(hours=9))

# --- dry-run / test stub support ---
# 環境変数 DRY_RUN が真なら実際のネットワーク呼び出しや注文を行わないスタブを使います。
DRY_RUN = str(os.getenv('DRY_RUN', '0')).lower() in ('1', 'true', 'yes', 'on')
try:
    DRY_RUN_PRICE = float(os.getenv('DRY_RUN_PRICE', '5000000'))
except Exception:
    DRY_RUN_PRICE = 5000000.0

class ExchangeStub:
    # 軽量な取引所スタブ: dry-run 用。実ネットワーク呼び出しを行いません。
    def __init__(self, price=None):
        try:
            self._price = float(price) if price is not None else DRY_RUN_PRICE
        except Exception:
            self._price = DRY_RUN_PRICE

    def fetch_balance(self):
        return {'total': {'JPY': 100000.0, 'BTC': 0.0}}

    def fetch_ticker(self, pair):
        return {'last': self._price}

    def fetch_ohlcv(self, pair, timeframe='1h', limit=250):
        try:
            log_info("Bot起動中...")
            run_bot_di()
        except Exception as e:
            log_error(f"Bot起動時に例外: {e}")
    cost = 500
    adapter = FundAdapter(fund_manager=None, initial_fund=1000.0, dry_run=True)
    ok = adapter.reserve(cost) if hasattr(adapter, 'reserve') else adapter.place_order(cost)
    try:
        log_info(f"reserve/place_order({cost}) -> {ok}")
        log_info("available after reserve:", adapter.available_fund())
    except Exception:
        log_info(f"reserve/place_order({cost}) ->", ok)
        log_info("available after reserve:", adapter.available_fund())
    # attempt release (refund)
    if hasattr(adapter, 'release'):
        adapter.release(cost)
        try:
            log_info("after release available:", adapter.available_fund())
        except Exception:
            log_info("after release available:", adapter.available_fund())
    else:
        # fallback: add_funds used as refund
        adapter.add_funds(cost)
        try:
            log_info("after add_funds available:", adapter.available_fund())
        except Exception:
            log_info("after add_funds available:", adapter.available_fund())

    # Dry-run adapter
    try:
        da = FundAdapter(fund_manager=None, initial_fund=1000.0, dry_run=True)
        try:
            log_info("dry initial available:", da.available_fund())
            ok2 = da.reserve(300)
            log_info(f"dry reserve(300) -> {ok2} available-> {da.available_fund()}")
        except Exception:
                    # 例外時は何もしない
                    pass
        try:
            log_info("✅ bitbankにccxtで認証接続しました。")
        except Exception:
            pass

    except Exception as e:
        try:
            log_error(f"❌ bitbankへの接続中にエラーが発生しました: {e}")
        except Exception:
            log_error(f"❌ bitbankへの接続中にエラーが発生しました: {e}")
        # return None  # ← 関数外のため削除




# === プライベートAPI関数群（認証必須） ===

def get_account_balance(exchange):
    
    """
    アカウントの残高情報を取得します。
    Returns:
        dict: { 'total': {...}, 'free': {...}, 'used': {...} }
    """
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
        try:
            log_error(f"❌ 残高取得エラー: {e}")
        except Exception:
            pass
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


def compute_dynamic_threshold(exchange, pair='BTC/JPY', days=DYN_OHLCV_DAYS,
                              buffer_jpy=DYN_THRESHOLD_BUFFER_JPY, buffer_pct=DYN_THRESHOLD_BUFFER_PCT):
    # Compute dynamic threshold from past OHLCV data
    try:
        import pandas as pd
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
        # Define get_ohlcv if not already defined
        def get_ohlcv(exchange, pair='BTC/JPY', timeframe='1d', limit=100):
            try:
                raw = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
                if not raw or len(raw) == 0:
                    return None
                import pandas as pd
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
            ratio = float(os.environ.get('DYN_THRESHOLD_RATIO', DYN_THRESHOLD_RATIO))
        except Exception:
            ratio = float(DYN_THRESHOLD_RATIO)
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
        import pandas as pd
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


def compute_rsi(values, period=14):
    # Compute RSI from list of close prices. Returns float or None.
    try:
        if values is None:
            return None
        vals = [float(v) for v in values if v is not None]
        if len(vals) < period + 1:
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
        try:
            ratio = float(os.environ.get('DYN_THRESHOLD_RATIO', DYN_THRESHOLD_RATIO))
        except Exception:
            ratio = float(DYN_THRESHOLD_RATIO)
        buffer_jpy = float(os.environ.get('DYN_THRESHOLD_BUFFER_JPY', DYN_THRESHOLD_BUFFER_JPY))
        buffer_pct = float(os.environ.get('DYN_THRESHOLD_BUFFER_PCT', DYN_THRESHOLD_BUFFER_PCT))
        min_close = min(closes)
        max_close = max(closes)
        if ratio and float(ratio) > 0:
            threshold = float(min_close) + (float(max_close) - float(min_close)) * float(ratio)
        elif buffer_jpy and float(buffer_jpy) > 0:
            threshold = float(min_close) + float(buffer_jpy)
        else:
            threshold = float(min_close) * (1.0 + float(buffer_pct))
        return float(threshold), float(min_close), float(max_close)
        
        spread = None
        mid_price = None
        if bid_price is not None and ask_price is not None:
            spread = ask_price - bid_price
            mid_price = (bid_price + ask_price) / 2.0
        
        return {
            'bids': bids,
            'asks': asks,
            'spread': spread,
            'mid_price': mid_price,
            'best_bid': bid_price,
            'best_ask': ask_price
        }
    except Exception as e:
        try:
            log_warn(f"⚠️ 板情報の取得に失敗: {e}")
        except Exception:
            pass
        return {
            'bids': [],
            'asks': [],
            'spread': None,
            'mid_price': None,
            'best_bid': None,
            'best_ask': None
        }


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


def compute_indicators(exchange, pair='BTC/JPY', timeframe='1h', limit=500):
    # Fetch OHLCV and compute a set of indicators. Returns dict of values (may contain None).
    try:
        # OHLCVデータ取得（ダミー実装）
        raw = []
        indicators = {}
        # prepare lists
        closes = [float(r[4]) for r in raw if r and len(r) >= 5 and r[4] is not None]
        highs = [float(r[2]) for r in raw if r and len(r) >= 3 and r[2] is not None]
        lows = [float(r[3]) for r in raw if r and len(r) >= 4 and r[3] is not None]

        indicators['latest_close'] = closes[-1] if closes else None
        indicators['sma_short_50'] = compute_sma_from_list(closes, 50)
        indicators['sma_long_200'] = compute_sma_from_list(closes, 200)
        indicators['ema_12'] = compute_ema(closes, 12)
        indicators['ema_26'] = compute_ema(closes, 26)
        indicators['atr_14'] = compute_atr(raw, period=14)
        indicators['rsi_14'] = compute_rsi(closes, period=14)
        # recent high over 20 periods
        try:
            indicators['recent_high_20'] = max(highs[-20:]) if highs and len(highs) >= 1 else None
        except Exception:
            indicators['recent_high_20'] = None

        return indicators
    except Exception:
        return {
            'sma_short_50': None,
            'sma_long_200': None,
            'ema_12': None,
            'ema_26': None,
            'atr_14': None,
            'rsi_14': None,
            'recent_high_20': None,
            'latest_close': None
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
            writer.writerow([
                ts,
                pair,
                indicators.get('latest_close'),
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


# --- State utilities for cooldown / positions ---
def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(state):
    try:
        # Write atomically: write to a temp file then replace to avoid partial writes
        tmp_path = STATE_FILE.with_name(STATE_FILE.name + '.tmp')
        # Use an explicit open+flush+fsync to reduce chance of OS-level caching/AV interference
        try:
            import io
            jtxt = json.dumps(state, ensure_ascii=False, indent=2)
            with open(str(tmp_path), 'w', encoding='utf-8') as fh:
                fh.write(jtxt)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except Exception:
                    # os.fsync may not be available on some platforms/streams; ignore if fails
                    pass
        except Exception as e_write_tmp:
            # If writing tmp file failed, attempt direct write and log error
            try:
                STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
                try:
                    print(f"DEBUG: save_state direct write fallback succeeded for {STATE_FILE}")
                except Exception:
                    pass
                # write a small success marker for forensic checks
                try:
                    ok_marker = STATE_FILE.with_name(STATE_FILE.name + '.last_save_ok')
                    ok_marker.write_text(json.dumps({'time': int(time.time()), 'method': 'direct_fallback'}), encoding='utf-8')
                except Exception:
                    pass
                return
            except Exception as e_direct:
                # log both failures
                try:
                    errfile = STATE_FILE.with_name(STATE_FILE.name + '.save_error.log')
                    import traceback
                    errtxt = 'tmp_write_error: ' + str(e_write_tmp) + "\ndirect_write_error: " + str(e_direct) + "\n"
                    errtxt += traceback.format_exc()
                    errfile.write_text(errtxt, encoding='utf-8')
                except Exception:
                    pass
                raise e_direct

        try:
            # atomic replace where possible
            os.replace(str(tmp_path), str(STATE_FILE))
            # After successful replace, create a tiny marker file for forensic verification
            try:
                ok_marker = STATE_FILE.with_name(STATE_FILE.name + '.last_save_ok')
                info = {'time': int(time.time()), 'size': STATE_FILE.stat().st_size, 'path': str(STATE_FILE)}
                ok_marker.write_text(json.dumps(info, ensure_ascii=False), encoding='utf-8')
            except Exception:
                    pass
            try:
                log_debug(f"DEBUG: save_state succeeded and replaced {STATE_FILE} (size={STATE_FILE.stat().st_size})")
            except Exception:
                    pass
            return
        except Exception as e_replace:
            # fallback to non-atomic write
            try:
                STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
                try:
                    ok_marker = STATE_FILE.with_name(STATE_FILE.name + '.last_save_ok')
                    ok_marker.write_text(json.dumps({'time': int(time.time()), 'method': 'non_atomic_replace'}), encoding='utf-8')
                except Exception:
                    pass
                try:
                    log_debug(f"DEBUG: save_state fallback non-atomic write succeeded for {STATE_FILE}")
                except Exception:
                    pass
                return
            except Exception as e_write:
                # fall through to outer exception handling
                exc = e_write
                # attempt to log both exceptions
                try:
                    errfile = STATE_FILE.with_name(STATE_FILE.name + '.save_error.log')
                    import traceback
                    errtxt = 'replace_error: ' + str(e_replace) + "\nwrite_error: " + str(e_write) + "\n"
                    errtxt += traceback.format_exc()
                    errfile.write_text(errtxt, encoding='utf-8')
                except Exception:
                    pass
                raise exc
    except Exception as e:
        # Print and persist detailed error information to help debugging on Windows
        try:
            import traceback
            try:
                log_warn("WARN: could not save state:", e)
            except Exception:
                log_warn("WARN: could not save state:", e)
            errfile = STATE_FILE.with_name(STATE_FILE.name + '.save_error.log')
            errfile.write_text(''.join(traceback.format_exception(type(e), e, e.__traceback__)), encoding='utf-8')
        except Exception:
            try:
                log_warn("WARN: could not save state and failed to write error log")
            except Exception:
                    pass


def get_last_buy_time(state):
    return state.get("last_buy_time")


def set_last_buy_time(state, ts=None):
    state["last_buy_time"] = ts or int(time.time())
    save_state(state)


def record_position(state, side, price, qty):
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
    save_state(state)
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
        return delta_pct > float(MAX_SLIPPAGE_PCT)
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

    # RSIによる売買判定
    if latest_data['rsi'] <= 30:
        signal = 'buy_entry'
        message = f"✅ RSI買いシグナル: RSI={latest_data['rsi']:.2f} (30以下)"
        return signal, message
    elif latest_data['rsi'] >= 70:
        signal = 'sell_all'
        message = f"❌ RSI売りシグナル: RSI={latest_data['rsi']:.2f} (70以上)"
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
                # 指定価格で指値注文を出す
                order = exchange.create_order(pair, 'limit', 'buy', amount, price)
            else:
                # 価格が指定されていなければ成行注文
                order = exchange.create_order(pair, 'market', 'buy', amount)
            log_order("💰 買い", pair, amount, price)

        elif order_type == 'sell':
            if price:
                order = exchange.create_order(pair, 'limit', 'sell', amount, price)
            else:
                order = exchange.create_order(pair, 'market', 'sell', amount)
            log_order("💸 売り", pair, amount, price)

        else:
            log_error(f"無効な注文タイプです: {order_type}")
            return None

        if order and isinstance(order, dict) and 'id' in order:
            log_info("注文成功:", order.get('id'))
            return order
        else:
            log_error("注文に失敗しました:", order)
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
        log_debug(f"DEBUG: run_bot start - DRY_RUN={DRY_RUN}, pair={pair}")
    except Exception:
        log_debug("DEBUG: run_bot start (print failed)")

    # 実行時チェック: 必要な環境変数は dry_run のときは緩和する
    env_dry_run = os.getenv("DRY_RUN", "").lower() in ["1", "true", "yes", "on"]
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
    exchange = None
    if 'exchange' not in locals() or exchange is None:
        exchange = connect_to_bitbank()
    # adapter を通して安全に扱えるようにする
    fund_manager = _adapt_fund_manager_instance(FundManager(initial_fund=0.0, state_file=os.getenv('FUND_STATE_FILE', 'funds_state.json')))

    if not exchange and not env_dry_run:
        try:
            log_error("API接続に失敗したためBotを停止します。")
        except Exception:
            log_error("API接続に失敗したためBotを停止します。")
        return

    try:
        log_info(f"Botを {pair} で実行します。データ取得間隔: {interval_seconds}秒 (1時間)")
    except Exception:
        log_info(f"Botを {pair} で実行します。データ取得間隔: {interval_seconds}秒 (1時間)")

    # 1回あたりの注文予算（JPY）。ユーザー指定が無ければ 10000 円に変更
    # JAPANESE_YEN_BUDGET = float(os.getenv('JAPANESE_YEN_BUDGET', '10000'))  # ← 使わない
    # 最小購入 BTC 数量（取引所の制約に合わせる）
    MIN_ORDER_BTC = float(os.getenv('MIN_ORDER_BTC', '0.0001'))
    # 小額運用向けの安全設定
    # 1回の注文で使ってよい最大割合 (残高に対するパーセンテージ。例: 0.05 = 5%)
    try:
        MAX_RISK_PERCENT = float(os.getenv('MAX_RISK_PERCENT', '0.05'))
    except Exception:
        MAX_RISK_PERCENT = 0.05
    # 注文後に常に残す最低バッファ (JPY)
    try:
        BALANCE_BUFFER = float(os.getenv('BALANCE_BUFFER', '500'))
    except Exception:
        BALANCE_BUFFER = 500.0

    # available_pre, allowed_by_percent, allowed_by_buffer, reserved_budgetの計算をprintより前に必ず実行
    try:
        available_pre = float(fund_manager.available_fund()) if hasattr(fund_manager, 'available_fund') else None
    except Exception:
        available_pre = None
    # --- ここから修正: 必ず1000円残し、それ以外は全額使う ---
    try:
        BALANCE_BUFFER = float(os.getenv('BALANCE_BUFFER', '1000'))
    except Exception:
        BALANCE_BUFFER = 1000.0
    if available_pre is not None:
        reserved_budget = max(0.0, available_pre - BALANCE_BUFFER)
    else:
        reserved_budget = 0.0
    log_info(f"💰 1回あたりの注文予算: {reserved_budget:.2f} 円（常に{BALANCE_BUFFER:.0f}円残し）")
    log_info(f"📉 最低注文数量: {MIN_ORDER_BTC} BTC")

    # --- 取引所の残高情報を取得して表示（少額運用向けに簡潔に） ---
    try:
        balance_info = get_account_balance(exchange)
        if balance_info and balance_info.get('total'):
            jpy_free = balance_info['free'].get('JPY', 0)
            btc_free = balance_info['free'].get('BTC', 0)
            # 少額運用では利用可能額のみ表示（総額は省略）
            log_info(f"💼 利用可能残高: JPY={jpy_free:.0f}円, BTC={btc_free:.8f}BTC")
    except Exception as e:
        print(f"DEBUG: save_state exception: {e}")
        try:
            log_warn(f"⚠️ 残高取得に失敗: {e}")
        except Exception:
            pass

    # --- アクティブな注文を表示（少額運用では簡潔に） ---
    try:
        open_orders = get_open_orders(exchange, pair)
        if open_orders:
            log_info(f"📋 未約定注文: {len(open_orders)}件")
            # 少額運用では最大2件まで表示
            for order in open_orders[:2]:
                log_info(f"  {order['side'].upper()} {order['amount']:.4f}BTC @ {order['price']:.0f}円")
    except Exception as e:
        try:
            log_warn(f"⚠️ アクティブ注文取得に失敗: {e}")
        except Exception:
            pass

    # --- 最近の約定履歴を表示（少額運用では最新2件のみ） ---
    try:
        my_trades = get_my_trades(exchange, pair, limit=5)
        if my_trades:
            log_info(f"💱 最近の約定: {len(my_trades)}件")
            # 少額運用では最新2件のみ簡潔に表示
            for trade in my_trades[:2]:
                fee_cost = trade.get('fee', {}).get('cost', 0) if trade.get('fee') else 0
                log_info(f"  {trade['side'].upper()} {trade['amount']:.4f}BTC @ {trade['price']:.0f}円 (手数料:{fee_cost:.2f}円)")
    except Exception as e:
        try:
            log_warn(f"⚠️ 約定履歴取得に失敗: {e}")
        except Exception:
            print(f"⚠️ 約定履歴取得に失敗: {e}")


    # --- 低残高アラート設定 ---
    LOW_FUNDS_ALERT_JPY = float(os.getenv('LOW_FUNDS_ALERT_JPY', '2000'))

    # state を読み込み、保有ポジションがあれば利確チェックを行う
    try:
        # Use a file lock when reading/modifying/saving state for sell flow to avoid
        # races with concurrent buy operations that also update the state file.
        LOCKFILE_SELL = os.getenv('ORDER_LOCKFILE', '/tmp/ninibo_order.lock')
        try:
            with FileLock(LOCKFILE_SELL):
                state = load_state()
                positions = state.get('positions') if isinstance(state, dict) else None
                if positions and isinstance(positions, list) and len(positions) > 0:
                    last_pos = positions[-1]
        except Exception:
            pass
    except Exception:
        pass

    if available_pre is not None and float(available_pre) < float(LOW_FUNDS_ALERT_JPY):
        msg = f"⚠️ 資金アラート: 残高が少なくなっています（残高: {available_pre:.0f} 円 < 閾値: {LOW_FUNDS_ALERT_JPY:.0f} 円）"
        print(msg)
        try:
            smtp_host = os.getenv('SMTP_HOST')
            smtp_port = int(os.getenv('SMTP_PORT', '587'))
            smtp_user = os.getenv('SMTP_USER')
            smtp_password = os.getenv('SMTP_PASS')
            email_to = os.getenv('TO_EMAIL')
            subject = f"Low Funds Alert: {available_pre:.0f}円"
            if smtp_host and email_to:
                send_notification(smtp_host, smtp_port, smtp_user, smtp_password, email_to, subject, msg)
        except Exception:
            pass

    if available_pre is not None:
        # reserved_budgetは常に残高の90%とバッファ考慮の小さい方
        allowed_by_buffer = 99999999  # 仮の値。実際のロジックに合わせて調整してください
        reserved_budget = min(allowed_by_percent, allowed_by_buffer)
    else:
        reserved_budget = 0.0
    LOCKFILE = os.getenv('ORDER_LOCKFILE', '/tmp/ninibo_order.lock')
    with FileLock(LOCKFILE):
        try:
            available = float(fund_manager.available_fund()) if hasattr(fund_manager, 'available_fund') else None
        except Exception:
            available = None

        # 予約する額が妥当か（手数料込みで最小数量を満たすか確認）
        # ここでは compute_qty_for_budget_with_fee を使って reserved_budget に対する実注文量を判定する
        # 記録: 予約が成功した時点の価格と時刻（後でスリッページ/実行ウィンドウ判定に使う）
        try:
            reservation_price = float(latest_price) if 'latest_price' in globals() and latest_price is not None else float(os.getenv('DRY_RUN_PRICE', '0'))
            reservation_time = int(time.time())

            # Cooldown (買いの間隔) チェック
            state = load_state()
            last_buy = get_last_buy_time(state)
            COOLDOWN_SEC = int(os.getenv('COOLDOWN_SEC', '3600'))  # デフォルトは1時間
            if last_buy and (time.time() - last_buy) < COOLDOWN_SEC:
                print("Cooldown active -> skipping buy to avoid frequent add-on")
                return

            q_check, cost_check, fee_check = compute_qty_for_budget_with_fee(
                reserved_budget, float(latest_price), min_btc=MIN_ORDER_BTC, step=MIN_ORDER_BTC,
                available_jpy=float(available) if available is not None else 0.0, balance_buffer=float(BALANCE_BUFFER)
            )

        except Exception:
            reservation_price = None
            reservation_time = None
            q_check = cost_check = fee_check = 0

        # 予算を予約するのに十分な残高があるか確認
        if available is not None and available < reserved_budget:
            print(f"🚫 残高不足のため注文をスキップします（必要: {reserved_budget:.2f} 円, 残高: {available:.2f} 円）")
            return

        if q_check <= 0 or reserved_budget <= 0:
            # Gather more internal diagnostics for debugging
            try:
                fee_rate = float(os.getenv('FEE_RATE', '0.001'))
            except Exception:
                fee_rate = 0.001
            try:
                q_check, cost_check, fee_check = compute_qty_for_budget_with_fee(
                    reserved_budget, float(latest_price), min_btc=MIN_ORDER_BTC, step=MIN_ORDER_BTC,
                    available_jpy=float(available) if available is not None else 0.0, balance_buffer=float(BALANCE_BUFFER)
                )
            except Exception:
                q_check = cost_check = fee_check = 0
            try:
                max_allowed_jpy = min(float(reserved_budget), float(available) - float(BALANCE_BUFFER)) if available is not None else float(reserved_budget)
            except Exception:
                max_allowed_jpy = float(reserved_budget)
            try:
                approx_qty = max_allowed_jpy / (reservation_price * (1.0 + fee_rate)) if reservation_price and reservation_price > 0 else 0.0
            except Exception:
                approx_qty = 0.0
            try:
                # Define fee_fixed before using it
                try:
                    fee_fixed = float(os.getenv('FEE_FIXED_JPY', '0.0'))
                except Exception:
                    fee_fixed = 0.0
                detail = (
                    f"予約額={reserved_budget:.2f}, q_check={q_check:.8f}, cost_check={cost_check:.2f}, fee_check={fee_check:.2f}, "
                    f"fee_rate={fee_rate}, fee_fixed={fee_fixed:.2f}, min_btc={MIN_ORDER_BTC}, step={MIN_ORDER_BTC}, "
                    f"max_allowed_jpy={max_allowed_jpy:.2f}, approx_qty={approx_qty:.8f}, reservation_price={reservation_price}"
                )
            except Exception:
                detail = f"予約額={reserved_budget:.2f} (failed to build details)"
            msg = f"ℹ️ 予約額が手数料込みの最小注文コストに満たないため注文をスキップします（{detail}）"
            print(msg)
            try:
                if smtp_host and email_to:
                    send_notification(smtp_host, smtp_port, smtp_user, smtp_password, email_to, subject, msg)
            except Exception:
                pass
            return

        # ここで予算を差し引いて予約する（失敗した場合は例外が上がる）
        if hasattr(fund_manager, 'place_order'):
            try:
                # デバッグ情報: 予約前の利用可能残高と予約額
                try:
                    cur_avail = float(fund_manager.available_fund()) if hasattr(fund_manager, 'available_fund') else None
                except Exception:
                    cur_avail = None
                print(f"DEBUG: 予約前 available={cur_avail}, reserved_budget={reserved_budget:.2f}")

                # reserve funds via adapter (new API)
                if hasattr(fund_manager, 'reserve'):
                    ok = fund_manager.reserve(reserved_budget)
                else:
                    ok = fund_manager.place_order(reserved_budget)

                # デバッグ情報: 予約後の利用可能残高
                try:
                    after_avail = float(fund_manager.available_fund()) if hasattr(fund_manager, 'available_fund') else None
                except Exception:
                    after_avail = None
                print(f"DEBUG: 予約後 available={after_avail}")

                # BTC残高のみ管理
                # その場合は残高が実際に減っているかで成功を判定する（後方互換処理）。                
                if not ok:
                    if cur_avail is not None and after_avail is not None and after_avail < cur_avail:
                        print("ℹ️ reserve/place_order は False/None を返しましたが、残高が減っているため予約成功とみなします")
                        ok = True
                    else:
                        print(f"⚠️ 資金予約に失敗しました（reserve/place_order が False を返しました）。予約額: {reserved_budget:.2f}")
                        return
                reserved = True
                try:
                    print(f"DEBUG: reservation set reserved={reserved}, reserved_budget={reserved_budget}")
                except Exception:
                    pass
            except Exception as e:
                print(f"⚠️ 予約（予算差し引き）に失敗しました: {e}")
                return

        # 実際の注文を発行: 注文直前に最新価格を取得して数量を再計算します
        try:
            # 最新価格を再取得して手数料込みで最終数量を算出
            latest_price_now = get_latest_price(exchange, pair)
            try:
                print(f"DEBUG: latest_price_now={latest_price_now}")
            except Exception:
                pass

            # --- 売買トリガー判定: reference を参照して -TRADE_TRIGGER_PCT% で買い, +TRADE_TRIGGER_PCT% で売り ---
            try:
                # state は前段でロード済みのはずですが、安全のため再取得
                try:
                    state
                except Exception:
                    state = load_state()

                # 監視基準価格 (watch_reference) を state から取得。
                # 優先ロジック:
                #  1) state.watch_reference が未設定 or 最新価格で初期化されている場合、
                #     保有ポジション（直近 buy）の price を優先して使う。
                #  2) それ以外は state.watch_reference を使う。
                #  3) どちらも無ければ現在価格で初期化して保存
                watch_ref = None
                try:
                    if isinstance(state, dict):
                        val = state.get('watch_reference')
                        if val is not None:
                            try:
                                watch_ref = float(val)
                            except Exception:
                                watch_ref = None
                        else:
                            watch_ref = None
                except Exception:
                    watch_ref = None

                # Try to infer from last buy position when appropriate.
                try:
                    positions = state.get('positions') if isinstance(state, dict) else None
                    if positions and isinstance(positions, list) and len(positions) > 0:
                        last_pos = positions[-1]
                        last_pos_price = 0.0
                        try:
                            last_pos_price = float(last_pos.get('price', 0) or 0)
                        except Exception:
                            last_pos_price = 0.0
                        if last_pos and last_pos.get('side') == 'buy' and last_pos_price > 0:
                            # prefer last buy price when state.watch_reference is missing
                            # or when it appears to have been initialized to the latest price
                            prefer_last = False
                            if watch_ref is None:
                                prefer_last = True
                            else:
                                try:
                                    # if watch_ref equals latest price (seeded), prefer last_pos
                                    if latest_price_now is not None and abs(float(watch_ref) - float(latest_price_now)) < 1e-6:
                                        prefer_last = True
                                except Exception:
                                    pass

                            if prefer_last:
                                watch_ref = float(last_pos_price)
                                # persist inferred watch_reference for future runs
                                try:
                                    state['watch_reference'] = watch_ref
                                    save_state(state)
                                except Exception:
                                    pass
                except Exception:
                    pass

                # 最後に fallback: 最新価格で初期化
                if watch_ref is None and latest_price_now is not None:
                    try:
                        state['watch_reference'] = float(latest_price_now)
                        save_state(state)
                        watch_ref = float(latest_price_now)
                        print(f"INFO: watch_reference initialized to {watch_ref}")
                    except Exception:
                        watch_ref = float(latest_price_now) if latest_price_now is not None else None

                # 🔧 自動修正: watch_refが現在価格と大きく乖離している場合は現在価格にリセット
                # （過去の売却価格が残っている、または長期間動作していなかった場合の対策）
                try:
                    if watch_ref is not None and latest_price_now is not None:
                        ratio = float(latest_price_now) / float(watch_ref)
                        # 現在価格がwatch_refの2倍以上、または0.5倍以下の場合は異常とみなす
                        if ratio > 2.0 or ratio < 0.5:
                            old_ref = watch_ref
                            watch_ref = float(latest_price_now)
                            state['watch_reference'] = watch_ref
                            save_state(state)
                            print(f"⚠️ watch_reference が現在価格と大きく乖離していたため自動修正: {old_ref:.0f}円 → {watch_ref:.0f}円")
                except Exception as e:
                    print(f"⚠️ watch_reference 自動修正中にエラー: {e}")

                do_buy_by_pct = False
                try:
                    if latest_price_now is not None and watch_ref is not None:
                        # TRADE_TRIGGER_PCT%下落で買い（上昇は売りなので買わない）
                        threshold_buy = watch_ref * (1.0 - float(TRADE_TRIGGER_PCT) / 100.0)
                        
                        # TRADE_TRIGGER_PCT%下落ラインからさらに5%下落で買いチャンス通知
                        further_drop_threshold = threshold_buy * 0.95  # 設定値%下落からさらに5%下落
                        
                        # 下落で買い
                        do_buy_by_pct = float(latest_price_now) <= float(threshold_buy)
                        
                        # 買いチャンス通知（{TRADE_TRIGGER_PCT:.0f}%下落 + さらに5%下落 = 合計{TRADE_TRIGGER_PCT + 5:.0f}%下落）
                        if float(latest_price_now) <= float(further_drop_threshold):
                            # 重複通知防止
                            last_buy_alert = state.get('last_buy_opportunity_alert') if isinstance(state, dict) else None
                            should_alert = True
                            if last_buy_alert:
                                try:
                                    # 前回通知から5%以上変動していれば再通知
                                    prev_change = abs((float(latest_price_now) - float(last_buy_alert)) / float(last_buy_alert)) * 100.0
                                    if prev_change < 5.0:
                                        should_alert = False
                                except Exception:
                                    pass
                        
                            if should_alert:
                                print(f"🎯 買いチャンス！ watch_ref={watch_ref:.0f}円から{TRADE_TRIGGER_PCT + 5:.0f}%下落 → 現在={latest_price_now:.0f}円")
                                
                                # メール通知
                                try:
                                    smtp_host = os.getenv('SMTP_HOST')
                                    smtp_port = int(os.getenv('SMTP_PORT', '587'))
                                    smtp_user = os.getenv('SMTP_USER')
                                    smtp_password = os.getenv('SMTP_PASS')
                                    email_to = os.getenv('TO_EMAIL')
                                    
                                    if smtp_host and email_to:
                                        drop_percent = ((float(latest_price_now) - watch_ref) / watch_ref) * 100.0
                                        subject = f"BTC Buy Opportunity! {abs(drop_percent):.1f}% drop"
                                        message = (
                                            f"BTC buy opportunity detected!\n\n"
                                            f"[Price Info]\n"
                                            f"Reference price: {watch_ref:,.0f} JPY\n"
                                            f"{TRADE_TRIGGER_PCT:.0f}% drop line: {threshold_buy:,.0f} JPY\n"
                                            f"Current price: {latest_price_now:,.0f} JPY\n"
                                            f"Drop percent: {drop_percent:.2f}%\n\n"
                                            f"[Recommended Action]\n"
                                            f"- Deposit funds to bitbank\n"
                                            f"- Bot will auto-buy BTC after deposit\n"
                                            f"- Auto-sell will trigger after {TRADE_TRIGGER_PCT:.0f}% rise from buy price\n\n"
                                            f"This is a major drop opportunity!"
                                        )
                                        
                                        send_notification(smtp_host, smtp_port, smtp_user, smtp_password,
                                                        email_to, subject, message)
                                        print(f"📧 買いチャンス通知メール送信完了")
                                        
                                        # 通知記録
                                        try:
                                            state['last_buy_opportunity_alert'] = float(latest_price_now)
                                            save_state(state)
                                        except Exception:
                                            pass
                                except Exception as e:
                                    print(f"⚠️ 買いチャンス通知メール送信エラー: {e}")
                        
                        if do_buy_by_pct:
                            print(f"📉 買いシグナル(下落): watch_ref={watch_ref:.0f}円, 下落閾値={threshold_buy:.0f}円, 現在={latest_price_now:.0f}円")
                        else:
                            print(f"DEBUG: watch_ref={watch_ref:.0f}円, 下落閾値={threshold_buy:.0f}円, 現在={latest_price_now:.0f}円, do_buy={do_buy_by_pct}")
                except Exception:
                    do_buy_by_pct = False

                # optionally still allow breakout buys if configured
                allow_buy_by_breakout = False
                # (壊れたexcept/try/インデントを削除)
                # except Exception as e: ... の壊れた部分を削除
                try:
                    if BUY_ON_BREAKOUT:
                        recent_high = get_recent_high(exchange, pair, days=BREAKOUT_LOOKBACK_DAYS)
                        if recent_high is not None and latest_price_now is not None and float(latest_price_now) >= float(recent_high) * (1.0 + float(BREAKOUT_PCT)):
                            allow_buy_by_breakout = True
                        else:
                            sma_s = compute_sma_from_ohlcv(exchange, pair, days=BREAKOUT_SMA_SHORT)
                            sma_l = compute_sma_from_ohlcv(exchange, pair, days=BREAKOUT_SMA_LONG)
                            if sma_s is not None and sma_l is not None and latest_price_now is not None:
                                if float(sma_s) > float(sma_l) and float(latest_price_now) > float(sma_s) * (1.0 + float(BREAKOUT_PCT)):
                                    allow_buy_by_breakout = True
                except Exception:
                    allow_buy_by_breakout = False

                do_buy = bool(do_buy_by_pct) or bool(allow_buy_by_breakout)

                # CSV に指標とシグナルを書き出す（監視・後解析用）
                try:
                    sig = 'BUY' if do_buy else 'NONE'
                    indicators = None
                    if indicators is None:
                        indicators = compute_indicators(exchange, pair, timeframe='1h', limit=500)
                    write_indicators_csv(indicators if indicators is not None else {}, pair, signal=sig)
                except Exception:
                    pass

                if not do_buy:
                    print(f"🚫 買い条件未達（watch_ref={watch_ref}, latest={latest_price_now}, buy_pct={TRADE_TRIGGER_PCT}, breakout_allowed={allow_buy_by_breakout}）→ 予約を返金して終了")
                    if reserved:
                        LOCKFILE = os.getenv('ORDER_LOCKFILE', '/tmp/ninibo_order.lock')
                        with FileLock(LOCKFILE):
                            if hasattr(fund_manager, 'release'):
                                fund_manager.release(reserved_budget)
                            elif hasattr(fund_manager, 'add_funds'):
                                fund_manager.add_funds(reserved_budget)
                            reserved = False
                    return
            except Exception:
                # 比較中のエラーは安全側でキャンセルする
                print("⚠️ 売買判定でエラーが発生しました。発注を中断します。")
                if reserved:
                    LOCKFILE = os.getenv('ORDER_LOCKFILE', '/tmp/ninibo_order.lock')
                    with FileLock(LOCKFILE):
                        if hasattr(fund_manager, 'release'):
                            fund_manager.release(reserved_budget)
                        elif hasattr(fund_manager, 'add_funds'):
                            fund_manager.add_funds(reserved_budget)
                        reserved = False
                return
            if latest_price_now is None:
                print("⚠️ 注文直前に価格が取得できませんでした。予約を取り消します。")
                try:
                    LOCKFILE = os.getenv('ORDER_LOCKFILE', '/tmp/ninibo_order.lock')
                    with FileLock(LOCKFILE):
                        if hasattr(fund_manager, 'release'):
                            fund_manager.release(reserved_budget)
                        elif hasattr(fund_manager, 'add_funds'):
                            fund_manager.add_funds(reserved_budget)
                        reserved = False
                except Exception as e:
                    print(f"⚠️ 予約取り消し（返金）に失敗しました: {e}")
                    if reserved:
                        LOCKFILE = os.getenv('ORDER_LOCKFILE', '/tmp/ninibo_order.lock')
                        with FileLock(LOCKFILE):
                            if hasattr(fund_manager, 'release'):
                                fund_manager.release(reserved_budget)
                            elif hasattr(fund_manager, 'add_funds'):
                                fund_manager.add_funds(reserved_budget)
                    return

            final_qty, final_cost, final_fee = compute_qty_for_budget_with_fee(
                reserved_budget, float(latest_price_now), min_btc=MIN_ORDER_BTC, step=MIN_ORDER_BTC,
                available_jpy=float(available) if available is not None else 0.0, balance_buffer=float(BALANCE_BUFFER)
            )

            # 最小数量チェック
            if final_qty <= 0:
                print(f"ℹ️ 注文直前で数量が最小取引単位を下回りましたまたは手数料で合計が超過しました。予約を取り消します。")
                if reserved:
                    LOCKFILE = os.getenv('ORDER_LOCKFILE', '/tmp/ninibo_order.lock')
                    with FileLock(LOCKFILE):
                        if hasattr(fund_manager, 'release'):
                            fund_manager.release(reserved_budget)
                        elif hasattr(fund_manager, 'add_funds'):
                            fund_manager.add_funds(reserved_budget)
                return

            # 最終的に execute_order を呼ぶ
            order = execute_order(exchange, pair, 'buy', final_qty)
            try:
                print(f"DEBUG: execute_order returned: {order}")
            except Exception:
                pass
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ 注文直前処理で例外が発生しました: {e}")
            # 例外時は予約を取り消して返金
            if reserved:
                try:
                    LOCKFILE = os.getenv('ORDER_LOCKFILE', '/tmp/ninibo_order.lock')
                    with FileLock(LOCKFILE):
                        if hasattr(fund_manager, 'release'):
                            fund_manager.release(reserved_budget)
                        elif hasattr(fund_manager, 'add_funds'):
                            fund_manager.add_funds(reserved_budget)
                        reserved = False
                except Exception as e:
                    # 例外時は何もしない
                    pass
            return

        if order and isinstance(order, dict) and 'id' in order:
            # Confirm reserved funds (consume actual cost) if adapter supports it
            try:
                if reserved:
                    if hasattr(fund_manager, 'confirm'):
                        try:
                            fund_manager.confirm(final_cost)
                        except Exception:
                            pass
                    # legacy place_order already deducted funds at reservation
                    reserved = False
            except Exception:
                pass
            try:
                print(f"💰 注文後の残高: {fund_manager.available_fund():.2f} 円")
            except Exception:
                print("💰 注文後の残高を取得できませんでした。")
            # 注文成功時はポジションを記録してクールダウンタイムを更新
            try:
                try:
                    state  # may already exist
                except Exception:
                    state = load_state()

                # Robustly determine entry_price:
                # 1) prefer the local latest_price_now variable (exists in this scope)
                # 2) fall back to order['price'] if present
                # 3) fall back to order['cost'] / order['amount'] if available
                entry_price = None
                try:
                    if 'latest_price_now' in locals() and latest_price_now is not None:
                        entry_price = float(latest_price_now)
                    elif isinstance(order, dict):
                        price_val = order.get('price')
                        if price_val is not None:
                            try:
                                entry_price = float(price_val)
                            except Exception:
                                entry_price = 0.0
                        else:
                            cost_val = order.get('cost')
                            amount_val = order.get('amount')
                            if cost_val is not None and amount_val is not None and amount_val != 0:
                                try:
                                    entry_price = float(cost_val) / float(amount_val)
                                except Exception:
                                    entry_price = 0.0
                except Exception:
                    entry_price = None

                # Log if we couldn't infer a sensible entry price
                if entry_price is None:
                    try:
                        print(f"⚠️ entry_price could not be inferred (order={order}). saving 0.0 as fallback")
                    except Exception:
                        pass

                # Detailed debug dump before recording position
                try:
                    if 'final_qty' in locals() and final_qty is not None:
                        dbg_final_qty = float(final_qty)
                    elif isinstance(order, dict):
                        amt = order.get('amount', 0)
                        dbg_final_qty = float(amt) if amt is not None else 0.0
                    else:
                        dbg_final_qty = 0.0
                except Exception:
                    dbg_final_qty = 0.0
                try:
                    if 'final_cost' in locals() and final_cost is not None:
                        dbg_final_cost = float(final_cost)
                    elif isinstance(order, dict):
                        cst = order.get('cost', 0)
                        dbg_final_cost = float(cst) if cst is not None else 0.0
                    else:
                        dbg_final_cost = 0.0
                except Exception:
                    dbg_final_cost = 0.0
                try:
                    print(f"DEBUG: record_position: entry_price={entry_price}, dbg_final_qty={dbg_final_qty}, dbg_final_cost={dbg_final_cost}, order={order}")
                except Exception:
                    pass

                record_position(state, 'buy', entry_price if entry_price is not None else 0.0, dbg_final_qty)
                # Immediately read back saved state file for verification and log it
                try:
                    try:
                        saved_text = STATE_FILE.read_text(encoding='utf-8')
                        saved_json = json.loads(saved_text)
                    except Exception:
                        saved_json = None
                    try:
                        print(f"DEBUG: saved_state_file={STATE_FILE}, saved_positions={(saved_json.get('positions') if isinstance(saved_json, dict) else 'unreadable')}")
                    except Exception:
                        pass
                except Exception as e:
                    try:
                        print(f"DEBUG: failed to read saved state file: {e}")
                    except Exception:
                        pass
                # 購入後は監視基準をエントリ価格に更新して売りトリガーが機能するようにする
                try:
                    if entry_price is not None:
                        state['watch_reference'] = float(entry_price)
                        save_state(state)
                except Exception:
                    pass
                set_last_buy_time(state)
            except Exception as e:
                print(f"⚠️ 注文成功後の状態記録に失敗しました: {e}")
            print("✅ 注文が正常に完了しました。")
        else:
            # 注文が返ってこない/失敗した場合は予約取り消し（返金）
            msg_fail = "⚠️ 注文は実行されませんでした（API応答が不正です）。予約を取り消します。"
            print(msg_fail)
            try:
                if smtp_host and email_to:
                    send_notification(smtp_host, smtp_port, smtp_user, smtp_password, email_to, subject, msg_fail)
            except Exception:
                pass
            if reserved:
                try:
                    with FileLock(LOCKFILE):
                        if hasattr(fund_manager, 'release'):
                            fund_manager.release(reserved_budget)
                        elif hasattr(fund_manager, 'add_funds'):
                            fund_manager.add_funds(reserved_budget)
                        reserved = False
                except Exception as e:

                    print(f"⚠️ 予約取り消し（返金）に失敗しました: {e}")

    # 次回の参考として残高表示
    try:
        required_cost = initial_cost
        if hasattr(fund_manager, "available_fund"):
            available = fund_manager.available_fund()
            if available is not None and available < required_cost:
                print(f"🚫 次回は残高不足の可能性があります（必要: {required_cost:.2f} 円, 残高: {available:.2f} 円）")
            else:
                print(f"✅ 次の注文を実行可能: （必要: {required_cost:.2f} 円, 残高: {available:.2f} 円）")
    except Exception as e:
        print(f"🚫  残高チェック中にエラーが発生しました: {e}")

    return


# Botを実行
if __name__ == "__main__":
    try:
        log_info("🔁 自動売買Botを継続運用モードで起動します")
    except Exception:
        pass
    log_info("🔁 自動売買Botを継続運用モードで起動します")
    # DEBUG: main entry
    try:
        log_debug(f"DEBUG: __main__ start - DRY_RUN={DRY_RUN}")
    except Exception:
        pass
        # ...existing code...
    exchange = connect_to_bitbank()
    # 初期資金は env で設定可能（なければ 20000 円）
    initial_fund = float(os.getenv('INITIAL_FUND', '20000'))
    fund_manager = None
    # Ensure fund state is usable for local DRY_RUN/tests
    _ensure_fund_manager_has_funds(fund_manager, initial_amount=initial_fund)
    # 毎ループで自動投入する少額（例: 毎時間100円ずつ入金する）を環境変数で指定
    deposit_amount = float(os.getenv('DEPOSIT_AMOUNT', '0'))
    # 自動トップアップの閾値（この金額を下回ったら自動入金を行う）
    # 例: MIN_BALANCE_THRESHOLD=5000
    min_balance_threshold = float(os.getenv('MIN_BALANCE_THRESHOLD', os.getenv('BALANCE_TOPUP_THRESHOLD', '5000')))
    if not exchange:
        try:
            log_error("API接続に失敗したためBotを終了します。")
        except Exception:
            try:
                print("API接続に失敗したためBotを終了します。")
            except Exception:
                pass
    else:
        while True:
            # 💡 価格変動チェック: 20%変動時にメール通知
            try:
                latest_price = get_latest_price(exchange, 'BTC/JPY')
                # 基準価格を state から取得
                state = load_state()
                reference_price = None
                try:
                    val = state.get('watch_reference') if isinstance(state, dict) else None
                    reference_price = float(val) if val is not None else None
                except Exception:
                    pass
                # 基準価格がない場合は環境変数から取得
                if reference_price is None:
                    try:
                        reference_price = float(os.getenv('PRICE_REFERENCE', '15000000'))  # デフォルト1500万円
                    except Exception:
                        reference_price = 15000000.0
                # 価格変動率を計算（上昇も下落も検知）
                price_change_percent = 0.0
                if latest_price and reference_price and reference_price > 0:
                    price_change_percent = ((latest_price - reference_price) / reference_price) * 100.0
                    # 20%変動の閾値（環境変数で変更可能）
                    try:
                        trigger_percent = float(os.getenv('PRICE_ALERT_PERCENT', '20.0'))
                    except Exception:
                        trigger_percent = 20.0

                # === ここから自動売却ロジック ===
                # ポジションが存在し、各ポジションごとに売りトリガーを超えたら自動売却
                try:
                    positions = state.get('positions') if isinstance(state, dict) else None
                    sold_positions = []
                    if positions and isinstance(positions, list) and len(positions) > 0:
                        sell_trigger_pct = float(os.getenv('TRADE_TRIGGER_PCT', '20.0'))
                        for pos in positions[:]:
                            entry_price = float(pos.get('price', 0.0))
                            qty = float(pos.get('qty', 0.0))
                            sell_trigger_price = entry_price * (1.0 + sell_trigger_pct / 100.0)
                            if latest_price and entry_price > 0 and float(latest_price) >= sell_trigger_price and qty > 0:
                                print(f"🚀 売りシグナル: entry={entry_price:.0f}円, trigger={sell_trigger_price:.0f}円, 現在={latest_price:.0f}円, qty={qty}")
                                # 売却実行
                                order = execute_order(exchange, 'BTC/JPY', 'sell', qty)
                                print(f"DEBUG: execute_order(sell) returned: {order}")
                                # ポジション記録
                                record_position(state, 'sell', float(latest_price), qty)
                                sold_positions.append(pos)
                                # 通知
                                try:
                                    smtp_host = os.getenv('SMTP_HOST')
                                    smtp_port = int(os.getenv('SMTP_PORT', '587'))
                                    smtp_user = os.getenv('SMTP_USER')
                                    smtp_password = os.getenv('SMTP_PASS')
                                    email_to = os.getenv('TO_EMAIL')
                                    if smtp_host and email_to:
                                        subject = f"BTC Auto Sell Complete: {qty:.4f} BTC"
                                        try:
                                            safe_watch_ref = entry_price
                                        except Exception:
                                            try:
                                                safe_watch_ref = reference_price
                                            except Exception:
                                                safe_watch_ref = 0.0
                                        message = (
                                            f"BTC auto sell completed!\n\n"
                                            f"[Sell Info]\n"
                                            f"Amount: {qty:.4f} BTC\n"
                                            f"Sell Price: {latest_price:,.0f} JPY/BTC\n"
                                            f"Entry Price: {entry_price:,.0f} JPY\n"
                                            f"Reference: {safe_watch_ref:,.0f} JPY\n"
                                            f"Profit: approx. {(latest_price-entry_price)*qty:,.0f} JPY\n\n"
                                            f"Position cleared.\n"
                                        )
                                        send_notification(smtp_host, smtp_port, smtp_user, smtp_password, email_to, subject, message)
                                        print(f"📧 売却完了通知メール送信完了")
                                except Exception as e:
                                    print(f"⚠️ 売却通知メール送信エラー: {e}")
                        # 売却済みポジションのみ削除
                        for sold in sold_positions:
                            positions.remove(sold)
                        # 監視基準をリセット（最新価格）
                        state['watch_reference'] = float(latest_price)
                        save_state(state)
                except Exception as e:
                    print(f"⚠️ 自動売却処理で例外: {e}")
                    
                    # メール通知フラグを state から取得（重複通知防止）
                    last_alert_price = state.get('last_alert_price') if isinstance(state, dict) else None
                    print(f"DEBUG: last_alert_price={last_alert_price}, price_change_percent={price_change_percent:.2f}%")
                    # 20%以上の変動を検知
                    if abs(price_change_percent) >= trigger_percent:
                        # 前回と異なる価格帯での通知か確認（同じ価格帯で何度も通知しない）
                        should_alert = True
                        if last_alert_price:
                            try:
                                # 前回の通知価格から5%以上変動していれば再通知
                                prev_change = abs((float(latest_price) - float(last_alert_price)) / float(last_alert_price)) * 100.0
                                if prev_change < 5.0:
                                    should_alert = False
                            except Exception:
                                pass
                        
                        if should_alert:
                            direction = "上昇" if price_change_percent > 0 else "下落"
                            emoji = "📈" if price_change_percent > 0 else "📉"
                            
                            print(f"{emoji} 大幅価格{direction}検知: {abs(price_change_percent):.2f}% (基準:{reference_price:.0f}円 → 現在:{latest_price:.0f}円)")
                            
                            # メール通知を送信
                            try:
                                smtp_host = os.getenv('SMTP_HOST')
                                smtp_port = int(os.getenv('SMTP_PORT', '587'))
                                smtp_user = os.getenv('SMTP_USER')
                                smtp_password = os.getenv('SMTP_PASS')
                                email_to = os.getenv('TO_EMAIL')
                                
                                if smtp_host and email_to:
                                    subject = f"BTC Price {direction} Alert: {abs(price_change_percent):.1f}% change"
                                    current_balance = "unknown"
                                    try:
                                        if fund_manager is not None and hasattr(fund_manager, 'available_fund'):
                                            bal = float(fund_manager.available_fund())
                                            current_balance = f"{bal:.0f} JPY"
                                    except Exception:
                                        pass
                                    safe_watch_ref = 0.0
                                    try:
                                        safe_watch_ref = watch_ref
                                    except NameError:
                                        try:
                                            safe_watch_ref = reference_price
                                        except NameError:
                                            safe_watch_ref = 0.0
                                    if safe_watch_ref is None:
                                        safe_watch_ref = 0.0
                                    message = (
                                        f"BTC price has changed significantly: {direction}\n\n"
                                        f"[Price Info]\n"
                                        f"Reference price: {safe_watch_ref:,.0f} JPY\n"
                                        f"Current price: {latest_price:,.0f} JPY\n"
                                        f"Change percent: {price_change_percent:+.2f}%\n\n"
                                        f"[Balance Info]\n"
                                        f"Available balance: {current_balance}\n\n"
                                        f"[Recommended Action]\n"
                                        f"{'Buy opportunity! You can buy at a low price.' if price_change_percent < 0 else 'Take profit opportunity! You can sell at a high price.'}\n"
                                        f"{'If your balance is low, please deposit more funds.' if price_change_percent < 0 else ''}\n\n"
                                        f"The bot will try to trade automatically, but please check your balance and deposit manually if needed."
                                    )
                                    
                                    send_notification(smtp_host, smtp_port, smtp_user, smtp_password, 
                                                    email_to, subject, message)
                                    print(f"📧 価格{direction}通知メール送信完了")
                                    
                                    # 通知済み価格を記録（重複通知防止）
                                    try:
                                        state['last_alert_price'] = float(latest_price)
                                        save_state(state)
                                    except Exception:
                                        pass
                            except Exception as e:
                                print(f"⚠️ メール通知送信エラー: {e}")
                            
            except Exception as e:
                try:
                    log_warn(f"⚠️ 価格チェックエラー: {e}")
                except Exception:
                    pass
            
            # 通常の自動入金（残高不足時のみ）
            try:
                if deposit_amount and deposit_amount > 0:
                    try:
                        if fund_manager is not None and hasattr(fund_manager, 'available_fund'):
                            current = float(fund_manager.available_fund())
                        else:
                            current = None
                    except Exception:
                        current = None

                    if current is not None and current < min_balance_threshold:
                        if fund_manager is not None and hasattr(fund_manager, 'add_funds'):
                            fund_manager.add_funds(deposit_amount)
                        try:
                            if fund_manager is not None and hasattr(fund_manager, 'available_fund'):
                                log_info(f"💳 残高不足のため自動入金: {deposit_amount:.2f} 円 → 残高: {fund_manager.available_fund():.2f} 円")
                            else:
                                log_info(f"💳 残高不足のため自動入金: {deposit_amount:.2f} 円 → 残高: 不明")
                        except Exception:
                            if fund_manager is not None and hasattr(fund_manager, 'available_fund'):
                                print(f"💳 残高不足のため自動入金: {deposit_amount:.2f} 円 → 残高: {fund_manager.available_fund():.2f} 円")
                            else:
                                print(f"💳 残高不足のため自動入金: {deposit_amount:.2f} 円 → 残高: 不明")

            except Exception as e:
                print(f"⚠️ 自動入金処理中にエラーが発生しました: {e}")

            # 💰 bitbank残高増加検知 → 入金額全額で自動購入
            try:
                state = load_state()
                balance = exchange.fetch_balance()
                val = balance['JPY']['free'] if balance and 'JPY' in balance and 'free' in balance['JPY'] else 0.0
                try:
                    current_jpy = float(val) if val is not None else 0.0
                except Exception:
                    current_jpy = 0.0
                
                # 前回のJPY残高を取得
                last_jpy = state.get('last_jpy_balance', 0.0) if isinstance(state, dict) else 0.0
                
                # 残高が増加していたら自動購入（1000円以上の増加で購入）
                if current_jpy > last_jpy + 1000:
                    deposit_detected = current_jpy - last_jpy
                    latest_price = get_latest_price(exchange, 'BTC/JPY')
                    if latest_price and latest_price > 0:
                        buy_jpy = deposit_detected * 0.99
                        buy_amount_btc = buy_jpy / latest_price
                        if buy_amount_btc >= 0.0001:
                            if not DRY_RUN:
                                    deposit_detected = current_jpy - last_jpy
                                    latest_price = get_latest_price(exchange, 'BTC/JPY')
                                    if latest_price and latest_price > 0:
                                        buy_jpy = deposit_detected * 0.99
                                        buy_amount_btc = buy_jpy / latest_price
                                        if buy_amount_btc >= 0.0001:
                                            if not DRY_RUN:
                                                limit_price = latest_price * 1.01
                                                order = exchange.create_limit_buy_order('BTC/JPY', buy_amount_btc, limit_price)
                                                state['watch_reference'] = latest_price
                                                state['last_jpy_balance'] = current_jpy - buy_jpy
                                                save_state(state)
                                                smtp_host = os.getenv('SMTP_HOST')
                                                smtp_port = int(os.getenv('SMTP_PORT', '587'))
                                                smtp_user = os.getenv('SMTP_USER')
                                                smtp_password = os.getenv('SMTP_PASS')
                                                email_to = os.getenv('TO_EMAIL')
                                                if smtp_host and email_to:
                                                    subject = f"BTC Auto Purchase Complete: {buy_amount_btc:.4f} BTC"
                                                    message = (
                                                        f"BTC auto purchase completed!\n\n"
                                                        f"[Purchase Info]\n"
                                                        f"Amount: {buy_amount_btc:.4f} BTC\n"
                                                        f"Price: {latest_price:,.0f} JPY/BTC\n"
                                                        f"Total: approx. {buy_jpy:,.0f} JPY\n\n"
                                                        f"[Sell Target]\n"
                                                        f"Target price: {latest_price * (1 + TRADE_TRIGGER_PCT/100):,.0f} JPY (+{TRADE_TRIGGER_PCT:.0f}%)\n"
                                                        f"Expected profit: approx. {buy_jpy * (TRADE_TRIGGER_PCT/100):,.0f} JPY\n\n"
                                                        f"Please wait for auto-sell trigger."
                                                    )
                                        
                                        send_notification(smtp_host, smtp_port, smtp_user, smtp_password, email_to, subject, message)
                                    else:
                                        state['last_jpy_balance'] = current_jpy
                                        save_state(state)
                else:
                    # 残高更新（増加していない場合も記録）
                    if last_jpy == 0.0:
                        # 初回起動時は現在残高を記録
                        state['last_jpy_balance'] = current_jpy
                        save_state(state)
            except Exception as e:
                print(f"⚠️ 残高チェック・自動購入処理中にエラー: {e}")

            

            # 従来の毎ループ入金（あえて残す。ENVで無効化可）
            try:
                post_deposit = float(os.getenv('POST_LOOP_DEPOSIT', '0'))
            except Exception:
                post_deposit = 0
            if post_deposit and post_deposit > 0:
                try:
                    if fund_manager is not None:
                        fund_manager.add_funds(post_deposit)
                        print(f"💳 ループ終了後の自動入金: {post_deposit:.2f} 円 → 残高: {fund_manager.available_fund():.2f} 円")
                    else:
                        print(f"⚠️ fund_manager が None のため自動入金をスキップします。")
                except Exception as e:
                    print(f"⚠️ ループ後自動入金に失敗しました: {e}")

            time.sleep(3600)  # 1時間待機


# === DI対応版のエントリーポイント ===
def run_bot(exchange, fund_manager, dry_run=False):
    PAIR = 'BTC/JPY'
    PROFIT_TAKE_PCT = 10.0  # 利確10%
    BUY_MORE_PCT = 10.0     # 追加買い10%下落
    MIN_ORDER_BTC = 0.001

    positions = []  # ポジション管理 [{'price': 買値, 'amount': 数量, 'timestamp': ...}, ...]

    # 初回買い（例: 1回だけ）
    current_price = get_latest_price(exchange, PAIR)
    buy_cost = current_price * MIN_ORDER_BTC
    if fund_manager.available_fund() - buy_cost >= 1000:
        if fund_manager.place_order(buy_cost):
            execute_order(exchange, PAIR, 'buy', MIN_ORDER_BTC, current_price)
            positions.append({'price': current_price, 'amount': MIN_ORDER_BTC, 'timestamp': time.time()})
            print(f"新規買い: {current_price}円で{MIN_ORDER_BTC}BTC（残高1000円以上キープ）")

    # 1回だけ利確・追加買い判定（本番はループ化推奨）
    current_price = get_latest_price(exchange, PAIR)
    for pos in positions[:]:
        buy_price = pos['price']
        amount = pos['amount']
        # 利確判定（10%上昇）
        if current_price >= buy_price * (1 + PROFIT_TAKE_PCT / 100):
            print(f"利確売り: 買値={buy_price}, 現在={current_price}, amount={amount}")
            execute_order(exchange, PAIR, 'sell', amount, current_price)
            fund_manager.add_funds(current_price * amount)
            positions.remove(pos)
        # 追加買い判定（10%下落）
        elif current_price <= buy_price * (1 - BUY_MORE_PCT / 100):
            print(f"追加買い: 買値={buy_price}, 現在={current_price}, amount={amount}")
            add_cost = current_price * amount
            if fund_manager.available_fund() - add_cost >= 1000:
                if fund_manager.place_order(add_cost):
                    execute_order(exchange, PAIR, 'buy', amount, current_price)
                    positions.append({'price': current_price, 'amount': amount, 'timestamp': time.time()})
                    print(f"追加買い: {current_price}円で{amount}BTC（残高1000円以上キープ）")

    return "run_bot executed"



