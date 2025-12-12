<<<<<<< HEAD
# python 3.14環境で動作確認済み (仮想環境venv314を使用)
# === 必要なライブラリを1回ずつインポート（心臓部の準備） ===

# ccxt がインストールされていない環境でもファイルが読み込めるよう、フォールバックのスタブを用意します。
try:
    import ccxt  # type: ignore
except Exception:
    # 最低限のインターフェースを持つスタブ実装
    class AuthenticationError(Exception):
        pass

    class BitbankStub:
        def __init__(self, config=None):
            self.apiKey = (config or {}).get('apiKey')
            self.secret = (config or {}).get('secret')
        # BitbankStub: テスト/フォールバック用の最小限インターフェース実装
        # このクラスはファイル冒頭での ccxt フォールバック用に使われます。
        # 注意: ここでは副作用を最小にするため簡素な実装に留めます。
        def fetch_balance(self):
            # テスト用のダミー残高
            return {'total': {'JPY': 0.0, 'BTC': 0.0}}

        def fetch_ticker(self, pair):
            # テスト用のダミー価格（0.0 で返すことで呼び出し側が安全に扱える）
            return {'last': 0.0}

        def fetch_ohlcv(self, pair, timeframe='1h', limit=250):
            # 空のデータを返して呼び出し側で安全に扱えるようにする
            return []

        def create_order(self, pair, type_, side, amount, price=None):
            # ダミー注文レスポンスを返す（cost は計算できる場合のみ設定）
            cost = None
            try:
                p = float(price) if price is not None else 0.0
                cost = float(amount) * p
            except Exception:
                cost = None
            return {'id': 'stub_order', 'pair': pair, 'type': type_, 'side': side, 'amount': amount, 'price': price, 'cost': cost}

    class _CCXTModule:
        AuthenticationError = AuthenticationError
        def bitbank(self, config=None):
            return BitbankStub(config)

    ccxt = _CCXTModule()

# 後続コードが使うために名前を揃える
# ccxt を直接参照する代わりに、このモジュール内で使う共通の例外参照を作成します。
# 実環境では ccxt.AuthenticationError が存在します。スタブ環境では上で定義したものが入ります。
AuthenticationError = getattr(ccxt, 'AuthenticationError', Exception)

# 互換性対策: 一部のコードやライブラリは ccxt.base.errors.AuthenticationError を参照する
# ことがあるため、ccxt.base.errors.AuthenticationError が存在しない場合は補完しておきます。
try:
    base_obj = getattr(ccxt, 'base', None)
    if base_obj is None or not hasattr(base_obj, 'errors'):
        class _BaseErrors:
            pass
        setattr(_BaseErrors, 'AuthenticationError', getattr(ccxt, 'AuthenticationError', Exception))

        class _Base:
            pass
        setattr(_Base, 'errors', _BaseErrors)
        setattr(ccxt, 'base', _Base)
except Exception:
    # 保険: 何か問題があれば無視して既存の AuthenticationError を使う
    pass

# typing の Optional を使う箇所があるため明示的にインポートしておく
from typing import Optional
import json
from pathlib import Path
import sys

# Replace built-in print with a safe wrapper to avoid UnicodeEncodeError on Windows consoles
import builtins
_orig_print = builtins.print
def _safe_print(*args, **kwargs):
    try:
        _orig_print(*args, **kwargs)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, 'encoding', None) or 'utf-8'
        new_args = []
        for a in args:
            s = str(a)
            try:
                s.encode(enc)
            except UnicodeEncodeError:
                s = s.encode(enc, errors='replace').decode(enc)
            new_args.append(s)
        _orig_print(*new_args, **kwargs)
    except Exception:
        try:
            _orig_print(*[str(a) for a in args], **kwargs)
        except Exception:
            pass

builtins.print = _safe_print

# Try to reconfigure stdout to UTF-8 to avoid encoding errors on Windows consoles
try:
    if hasattr(sys, 'stdout') and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# funds モジュールが存在しても、外部の FundManager がこのスクリプトの期待する
# メソッド (available_fund, place_order, add_funds) を持たない場合があるため、
# 互換性チェックをして不一致なら内部スタブを使うようにします。
def _make_internal_fund_manager_class():
    # Minimal internal FundManager class used when external `funds` module is unavailable.
    class FundManagerStub:
        """Simple persistent fund manager for DRY_RUN / tests.

        - Persists a JSON file with {"available": <float>, "reserved": <float>} when state_file is provided.
        - Provides methods: available_fund(), place_order(cost), add_funds(amount), reserve(cost), confirm(cost), release(cost).
        """
        def __init__(self, initial_fund: float = 0.0, state_file: Optional[str] = None):
            import threading
            self._lock = threading.Lock()
            self._state_file = Path(state_file) if state_file else None
            self._available = float(initial_fund or 0.0)
            self._reserved = 0.0
            # load persisted state if present
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
            with self._lock:
                try:
                    return float(self._available)
                except Exception:
                    return 0.0

        def place_order(self, cost: float) -> bool:
            """Legacy immediate-deduct behavior: consume available balance if enough."""
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

        # Reservation-style API
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
                # consumed reserved amount
                self._reserved = max(0.0, float(self._reserved) - c)
                # no change to available (already deducted at reservation or by place_order)
                self._persist()

        def release(self, cost: float) -> None:
            try:
                c = float(cost)
            except Exception:
                return
            with self._lock:
                # move from reserved back to available
                self._reserved = max(0.0, float(self._reserved) - c)
                self._available = float(self._available) + c
                self._persist()

    return FundManagerStub


_InternalFundManager = _make_internal_fund_manager_class()

try:
    from funds import FundManager as _ImportedFundManager  # type: ignore
    # 簡易的な互換性チェック: クラスに想定するメソッドが存在するか
    required = ('available_fund', 'place_order', 'add_funds')
    if all(hasattr(_ImportedFundManager, name) for name in required):
        FundManager = _ImportedFundManager
    else:
        # 互換性なし → 内部スタブを利用
        FundManager = _InternalFundManager
except Exception:
    # インポート失敗時は内部スタブを利用
    FundManager = _InternalFundManager

import os

fund_manager = FundManager(initial_fund=0.0, state_file=os.getenv('FUND_STATE_FILE', 'funds_state.json'))


class FundAdapter:
    """Module-level FundAdapter used when adapting external FundManager instances.

    Provides reserve/confirm/release semantics and a local dry-run fallback.
    """
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
            return float(self._local_total - self._local_used)

    def reserve(self, cost: float) -> bool:
        c = float(cost or 0.0)
        if c <= 0:
            return False
        if self._fund is not None and not self._dry_run:
            try:
                if hasattr(self._fund, 'reserve'):
                    return bool(self._fund.reserve(c))
                if hasattr(self._fund, 'place_order'):
                    return bool(self._fund.place_order(c))
            except Exception:
                return False
        with self._lock:
            if (self._local_total - self._local_used) < c:
                return False
            self._local_used += c
            return True

    def confirm(self, cost: float) -> None:
        c = float(cost or 0.0)
        if c <= 0:
            return
        if self._fund is not None and not self._dry_run:
            try:
                if hasattr(self._fund, 'confirm'):
                    return self._fund.confirm(c)
                return
            except Exception:
                return
        with self._lock:
            self._local_total = max(0.0, self._local_total - c)
            self._local_used = max(0.0, self._local_used - c)

    def release(self, cost: float) -> None:
        c = float(cost or 0.0)
        if c <= 0:
            return
        if self._fund is not None and not self._dry_run:
            try:
                if hasattr(self._fund, 'release'):
                    return self._fund.release(c)
                print("⚠️ FundManager has no reservation-cancel API; manual correction may be required.")
                return
            except Exception:
                return
        with self._lock:
            self._local_used = max(0.0, self._local_used - c)

    def add_funds(self, amount: float) -> None:
        try:
            a = float(amount)
        except Exception:
            return
        if self._fund is not None and not self._dry_run and hasattr(self._fund, 'add_funds'):
            try:
                return self._fund.add_funds(a)
            except Exception:
                pass
        with self._lock:
            self._local_total = float(self._local_total) + a

def _adapt_fund_manager_instance(fm):
    """
    Wrap an external FundManager instance so it exposes the small API
    this script expects: available_fund(), place_order(cost), add_funds(amount).
    If the passed object already has these methods, return it unchanged.
    Otherwise return an adapter that attempts to call the underlying
    implementation when possible and falls back to an internal counter.
    """
    # Build an adapter that exposes both the old API (available_fund, place_order, add_funds)
    # and the new reservation API (reserve, confirm, release). This keeps backward
    # compatibility while allowing run_bot to use reserve/confirm/release semantics.
    try:
        dry_run_env = str(os.getenv('DRY_RUN', '')).lower() in ('1', 'true', 'yes', 'on')
    except Exception:
        dry_run_env = False

    # If the passed object already supports reserve/confirm/release, return it as-is
    if fm is not None and all(hasattr(fm, name) for name in ('reserve', 'confirm', 'release', 'available_fund')):
        return fm

    class FundAdapter:
        """Adapter providing reservation-style API and a local fallback for dry-run.

        This is a small, self-contained adapter used when an external FundManager
        does not implement reserve/confirm/release. It keeps local counters when
        dry_run is True or when no underlying manager is provided.
        """
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
                return float(self._local_total - self._local_used)

        def reserve(self, cost: float) -> bool:
            c = float(cost or 0.0)
            if c <= 0:
                return False
            # Delegate to underlying if possible and not dry-run
            if self._fund is not None and not self._dry_run:
                try:
                    # prefer reserve() if provided
                    if hasattr(self._fund, 'reserve'):
                        return bool(self._fund.reserve(c))
                    # fall back to place_order (immediate deduct)
                    if hasattr(self._fund, 'place_order'):
                        return bool(self._fund.place_order(c))
                except Exception:
                    return False
            # simulate local reservation
            with self._lock:
                if (self._local_total - self._local_used) < c:
                    return False
                self._local_used += c
                return True

        def confirm(self, cost: float) -> None:
            c = float(cost or 0.0)
            if c <= 0:
                return
            if self._fund is not None and not self._dry_run:
                try:
                    if hasattr(self._fund, 'confirm'):
                        return self._fund.confirm(c)
                    # if only place_order existed, it's already consumed
                    return
                except Exception:
                    return
            with self._lock:
                # consume reserved funds
                self._local_total = max(0.0, self._local_total - c)
                self._local_used = max(0.0, self._local_used - c)

        def release(self, cost: float) -> None:
            c = float(cost or 0.0)
            if c <= 0:
                return
            if self._fund is not None and not self._dry_run:
                try:
                    if hasattr(self._fund, 'release'):
                        return self._fund.release(c)
                    # no generic undo for place_order
                    print("⚠️ FundManager has no reservation-cancel API; manual correction may be required.")
                    return
                except Exception:
                    return
            with self._lock:
                self._local_used = max(0.0, self._local_used - c)

        def add_funds(self, amount: float) -> None:
            try:
                a = float(amount)
            except Exception:
                return
            if self._fund is not None and not self._dry_run and hasattr(self._fund, 'add_funds'):
                try:
                    return self._fund.add_funds(a)
                except Exception:
                    pass
            with self._lock:
                self._local_total = float(self._local_total) + a

    class _Adapter:
        def __init__(self, underlying, initial=0.0, dry_run=False):
            self._underlying = underlying
            self._fund_adapter = FundAdapter(fund_manager=underlying if underlying is not None else None,
                                             initial_fund=initial, dry_run=dry_run)

        # New reservation-style API
        def reserve(self, cost: float) -> bool:
            return self._fund_adapter.reserve(cost)

        def confirm(self, cost: float) -> None:
            return self._fund_adapter.confirm(cost)

        def release(self, cost: float) -> None:
            return self._fund_adapter.release(cost)

        # Backward-compatible legacy API
        def available_fund(self):
            return self._fund_adapter.available_fund()

        def place_order(self, cost):
            """Legacy behavior: attempt to delegate to underlying.place_order if available.
            Otherwise behave as reserve+confirm (consume immediately)."""
            try:
                if self._underlying is not None and hasattr(self._underlying, 'place_order'):
                    res = self._underlying.place_order(cost)
                    # If underlying returns a boolean, respect it
                    if isinstance(res, bool):
                        return res
                    # If underlying returns None/other, assume success if available decreased
                    return True
            except Exception:
                pass
            # fallback: reserve then confirm (atomic from caller's perspective)
            ok = self._fund_adapter.reserve(cost)
            if ok:
                self._fund_adapter.confirm(cost)
                return True
            return False

        def add_funds(self, amount):
            try:
                if self._underlying is not None and hasattr(self._underlying, 'add_funds'):
                    return self._underlying.add_funds(amount)
            except Exception:
                pass
            # fallback to fund_adapter local top-up
            try:
                with self._fund_adapter._lock:
                    self._fund_adapter._local_total = float(self._fund_adapter._local_total) + float(amount)
            except Exception:
                pass

    # Try to seed the adapter with the underlying available balance when possible
    init_bal = 0.0
    try:
        if fm is not None and hasattr(fm, 'available_fund'):
            try:
                init_bal = float(fm.available_fund())
            except Exception:
                init_bal = float(getattr(fm, '_available', 0.0))
        else:
            init_bal = float(getattr(fm, '_available', 0.0) if fm is not None else 0.0)
    except Exception:
        init_bal = 0.0

    return _Adapter(fm, initial=init_bal, dry_run=dry_run_env)

import os
import time
import datetime
import math
import sys
try:
    from filelock import FileLock
except Exception:
    # Minimal FileLock fallback using fcntl for Unix-like systems
    try:
        import fcntl

        class FileLock:
            def __init__(self, path, timeout=None):
                self.path = path
                self.timeout = timeout
                self.fd = None

            def __enter__(self):
                start = time.time()
                while True:
                    try:
                        # open file for writing (create if not exists)
                        self.fd = open(self.path, 'w')
                        # try to acquire non-blocking exclusive lock
                        fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        return self
                    except BlockingIOError:
                        # timeout handling
                        if self.timeout is not None and (time.time() - start) >= float(self.timeout):
                            raise TimeoutError(f"Timeout acquiring lock {self.path}")
                        time.sleep(0.05)

            def __exit__(self, exc_type, exc, tb):
                try:
                    if self.fd:
                        fcntl.flock(self.fd, fcntl.LOCK_UN)
                        self.fd.close()
                except Exception:
                    pass

    except Exception:
        # Last-resort dummy lock (no-op) for non-Unix platforms; safer to install filelock package
        class FileLock:
            def __init__(self, path, timeout=None):
                self.path = path

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return
try:
    import pandas as pd  # type: ignore
except Exception:
    # Minimal pandas-like stub to avoid import errors and provide the small API used in this script.
    # NOTE: This is a lightweight compatibility shim for parsing/testing and does NOT replace real pandas.
    class Series:
        def __init__(self, values):
            self.values = list(values) if values is not None else []
            self._window = None

        def rolling(self, window):
            self._window = int(window)
            return self

        def mean(self):
            vals = self.values
            w = self._window or 1
            if not vals:
                return []
            res = []
            for i in range(len(vals)):
                if i + 1 < w:
                    res.append(None)
                else:
                    window_vals = [v for v in vals[i + 1 - w:i + 1] if v is not None]
                    res.append(sum(window_vals) / len(window_vals) if window_vals else None)
            return res

        def __iter__(self):
            return iter(self.values)

    class Row:
        def __init__(self, data):
            self._data = data or {}

        def __getitem__(self, key):
            return self._data.get(key)

        def __getattr__(self, name):
            if name in self._data:
                return self._data[name]
            raise AttributeError(name)

    class DataFrame:
        def __init__(self, data=None, columns=None):
            # data: list of lists (rows) or list of dicts
            self._columns = list(columns) if columns else []
            self._rows = []
            if data:
                if self._columns and all(isinstance(r, (list, tuple)) for r in data):
                    for row in data:
                        self._rows.append({c: v for c, v in zip(self._columns, row)})
                elif all(isinstance(r, dict) for r in data):
                    self._rows = [dict(r) for r in data]
                    if not self._columns:
                        cols = set()
                        for r in self._rows:
                            cols.update(r.keys())
                        self._columns = list(cols)
                else:
                    # fallback: single column
                    col = self._columns[0] if self._columns else "data"
                    for r in data:
                        self._rows.append({col: r})
            self.index = None

        def __len__(self):
            return len(self._rows)

        def __getitem__(self, key):
            if isinstance(key, str):
                vals = [row.get(key) for row in self._rows]
                return Series(vals)
            raise KeyError(key)

        def __setitem__(self, key, value):
            # value can be Series or iterable; align by index
            vals = list(value) if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)) else [value] * len(self._rows)
            if not self._rows and vals:
                for v in vals:
                    self._rows.append({key: v})
            else:
                for i, v in enumerate(vals):
                    if i < len(self._rows):
                        self._rows[i][key] = v
                    else:
                        self._rows.append({key: v})
            if key not in self._columns:
                self._columns.append(key)

        @property
        def iloc(self):
            class _Loc:
                def __init__(self, rows):
                    self._rows = rows

                def __getitem__(self, idx):
                    return Row(self._rows[idx])
            return _Loc(self._rows)

        def set_index(self, key):
            self.index = key
            return self

    def to_datetime(values, unit='ms'):
        out = []
        for v in values:
            try:
                if v is None:
                    out.append(None)
                    continue
                if unit == 'ms':
                    ts = float(v) / 1000.0
                else:
                    ts = float(v)
                out.append(datetime.datetime.fromtimestamp(ts))
            except Exception:
                out.append(None)
        return out

    import types
    pd = types.SimpleNamespace(DataFrame=DataFrame, to_datetime=to_datetime)

from zoneinfo import ZoneInfo  # 標準ライブラリのタイムゾーン処理
#import time
import json
from pathlib import Path
import math
import os

# Safety & strategy environment variables (defaults)
MAX_SLIPPAGE_PCT = float(os.environ.get("MAX_SLIPPAGE_PCT", "0.5"))  # percent
ORDER_EXECUTION_WINDOW_SEC = int(os.environ.get("ORDER_EXECUTION_WINDOW_SEC", "30"))
COOLDOWN_SEC = int(os.environ.get("COOLDOWN_SEC", "3600"))  # 秒
TAKE_PROFIT_PCT = float(os.environ.get("TAKE_PROFIT_PCT", "5.0"))  # percent
MAX_ORDER_JPY = int(os.environ.get("MAX_ORDER_JPY", "2000"))  # 1注文上限（JPY）
STATE_FILE = Path(os.environ.get("TRADING_STATE_FILE", "bot_state.json"))
# 売買トリガー（%）。ユーザー指定が無ければ 20% を使う。
TRADE_TRIGGER_PCT = float(os.environ.get('TRADE_TRIGGER_PCT', '20.0'))
# 新規: 注文を行うための市場価格閾値（JPY）。この値未満なら発注を行わない。
MIN_PRICE_THRESHOLD_JPY = float(os.environ.get("MIN_PRICE_THRESHOLD_JPY", "12000000"))
USE_DYNAMIC_THRESHOLD = str(os.environ.get('USE_DYNAMIC_THRESHOLD', '1')).lower() in ('1', 'true', 'yes', 'on')
# 動的閾値用の設定: 過去何日分を参照するか、バッファを固定円 or 割合で指定
DYN_OHLCV_DAYS = int(os.environ.get('DYN_OHLCV_DAYS', '365'))
DYN_THRESHOLD_BUFFER_JPY = float(os.environ.get('DYN_THRESHOLD_BUFFER_JPY', '20000'))
DYN_THRESHOLD_BUFFER_PCT = float(os.environ.get('DYN_THRESHOLD_BUFFER_PCT', '0.01'))
DYN_THRESHOLD_RATIO = float(os.environ.get('DYN_THRESHOLD_RATIO', '0.15'))
# ブレイクアウト設定（環境変数で上書き可）
BUY_ON_BREAKOUT = str(os.environ.get('BUY_ON_BREAKOUT', '1')).lower() in ('1','true','yes','on')
BREAKOUT_PCT = float(os.environ.get('BREAKOUT_PCT', '0.01'))  # 例: 0.01 = 1%
BREAKOUT_SMA_SHORT = int(os.environ.get('BREAKOUT_SMA_SHORT', '30'))  # 日数
BREAKOUT_SMA_LONG = int(os.environ.get('BREAKOUT_SMA_LONG', '75'))  # 日数（比較用）
BREAKOUT_LOOKBACK_DAYS = int(os.environ.get('BREAKOUT_LOOKBACK_DAYS', '30'))  # 直近高値参照日数

try:
    from dotenv import load_dotenv  # type: ignore[reportMissingImports]
except Exception:
    # minimal stub for load_dotenv to allow .env loading when python-dotenv is not installed
    def load_dotenv(dotenv_path=None):
        """
        Very small implementation that reads KEY=VALUE lines from a file and sets os.environ entries.
        Returns True if a file was read, False otherwise.
        """
        if not dotenv_path or not os.path.exists(dotenv_path):
            return False
        try:
            with open(dotenv_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        k, v = line.split('=', 1)
                        k = k.strip()
                        v = v.strip()
                        # strip optional surrounding quotes
                        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                            v = v[1:-1]
                        # do not overwrite existing environment variables
                        os.environ.setdefault(k, v)
            return True
        except Exception:
            return False


# === .envファイルの読み込み（複数の場所を試す） ===
# systemd サービスでは EnvironmentFile が優先されるが、
# 手動実行時は .env ファイルを読み込む
env_paths = [
    '/home/ninitan/.secrets/.env',  # systemd で使用
    '.env',                         # プロジェクト直下
    'config.env'                    # 旧形式
]

env_loaded = False
def safe_print(s: str) -> None:
    """Print safely even when the console encoding can't represent some characters.

    Replaces unencodable characters with the platform replacement character.
    """
    try:
        print(s)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, 'encoding', None) or 'utf-8'
        try:
            safe = s.encode(enc, errors='replace').decode(enc)
        except Exception:
            # fallback: remove non-ascii
            safe = ''.join(ch if ord(ch) < 128 else '?' for ch in s)
        print(safe)
    except Exception:
        # last resort
        try:
            print(str(s))
        except Exception:
            pass
for env_path in env_paths:
    if load_dotenv(dotenv_path=env_path):
        safe_print(f"[OK] 環境変数を {env_path} から読み込みました")
        env_loaded = True
        break

if not env_loaded:
    safe_print("[WARN] .env ファイルが見つかりません。環境変数は systemd EnvironmentFile から読み込まれます。")

# グローバルに使う API キーを一度だけ読み込む
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

# 日本標準時 (JST) のタイムゾーンオブジェクトを作成
try:
    JST = ZoneInfo('Asia/Tokyo')
except Exception:
    # Windows 等で tzdata が無い環境では ZoneInfo が ZoneInfoNotFoundError を出すことがあるため
    # 安全に固定オフセットで JST を作成してフォールバックします（UTC+9）。
    JST = datetime.timezone(datetime.timedelta(hours=9))

# === 環境変数の取得（実行時チェック用にグローバル変数として定義） ===
smtp_user = os.getenv("SMTP_USER")
smtp_password = os.getenv("SMTP_PASS")  # .env.newに合わせてSMTP_PASSに変更
email_to = os.getenv("TO_EMAIL")  # .env.newに合わせてTO_EMAILに変更
smtp_host = os.getenv("SMTP_HOST")  # キー名を取得する変数を smtp_host に変更
# モジュールレベルでのチェックを削除（実行時にチェックするように変更）

subject = os.getenv("SUBJECT", "📬 通知")

# === SMTP_PORT の安全な読み込み ===
# デフォルトを 465 (SMTPS) にしておきます。環境変数があればそれを使い、整数変換に失敗したら 465 にフォールバックします。
port_str = os.getenv("SMTP_PORT", "465")
try:
    smtp_port = int(port_str)
except Exception:
    smtp_port = 465

# === メール送信関数 ===
def send_notification(smtp_host, smtp_port, smtp_user, smtp_password, to, subject, body):
    """安全化したメール送信ラッパー。
    - DRY_RUN のときは送信をスキップする。
    - SMTP ホスト/宛先が未設定のときは送信をスキップする。
    - 接続タイムアウトを短くしてブロックを避ける。
    Returns True on success, False otherwise.
    """
    from email.mime.text import MIMEText
    import smtplib
    import os

    # DRY_RUN のときは送信をスキップ (成功扱いにすることで通知ループを防ぐ)
    if str(os.getenv('DRY_RUN', '0')).lower() in ('1', 'true', 'yes', 'on'):
        print('ℹ️ DRY_RUN が有効のためメール送信をスキップします')
        return True

    # 必須情報が無ければ送信をスキップ
    if not smtp_host or not to:
        print('ℹ️ SMTP ホストまたは宛先が未設定のためメール送信をスキップします')
        return False

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = smtp_user or ''
    msg["To"] = to

    # SMTPS 判定（環境変数が無ければポート465を SMTPS と判断）
    use_ssl_env = os.getenv("SMTP_USE_SSL")
    if use_ssl_env is None:
        try:
            use_ssl = (int(smtp_port) == 465)
        except Exception:
            use_ssl = True
    else:
        use_ssl = str(use_ssl_env).lower() in ("1", "true", "yes", "on")

    # 接続タイムアウト（秒）を短めにする
    try:
        timeout_sec = float(os.getenv('SMTP_CONNECT_TIMEOUT', '10'))
    except Exception:
        timeout_sec = 10.0

    try:
        if use_ssl:
            with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=timeout_sec) as server:
                if smtp_user and smtp_password:
                    try:
                        server.login(smtp_user, smtp_password)
                    except Exception as e:
                        print(f'⚠️ SMTP 認証失敗: {e}')
                server.send_message(msg)
        else:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=timeout_sec) as server:
                try:
                    server.starttls()
                except Exception:
                    # StartTLS が使えない環境でもログは残す
                    pass
                if smtp_user and smtp_password:
                    try:
                        server.login(smtp_user, smtp_password)
                    except Exception as e:
                        print(f'⚠️ SMTP 認証失敗: {e}')
                server.send_message(msg)

        print("✅ メール送信成功")
        return True

    except Exception as e:
        # 失敗しても Bot の実行を止めない（ログを残すだけ）
        print(f"❌ メール送信失敗: {e}")
        return False


# 取引所の設定を取得
exchange_name = os.getenv("EXCHANGE", "bitbank")


# === メイン処理開始（Botの心臓が動き出す） ===
if __name__ == "__main__":
    print("Bot起動中...")
    # run_botの定義後に呼び出すように移動しました

# 1. 初期設定と認証 (APIキーの読み込みはここにあります)

# .envファイルからAPIキーを読み込みます（config.envから統合済み）

 
# ※注意: APIキー読み込みと bitbank インスタンスの直接作成は
# connect_to_bitbank() に統合しました。元の直接作成コードを削除しています。
# 必要であれば、環境変数の確認は connect_to_bitbank() 呼び出し時に行われます。

# 旧来の直接接続テスト/監視ループは削除しました。
# 取引所接続とループは connect_to_bitbank() と run_bot() に統合されています。


# ==========================================================
# 🔑 2. グローバルキー読み込みと定義 (修正点: 最上部に移動)
# ==========================================================
#.env# config.envからAPIキーを読み込みます

load_dotenv(dotenv_path='.env') 
API_KEY = os.getenv("API_KEY") # グローバル定数として定義
SECRET_KEY = os.getenv("SECRET_KEY") # グローバル定数として定義

# 日本標準時 (JST) のタイムゾーンオブジェクトを作成
try:
    JST = ZoneInfo('Asia/Tokyo')
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
    """軽量な取引所スタブ: dry-run 用。実ネットワーク呼び出しを行いません。"""
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
        return []

    def create_order(self, pair, type_, side, amount, price=None):
        cost = None
        try:
            p = float(price) if price is not None else float(self._price)
            cost = float(amount) * p
        except Exception:
            cost = None
        return {'id': 'dry_order', 'pair': pair, 'type': type_, 'side': side, 'amount': amount, 'price': price, 'cost': cost}


def test_fund_adapter():
    """Quick smoke test for FundAdapter/_adapt_fund_manager_instance.
    Prints expected behavior for reserve/confirm/release in DRY_RUN and live modes.
    """
    print("--- FundAdapter smoke test ---")
    # live-like stub
    fm = FundManager(initial_fund=2000, state_file=os.getenv('FUND_STATE_FILE', 'funds_state.json'))
    adapter = _adapt_fund_manager_instance(fm)
    print("initial available (live stub):", adapter.available_fund())
    cost = 500
    ok = adapter.reserve(cost) if hasattr(adapter, 'reserve') else adapter.place_order(cost)
    print(f"reserve/place_order({cost}) ->", ok)
    print("available after reserve:", adapter.available_fund())
    # attempt release (refund)
    if hasattr(adapter, 'release'):
        adapter.release(cost)
        print("after release available:", adapter.available_fund())
    else:
        # fallback: add_funds used as refund
        adapter.add_funds(cost)
        print("after add_funds available:", adapter.available_fund())

    # Dry-run adapter
    try:
        da = FundAdapter(fund_manager=None, initial_fund=1000.0, dry_run=True)
        print("dry initial available:", da.available_fund())
        ok2 = da.reserve(300)
        print("dry reserve(300) ->", ok2, "available->", da.available_fund())
        da.confirm(300)
        print("dry confirm(300) -> available->", da.available_fund())
    except Exception as e:
        print("dry adapter test failed:", e)


# === 1. 取引所への接続 ===
# 修正点: グローバルキーを使用するため引数を削除し、冗長なコードを削除
def connect_to_bitbank():
    """bitbankに接続します。グローバルで読み込んだAPIキーを使用します。"""
    try:
        # dry-run が有効な場合はネットワークを使わないスタブを返す
        if str(os.getenv('DRY_RUN', '0')).lower() in ('1', 'true', 'yes', 'on'):
            price = os.getenv('DRY_RUN_PRICE')
            try:
                price = float(price) if price is not None else None
            except Exception:
                price = None
            print("🔧 DRY_RUN enabled — using ExchangeStub (no network calls).")
            return ExchangeStub(price)

        # API_KEYとSECRET_KEYはファイルの最上部で既に読み込まれている
        if not API_KEY or not SECRET_KEY:
            print("エラー：APIキーまたはシークレットキーが未定義です。config.envを確認してください。")
            return None

        # ccxtを使ってbitbankに接続
        exchange = ccxt.bitbank({
            'apiKey': API_KEY,
            'secret': SECRET_KEY,
        })
        print("✅ bitbankにccxtで認証接続しました。")
        return exchange

    except Exception as e:
        print(f"❌ bitbankへの接続中にエラーが発生しました: {e}")
        return None
     
        print("✅ bitbankにccxtで認証接続しました。")
        return exchange    

# === 2. 価格データの取得 ===
def get_ohlcv(exchange, pair='BTC/JPY', timeframe='1h', limit=250):
    """
    指定した通貨ペアのOHLCVデータを取得します。(ccxt使用)
    """
    try:
        ohlcv_data = exchange.fetch_ohlcv(pair, timeframe, limit=limit)

        if ohlcv_data:
            # データをDataFrameに変換
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            return df
        else:
            print(f"{pair} のOHLCVデータを取得できませんでした。")
            return None

    except Exception as e:
        print(f"OHLCVデータの取得中にエラーが発生しました: {e}")
        return None


def get_latest_price(exchange, pair='BTC/JPY', retries=3, backoff=1.0):
    """取引所から最新の価格 (last) を取得するユーティリティ。
    - DRY_RUN が有効な場合は環境変数 DRY_RUN_PRICE を返す。
    - ネットワークエラーが起きた場合はリトライ（指数バックオフ）する。
    - 成功時は float の価格、取得不可なら None を返す。
    """
    try:
        if str(os.getenv('DRY_RUN', '0')).lower() in ('1', 'true', 'yes', 'on'):
            try:
                return float(os.getenv('DRY_RUN_PRICE', str(DRY_RUN_PRICE)))
            except Exception:
                return float(DRY_RUN_PRICE)
    except Exception:
        pass

    attempt = 0
    while attempt < retries:
        try:
            if exchange is None:
                # ネット接続無しのテスト環境等では DRY_RUN_PRICE を返す
                try:
                    return float(os.getenv('DRY_RUN_PRICE', str(DRY_RUN_PRICE)))
                except Exception:
                    return float(DRY_RUN_PRICE)

            ticker = exchange.fetch_ticker(pair)
            if isinstance(ticker, dict) and 'last' in ticker:
                try:
                    return float(ticker.get('last'))
                except Exception:
                    return None
            # ccxt の一部実装はオブジェクトや異なる形で返す可能性がある
            # 呼び出し側で安全に扱える形で None を返す
            return None

        except Exception as e:
            attempt += 1
            print(f"⚠️ 価格取得失敗（試行 {attempt}/{retries}）: {e}")
            if attempt >= retries:
                break
            sleep_sec = backoff * (2 ** (attempt - 1))
            try:
                time.sleep(sleep_sec)
            except Exception:
                pass

    return None


def compute_dynamic_threshold(exchange, pair='BTC/JPY', days=DYN_OHLCV_DAYS,
                              buffer_jpy=DYN_THRESHOLD_BUFFER_JPY, buffer_pct=DYN_THRESHOLD_BUFFER_PCT):
    """過去 `days` 日の OHLCV を取得し、1年レンジの最安値を基に閾値を算出します。
    戻り値: (threshold_jpy, min_jpy, max_jpy) または (None, None, None) on failure
    算出方法:
      - 可能なら固定バッファ (buffer_jpy) を優先して閾値 = min_close + buffer_jpy
      - buffer_jpy が <= 0 のときは buffer_pct を使って閾値 = min_close * (1 + buffer_pct)
    """
    try:
        # get_ohlcv は timeframe='1d' で日次データを返す
        df = get_ohlcv(exchange, pair, timeframe='1d', limit=max(10, int(days)))
        if df is None or len(df) == 0:
            return None, None, None
        # close 列の最小・最大
        try:
            closes = [float(v) for v in df['close'] if v is not None]
        except Exception:
            # データフレーム互換処理
            closes = []
            for i in range(len(df)):
                try:
                    closes.append(float(df.iloc[i]['close']))
                except Exception:
                    pass
        if not closes:
            return None, None, None
        min_close = min(closes)
        max_close = max(closes)
        # 優先順位:
        # 1) ratio が指定されている（>0）なら、1年レンジの比率で閾値を決定
        # 2) buffer_jpy が指定されている (>0) なら最安値 + 固定円で閾値
        # 3) それ以外は割合バッファで閾値
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
        print(f"⚠️ dynamic threshold computation failed: {e}")
        return None, None, None


def compute_sma_from_ohlcv(exchange, pair='BTC/JPY', days=30):
    """日次 OHLCV から単純移動平均 (SMA) を計算して返す。失敗時は None を返す。"""
    try:
        df = get_ohlcv(exchange, pair, timeframe='1d', limit=max(10, days + 5))
        if df is None or len(df) < days:
            return None
        vals = []
        try:
            for i in range(-days, 0):
                vals.append(float(df['close'].iloc[i]))
            return sum(vals) / len(vals) if vals else None
        except Exception:
            # DataFrame 互換ループ
            vals = []
            for i in range(len(df) - days, len(df)):
                try:
                    vals.append(float(df.iloc[i]['close']))
                except Exception:
                    pass
            return sum(vals) / len(vals) if vals else None
    except Exception:
        return None


def get_recent_high(exchange, pair='BTC/JPY', days=30):
    """直近 days 日の high の最大値を返す。失敗時は None を返す。"""
    try:
        df = get_ohlcv(exchange, pair, timeframe='1d', limit=max(10, days + 5))
        if df is None or len(df) == 0:
            return None
        try:
            highs = [float(v) for v in df['high'] if v is not None]
            return max(highs) if highs else None
        except Exception:
            hs = []
            for i in range(len(df)):
                try:
                    hs.append(float(df.iloc[i]['high']))
                except Exception:
                    pass
            return max(hs) if hs else None
    except Exception:
        return None


# === 指標計算ユーティリティ (SMA/EMA/ATR/RSI) ===
def compute_sma(values, period):
    """Simple moving average for last `period` values. Returns None if insufficient data."""
    try:
        if values is None:
            return None
        vals = [float(v) for v in values if v is not None]
        if len(vals) < period or period <= 0:
            return None
        return sum(vals[-period:]) / float(period)
    except Exception:
        return None


def compute_ema(values, period):
    """Exponential moving average for last `period` values. Returns None if insufficient data."""
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
    """Compute ATR (Average True Range) from OHLCV rows (list of [ts, o, h, l, c, v] or DataFrame-like).
    Returns ATR float or None."""
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
    """Compute RSI from list of close prices. Returns float or None."""
    try:
        if values is None:
            return None
        vals = [float(v) for v in values if v is not None]
        if len(vals) < period + 1:
            return None
        gains = []
        losses = []
        for i in range(1, len(vals)):
            diff = vals[i] - vals[i - 1]
            if diff > 0:
                gains.append(diff)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(abs(diff))
        # use Wilder's smoothing (simple average for first)
        avg_gain = sum(gains[-period:]) / float(period)
        avg_loss = sum(losses[-period:]) / float(period)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi
    except Exception:
        return None


def compute_indicators(exchange, pair='BTC/JPY', timeframe='1h', limit=500):
    """Fetch OHLCV and compute a set of indicators. Returns dict of values (may contain None)."""
    try:
        ohlcv = get_ohlcv(exchange, pair, timeframe=timeframe, limit=limit)
        indicators = {
            'sma_short_50': None,
            'sma_long_200': None,
            'ema_12': None,
            'ema_26': None,
            'atr_14': None,
            'rsi_14': None,
            'recent_high_20': None,
            'latest_close': None,
        }

        # If get_ohlcv returned None, try to fallback to exchange.fetch_ohlcv directly
        raw = None
        if ohlcv is None:
            try:
                raw = exchange.fetch_ohlcv(pair, timeframe, limit=limit)
            except Exception:
                raw = None
        else:
            # convert DataFrame-like to list of rows
            try:
                closes = [float(v) for v in ohlcv['close'] if v is not None]
                highs = [float(v) for v in ohlcv['high'] if v is not None]
                lows = [float(v) for v in ohlcv['low'] if v is not None]
                raw = []
                for i in range(len(closes)):
                    # timestamp may be index
                    try:
                        ts = ohlcv.index[i]
                        raw.append([ts, ohlcv['open'].iloc[i], highs[i], lows[i], closes[i], ohlcv['volume'].iloc[i]])
                    except Exception:
                        raw.append([None, None, highs[i], lows[i], closes[i], None])
            except Exception:
                raw = None
        if raw is None:
            return indicators

        # prepare lists
        closes = [float(r[4]) for r in raw if r and len(r) >= 5 and r[4] is not None]
        highs = [float(r[2]) for r in raw if r and len(r) >= 3 and r[2] is not None]
        lows = [float(r[3]) for r in raw if r and len(r) >= 4 and r[3] is not None]

        indicators['latest_close'] = closes[-1] if closes else None
        indicators['sma_short_50'] = compute_sma(closes, 50)
        indicators['sma_long_200'] = compute_sma(closes, 200)
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
            'latest_close': None,
        }


def write_indicators_csv(indicators: dict, pair: str, signal: str = 'NONE', csv_path='indicators.csv'):
    """Append indicators as a CSV row. Creates header if file does not exist."""
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
def round_down_qty(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    factor = 1.0 / step
    return math.floor(qty * factor) / factor


def compute_qty_for_budget_with_fee(reserved_jpy: float, price_jpy: float,
                                    min_btc: float = 0.0001, step: float = 0.0001,
                                    available_jpy: float = None, balance_buffer: float = 0.0):
    """
    返り値: (qty, cost_jpy, fee_jpy)
    qty: 注文数量（step に合わせて丸め）。0 の場合は注文不可。
    cost_jpy: qty * price_jpy
    fee_jpy: cost_jpy * FEE_RATE + FEE_FIXED_JPY
    """
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
            alt_max_jpy = min(reserved_jpy * max_mult, (float(available_jpy) - float(balance_buffer)) if available_jpy is not None else reserved_jpy * max_mult)
            if alt_max_jpy > 0:
                approx_qty = alt_max_jpy / (price_jpy * (1.0 + fee_rate))
                alt_qty = round_down_qty(approx_qty, step)
                cost = alt_qty * price_jpy
                fee = cost * fee_rate + fee_fixed
                if alt_qty >= min_btc and (cost + fee) <= alt_max_jpy:
                    return alt_qty, cost, fee
        return 0.0, 0.0, 0.0

    # fee を計算して合計が上限を超えないか確認
    cost = qty * price_jpy
    fee = cost * fee_rate + fee_fixed
    if (cost + fee) <= max_allowed_jpy:
        return qty, cost, fee

    # 少しずつ qty を減らしてチェック
    while qty >= min_btc:
        cost = qty * price_jpy
        fee = cost * fee_rate + fee_fixed
        if (cost + fee) <= max_allowed_jpy:
            return qty, cost, fee
        qty = round_down_qty(qty - step, step)

    return 0.0, 0.0, 0.0


def is_trade_day(now=None):
    # now は timezone-aware datetime を想定
    if now is None:
        now = datetime.datetime.now()
    forced = os.getenv('FORCE_TRADE_DAY')
    if forced:
        allowed = [d.strip().lower() for d in forced.split(',') if d.strip()]
        dow = now.strftime('%a').lower()[:3]
        return dow in allowed
    # デフォルトで土日限定にする（環境変数で上書き可）
    if os.getenv('TRADE_ONLY_WEEKENDS', '1') == '1':
        return now.weekday() in (5, 6)
    return True


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
                print(f"DEBUG: save_state succeeded and replaced {STATE_FILE} (size={STATE_FILE.stat().st_size})")
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
                    print(f"DEBUG: save_state fallback non-atomic write succeeded for {STATE_FILE}")
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
            print("WARN: could not save state:", e)
            errfile = STATE_FILE.with_name(STATE_FILE.name + '.save_error.log')
            errfile.write_text(''.join(traceback.format_exception(type(e), e, e.__traceback__)), encoding='utf-8')
        except Exception:
            try:
                print("WARN: could not save state and failed to write error log")
            except Exception:
                pass


def get_last_buy_time(state):
    return state.get("last_buy_time")


def set_last_buy_time(state, ts=None):
    state["last_buy_time"] = ts or int(time.time())
    save_state(state)


def record_position(state, side, price, qty):
    try:
        # Use the same ORDER_LOCKFILE as other parts of the code to serialize
        # state updates (buy reservation, record_position, sell flow).
        lockfile = os.getenv('ORDER_LOCKFILE')
        if not lockfile:
            try:
                lockfile = str(STATE_FILE.with_name('.ninibo_order.lock'))
            except Exception:
                lockfile = os.path.join(os.getcwd(), '.ninibo_order.lock')
        lock_timeout = float(os.getenv('ORDER_LOCK_TIMEOUT', '10'))
        with FileLock(lockfile, timeout=lock_timeout):
            # Read latest on-disk state to avoid overwriting concurrent updates
            disk_state = load_state()
            if not isinstance(disk_state, dict):
                disk_state = {}
            disk_state.setdefault("positions", [])
            disk_state["positions"].append({
                "side": side,
                "price": float(price),
                "qty": float(qty),
                "time": int(time.time())
            })
            if len(disk_state["positions"]) > 50:
                disk_state["positions"] = disk_state["positions"][-50:]
            # persist atomically
            save_state(disk_state)
            # update provided in-memory state object for caller convenience
            try:
                if isinstance(state, dict):
                    state["positions"] = disk_state["positions"]
            except Exception:
                pass
    except Exception:
        # fallback to original behavior if something goes wrong
        try:
            state.setdefault("positions", [])
            state["positions"].append({
                "side": side,
                "price": float(price),
                "qty": float(qty),
                "time": int(time.time())
            })
            if len(state["positions"]) > 50:
                state["positions"] = state["positions"][-50:]
            save_state(state)
        except Exception:
            pass


def is_slippage_too_large(reference_price, latest_price):
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
    """
    価格データに基づいて売買シグナルを生成します。
    """
    # データ数が200本必要
    if df is None or len(df) < 200:
        # エラーメッセージを改善
        print(f"⚠️ データが不足しています。最低200本必要ですが、{len(df) if df is not None else 0}本しかありません。")
        return None

    # 短期25、中期75、長期200を追加
    df['short_mavg'] = df['close'].rolling(window=25).mean()
    df['mid_mavg'] = df['close'].rolling(window=75).mean() # 75をmidに名称変更
    df['long_mavg'] = df['close'].rolling(window=200).mean() # 新しい長期MA

    latest_data = df.iloc[-1]
    previous_data = df.iloc[-2]

    signal = None
    message = None

    # 🔑 トレンドフィルター
    is_uptrend = latest_data['mid_mavg'] > latest_data['long_mavg']
    mid_mavg_is_rising = latest_data['mid_mavg'] > previous_data['mid_mavg']

    # --- 買いシグナル 1：新規エントリー (ゴールデンクロス) ---
    if (previous_data['short_mavg'] <= previous_data['mid_mavg'] and
        latest_data['short_mavg'] > latest_data['mid_mavg'] and
        is_uptrend and mid_mavg_is_rising):
        signal = 'buy_entry' # 新規エントリーシグナル
        message = "✅ 新規エントリーシグナル (GC 25/75、トレンド確認) が発生しました。"
        return signal, message

    # --- 買いシグナル 2：買い増し (押し目) ---
    # 注: GC後、ポジション保有中に価格がMA25を上回っている（押し目買い）でトレンド上昇中
    elif latest_data['close'] > latest_data['short_mavg'] and is_uptrend:
        signal = 'buy_add' 
        message =  "📈 買い増しシグナル (押し目買い) が発生しました。"
        
    # --- 売りシグナル：全決済 (トレンド終了) ---
    # MA75がMA200を下回った、またはMA75が下向きに転じた
    elif not is_uptrend or latest_data['mid_mavg'] < previous_data['mid_mavg']:
        signal = 'sell_all'
        message = "❌ 全決済シグナル (長期トレンド終了/反転) が発生しました。"
    
    return signal, message


# === 4. 注文の整形 ===

def log_order(action, pair, amount, price=None):
    """
  注文内容を整形してログメッセージを返します。
    """
    msg = f"{action}注文: {amount:.4f} {pair.split('/')[0]} {'@ ' + str(price) if price else '（成行）'}"
    print(msg)
    return msg

# === 5. 注文の実行 ===

def execute_order(exchange, pair, order_type, amount, price=None):
    """
    Bitbankに注文を出します。(ccxt使用)
    """
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
            print("ℹ️ DRY_RUN: 注文は実行されませんでした（シミュレーション）")
            return simulated

        if order_type == 'buy':
            if price:
                # 指定価格で成行ではなく指値注文を出す
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
            print(f"無効な注文タイプです: {order_type}")
            return None

        if order and isinstance(order, dict) and 'id' in order:
            print("注文成功:", order['id'])  # IDのみ表示に修正
            return order
        else:
            print("注文に失敗しました:", order)
            return None

    except Exception as e:
        import traceback
        traceback.print_exc()  # ← これでエラーの詳細が表示されます

        print(f"❌ 注文実行中にエラーが発生しました: {e}")
        return None

# === 6. メインループ（Botの実行部分） ===
# Small helper: in DRY_RUN or when AUTO_FIX_FUNDS is enabled, ensure FundManager has a reasonable balance
def _ensure_fund_manager_has_funds(fm, initial_amount=None):
    """If fm.available_fund() is zero or unavailable, optionally top-up using add_funds().

    This is intentionally conservative: it only performs the auto-fix when DRY_RUN is
    enabled or when AUTO_FIX_FUNDS environment variable is set. It helps tests and
    local DRY_RUN runs where state files may be missing or zeroed.
    """
    # Auto-fix funds is now opt-in via AUTO_FIX_FUNDS. This avoids silently
    # modifying funds during regular DRY_RUNs — operator must explicitly enable it.
    try:
        auto_fix = str(os.getenv('AUTO_FIX_FUNDS', '')).lower() in ('1', 'true', 'yes', 'on')
    except Exception:
        auto_fix = False

    if not auto_fix:
        return

    try:
        avail = float(fm.available_fund())
    except Exception:
        avail = 0.0

    if avail and avail > 0:
        return

    # determine top-up amount; allow override via AUTO_FIX_AMOUNT (JPY)
    try:
        if initial_amount is not None:
            initial = float(initial_amount)
        else:
            initial = float(os.getenv('AUTO_FIX_AMOUNT', os.getenv('INITIAL_FUND', '20000')))
    except Exception:
        initial = 20000.0

    try:
        fm.add_funds(initial)
        print(f"🔧 funds were zero; auto-added {initial:.0f} JPY to fund_manager (AUTO_FIX_FUNDS)")
    except Exception as e:
        print(f"⚠️ failed to auto-fix fund_manager funds: {e}")

# 修正点: グローバルキーを使用するため、api_keyとsecret_keyの引数を削除
def run_bot(exchange, fund_manager_instance):
    """自動売買Botのメイン実行（1回分を実行する）。

    この実装は次を満たします:
    - 注文を出す前に最新残高を確認する
    - 注文が成功したときのみ fund_manager.place_order(cost) により残高を差し引く
    - DRY_RUN を尊重して、実運用時のみ API 呼び出しを行う
    """
    pair = 'BTC/JPY'
    interval_seconds = 3600

    # DEBUG: run_bot entry
    try:
        print(f"DEBUG: run_bot start - DRY_RUN={DRY_RUN}, pair={pair}, JAPANESE_YEN_BUDGET={JAPANESE_YEN_BUDGET}")
    except Exception:
        print("DEBUG: run_bot start (print failed)")

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
    if exchange is None:
        exchange = connect_to_bitbank()
    # adapter を通して安全に扱えるようにする
    fund_manager = _adapt_fund_manager_instance(fund_manager_instance) if fund_manager_instance is not None else _adapt_fund_manager_instance(FundManager(initial_fund=float(os.getenv('INITIAL_FUND', '20000')), state_file=os.getenv('FUND_STATE_FILE', 'funds_state.json')))

    if not exchange and not env_dry_run:
        print("API接続に失敗したためBotを停止します。")
        return

    print(f"Botを {pair} で実行します。データ取得間隔: {interval_seconds}秒 (1時間)")

    # 1回あたりの注文予算（JPY）。ユーザー指定が無ければ 10000 円に変更
    JAPANESE_YEN_BUDGET = float(os.getenv('JAPANESE_YEN_BUDGET', '10000'))
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
        BALANCE_BUFFER = float(os.getenv('BALANCE_BUFFER', '1000'))
    except Exception:
        BALANCE_BUFFER = 1000.0

    print(f"💰 1回あたりの注文予算: {JAPANESE_YEN_BUDGET} 円")
    print(f"📉 最低注文数量: {MIN_ORDER_BTC} BTC")


    # state を読み込み、保有ポジションがあれば利確チェックを行う
    # --- 低残高アラート設定 ---
    LOW_FUNDS_ALERT_JPY = float(os.getenv('LOW_FUNDS_ALERT_JPY', '2000'))

    # state を読み込み、保有ポジションがあれば利確チェックを行う
    try:
        # Use a file lock when reading/modifying/saving state for sell flow to avoid
        # races with concurrent buy operations that also update the state file.
        LOCKFILE_SELL = os.getenv('ORDER_LOCKFILE', '/tmp/ninibo_order.lock')
        with FileLock(LOCKFILE_SELL, timeout=10):
            state = load_state()
            positions = state.get('positions', []) if isinstance(state, dict) else []
            if positions:
                # Find the most recent BUY position that has a valid (non-zero) price.
                last_pos = None
                for p in reversed(positions):
                    try:
                        if p.get('side') == 'buy' and float(p.get('price', 0) or 0) > 0:
                            last_pos = p
                            break
                    except Exception:
                        continue

                if last_pos is not None:
                    entry_price = float(last_pos.get('price', 0))
                    entry_qty = float(last_pos.get('qty', 0))
                    lp = get_latest_price(exchange, pair)
                    if lp is not None:
                        # TRADE_TRIGGER_PCT を使って利確（デフォルト 20%）
                        gain_pct = (float(lp) - entry_price) / float(entry_price) * 100.0 if entry_price and entry_price > 0 else 0.0
                        if gain_pct >= float(TRADE_TRIGGER_PCT):
                            print(f"INFO: Trigger sell: gain={gain_pct:.2f}% >= {TRADE_TRIGGER_PCT}% -> selling {entry_qty} at {lp}")
                            sell_order = execute_order(exchange, pair, 'sell', entry_qty)
                            try:
                                print(f"DEBUG: post-execute_order sell_order={sell_order}")
                            except Exception:
                                pass
                            if sell_order and isinstance(sell_order, dict) and 'id' in sell_order:
                                # 売却成功: remove the sold position (the last valid one) and save state
                                try:
                                    removed = False
                                    for i in range(len(positions)-1, -1, -1):
                                        try:
                                            p = positions[i]
                                            if p.get('side') == 'buy' and float(p.get('price', 0) or 0) == float(entry_price) and float(p.get('qty', 0) or 0) == float(entry_qty):
                                                try:
                                                    print(f"DEBUG: removing position at index={i} -> {p}")
                                                except Exception:
                                                    pass
                                                # perform deletion
                                                del positions[i]
                                                removed = True
                                                # Immediately persist a tiny marker and an in-memory snapshot
                                                try:
                                                    marker = STATE_FILE.with_name(STATE_FILE.name + f'.after_sell.marker')
                                                    with open(str(marker), 'wb') as mf:
                                                        mf.write(b'REACHED_AFTER_SELL')
                                                    dbg_path = STATE_FILE.with_name(STATE_FILE.name + f'.after_sell.immediate.json')
                                                    dbg_path.write_text(json.dumps({'positions': positions, 'watch_reference': state.get('watch_reference')}, ensure_ascii=False, indent=2), encoding='utf-8')
                                                    try:
                                                        print(f"DEBUG: immediate marker and snapshot written: {marker}, {dbg_path}")
                                                    except Exception:
                                                        pass
                                                except Exception as e_immediate:
                                                    try:
                                                        print(f"DEBUG: failed immediate marker/snapshot write: {e_immediate}")
                                                    except Exception:
                                                        pass
                                                break
                                        except Exception as e_rem:
                                            try:
                                                print(f"DEBUG: exception while scanning positions for removal: {e_rem}")
                                            except Exception:
                                                pass
                                            continue
                                    if not removed:
                                        try:
                                            print("DEBUG: no exact matching position found to remove; will attempt to pop last element")
                                        except Exception:
                                            pass
                                        try:
                                            positions = positions[:-1]
                                        except Exception as e_pop:
                                            try:
                                                print(f"DEBUG: failed to pop last position: {e_pop}")
                                            except Exception:
                                                pass
                                    state['positions'] = positions
                                except Exception as e_state:
                                    try:
                                        print(f"DEBUG: exception while removing position: {e_state}")
                                    except Exception:
                                        pass
                                    # fallback: pop the last element
                                    try:
                                        state['positions'] = positions[:-1]
                                    except Exception:
                                        pass
                                # 売却後は監視基準価格を最新価格にリセット
                                try:
                                    state['watch_reference'] = float(lp)
                                except Exception:
                                    pass
                                # 売却成功時に売却代金をファンドへ戻す（実運用／DRY_RUN に対して適切な API を呼ぶ）
                                try:
                                    sell_proceeds = None
                                    if isinstance(sell_order, dict):
                                        sell_proceeds = sell_order.get('cost')
                                    if not sell_proceeds:
                                        # フォールバック: 最新価格 * 数量
                                        try:
                                            sell_proceeds = float(entry_qty) * float(lp)
                                        except Exception:
                                            sell_proceeds = None
                                    if sell_proceeds is not None:
                                        try:
                                            lock_timeout_local = float(os.getenv('ORDER_LOCK_TIMEOUT', '10'))
                                            with FileLock(LOCKFILE_SELL, timeout=lock_timeout_local):
                                                if hasattr(fund_manager, 'add_funds'):
                                                    fund_manager.add_funds(float(sell_proceeds))
                                                else:
                                                    # もし underlying が reservation-style を持たない場合は警告
                                                    try:
                                                        print("⚠️ fund_manager に add_funds メソッドがありません。手動で残高調整が必要です。")
                                                    except Exception:
                                                        pass
                                        except Exception as e_add:
                                            try:
                                                print(f"⚠️ 売却代金のファンド加算に失敗しました: {e_add}")
                                            except Exception:
                                                pass
                                except Exception:
                                    pass
                                # 保存前に内容をデバッグ出力
                                try:
                                    print(f"DEBUG: saving state after sell: watch_reference={state.get('watch_reference')} positions_count={len(state.get('positions', []))}")
                                except Exception:
                                    pass
                                # Immediately create a lightweight marker file to prove we reached
                                # this point. Use binary write to avoid encoding surprises.
                                try:
                                    marker = STATE_FILE.with_name(STATE_FILE.name + '.after_sell.marker')
                                    with open(str(marker), 'wb') as mf:
                                        mf.write(b'REACHED_AFTER_SELL')
                                    try:
                                        print(f"DEBUG: wrote marker file {marker}")
                                    except Exception:
                                        pass
                                except Exception as e_marker:
                                    try:
                                        print(f"DEBUG: failed to write marker file: {e_marker}")
                                    except Exception:
                                        pass
                                # Immediately dump an auxiliary debug file to make the in-memory
                                # state observable even if save_state fails or gets overwritten.
                                try:
                                    dbg_path = STATE_FILE.with_name(STATE_FILE.name + '.after_sell.debug.json')
                                    dbg_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
                                    try:
                                        print(f"DEBUG: wrote debug snapshot to {dbg_path}")
                                    except Exception:
                                        pass
                                except Exception as e_dbg:
                                    try:
                                        print(f"DEBUG: failed to write debug snapshot: {e_dbg}")
                                    except Exception:
                                        pass
                                save_state(state)
                                try:
                                    if STATE_FILE.exists():
                                        txt = STATE_FILE.read_text(encoding='utf-8')
                                        try:
                                            print(f"DEBUG: post-save STATE_FILE len={len(txt)}")
                                        except Exception:
                                            pass
                                except Exception:
                                    try:
                                        print(f"DEBUG: could not read state file after save")
                                    except Exception:
                                        pass

                                # Ensure proceeds are persisted to underlying fund file in DRY_RUN tests too
                                try:
                                    # primary: adapter-level add_funds (may be local in DRY_RUN)
                                    if sell_proceeds is not None:
                                        try:
                                            fund_manager.add_funds(float(sell_proceeds))
                                        except Exception:
                                            pass
                                    # fallback: if adapter wraps an underlying object that supports add_funds, call it to persist
                                    underlying = getattr(fund_manager, '_underlying', None)
                                    if underlying is not None and hasattr(underlying, 'add_funds'):
                                        try:
                                            # write under lock to avoid races
                                            lockfile_main = os.getenv('ORDER_LOCKFILE') or str(STATE_FILE.with_name('.ninibo_order.lock'))
                                            lock_timeout_local = float(os.getenv('ORDER_LOCK_TIMEOUT', '10'))
                                            with FileLock(lockfile_main, timeout=lock_timeout_local):
                                                underlying.add_funds(float(sell_proceeds))
                                        except Exception:
                                            pass
                                except Exception:
                                    pass

                                print("✅ Trigger sell: position sold and state updated")
                                # Exit this run after successful sell to avoid later logic
                                # (e.g. buy path) overwriting the updated state file.
                                return
                                # After a successful sell and state persist, return early to avoid subsequent
                                # buy logic in the same run from overwriting the state file.
                                return
    except Exception as e:
        print(f"⚠️ 利確チェック中にエラー: {e}")

    try:
        # 取引許可日のチェック (週末限定など)
        tz_name = os.getenv('TRADE_TIMEZONE')
        if tz_name:
            now = datetime.datetime.now(ZoneInfo(tz_name))
        else:
            now = datetime.datetime.now(JST)
        if not is_trade_day(now):
            print(f"取引制限: 本日は取引不可です（{now.strftime('%a %Y-%m-%d %H:%M:%S')}）。スキップします。")
            return

        latest_price = get_latest_price(exchange, pair)
        if latest_price is None:
            print("エラー: 価格が取得できませんでした。")
            return
        # 動的閾値を使う場合は1年レンジ等から閾値を計算する
        effective_threshold = float(MIN_PRICE_THRESHOLD_JPY)
        if USE_DYNAMIC_THRESHOLD:
            thr, min1y, max1y = compute_dynamic_threshold(exchange, pair, days=DYN_OHLCV_DAYS,
                                                          buffer_jpy=DYN_THRESHOLD_BUFFER_JPY,
                                                          buffer_pct=DYN_THRESHOLD_BUFFER_PCT)
            if thr is not None:
                effective_threshold = float(thr)
                print(f"🔎 dynamic threshold computed: threshold={effective_threshold}, 1y_min={min1y}, 1y_max={max1y}")
            else:
                print("⚠️ dynamic threshold could not be computed; using static MIN_PRICE_THRESHOLD_JPY")

        print(f"💵 最新の市場価格: {latest_price} 円 (buy_when_price <= {effective_threshold})")

        # --- 指標を計算してログに出力 ---
        try:
            indicators = compute_indicators(exchange, pair, timeframe='1h', limit=500)
            try:
                print(f"Indicators: price={indicators.get('latest_close')}, sma50={indicators.get('sma_short_50')}, sma200={indicators.get('sma_long_200')}, ema12={indicators.get('ema_12')}, rsi14={indicators.get('rsi_14')}, atr14={indicators.get('atr_14')}")
            except Exception:
                pass
        except Exception:
            indicators = None

        # 初期表示用に手数料を考慮した数量を算出
        initial_qty, initial_cost, initial_fee = compute_qty_for_budget_with_fee(
            float(JAPANESE_YEN_BUDGET), float(latest_price), min_btc=MIN_ORDER_BTC, step=MIN_ORDER_BTC,
            available_jpy=None, balance_buffer=float(BALANCE_BUFFER)
        )
        if initial_qty <= 0:
            print(f"ℹ️ 注文数量が最小取引単位を下回る（手数料込み）。AUTO_RESIZE={os.getenv('AUTO_RESIZE')} を確認してください。")
            return
        print(f"✅ 注文可能: {initial_qty} BTC (約 {initial_cost:.2f} 円, 手数料: {initial_fee:.2f} 円)")

    except Exception as e:
        print(f"エラー: 最新価格の取得に失敗しました。Botを停止します: {e}")
        return

    # 注文前に最新残高を確認して、不足であればスキップする
    try:
        available = None
        if hasattr(fund_manager, "available_fund"):
            try:
                available = float(fund_manager.available_fund())
            except Exception:
                available = None

        # Note: use initial_cost (fee-aware estimation) for pre-check
        if available is not None and available < initial_cost:
            print(f"🚫 残高不足のため注文をスキップします（必要: {initial_cost:.2f} 円, 残高: {available:.2f} 円）")
            return
    except Exception as e:
        print(f"🚫 残高チェック中にエラーが発生しました: {e}")
        return

    # 注文実行: ここでは「予約 (reserve)」戦略を採用します。
    # 1) ロックを取得して残高を再確認 -> 残高を差し引いて予約
    # 2) ロックを解放して実際の注文を発行
    # 3) 注文失敗時はロック下で返金（予約取り消し）
    LOCKFILE = os.getenv('ORDER_LOCKFILE', '/tmp/ninibo_order.lock')
    LOCK_TIMEOUT = float(os.getenv('ORDER_LOCK_TIMEOUT', '10'))
    reserved = False
    try:
        # 予約フェーズ: 価格変動に対応するため、"予算" を先に予約し
        # 注文直前に最新価格を再取得して数量を再計算します。
        # 小額運用向けに、残高の割合やバッファを尊重して予約額を決める
        reserved_budget = JAPANESE_YEN_BUDGET
        try:
            # available を事前取得（lock の外での読み取りで概算を取る）
            available_pre = float(fund_manager.available_fund()) if hasattr(fund_manager, 'available_fund') else None
        except Exception:
            available_pre = None
        # DEBUG: show pre-reservation estimates (より詳細に出力)
        try:
            allowed_by_percent = max(0.0, available_pre * float(MAX_RISK_PERCENT)) if available_pre is not None else None
            allowed_by_buffer = max(0.0, available_pre - float(BALANCE_BUFFER)) if available_pre is not None else None
            print(f"DEBUG: pre-reservation: available_pre={available_pre}, reserved_budget={reserved_budget}, allowed_by_percent={allowed_by_percent}, allowed_by_buffer={allowed_by_buffer}")
        except Exception:
            pass

        # 低残高アラート: available_pre がある場合に閾値を下回っていたら通知
        try:
            if available_pre is not None and float(available_pre) < float(LOW_FUNDS_ALERT_JPY):
                msg = f"⚠️ 資金アラート: 残高が少なくなっています（残高: {available_pre:.0f} 円 < 閾値: {LOW_FUNDS_ALERT_JPY:.0f} 円）"
                print(msg)
                try:
                    if smtp_host and email_to:
                        send_notification(smtp_host, smtp_port, smtp_user, smtp_password, email_to, subject, msg)
                except Exception:
                    pass
        except Exception:
            pass

        if available_pre is not None:
            # 利用可能残高に対する上限 (割合)
            allowed_by_percent = max(0.0, available_pre * float(MAX_RISK_PERCENT))
            # 残しておく最低バッファを考慮
            allowed_by_buffer = max(0.0, available_pre - float(BALANCE_BUFFER))
            # 実際に予約する金額は、環境変数での予算と上限の小さい方
            reserved_budget = min(float(JAPANESE_YEN_BUDGET), allowed_by_percent, allowed_by_buffer)
            # 小額になりすぎないよう安全下限チェックはロック内で行う
        with FileLock(LOCKFILE, timeout=LOCK_TIMEOUT):
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
                if last_buy and (time.time() - last_buy) < COOLDOWN_SEC:
                    print("Cooldown active -> skipping buy to avoid frequent add-on")
                    return

                q_check, cost_check, fee_check = compute_qty_for_budget_with_fee(
                    reserved_budget, float(latest_price), min_btc=MIN_ORDER_BTC, step=MIN_ORDER_BTC,
                    available_jpy=available, balance_buffer=float(BALANCE_BUFFER)
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
                    fee_fixed = float(os.getenv('FEE_FIXED_JPY', '0.0'))
                except Exception:
                    fee_fixed = 0.0
                # estimate max_allowed_jpy used inside compute_qty_for_budget_with_fee
                try:
                    max_allowed_jpy = min(float(reserved_budget), float(available) - float(BALANCE_BUFFER)) if available is not None else float(reserved_budget)
                except Exception:
                    max_allowed_jpy = float(reserved_budget)
                try:
                    approx_qty = max_allowed_jpy / (reservation_price * (1.0 + fee_rate)) if reservation_price and reservation_price > 0 else 0.0
                except Exception:
                    approx_qty = 0.0
                try:
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

                    # 一部の FundManager 実装は place_order が真/偽を返さず None を返す場合がある。
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
                    if isinstance(state, dict) and state.get('watch_reference') is not None:
                        try:
                            watch_ref = float(state.get('watch_reference'))
                        except Exception:
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

                do_buy_by_pct = False
                try:
                    if latest_price_now is not None and watch_ref is not None:
                        threshold_buy = watch_ref * (1.0 - float(TRADE_TRIGGER_PCT) / 100.0)
                        do_buy_by_pct = float(latest_price_now) <= float(threshold_buy)
                        print(f"DEBUG: watch_ref={watch_ref}, threshold_buy={threshold_buy}, latest={latest_price_now}, do_buy_by_pct={do_buy_by_pct}")
                except Exception:
                    do_buy_by_pct = False

                # optionally still allow breakout buys if configured
                allow_buy_by_breakout = False
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
                    if indicators is None:
                        indicators = compute_indicators(exchange, pair, timeframe='1h', limit=500)
                    write_indicators_csv(indicators if indicators is not None else {}, pair, signal=sig)
                except Exception:
                    pass

                if not do_buy:
                    print(f"🚫 買い条件未達（watch_ref={watch_ref}, latest={latest_price_now}, buy_pct={TRADE_TRIGGER_PCT}, breakout_allowed={allow_buy_by_breakout}）→ 予約を返金して終了")
                    if reserved:
                        with FileLock(LOCKFILE, timeout=LOCK_TIMEOUT):
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
                    with FileLock(LOCKFILE, timeout=LOCK_TIMEOUT):
                        if hasattr(fund_manager, 'release'):
                            fund_manager.release(reserved_budget)
                        elif hasattr(fund_manager, 'add_funds'):
                            fund_manager.add_funds(reserved_budget)
                        reserved = False
                return
            if latest_price_now is None:
                print("⚠️ 注文直前に価格が取得できませんでした。予約を取り消します。")
                # 返金
                if reserved:
                    with FileLock(LOCKFILE, timeout=LOCK_TIMEOUT):
                        if hasattr(fund_manager, 'release'):
                            fund_manager.release(reserved_budget)
                        elif hasattr(fund_manager, 'add_funds'):
                            fund_manager.add_funds(reserved_budget)
                return

            # 最新価格取得後にスリッページ最終判定
            if 'reservation_price' in globals() and reservation_price is not None:
                if is_slippage_too_large(reservation_price, latest_price_now):
                    print(f"Slippage too large (ref={reservation_price}, latest={latest_price_now}) -> cancelling & refund")
                    if reserved:
                        with FileLock(LOCKFILE, timeout=LOCK_TIMEOUT):
                            if hasattr(fund_manager, 'release'):
                                fund_manager.release(reserved_budget)
                            elif hasattr(fund_manager, 'add_funds'):
                                fund_manager.add_funds(reserved_budget)
                    return

            final_qty, final_cost, final_fee = compute_qty_for_budget_with_fee(
                reserved_budget, float(latest_price_now), min_btc=MIN_ORDER_BTC, step=MIN_ORDER_BTC,
                available_jpy=available, balance_buffer=float(BALANCE_BUFFER)
            )

            # 最小数量チェック
            if final_qty <= 0:
                print(f"ℹ️ 注文直前で数量が最小取引単位を下回りましたまたは手数料で合計が超過しました。予約を取り消します。")
                if reserved:
                    with FileLock(LOCKFILE, timeout=LOCK_TIMEOUT):
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
                    with FileLock(LOCKFILE, timeout=LOCK_TIMEOUT):
                        if hasattr(fund_manager, 'release'):
                            fund_manager.release(reserved_budget)
                        elif hasattr(fund_manager, 'add_funds'):
                            fund_manager.add_funds(reserved_budget)
                        reserved = False
                except Exception as e2:
                    print(f"⚠️ 予約取り消しに失敗しました: {e2}")
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
                        if 'price' in order and order.get('price') is not None:
                            try:
                                entry_price = float(order.get('price'))
                            except Exception:
                                entry_price = None
                        elif 'cost' in order and order.get('amount'):
                            try:
                                entry_price = float(order.get('cost')) / float(order.get('amount'))
                            except Exception:
                                entry_price = None
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
                    dbg_final_qty = float(final_qty) if 'final_qty' in locals() else (float(order.get('amount', 0)) if isinstance(order, dict) and order.get('amount') is not None else 0.0)
                except Exception:
                    dbg_final_qty = 0.0
                try:
                    dbg_final_cost = float(final_cost) if 'final_cost' in locals() else (float(order.get('cost')) if isinstance(order, dict) and order.get('cost') is not None else None)
                except Exception:
                    dbg_final_cost = None
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
                    with FileLock(LOCKFILE, timeout=LOCK_TIMEOUT):
                        if hasattr(fund_manager, 'release'):
                            fund_manager.release(reserved_budget)
                        elif hasattr(fund_manager, 'add_funds'):
                            fund_manager.add_funds(reserved_budget)
                        reserved = False
                except Exception as e:
                    print(f"⚠️ 予約取り消し（返金）に失敗しました: {e}")

    except Exception as e:
        # 途中例外発生時は予約取り消しを試みる
        if reserved:
            try:
                with FileLock(LOCKFILE, timeout=LOCK_TIMEOUT):
                    if hasattr(fund_manager, 'release'):
                        fund_manager.release(reserved_budget)
                    elif hasattr(fund_manager, 'add_funds'):
                        # refund reserved_budget on unexpected exception
                        fund_manager.add_funds(reserved_budget)
                    reserved = False
            except Exception as e2:
                print(f"⚠️ 予約取り消しに失敗しました: {e2}")
        print(f"⚠️ 注文に失敗しました: {e}")

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
    print("🔁 自動売買Botを継続運用モードで起動します")
    # DEBUG: main entry
    try:
        print(f"DEBUG: __main__ start - DRY_RUN={DRY_RUN}")
    except Exception:
        pass
    # CLI helper: run small adapter test and exit
    try:
        if len(sys.argv) > 1 and sys.argv[1] == 'test_adapter':
            test_fund_adapter()
            sys.exit(0)
    except Exception:
        pass
    exchange = connect_to_bitbank()
    # 初期資金は env で設定可能（なければ 20000 円）
    initial_fund = float(os.getenv('INITIAL_FUND', '20000'))
    fund_manager = FundManager(initial_fund=initial_fund, state_file=os.getenv('FUND_STATE_FILE', 'funds_state.json'))
    # Ensure fund state is usable for local DRY_RUN/tests
    _ensure_fund_manager_has_funds(fund_manager, initial_amount=initial_fund)
    # 毎ループで自動投入する少額（例: 毎時間100円ずつ入金する）を環境変数で指定
    deposit_amount = float(os.getenv('DEPOSIT_AMOUNT', '0'))
    # 自動トップアップの閾値（この金額を下回ったら自動入金を行う）
    # 例: MIN_BALANCE_THRESHOLD=5000
    min_balance_threshold = float(os.getenv('MIN_BALANCE_THRESHOLD', os.getenv('BALANCE_TOPUP_THRESHOLD', '5000')))
    if not exchange:
        print("API接続に失敗したためBotを終了します。")
    else:
        while True:
            # 事前トップアップ（安全な段階的入金）: 残高が閾値未満なら段階的に入金
            try:
                if deposit_amount and deposit_amount > 0:
                    try:
                        current = float(fund_manager.available_fund())
                    except Exception:
                        current = None

                    # 目標残高: 環境変数 TOPUP_TARGET があればそれを使い、無ければ閾値の2倍を目標にする
                    try:
                        topup_target = float(os.getenv('TOPUP_TARGET', str(min_balance_threshold * 2)))
                    except Exception:
                        topup_target = min_balance_threshold * 2

                    if current is None:
                        # available_fund が使えない場合は通常の自動入金を行う
                        fund_manager.add_funds(deposit_amount)
                        print(f"💳 自動入金(保険): {deposit_amount:.2f} 円 → 残高: {fund_manager.available_fund():.2f} 円")
                    else:
                        if current < min_balance_threshold:
                            # 実際に入金する額は、目標との差分と deposit_amount の小さい方
                            to_add = min(deposit_amount, max(0.0, topup_target - current))
                            if to_add > 0:
                                fund_manager.add_funds(to_add)
                                print(f"💳 閾値以下のため自動入金: {to_add:.2f} 円 → 残高: {fund_manager.available_fund():.2f} 円 (閾値: {min_balance_threshold:.2f}, 目標: {topup_target:.2f})")
                            else:
                                print(f"ℹ️ 自動入金は不要（目標残高に達しています）: 現在 {current:.2f} 円, 目標 {topup_target:.2f} 円")

            except Exception as e:
                print(f"⚠️ 自動入金処理中にエラーが発生しました: {e}")

            run_bot(exchange, fund_manager)

            # 従来の毎ループ入金（あえて残す。ENVで無効化可）
            try:
                post_deposit = float(os.getenv('POST_LOOP_DEPOSIT', '0'))
            except Exception:
                post_deposit = 0
            if post_deposit and post_deposit > 0:
                try:
                    fund_manager.add_funds(post_deposit)
                    print(f"💳 ループ終了後の自動入金: {post_deposit:.2f} 円 → 残高: {fund_manager.available_fund():.2f} 円")
                except Exception as e:
                    print(f"⚠️ ループ後自動入金に失敗しました: {e}")

            time.sleep(3600)  # 1時間待機


# === DI対応版のエントリーポイント ===
def run_bot_di(dry_run=False, exchange_override=None):
    """
    Bot のメインエントリーポイント（DI対応）
    
    Args:
        dry_run (bool): True の場合、実際の取引を行わずログ出力のみ
        exchange_override: テスト用の Exchange オブジェクト（None の場合は実際の取引所に接続）
    
    Returns:
        dict: 実行結果の辞書
    """
    # 環境変数チェック（dry_run の場合は必須チェックを緩和する）
    # DRY_RUN 実行時は外部サービス（SMTP/APIキー等）を必須にしない
    env_dry_run = os.getenv("DRY_RUN", "").lower() in ["1", "true", "yes", "on"]
    actual_dry_run = dry_run or env_dry_run

    if not actual_dry_run:
        required_env_vars = ["SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS", "TO_EMAIL", "API_KEY", "SECRET_KEY"]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"以下の環境変数が .env に設定されていません: {', '.join(missing_vars)}")

    # DRY_RUN フラグの確認
    env_dry_run = os.getenv("DRY_RUN", "").lower() in ["1", "true", "yes", "on"]
    actual_dry_run = dry_run or env_dry_run
    
    print(f"🚀 Bot開始 (DRY_RUN: {actual_dry_run})")
    
    # Exchange の準備
    if exchange_override:
        exchange = exchange_override
    elif actual_dry_run:
        exchange = ExchangeStub()
        print("🔧 DRY_RUN モード: ExchangeStub を使用")
    else:
        exchange = connect_to_bitbank()
        if not exchange:
            return {"status": "error", "message": "取引所接続に失敗"}
    
    # FundManager の準備
    initial_fund = float(os.getenv('INITIAL_FUND', '20000'))
    # Create raw FundManager instance, ensure it has funds when appropriate, then adapt
    _raw_fm = FundManager(initial_fund=initial_fund, state_file=os.getenv('FUND_STATE_FILE', 'funds_state.json'))
    _ensure_fund_manager_has_funds(_raw_fm, initial_amount=initial_fund)
    fund_manager = _adapt_fund_manager_instance(_raw_fm)
    
    try:
        run_bot(exchange, fund_manager)
        return {"status": "success", "message": "Bot実行完了"}
    except Exception as e:
        return {"status": "error", "message": f"Bot実行中にエラー: {e}"}


def test_fund_adapter():
    """Quick smoke test for FundAdapter/_adapt_fund_manager_instance.
    Prints expected behavior for reserve/confirm/release in DRY_RUN and live modes.
    """
    print("--- FundAdapter smoke test ---")
    # live-like stub
    fm = FundManager(initial_fund=2000, state_file=os.getenv('FUND_STATE_FILE', 'funds_state.json'))
    adapter = _adapt_fund_manager_instance(fm)
    print("initial available (live stub):", adapter.available_fund())
    cost = 500
    ok = adapter.reserve(cost) if hasattr(adapter, 'reserve') else adapter.place_order(cost)
    print(f"reserve/place_order({cost}) ->", ok)
    print("available after reserve:", adapter.available_fund())
    # attempt release (refund)
    if hasattr(adapter, 'release'):
        adapter.release(cost)
        print("after release available:", adapter.available_fund())
    else:
        # fallback: add_funds used as refund
        adapter.add_funds(cost)
        print("after add_funds available:", adapter.available_fund())

    # Dry-run adapter
    dry_adapter = _adapt_fund_manager_instance(None)
    # ensure it simulates local funds
    try:
        da = FundAdapter(fund_manager=None, initial_fund=1000.0, dry_run=True)
        print("dry initial available:", da.available_fund())
        ok2 = da.reserve(300)
        print("dry reserve(300) ->", ok2, "available->", da.available_fund())
        da.confirm(300)
        print("dry confirm(300) -> available->", da.available_fund())
    except Exception as e:
        print("dry adapter test failed:", e)

=======
# python 3.14環境で動作確認済み (仮想環境venv314を使用)
# === 必要なライブラリを1回ずつインポート（心臓部の準備） ===

# ccxt がインストールされていない環境でもファイルが読み込めるよう、フォールバックのスタブを用意します。
try:
    import ccxt  # type: ignore
except Exception:
    # 最低限のインターフェースを持つスタブ実装
    class AuthenticationError(Exception):
        pass

    class BitbankStub:
        def __init__(self, config=None):
            self.apiKey = (config or {}).get('apiKey')
            self.secret = (config or {}).get('secret')
        # BitbankStub: テスト/フォールバック用の最小限インターフェース実装
        # このクラスはファイル冒頭での ccxt フォールバック用に使われます。
        # 注意: ここでは副作用を最小にするため簡素な実装に留めます。
        def fetch_balance(self):
            # テスト用のダミー残高
            return {'total': {'JPY': 0.0, 'BTC': 0.0}}

        def fetch_ticker(self, pair):
            # テスト用のダミー価格（0.0 で返すことで呼び出し側が安全に扱える）
            return {'last': 0.0}

        def fetch_ohlcv(self, pair, timeframe='1h', limit=250):
            # 空のデータを返して呼び出し側で安全に扱えるようにする
            return []

        def create_order(self, pair, type_, side, amount, price=None):
            # ダミー注文レスポンスを返す（cost は計算できる場合のみ設定）
            cost = None
            try:
                p = float(price) if price is not None else 0.0
                cost = float(amount) * p
            except Exception:
                cost = None
            return {'id': 'stub_order', 'pair': pair, 'type': type_, 'side': side, 'amount': amount, 'price': price, 'cost': cost}

    class _CCXTModule:
        AuthenticationError = AuthenticationError
        def bitbank(self, config=None):
            return BitbankStub(config)

    ccxt = _CCXTModule()

# 後続コードが使うために名前を揃える
# ccxt を直接参照する代わりに、このモジュール内で使う共通の例外参照を作成します。
# 実環境では ccxt.AuthenticationError が存在します。スタブ環境では上で定義したものが入ります。
AuthenticationError = getattr(ccxt, 'AuthenticationError', Exception)

# 互換性対策: 一部のコードやライブラリは ccxt.base.errors.AuthenticationError を参照する
# ことがあるため、ccxt.base.errors.AuthenticationError が存在しない場合は補完しておきます。
try:
    base_obj = getattr(ccxt, 'base', None)
    if base_obj is None or not hasattr(base_obj, 'errors'):
        class _BaseErrors:
            pass
        setattr(_BaseErrors, 'AuthenticationError', getattr(ccxt, 'AuthenticationError', Exception))

        class _Base:
            pass
        setattr(_Base, 'errors', _BaseErrors)
        setattr(ccxt, 'base', _Base)
except Exception:
    # 保険: 何か問題があれば無視して既存の AuthenticationError を使う
    pass

# typing の Optional を使う箇所があるため明示的にインポートしておく
from typing import Optional
import json
from pathlib import Path
import sys

# Replace built-in print with a safe wrapper to avoid UnicodeEncodeError on Windows consoles
import builtins
_orig_print = builtins.print
def _safe_print(*args, **kwargs):
    try:
        _orig_print(*args, **kwargs)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, 'encoding', None) or 'utf-8'
        new_args = []
        for a in args:
            s = str(a)
            try:
                s.encode(enc)
            except UnicodeEncodeError:
                s = s.encode(enc, errors='replace').decode(enc)
            new_args.append(s)
        _orig_print(*new_args, **kwargs)
    except Exception:
        try:
            _orig_print(*[str(a) for a in args], **kwargs)
        except Exception:
            pass

builtins.print = _safe_print

# Try to reconfigure stdout to UTF-8 to avoid encoding errors on Windows consoles
try:
    if hasattr(sys, 'stdout') and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# funds モジュールが存在しても、外部の FundManager がこのスクリプトの期待する
# メソッド (available_fund, place_order, add_funds) を持たない場合があるため、
# 互換性チェックをして不一致なら内部スタブを使うようにします。
def _make_internal_fund_manager_class():
    # Minimal internal FundManager class used when external `funds` module is unavailable.
    class FundManagerStub:
        """Simple persistent fund manager for DRY_RUN / tests.

        - Persists a JSON file with {"available": <float>, "reserved": <float>} when state_file is provided.
        - Provides methods: available_fund(), place_order(cost), add_funds(amount), reserve(cost), confirm(cost), release(cost).
        """
        def __init__(self, initial_fund: float = 0.0, state_file: Optional[str] = None):
            import threading
            self._lock = threading.Lock()
            self._state_file = Path(state_file) if state_file else None
            self._available = float(initial_fund or 0.0)
            self._reserved = 0.0
            # load persisted state if present
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
            with self._lock:
                try:
                    return float(self._available)
                except Exception:
                    return 0.0

        def place_order(self, cost: float) -> bool:
            """Legacy immediate-deduct behavior: consume available balance if enough."""
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

        # Reservation-style API
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
                # consumed reserved amount
                self._reserved = max(0.0, float(self._reserved) - c)
                # no change to available (already deducted at reservation or by place_order)
                self._persist()

        def release(self, cost: float) -> None:
            try:
                c = float(cost)
            except Exception:
                return
            with self._lock:
                # move from reserved back to available
                self._reserved = max(0.0, float(self._reserved) - c)
                self._available = float(self._available) + c
                self._persist()

    return FundManagerStub


_InternalFundManager = _make_internal_fund_manager_class()

try:
    from funds import FundManager as _ImportedFundManager  # type: ignore
    # 簡易的な互換性チェック: クラスに想定するメソッドが存在するか
    required = ('available_fund', 'place_order', 'add_funds')
    if all(hasattr(_ImportedFundManager, name) for name in required):
        FundManager = _ImportedFundManager
    else:
        # 互換性なし → 内部スタブを利用
        FundManager = _InternalFundManager
except Exception:
    # インポート失敗時は内部スタブを利用
    FundManager = _InternalFundManager

import os

fund_manager = FundManager(initial_fund=0.0, state_file=os.getenv('FUND_STATE_FILE', 'funds_state.json'))


class FundAdapter:
    """Module-level FundAdapter used when adapting external FundManager instances.

    Provides reserve/confirm/release semantics and a local dry-run fallback.
    """
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
            return float(self._local_total - self._local_used)

    def reserve(self, cost: float) -> bool:
        c = float(cost or 0.0)
        if c <= 0:
            return False
        if self._fund is not None and not self._dry_run:
            try:
                if hasattr(self._fund, 'reserve'):
                    return bool(self._fund.reserve(c))
                if hasattr(self._fund, 'place_order'):
                    return bool(self._fund.place_order(c))
            except Exception:
                return False
        with self._lock:
            if (self._local_total - self._local_used) < c:
                return False
            self._local_used += c
            return True

    def confirm(self, cost: float) -> None:
        c = float(cost or 0.0)
        if c <= 0:
            return
        if self._fund is not None and not self._dry_run:
            try:
                if hasattr(self._fund, 'confirm'):
                    return self._fund.confirm(c)
                return
            except Exception:
                return
        with self._lock:
            self._local_total = max(0.0, self._local_total - c)
            self._local_used = max(0.0, self._local_used - c)

    def release(self, cost: float) -> None:
        c = float(cost or 0.0)
        if c <= 0:
            return
        if self._fund is not None and not self._dry_run:
            try:
                if hasattr(self._fund, 'release'):
                    return self._fund.release(c)
                print("⚠️ FundManager has no reservation-cancel API; manual correction may be required.")
                return
            except Exception:
                return
        with self._lock:
            self._local_used = max(0.0, self._local_used - c)

    def add_funds(self, amount: float) -> None:
        try:
            a = float(amount)
        except Exception:
            return
        if self._fund is not None and not self._dry_run and hasattr(self._fund, 'add_funds'):
            try:
                return self._fund.add_funds(a)
            except Exception:
                pass
        with self._lock:
            self._local_total = float(self._local_total) + a

def _adapt_fund_manager_instance(fm):
    """
    Wrap an external FundManager instance so it exposes the small API
    this script expects: available_fund(), place_order(cost), add_funds(amount).
    If the passed object already has these methods, return it unchanged.
    Otherwise return an adapter that attempts to call the underlying
    implementation when possible and falls back to an internal counter.
    """
    # Build an adapter that exposes both the old API (available_fund, place_order, add_funds)
    # and the new reservation API (reserve, confirm, release). This keeps backward
    # compatibility while allowing run_bot to use reserve/confirm/release semantics.
    try:
        dry_run_env = str(os.getenv('DRY_RUN', '')).lower() in ('1', 'true', 'yes', 'on')
    except Exception:
        dry_run_env = False

    # If the passed object already supports reserve/confirm/release, return it as-is
    if fm is not None and all(hasattr(fm, name) for name in ('reserve', 'confirm', 'release', 'available_fund')):
        return fm

    class FundAdapter:
        """Adapter providing reservation-style API and a local fallback for dry-run.

        This is a small, self-contained adapter used when an external FundManager
        does not implement reserve/confirm/release. It keeps local counters when
        dry_run is True or when no underlying manager is provided.
        """
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
                return float(self._local_total - self._local_used)

        def reserve(self, cost: float) -> bool:
            c = float(cost or 0.0)
            if c <= 0:
                return False
            # Delegate to underlying if possible and not dry-run
            if self._fund is not None and not self._dry_run:
                try:
                    # prefer reserve() if provided
                    if hasattr(self._fund, 'reserve'):
                        return bool(self._fund.reserve(c))
                    # fall back to place_order (immediate deduct)
                    if hasattr(self._fund, 'place_order'):
                        return bool(self._fund.place_order(c))
                except Exception:
                    return False
            # simulate local reservation
            with self._lock:
                if (self._local_total - self._local_used) < c:
                    return False
                self._local_used += c
                return True

        def confirm(self, cost: float) -> None:
            c = float(cost or 0.0)
            if c <= 0:
                return
            if self._fund is not None and not self._dry_run:
                try:
                    if hasattr(self._fund, 'confirm'):
                        return self._fund.confirm(c)
                    # if only place_order existed, it's already consumed
                    return
                except Exception:
                    return
            with self._lock:
                # consume reserved funds
                self._local_total = max(0.0, self._local_total - c)
                self._local_used = max(0.0, self._local_used - c)

        def release(self, cost: float) -> None:
            c = float(cost or 0.0)
            if c <= 0:
                return
            if self._fund is not None and not self._dry_run:
                try:
                    if hasattr(self._fund, 'release'):
                        return self._fund.release(c)
                    # no generic undo for place_order
                    print("⚠️ FundManager has no reservation-cancel API; manual correction may be required.")
                    return
                except Exception:
                    return
            with self._lock:
                self._local_used = max(0.0, self._local_used - c)

        def add_funds(self, amount: float) -> None:
            try:
                a = float(amount)
            except Exception:
                return
            if self._fund is not None and not self._dry_run and hasattr(self._fund, 'add_funds'):
                try:
                    return self._fund.add_funds(a)
                except Exception:
                    pass
            with self._lock:
                self._local_total = float(self._local_total) + a

    class _Adapter:
        def __init__(self, underlying, initial=0.0, dry_run=False):
            self._underlying = underlying
            self._fund_adapter = FundAdapter(fund_manager=underlying if underlying is not None else None,
                                             initial_fund=initial, dry_run=dry_run)

        # New reservation-style API
        def reserve(self, cost: float) -> bool:
            return self._fund_adapter.reserve(cost)

        def confirm(self, cost: float) -> None:
            return self._fund_adapter.confirm(cost)

        def release(self, cost: float) -> None:
            return self._fund_adapter.release(cost)

        # Backward-compatible legacy API
        def available_fund(self):
            return self._fund_adapter.available_fund()

        def place_order(self, cost):
            """Legacy behavior: attempt to delegate to underlying.place_order if available.
            Otherwise behave as reserve+confirm (consume immediately)."""
            try:
                if self._underlying is not None and hasattr(self._underlying, 'place_order'):
                    res = self._underlying.place_order(cost)
                    # If underlying returns a boolean, respect it
                    if isinstance(res, bool):
                        return res
                    # If underlying returns None/other, assume success if available decreased
                    return True
            except Exception:
                pass
            # fallback: reserve then confirm (atomic from caller's perspective)
            ok = self._fund_adapter.reserve(cost)
            if ok:
                self._fund_adapter.confirm(cost)
                return True
            return False

        def add_funds(self, amount):
            try:
                if self._underlying is not None and hasattr(self._underlying, 'add_funds'):
                    return self._underlying.add_funds(amount)
            except Exception:
                pass
            # fallback to fund_adapter local top-up
            try:
                with self._fund_adapter._lock:
                    self._fund_adapter._local_total = float(self._fund_adapter._local_total) + float(amount)
            except Exception:
                pass

    # Try to seed the adapter with the underlying available balance when possible
    init_bal = 0.0
    try:
        if fm is not None and hasattr(fm, 'available_fund'):
            try:
                init_bal = float(fm.available_fund())
            except Exception:
                init_bal = float(getattr(fm, '_available', 0.0))
        else:
            init_bal = float(getattr(fm, '_available', 0.0) if fm is not None else 0.0)
    except Exception:
        init_bal = 0.0

    return _Adapter(fm, initial=init_bal, dry_run=dry_run_env)

import os
import time
import datetime
import math
import sys
try:
    from filelock import FileLock
except Exception:
    # Minimal FileLock fallback using fcntl for Unix-like systems
    try:
        import fcntl

        class FileLock:
            def __init__(self, path, timeout=None):
                self.path = path
                self.timeout = timeout
                self.fd = None

            def __enter__(self):
                start = time.time()
                while True:
                    try:
                        # open file for writing (create if not exists)
                        self.fd = open(self.path, 'w')
                        # try to acquire non-blocking exclusive lock
                        fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        return self
                    except BlockingIOError:
                        # timeout handling
                        if self.timeout is not None and (time.time() - start) >= float(self.timeout):
                            raise TimeoutError(f"Timeout acquiring lock {self.path}")
                        time.sleep(0.05)

            def __exit__(self, exc_type, exc, tb):
                try:
                    if self.fd:
                        fcntl.flock(self.fd, fcntl.LOCK_UN)
                        self.fd.close()
                except Exception:
                    pass

    except Exception:
        # Last-resort dummy lock (no-op) for non-Unix platforms; safer to install filelock package
        class FileLock:
            def __init__(self, path, timeout=None):
                self.path = path

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return
try:
    import pandas as pd  # type: ignore
except Exception:
    # Minimal pandas-like stub to avoid import errors and provide the small API used in this script.
    # NOTE: This is a lightweight compatibility shim for parsing/testing and does NOT replace real pandas.
    class Series:
        def __init__(self, values):
            self.values = list(values) if values is not None else []
            self._window = None

        def rolling(self, window):
            self._window = int(window)
            return self

        def mean(self):
            vals = self.values
            w = self._window or 1
            if not vals:
                return []
            res = []
            for i in range(len(vals)):
                if i + 1 < w:
                    res.append(None)
                else:
                    window_vals = [v for v in vals[i + 1 - w:i + 1] if v is not None]
                    res.append(sum(window_vals) / len(window_vals) if window_vals else None)
            return res

        def __iter__(self):
            return iter(self.values)

    class Row:
        def __init__(self, data):
            self._data = data or {}

        def __getitem__(self, key):
            return self._data.get(key)

        def __getattr__(self, name):
            if name in self._data:
                return self._data[name]
            raise AttributeError(name)

    class DataFrame:
        def __init__(self, data=None, columns=None):
            # data: list of lists (rows) or list of dicts
            self._columns = list(columns) if columns else []
            self._rows = []
            if data:
                if self._columns and all(isinstance(r, (list, tuple)) for r in data):
                    for row in data:
                        self._rows.append({c: v for c, v in zip(self._columns, row)})
                elif all(isinstance(r, dict) for r in data):
                    self._rows = [dict(r) for r in data]
                    if not self._columns:
                        cols = set()
                        for r in self._rows:
                            cols.update(r.keys())
                        self._columns = list(cols)
                else:
                    # fallback: single column
                    col = self._columns[0] if self._columns else "data"
                    for r in data:
                        self._rows.append({col: r})
            self.index = None

        def __len__(self):
            return len(self._rows)

        def __getitem__(self, key):
            if isinstance(key, str):
                vals = [row.get(key) for row in self._rows]
                return Series(vals)
            raise KeyError(key)

        def __setitem__(self, key, value):
            # value can be Series or iterable; align by index
            vals = list(value) if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)) else [value] * len(self._rows)
            if not self._rows and vals:
                for v in vals:
                    self._rows.append({key: v})
            else:
                for i, v in enumerate(vals):
                    if i < len(self._rows):
                        self._rows[i][key] = v
                    else:
                        self._rows.append({key: v})
            if key not in self._columns:
                self._columns.append(key)

        @property
        def iloc(self):
            class _Loc:
                def __init__(self, rows):
                    self._rows = rows

                def __getitem__(self, idx):
                    return Row(self._rows[idx])
            return _Loc(self._rows)

        def set_index(self, key):
            self.index = key
            return self

    def to_datetime(values, unit='ms'):
        out = []
        for v in values:
            try:
                if v is None:
                    out.append(None)
                    continue
                if unit == 'ms':
                    ts = float(v) / 1000.0
                else:
                    ts = float(v)
                out.append(datetime.datetime.fromtimestamp(ts))
            except Exception:
                out.append(None)
        return out

    import types
    pd = types.SimpleNamespace(DataFrame=DataFrame, to_datetime=to_datetime)

from zoneinfo import ZoneInfo  # 標準ライブラリのタイムゾーン処理
#import time
import json
from pathlib import Path
import math
import os

# Safety & strategy environment variables (defaults)
MAX_SLIPPAGE_PCT = float(os.environ.get("MAX_SLIPPAGE_PCT", "0.5"))  # percent
ORDER_EXECUTION_WINDOW_SEC = int(os.environ.get("ORDER_EXECUTION_WINDOW_SEC", "30"))
COOLDOWN_SEC = int(os.environ.get("COOLDOWN_SEC", "3600"))  # 秒
TAKE_PROFIT_PCT = float(os.environ.get("TAKE_PROFIT_PCT", "5.0"))  # percent
MAX_ORDER_JPY = int(os.environ.get("MAX_ORDER_JPY", "2000"))  # 1注文上限（JPY）
STATE_FILE = Path(os.environ.get("TRADING_STATE_FILE", "bot_state.json"))
# 売買トリガー（%）。ユーザー指定が無ければ 20% を使う。
TRADE_TRIGGER_PCT = float(os.environ.get('TRADE_TRIGGER_PCT', '20.0'))
# 新規: 注文を行うための市場価格閾値（JPY）。この値未満なら発注を行わない。
MIN_PRICE_THRESHOLD_JPY = float(os.environ.get("MIN_PRICE_THRESHOLD_JPY", "12000000"))
USE_DYNAMIC_THRESHOLD = str(os.environ.get('USE_DYNAMIC_THRESHOLD', '1')).lower() in ('1', 'true', 'yes', 'on')
# 動的閾値用の設定: 過去何日分を参照するか、バッファを固定円 or 割合で指定
DYN_OHLCV_DAYS = int(os.environ.get('DYN_OHLCV_DAYS', '365'))
DYN_THRESHOLD_BUFFER_JPY = float(os.environ.get('DYN_THRESHOLD_BUFFER_JPY', '20000'))
DYN_THRESHOLD_BUFFER_PCT = float(os.environ.get('DYN_THRESHOLD_BUFFER_PCT', '0.01'))
DYN_THRESHOLD_RATIO = float(os.environ.get('DYN_THRESHOLD_RATIO', '0.15'))
# ブレイクアウト設定（環境変数で上書き可）
BUY_ON_BREAKOUT = str(os.environ.get('BUY_ON_BREAKOUT', '1')).lower() in ('1','true','yes','on')
BREAKOUT_PCT = float(os.environ.get('BREAKOUT_PCT', '0.01'))  # 例: 0.01 = 1%
BREAKOUT_SMA_SHORT = int(os.environ.get('BREAKOUT_SMA_SHORT', '30'))  # 日数
BREAKOUT_SMA_LONG = int(os.environ.get('BREAKOUT_SMA_LONG', '75'))  # 日数（比較用）
BREAKOUT_LOOKBACK_DAYS = int(os.environ.get('BREAKOUT_LOOKBACK_DAYS', '30'))  # 直近高値参照日数

try:
    from dotenv import load_dotenv  # type: ignore[reportMissingImports]
except Exception:
    # minimal stub for load_dotenv to allow .env loading when python-dotenv is not installed
    def load_dotenv(dotenv_path=None):
        """
        Very small implementation that reads KEY=VALUE lines from a file and sets os.environ entries.
        Returns True if a file was read, False otherwise.
        """
        if not dotenv_path or not os.path.exists(dotenv_path):
            return False
        try:
            with open(dotenv_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        k, v = line.split('=', 1)
                        k = k.strip()
                        v = v.strip()
                        # strip optional surrounding quotes
                        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                            v = v[1:-1]
                        # do not overwrite existing environment variables
                        os.environ.setdefault(k, v)
            return True
        except Exception:
            return False


# === .envファイルの読み込み（複数の場所を試す） ===
# systemd サービスでは EnvironmentFile が優先されるが、
# 手動実行時は .env ファイルを読み込む
env_paths = [
    '/home/ninitan/.secrets/.env',  # systemd で使用
    '.env',                         # プロジェクト直下
    'config.env'                    # 旧形式
]

env_loaded = False
def safe_print(s: str) -> None:
    """Print safely even when the console encoding can't represent some characters.

    Replaces unencodable characters with the platform replacement character.
    """
    try:
        print(s)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, 'encoding', None) or 'utf-8'
        try:
            safe = s.encode(enc, errors='replace').decode(enc)
        except Exception:
            # fallback: remove non-ascii
            safe = ''.join(ch if ord(ch) < 128 else '?' for ch in s)
        print(safe)
    except Exception:
        # last resort
        try:
            print(str(s))
        except Exception:
            pass
for env_path in env_paths:
    if load_dotenv(dotenv_path=env_path):
        safe_print(f"[OK] 環境変数を {env_path} から読み込みました")
        env_loaded = True
        break

if not env_loaded:
    safe_print("[WARN] .env ファイルが見つかりません。環境変数は systemd EnvironmentFile から読み込まれます。")

# グローバルに使う API キーを一度だけ読み込む
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

# 日本標準時 (JST) のタイムゾーンオブジェクトを作成
try:
    JST = ZoneInfo('Asia/Tokyo')
except Exception:
    # Windows 等で tzdata が無い環境では ZoneInfo が ZoneInfoNotFoundError を出すことがあるため
    # 安全に固定オフセットで JST を作成してフォールバックします（UTC+9）。
    JST = datetime.timezone(datetime.timedelta(hours=9))

# === 環境変数の取得（実行時チェック用にグローバル変数として定義） ===
smtp_user = os.getenv("SMTP_USER")
smtp_password = os.getenv("SMTP_PASS")  # .env.newに合わせてSMTP_PASSに変更
email_to = os.getenv("TO_EMAIL")  # .env.newに合わせてTO_EMAILに変更
smtp_host = os.getenv("SMTP_HOST")  # キー名を取得する変数を smtp_host に変更
# モジュールレベルでのチェックを削除（実行時にチェックするように変更）

subject = os.getenv("SUBJECT", "📬 通知")

# === SMTP_PORT の安全な読み込み ===
# デフォルトを 465 (SMTPS) にしておきます。環境変数があればそれを使い、整数変換に失敗したら 465 にフォールバックします。
port_str = os.getenv("SMTP_PORT", "465")
try:
    smtp_port = int(port_str)
except Exception:
    smtp_port = 465

# === メール送信関数 ===
def send_notification(smtp_host, smtp_port, smtp_user, smtp_password, to, subject, body):
    """安全化したメール送信ラッパー。
    - DRY_RUN のときは送信をスキップする。
    - SMTP ホスト/宛先が未設定のときは送信をスキップする。
    - 接続タイムアウトを短くしてブロックを避ける。
    Returns True on success, False otherwise.
    """
    from email.mime.text import MIMEText
    import smtplib
    import os

    # DRY_RUN のときは送信をスキップ (成功扱いにすることで通知ループを防ぐ)
    if str(os.getenv('DRY_RUN', '0')).lower() in ('1', 'true', 'yes', 'on'):
        print('ℹ️ DRY_RUN が有効のためメール送信をスキップします')
        return True

    # 必須情報が無ければ送信をスキップ
    if not smtp_host or not to:
        print('ℹ️ SMTP ホストまたは宛先が未設定のためメール送信をスキップします')
        return False

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = smtp_user or ''
    msg["To"] = to

    # SMTPS 判定（環境変数が無ければポート465を SMTPS と判断）
    use_ssl_env = os.getenv("SMTP_USE_SSL")
    if use_ssl_env is None:
        try:
            use_ssl = (int(smtp_port) == 465)
        except Exception:
            use_ssl = True
    else:
        use_ssl = str(use_ssl_env).lower() in ("1", "true", "yes", "on")

    # 接続タイムアウト（秒）を短めにする
    try:
        timeout_sec = float(os.getenv('SMTP_CONNECT_TIMEOUT', '10'))
    except Exception:
        timeout_sec = 10.0

    try:
        if use_ssl:
            with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=timeout_sec) as server:
                if smtp_user and smtp_password:
                    try:
                        server.login(smtp_user, smtp_password)
                    except Exception as e:
                        print(f'⚠️ SMTP 認証失敗: {e}')
                server.send_message(msg)
        else:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=timeout_sec) as server:
                try:
                    server.starttls()
                except Exception:
                    # StartTLS が使えない環境でもログは残す
                    pass
                if smtp_user and smtp_password:
                    try:
                        server.login(smtp_user, smtp_password)
                    except Exception as e:
                        print(f'⚠️ SMTP 認証失敗: {e}')
                server.send_message(msg)

        print("✅ メール送信成功")
        return True

    except Exception as e:
        # 失敗しても Bot の実行を止めない（ログを残すだけ）
        print(f"❌ メール送信失敗: {e}")
        return False


# 取引所の設定を取得
exchange_name = os.getenv("EXCHANGE", "bitbank")


# === メイン処理開始（Botの心臓が動き出す） ===
if __name__ == "__main__":
    print("Bot起動中...")
    # run_botの定義後に呼び出すように移動しました

# 1. 初期設定と認証 (APIキーの読み込みはここにあります)

# .envファイルからAPIキーを読み込みます（config.envから統合済み）

 
# ※注意: APIキー読み込みと bitbank インスタンスの直接作成は
# connect_to_bitbank() に統合しました。元の直接作成コードを削除しています。
# 必要であれば、環境変数の確認は connect_to_bitbank() 呼び出し時に行われます。

# 旧来の直接接続テスト/監視ループは削除しました。
# 取引所接続とループは connect_to_bitbank() と run_bot() に統合されています。


# ==========================================================
# 🔑 2. グローバルキー読み込みと定義 (修正点: 最上部に移動)
# ==========================================================
#.env# config.envからAPIキーを読み込みます

load_dotenv(dotenv_path='.env') 
API_KEY = os.getenv("API_KEY") # グローバル定数として定義
SECRET_KEY = os.getenv("SECRET_KEY") # グローバル定数として定義

# 日本標準時 (JST) のタイムゾーンオブジェクトを作成
try:
    JST = ZoneInfo('Asia/Tokyo')
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
    """軽量な取引所スタブ: dry-run 用。実ネットワーク呼び出しを行いません。"""
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
        return []

    def create_order(self, pair, type_, side, amount, price=None):
        cost = None
        try:
            p = float(price) if price is not None else float(self._price)
            cost = float(amount) * p
        except Exception:
            cost = None
        return {'id': 'dry_order', 'pair': pair, 'type': type_, 'side': side, 'amount': amount, 'price': price, 'cost': cost}


def test_fund_adapter():
    """Quick smoke test for FundAdapter/_adapt_fund_manager_instance.
    Prints expected behavior for reserve/confirm/release in DRY_RUN and live modes.
    """
    print("--- FundAdapter smoke test ---")
    # live-like stub
    fm = FundManager(initial_fund=2000, state_file=os.getenv('FUND_STATE_FILE', 'funds_state.json'))
    adapter = _adapt_fund_manager_instance(fm)
    print("initial available (live stub):", adapter.available_fund())
    cost = 500
    ok = adapter.reserve(cost) if hasattr(adapter, 'reserve') else adapter.place_order(cost)
    print(f"reserve/place_order({cost}) ->", ok)
    print("available after reserve:", adapter.available_fund())
    # attempt release (refund)
    if hasattr(adapter, 'release'):
        adapter.release(cost)
        print("after release available:", adapter.available_fund())
    else:
        # fallback: add_funds used as refund
        adapter.add_funds(cost)
        print("after add_funds available:", adapter.available_fund())

    # Dry-run adapter
    try:
        da = FundAdapter(fund_manager=None, initial_fund=1000.0, dry_run=True)
        print("dry initial available:", da.available_fund())
        ok2 = da.reserve(300)
        print("dry reserve(300) ->", ok2, "available->", da.available_fund())
        da.confirm(300)
        print("dry confirm(300) -> available->", da.available_fund())
    except Exception as e:
        print("dry adapter test failed:", e)


# === 1. 取引所への接続 ===
# 修正点: グローバルキーを使用するため引数を削除し、冗長なコードを削除
def connect_to_bitbank():
    """bitbankに接続します。グローバルで読み込んだAPIキーを使用します。"""
    try:
        # dry-run が有効な場合はネットワークを使わないスタブを返す
        if str(os.getenv('DRY_RUN', '0')).lower() in ('1', 'true', 'yes', 'on'):
            price = os.getenv('DRY_RUN_PRICE')
            try:
                price = float(price) if price is not None else None
            except Exception:
                price = None
            print("🔧 DRY_RUN enabled — using ExchangeStub (no network calls).")
            return ExchangeStub(price)

        # API_KEYとSECRET_KEYはファイルの最上部で既に読み込まれている
        if not API_KEY or not SECRET_KEY:
            print("エラー：APIキーまたはシークレットキーが未定義です。config.envを確認してください。")
            return None

        # ccxtを使ってbitbankに接続
        exchange = ccxt.bitbank({
            'apiKey': API_KEY,
            'secret': SECRET_KEY,
        })
        print("✅ bitbankにccxtで認証接続しました。")
        return exchange

    except Exception as e:
        print(f"❌ bitbankへの接続中にエラーが発生しました: {e}")
        return None
     
        print("✅ bitbankにccxtで認証接続しました。")
        return exchange    

# === 2. 価格データの取得 ===
def get_ohlcv(exchange, pair='BTC/JPY', timeframe='1h', limit=250):
    """
    指定した通貨ペアのOHLCVデータを取得します。(ccxt使用)
    """
    try:
        ohlcv_data = exchange.fetch_ohlcv(pair, timeframe, limit=limit)

        if ohlcv_data:
            # データをDataFrameに変換
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            return df
        else:
            print(f"{pair} のOHLCVデータを取得できませんでした。")
            return None

    except Exception as e:
        print(f"OHLCVデータの取得中にエラーが発生しました: {e}")
        return None


def get_latest_price(exchange, pair='BTC/JPY', retries=3, backoff=1.0):
    """取引所から最新の価格 (last) を取得するユーティリティ。
    - DRY_RUN が有効な場合は環境変数 DRY_RUN_PRICE を返す。
    - ネットワークエラーが起きた場合はリトライ（指数バックオフ）する。
    - 成功時は float の価格、取得不可なら None を返す。
    """
    try:
        if str(os.getenv('DRY_RUN', '0')).lower() in ('1', 'true', 'yes', 'on'):
            try:
                return float(os.getenv('DRY_RUN_PRICE', str(DRY_RUN_PRICE)))
            except Exception:
                return float(DRY_RUN_PRICE)
    except Exception:
        pass

    attempt = 0
    while attempt < retries:
        try:
            if exchange is None:
                # ネット接続無しのテスト環境等では DRY_RUN_PRICE を返す
                try:
                    return float(os.getenv('DRY_RUN_PRICE', str(DRY_RUN_PRICE)))
                except Exception:
                    return float(DRY_RUN_PRICE)

            ticker = exchange.fetch_ticker(pair)
            if isinstance(ticker, dict) and 'last' in ticker:
                try:
                    return float(ticker.get('last'))
                except Exception:
                    return None
            # ccxt の一部実装はオブジェクトや異なる形で返す可能性がある
            # 呼び出し側で安全に扱える形で None を返す
            return None

        except Exception as e:
            attempt += 1
            print(f"⚠️ 価格取得失敗（試行 {attempt}/{retries}）: {e}")
            if attempt >= retries:
                break
            sleep_sec = backoff * (2 ** (attempt - 1))
            try:
                time.sleep(sleep_sec)
            except Exception:
                pass

    return None


def compute_dynamic_threshold(exchange, pair='BTC/JPY', days=DYN_OHLCV_DAYS,
                              buffer_jpy=DYN_THRESHOLD_BUFFER_JPY, buffer_pct=DYN_THRESHOLD_BUFFER_PCT):
    """過去 `days` 日の OHLCV を取得し、1年レンジの最安値を基に閾値を算出します。
    戻り値: (threshold_jpy, min_jpy, max_jpy) または (None, None, None) on failure
    算出方法:
      - 可能なら固定バッファ (buffer_jpy) を優先して閾値 = min_close + buffer_jpy
      - buffer_jpy が <= 0 のときは buffer_pct を使って閾値 = min_close * (1 + buffer_pct)
    """
    try:
        # get_ohlcv は timeframe='1d' で日次データを返す
        df = get_ohlcv(exchange, pair, timeframe='1d', limit=max(10, int(days)))
        if df is None or len(df) == 0:
            return None, None, None
        # close 列の最小・最大
        try:
            closes = [float(v) for v in df['close'] if v is not None]
        except Exception:
            # データフレーム互換処理
            closes = []
            for i in range(len(df)):
                try:
                    closes.append(float(df.iloc[i]['close']))
                except Exception:
                    pass
        if not closes:
            return None, None, None
        min_close = min(closes)
        max_close = max(closes)
        # 優先順位:
        # 1) ratio が指定されている（>0）なら、1年レンジの比率で閾値を決定
        # 2) buffer_jpy が指定されている (>0) なら最安値 + 固定円で閾値
        # 3) それ以外は割合バッファで閾値
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
        print(f"⚠️ dynamic threshold computation failed: {e}")
        return None, None, None


def compute_sma_from_ohlcv(exchange, pair='BTC/JPY', days=30):
    """日次 OHLCV から単純移動平均 (SMA) を計算して返す。失敗時は None を返す。"""
    try:
        df = get_ohlcv(exchange, pair, timeframe='1d', limit=max(10, days + 5))
        if df is None or len(df) < days:
            return None
        vals = []
        try:
            for i in range(-days, 0):
                vals.append(float(df['close'].iloc[i]))
            return sum(vals) / len(vals) if vals else None
        except Exception:
            # DataFrame 互換ループ
            vals = []
            for i in range(len(df) - days, len(df)):
                try:
                    vals.append(float(df.iloc[i]['close']))
                except Exception:
                    pass
            return sum(vals) / len(vals) if vals else None
    except Exception:
        return None


def get_recent_high(exchange, pair='BTC/JPY', days=30):
    """直近 days 日の high の最大値を返す。失敗時は None を返す。"""
    try:
        df = get_ohlcv(exchange, pair, timeframe='1d', limit=max(10, days + 5))
        if df is None or len(df) == 0:
            return None
        try:
            highs = [float(v) for v in df['high'] if v is not None]
            return max(highs) if highs else None
        except Exception:
            hs = []
            for i in range(len(df)):
                try:
                    hs.append(float(df.iloc[i]['high']))
                except Exception:
                    pass
            return max(hs) if hs else None
    except Exception:
        return None


# === 指標計算ユーティリティ (SMA/EMA/ATR/RSI) ===
def compute_sma(values, period):
    """Simple moving average for last `period` values. Returns None if insufficient data."""
    try:
        if values is None:
            return None
        vals = [float(v) for v in values if v is not None]
        if len(vals) < period or period <= 0:
            return None
        return sum(vals[-period:]) / float(period)
    except Exception:
        return None


def compute_ema(values, period):
    """Exponential moving average for last `period` values. Returns None if insufficient data."""
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
    """Compute ATR (Average True Range) from OHLCV rows (list of [ts, o, h, l, c, v] or DataFrame-like).
    Returns ATR float or None."""
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
    """Compute RSI from list of close prices. Returns float or None."""
    try:
        if values is None:
            return None
        vals = [float(v) for v in values if v is not None]
        if len(vals) < period + 1:
            return None
        gains = []
        losses = []
        for i in range(1, len(vals)):
            diff = vals[i] - vals[i - 1]
            if diff > 0:
                gains.append(diff)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(abs(diff))
        # use Wilder's smoothing (simple average for first)
        avg_gain = sum(gains[-period:]) / float(period)
        avg_loss = sum(losses[-period:]) / float(period)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi
    except Exception:
        return None


def compute_indicators(exchange, pair='BTC/JPY', timeframe='1h', limit=500):
    """Fetch OHLCV and compute a set of indicators. Returns dict of values (may contain None)."""
    try:
        ohlcv = get_ohlcv(exchange, pair, timeframe=timeframe, limit=limit)
        indicators = {
            'sma_short_50': None,
            'sma_long_200': None,
            'ema_12': None,
            'ema_26': None,
            'atr_14': None,
            'rsi_14': None,
            'recent_high_20': None,
            'latest_close': None,
        }

        # If get_ohlcv returned None, try to fallback to exchange.fetch_ohlcv directly
        raw = None
        if ohlcv is None:
            try:
                raw = exchange.fetch_ohlcv(pair, timeframe, limit=limit)
            except Exception:
                raw = None
        else:
            # convert DataFrame-like to list of rows
            try:
                closes = [float(v) for v in ohlcv['close'] if v is not None]
                highs = [float(v) for v in ohlcv['high'] if v is not None]
                lows = [float(v) for v in ohlcv['low'] if v is not None]
                raw = []
                for i in range(len(closes)):
                    # timestamp may be index
                    try:
                        ts = ohlcv.index[i]
                        raw.append([ts, ohlcv['open'].iloc[i], highs[i], lows[i], closes[i], ohlcv['volume'].iloc[i]])
                    except Exception:
                        raw.append([None, None, highs[i], lows[i], closes[i], None])
            except Exception:
                raw = None
        if raw is None:
            return indicators

        # prepare lists
        closes = [float(r[4]) for r in raw if r and len(r) >= 5 and r[4] is not None]
        highs = [float(r[2]) for r in raw if r and len(r) >= 3 and r[2] is not None]
        lows = [float(r[3]) for r in raw if r and len(r) >= 4 and r[3] is not None]

        indicators['latest_close'] = closes[-1] if closes else None
        indicators['sma_short_50'] = compute_sma(closes, 50)
        indicators['sma_long_200'] = compute_sma(closes, 200)
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
            'latest_close': None,
        }


def write_indicators_csv(indicators: dict, pair: str, signal: str = 'NONE', csv_path='indicators.csv'):
    """Append indicators as a CSV row. Creates header if file does not exist."""
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
def round_down_qty(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    factor = 1.0 / step
    return math.floor(qty * factor) / factor


def compute_qty_for_budget_with_fee(reserved_jpy: float, price_jpy: float,
                                    min_btc: float = 0.0001, step: float = 0.0001,
                                    available_jpy: float = None, balance_buffer: float = 0.0):
    """
    返り値: (qty, cost_jpy, fee_jpy)
    qty: 注文数量（step に合わせて丸め）。0 の場合は注文不可。
    cost_jpy: qty * price_jpy
    fee_jpy: cost_jpy * FEE_RATE + FEE_FIXED_JPY
    """
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
            alt_max_jpy = min(reserved_jpy * max_mult, (float(available_jpy) - float(balance_buffer)) if available_jpy is not None else reserved_jpy * max_mult)
            if alt_max_jpy > 0:
                approx_qty = alt_max_jpy / (price_jpy * (1.0 + fee_rate))
                alt_qty = round_down_qty(approx_qty, step)
                cost = alt_qty * price_jpy
                fee = cost * fee_rate + fee_fixed
                if alt_qty >= min_btc and (cost + fee) <= alt_max_jpy:
                    return alt_qty, cost, fee
        return 0.0, 0.0, 0.0

    # fee を計算して合計が上限を超えないか確認
    cost = qty * price_jpy
    fee = cost * fee_rate + fee_fixed
    if (cost + fee) <= max_allowed_jpy:
        return qty, cost, fee

    # 少しずつ qty を減らしてチェック
    while qty >= min_btc:
        cost = qty * price_jpy
        fee = cost * fee_rate + fee_fixed
        if (cost + fee) <= max_allowed_jpy:
            return qty, cost, fee
        qty = round_down_qty(qty - step, step)

    return 0.0, 0.0, 0.0


def is_trade_day(now=None):
    # now は timezone-aware datetime を想定
    if now is None:
        now = datetime.datetime.now()
    forced = os.getenv('FORCE_TRADE_DAY')
    if forced:
        allowed = [d.strip().lower() for d in forced.split(',') if d.strip()]
        dow = now.strftime('%a').lower()[:3]
        return dow in allowed
    # デフォルトで土日限定にする（環境変数で上書き可）
    if os.getenv('TRADE_ONLY_WEEKENDS', '1') == '1':
        return now.weekday() in (5, 6)
    return True


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
                print(f"DEBUG: save_state succeeded and replaced {STATE_FILE} (size={STATE_FILE.stat().st_size})")
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
                    print(f"DEBUG: save_state fallback non-atomic write succeeded for {STATE_FILE}")
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
            print("WARN: could not save state:", e)
            errfile = STATE_FILE.with_name(STATE_FILE.name + '.save_error.log')
            errfile.write_text(''.join(traceback.format_exception(type(e), e, e.__traceback__)), encoding='utf-8')
        except Exception:
            try:
                print("WARN: could not save state and failed to write error log")
            except Exception:
                pass


def get_last_buy_time(state):
    return state.get("last_buy_time")


def set_last_buy_time(state, ts=None):
    state["last_buy_time"] = ts or int(time.time())
    save_state(state)


def record_position(state, side, price, qty):
    try:
        # Use the same ORDER_LOCKFILE as other parts of the code to serialize
        # state updates (buy reservation, record_position, sell flow).
        lockfile = os.getenv('ORDER_LOCKFILE')
        if not lockfile:
            try:
                lockfile = str(STATE_FILE.with_name('.ninibo_order.lock'))
            except Exception:
                lockfile = os.path.join(os.getcwd(), '.ninibo_order.lock')
        lock_timeout = float(os.getenv('ORDER_LOCK_TIMEOUT', '10'))
        with FileLock(lockfile, timeout=lock_timeout):
            # Read latest on-disk state to avoid overwriting concurrent updates
            disk_state = load_state()
            if not isinstance(disk_state, dict):
                disk_state = {}
            disk_state.setdefault("positions", [])
            disk_state["positions"].append({
                "side": side,
                "price": float(price),
                "qty": float(qty),
                "time": int(time.time())
            })
            if len(disk_state["positions"]) > 50:
                disk_state["positions"] = disk_state["positions"][-50:]
            # persist atomically
            save_state(disk_state)
            # update provided in-memory state object for caller convenience
            try:
                if isinstance(state, dict):
                    state["positions"] = disk_state["positions"]
            except Exception:
                pass
    except Exception:
        # fallback to original behavior if something goes wrong
        try:
            state.setdefault("positions", [])
            state["positions"].append({
                "side": side,
                "price": float(price),
                "qty": float(qty),
                "time": int(time.time())
            })
            if len(state["positions"]) > 50:
                state["positions"] = state["positions"][-50:]
            save_state(state)
        except Exception:
            pass


def is_slippage_too_large(reference_price, latest_price):
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
    """
    価格データに基づいて売買シグナルを生成します。
    """
    # データ数が200本必要
    if df is None or len(df) < 200:
        # エラーメッセージを改善
        print(f"⚠️ データが不足しています。最低200本必要ですが、{len(df) if df is not None else 0}本しかありません。")
        return None

    # 短期25、中期75、長期200を追加
    df['short_mavg'] = df['close'].rolling(window=25).mean()
    df['mid_mavg'] = df['close'].rolling(window=75).mean() # 75をmidに名称変更
    df['long_mavg'] = df['close'].rolling(window=200).mean() # 新しい長期MA

    latest_data = df.iloc[-1]
    previous_data = df.iloc[-2]

    signal = None
    message = None

    # 🔑 トレンドフィルター
    is_uptrend = latest_data['mid_mavg'] > latest_data['long_mavg']
    mid_mavg_is_rising = latest_data['mid_mavg'] > previous_data['mid_mavg']

    # --- 買いシグナル 1：新規エントリー (ゴールデンクロス) ---
    if (previous_data['short_mavg'] <= previous_data['mid_mavg'] and
        latest_data['short_mavg'] > latest_data['mid_mavg'] and
        is_uptrend and mid_mavg_is_rising):
        signal = 'buy_entry' # 新規エントリーシグナル
        message = "✅ 新規エントリーシグナル (GC 25/75、トレンド確認) が発生しました。"
        return signal, message

    # --- 買いシグナル 2：買い増し (押し目) ---
    # 注: GC後、ポジション保有中に価格がMA25を上回っている（押し目買い）でトレンド上昇中
    elif latest_data['close'] > latest_data['short_mavg'] and is_uptrend:
        signal = 'buy_add' 
        message =  "📈 買い増しシグナル (押し目買い) が発生しました。"
        
    # --- 売りシグナル：全決済 (トレンド終了) ---
    # MA75がMA200を下回った、またはMA75が下向きに転じた
    elif not is_uptrend or latest_data['mid_mavg'] < previous_data['mid_mavg']:
        signal = 'sell_all'
        message = "❌ 全決済シグナル (長期トレンド終了/反転) が発生しました。"
    
    return signal, message


# === 4. 注文の整形 ===

def log_order(action, pair, amount, price=None):
    """
  注文内容を整形してログメッセージを返します。
    """
    msg = f"{action}注文: {amount:.4f} {pair.split('/')[0]} {'@ ' + str(price) if price else '（成行）'}"
    print(msg)
    return msg

# === 5. 注文の実行 ===

def execute_order(exchange, pair, order_type, amount, price=None):
    """
    Bitbankに注文を出します。(ccxt使用)
    """
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
            print("ℹ️ DRY_RUN: 注文は実行されませんでした（シミュレーション）")
            return simulated

        if order_type == 'buy':
            if price:
                # 指定価格で成行ではなく指値注文を出す
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
            print(f"無効な注文タイプです: {order_type}")
            return None

        if order and isinstance(order, dict) and 'id' in order:
            print("注文成功:", order['id'])  # IDのみ表示に修正
            return order
        else:
            print("注文に失敗しました:", order)
            return None

    except Exception as e:
        import traceback
        traceback.print_exc()  # ← これでエラーの詳細が表示されます

        print(f"❌ 注文実行中にエラーが発生しました: {e}")
        return None

# === 6. メインループ（Botの実行部分） ===
# Small helper: in DRY_RUN or when AUTO_FIX_FUNDS is enabled, ensure FundManager has a reasonable balance
def _ensure_fund_manager_has_funds(fm, initial_amount=None):
    """If fm.available_fund() is zero or unavailable, optionally top-up using add_funds().

    This is intentionally conservative: it only performs the auto-fix when DRY_RUN is
    enabled or when AUTO_FIX_FUNDS environment variable is set. It helps tests and
    local DRY_RUN runs where state files may be missing or zeroed.
    """
    # Auto-fix funds is now opt-in via AUTO_FIX_FUNDS. This avoids silently
    # modifying funds during regular DRY_RUNs — operator must explicitly enable it.
    try:
        auto_fix = str(os.getenv('AUTO_FIX_FUNDS', '')).lower() in ('1', 'true', 'yes', 'on')
    except Exception:
        auto_fix = False

    if not auto_fix:
        return

    try:
        avail = float(fm.available_fund())
    except Exception:
        avail = 0.0

    if avail and avail > 0:
        return

    # determine top-up amount; allow override via AUTO_FIX_AMOUNT (JPY)
    try:
        if initial_amount is not None:
            initial = float(initial_amount)
        else:
            initial = float(os.getenv('AUTO_FIX_AMOUNT', os.getenv('INITIAL_FUND', '20000')))
    except Exception:
        initial = 20000.0

    try:
        fm.add_funds(initial)
        print(f"🔧 funds were zero; auto-added {initial:.0f} JPY to fund_manager (AUTO_FIX_FUNDS)")
    except Exception as e:
        print(f"⚠️ failed to auto-fix fund_manager funds: {e}")

# 修正点: グローバルキーを使用するため、api_keyとsecret_keyの引数を削除
def run_bot(exchange, fund_manager_instance):
    """自動売買Botのメイン実行（1回分を実行する）。

    この実装は次を満たします:
    - 注文を出す前に最新残高を確認する
    - 注文が成功したときのみ fund_manager.place_order(cost) により残高を差し引く
    - DRY_RUN を尊重して、実運用時のみ API 呼び出しを行う
    """
    pair = 'BTC/JPY'
    interval_seconds = 3600

    # DEBUG: run_bot entry
    try:
        print(f"DEBUG: run_bot start - DRY_RUN={DRY_RUN}, pair={pair}, JAPANESE_YEN_BUDGET={JAPANESE_YEN_BUDGET}")
    except Exception:
        print("DEBUG: run_bot start (print failed)")

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
    if exchange is None:
        exchange = connect_to_bitbank()
    # adapter を通して安全に扱えるようにする
    fund_manager = _adapt_fund_manager_instance(fund_manager_instance) if fund_manager_instance is not None else _adapt_fund_manager_instance(FundManager(initial_fund=float(os.getenv('INITIAL_FUND', '20000')), state_file=os.getenv('FUND_STATE_FILE', 'funds_state.json')))

    if not exchange and not env_dry_run:
        print("API接続に失敗したためBotを停止します。")
        return

    print(f"Botを {pair} で実行します。データ取得間隔: {interval_seconds}秒 (1時間)")

    # 1回あたりの注文予算（JPY）。ユーザー指定が無ければ 10000 円に変更
    JAPANESE_YEN_BUDGET = float(os.getenv('JAPANESE_YEN_BUDGET', '10000'))
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
        BALANCE_BUFFER = float(os.getenv('BALANCE_BUFFER', '1000'))
    except Exception:
        BALANCE_BUFFER = 1000.0

    print(f"💰 1回あたりの注文予算: {JAPANESE_YEN_BUDGET} 円")
    print(f"📉 最低注文数量: {MIN_ORDER_BTC} BTC")


    # state を読み込み、保有ポジションがあれば利確チェックを行う
    # --- 低残高アラート設定 ---
    LOW_FUNDS_ALERT_JPY = float(os.getenv('LOW_FUNDS_ALERT_JPY', '2000'))

    # state を読み込み、保有ポジションがあれば利確チェックを行う
    try:
        # Use a file lock when reading/modifying/saving state for sell flow to avoid
        # races with concurrent buy operations that also update the state file.
        LOCKFILE_SELL = os.getenv('ORDER_LOCKFILE', '/tmp/ninibo_order.lock')
        with FileLock(LOCKFILE_SELL, timeout=10):
            state = load_state()
            positions = state.get('positions', []) if isinstance(state, dict) else []
            if positions:
                # Find the most recent BUY position that has a valid (non-zero) price.
                last_pos = None
                for p in reversed(positions):
                    try:
                        if p.get('side') == 'buy' and float(p.get('price', 0) or 0) > 0:
                            last_pos = p
                            break
                    except Exception:
                        continue

                if last_pos is not None:
                    entry_price = float(last_pos.get('price', 0))
                    entry_qty = float(last_pos.get('qty', 0))
                    lp = get_latest_price(exchange, pair)
                    if lp is not None:
                        # TRADE_TRIGGER_PCT を使って利確（デフォルト 20%）
                        gain_pct = (float(lp) - entry_price) / float(entry_price) * 100.0 if entry_price and entry_price > 0 else 0.0
                        if gain_pct >= float(TRADE_TRIGGER_PCT):
                            print(f"INFO: Trigger sell: gain={gain_pct:.2f}% >= {TRADE_TRIGGER_PCT}% -> selling {entry_qty} at {lp}")
                            sell_order = execute_order(exchange, pair, 'sell', entry_qty)
                            try:
                                print(f"DEBUG: post-execute_order sell_order={sell_order}")
                            except Exception:
                                pass
                            if sell_order and isinstance(sell_order, dict) and 'id' in sell_order:
                                # 売却成功: remove the sold position (the last valid one) and save state
                                try:
                                    removed = False
                                    for i in range(len(positions)-1, -1, -1):
                                        try:
                                            p = positions[i]
                                            if p.get('side') == 'buy' and float(p.get('price', 0) or 0) == float(entry_price) and float(p.get('qty', 0) or 0) == float(entry_qty):
                                                try:
                                                    print(f"DEBUG: removing position at index={i} -> {p}")
                                                except Exception:
                                                    pass
                                                # perform deletion
                                                del positions[i]
                                                removed = True
                                                # Immediately persist a tiny marker and an in-memory snapshot
                                                try:
                                                    marker = STATE_FILE.with_name(STATE_FILE.name + f'.after_sell.marker')
                                                    with open(str(marker), 'wb') as mf:
                                                        mf.write(b'REACHED_AFTER_SELL')
                                                    dbg_path = STATE_FILE.with_name(STATE_FILE.name + f'.after_sell.immediate.json')
                                                    dbg_path.write_text(json.dumps({'positions': positions, 'watch_reference': state.get('watch_reference')}, ensure_ascii=False, indent=2), encoding='utf-8')
                                                    try:
                                                        print(f"DEBUG: immediate marker and snapshot written: {marker}, {dbg_path}")
                                                    except Exception:
                                                        pass
                                                except Exception as e_immediate:
                                                    try:
                                                        print(f"DEBUG: failed immediate marker/snapshot write: {e_immediate}")
                                                    except Exception:
                                                        pass
                                                break
                                        except Exception as e_rem:
                                            try:
                                                print(f"DEBUG: exception while scanning positions for removal: {e_rem}")
                                            except Exception:
                                                pass
                                            continue
                                    if not removed:
                                        try:
                                            print("DEBUG: no exact matching position found to remove; will attempt to pop last element")
                                        except Exception:
                                            pass
                                        try:
                                            positions = positions[:-1]
                                        except Exception as e_pop:
                                            try:
                                                print(f"DEBUG: failed to pop last position: {e_pop}")
                                            except Exception:
                                                pass
                                    state['positions'] = positions
                                except Exception as e_state:
                                    try:
                                        print(f"DEBUG: exception while removing position: {e_state}")
                                    except Exception:
                                        pass
                                    # fallback: pop the last element
                                    try:
                                        state['positions'] = positions[:-1]
                                    except Exception:
                                        pass
                                # 売却後は監視基準価格を最新価格にリセット
                                try:
                                    state['watch_reference'] = float(lp)
                                except Exception:
                                    pass
                                # 売却成功時に売却代金をファンドへ戻す（実運用／DRY_RUN に対して適切な API を呼ぶ）
                                try:
                                    sell_proceeds = None
                                    if isinstance(sell_order, dict):
                                        sell_proceeds = sell_order.get('cost')
                                    if not sell_proceeds:
                                        # フォールバック: 最新価格 * 数量
                                        try:
                                            sell_proceeds = float(entry_qty) * float(lp)
                                        except Exception:
                                            sell_proceeds = None
                                    if sell_proceeds is not None:
                                        try:
                                            lock_timeout_local = float(os.getenv('ORDER_LOCK_TIMEOUT', '10'))
                                            with FileLock(LOCKFILE_SELL, timeout=lock_timeout_local):
                                                if hasattr(fund_manager, 'add_funds'):
                                                    fund_manager.add_funds(float(sell_proceeds))
                                                else:
                                                    # もし underlying が reservation-style を持たない場合は警告
                                                    try:
                                                        print("⚠️ fund_manager に add_funds メソッドがありません。手動で残高調整が必要です。")
                                                    except Exception:
                                                        pass
                                        except Exception as e_add:
                                            try:
                                                print(f"⚠️ 売却代金のファンド加算に失敗しました: {e_add}")
                                            except Exception:
                                                pass
                                except Exception:
                                    pass
                                # 保存前に内容をデバッグ出力
                                try:
                                    print(f"DEBUG: saving state after sell: watch_reference={state.get('watch_reference')} positions_count={len(state.get('positions', []))}")
                                except Exception:
                                    pass
                                # Immediately create a lightweight marker file to prove we reached
                                # this point. Use binary write to avoid encoding surprises.
                                try:
                                    marker = STATE_FILE.with_name(STATE_FILE.name + '.after_sell.marker')
                                    with open(str(marker), 'wb') as mf:
                                        mf.write(b'REACHED_AFTER_SELL')
                                    try:
                                        print(f"DEBUG: wrote marker file {marker}")
                                    except Exception:
                                        pass
                                except Exception as e_marker:
                                    try:
                                        print(f"DEBUG: failed to write marker file: {e_marker}")
                                    except Exception:
                                        pass
                                # Immediately dump an auxiliary debug file to make the in-memory
                                # state observable even if save_state fails or gets overwritten.
                                try:
                                    dbg_path = STATE_FILE.with_name(STATE_FILE.name + '.after_sell.debug.json')
                                    dbg_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
                                    try:
                                        print(f"DEBUG: wrote debug snapshot to {dbg_path}")
                                    except Exception:
                                        pass
                                except Exception as e_dbg:
                                    try:
                                        print(f"DEBUG: failed to write debug snapshot: {e_dbg}")
                                    except Exception:
                                        pass
                                save_state(state)
                                try:
                                    if STATE_FILE.exists():
                                        txt = STATE_FILE.read_text(encoding='utf-8')
                                        try:
                                            print(f"DEBUG: post-save STATE_FILE len={len(txt)}")
                                        except Exception:
                                            pass
                                except Exception:
                                    try:
                                        print(f"DEBUG: could not read state file after save")
                                    except Exception:
                                        pass

                                # Ensure proceeds are persisted to underlying fund file in DRY_RUN tests too
                                try:
                                    # primary: adapter-level add_funds (may be local in DRY_RUN)
                                    if sell_proceeds is not None:
                                        try:
                                            fund_manager.add_funds(float(sell_proceeds))
                                        except Exception:
                                            pass
                                    # fallback: if adapter wraps an underlying object that supports add_funds, call it to persist
                                    underlying = getattr(fund_manager, '_underlying', None)
                                    if underlying is not None and hasattr(underlying, 'add_funds'):
                                        try:
                                            # write under lock to avoid races
                                            lockfile_main = os.getenv('ORDER_LOCKFILE') or str(STATE_FILE.with_name('.ninibo_order.lock'))
                                            lock_timeout_local = float(os.getenv('ORDER_LOCK_TIMEOUT', '10'))
                                            with FileLock(lockfile_main, timeout=lock_timeout_local):
                                                underlying.add_funds(float(sell_proceeds))
                                        except Exception:
                                            pass
                                except Exception:
                                    pass

                                print("✅ Trigger sell: position sold and state updated")
                                # Exit this run after successful sell to avoid later logic
                                # (e.g. buy path) overwriting the updated state file.
                                return
                                # After a successful sell and state persist, return early to avoid subsequent
                                # buy logic in the same run from overwriting the state file.
                                return
    except Exception as e:
        print(f"⚠️ 利確チェック中にエラー: {e}")

    try:
        # 取引許可日のチェック (週末限定など)
        tz_name = os.getenv('TRADE_TIMEZONE')
        if tz_name:
            now = datetime.datetime.now(ZoneInfo(tz_name))
        else:
            now = datetime.datetime.now(JST)
        if not is_trade_day(now):
            print(f"取引制限: 本日は取引不可です（{now.strftime('%a %Y-%m-%d %H:%M:%S')}）。スキップします。")
            return

        latest_price = get_latest_price(exchange, pair)
        if latest_price is None:
            print("エラー: 価格が取得できませんでした。")
            return
        # 動的閾値を使う場合は1年レンジ等から閾値を計算する
        effective_threshold = float(MIN_PRICE_THRESHOLD_JPY)
        if USE_DYNAMIC_THRESHOLD:
            thr, min1y, max1y = compute_dynamic_threshold(exchange, pair, days=DYN_OHLCV_DAYS,
                                                          buffer_jpy=DYN_THRESHOLD_BUFFER_JPY,
                                                          buffer_pct=DYN_THRESHOLD_BUFFER_PCT)
            if thr is not None:
                effective_threshold = float(thr)
                print(f"🔎 dynamic threshold computed: threshold={effective_threshold}, 1y_min={min1y}, 1y_max={max1y}")
            else:
                print("⚠️ dynamic threshold could not be computed; using static MIN_PRICE_THRESHOLD_JPY")

        print(f"💵 最新の市場価格: {latest_price} 円 (buy_when_price <= {effective_threshold})")

        # --- 指標を計算してログに出力 ---
        try:
            indicators = compute_indicators(exchange, pair, timeframe='1h', limit=500)
            try:
                print(f"Indicators: price={indicators.get('latest_close')}, sma50={indicators.get('sma_short_50')}, sma200={indicators.get('sma_long_200')}, ema12={indicators.get('ema_12')}, rsi14={indicators.get('rsi_14')}, atr14={indicators.get('atr_14')}")
            except Exception:
                pass
        except Exception:
            indicators = None

        # 初期表示用に手数料を考慮した数量を算出
        initial_qty, initial_cost, initial_fee = compute_qty_for_budget_with_fee(
            float(JAPANESE_YEN_BUDGET), float(latest_price), min_btc=MIN_ORDER_BTC, step=MIN_ORDER_BTC,
            available_jpy=None, balance_buffer=float(BALANCE_BUFFER)
        )
        if initial_qty <= 0:
            print(f"ℹ️ 注文数量が最小取引単位を下回る（手数料込み）。AUTO_RESIZE={os.getenv('AUTO_RESIZE')} を確認してください。")
            return
        print(f"✅ 注文可能: {initial_qty} BTC (約 {initial_cost:.2f} 円, 手数料: {initial_fee:.2f} 円)")

    except Exception as e:
        print(f"エラー: 最新価格の取得に失敗しました。Botを停止します: {e}")
        return

    # 注文前に最新残高を確認して、不足であればスキップする
    try:
        available = None
        if hasattr(fund_manager, "available_fund"):
            try:
                available = float(fund_manager.available_fund())
            except Exception:
                available = None

        # Note: use initial_cost (fee-aware estimation) for pre-check
        if available is not None and available < initial_cost:
            print(f"🚫 残高不足のため注文をスキップします（必要: {initial_cost:.2f} 円, 残高: {available:.2f} 円）")
            return
    except Exception as e:
        print(f"🚫 残高チェック中にエラーが発生しました: {e}")
        return

    # 注文実行: ここでは「予約 (reserve)」戦略を採用します。
    # 1) ロックを取得して残高を再確認 -> 残高を差し引いて予約
    # 2) ロックを解放して実際の注文を発行
    # 3) 注文失敗時はロック下で返金（予約取り消し）
    LOCKFILE = os.getenv('ORDER_LOCKFILE', '/tmp/ninibo_order.lock')
    LOCK_TIMEOUT = float(os.getenv('ORDER_LOCK_TIMEOUT', '10'))
    reserved = False
    try:
        # 予約フェーズ: 価格変動に対応するため、"予算" を先に予約し
        # 注文直前に最新価格を再取得して数量を再計算します。
        # 小額運用向けに、残高の割合やバッファを尊重して予約額を決める
        reserved_budget = JAPANESE_YEN_BUDGET
        try:
            # available を事前取得（lock の外での読み取りで概算を取る）
            available_pre = float(fund_manager.available_fund()) if hasattr(fund_manager, 'available_fund') else None
        except Exception:
            available_pre = None
        # DEBUG: show pre-reservation estimates (より詳細に出力)
        try:
            allowed_by_percent = max(0.0, available_pre * float(MAX_RISK_PERCENT)) if available_pre is not None else None
            allowed_by_buffer = max(0.0, available_pre - float(BALANCE_BUFFER)) if available_pre is not None else None
            print(f"DEBUG: pre-reservation: available_pre={available_pre}, reserved_budget={reserved_budget}, allowed_by_percent={allowed_by_percent}, allowed_by_buffer={allowed_by_buffer}")
        except Exception:
            pass

        # 低残高アラート: available_pre がある場合に閾値を下回っていたら通知
        try:
            if available_pre is not None and float(available_pre) < float(LOW_FUNDS_ALERT_JPY):
                msg = f"⚠️ 資金アラート: 残高が少なくなっています（残高: {available_pre:.0f} 円 < 閾値: {LOW_FUNDS_ALERT_JPY:.0f} 円）"
                print(msg)
                try:
                    if smtp_host and email_to:
                        send_notification(smtp_host, smtp_port, smtp_user, smtp_password, email_to, subject, msg)
                except Exception:
                    pass
        except Exception:
            pass

        if available_pre is not None:
            # 利用可能残高に対する上限 (割合)
            allowed_by_percent = max(0.0, available_pre * float(MAX_RISK_PERCENT))
            # 残しておく最低バッファを考慮
            allowed_by_buffer = max(0.0, available_pre - float(BALANCE_BUFFER))
            # 実際に予約する金額は、環境変数での予算と上限の小さい方
            reserved_budget = min(float(JAPANESE_YEN_BUDGET), allowed_by_percent, allowed_by_buffer)
            # 小額になりすぎないよう安全下限チェックはロック内で行う
        with FileLock(LOCKFILE, timeout=LOCK_TIMEOUT):
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
                if last_buy and (time.time() - last_buy) < COOLDOWN_SEC:
                    print("Cooldown active -> skipping buy to avoid frequent add-on")
                    return

                q_check, cost_check, fee_check = compute_qty_for_budget_with_fee(
                    reserved_budget, float(latest_price), min_btc=MIN_ORDER_BTC, step=MIN_ORDER_BTC,
                    available_jpy=available, balance_buffer=float(BALANCE_BUFFER)
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
                    fee_fixed = float(os.getenv('FEE_FIXED_JPY', '0.0'))
                except Exception:
                    fee_fixed = 0.0
                # estimate max_allowed_jpy used inside compute_qty_for_budget_with_fee
                try:
                    max_allowed_jpy = min(float(reserved_budget), float(available) - float(BALANCE_BUFFER)) if available is not None else float(reserved_budget)
                except Exception:
                    max_allowed_jpy = float(reserved_budget)
                try:
                    approx_qty = max_allowed_jpy / (reservation_price * (1.0 + fee_rate)) if reservation_price and reservation_price > 0 else 0.0
                except Exception:
                    approx_qty = 0.0
                try:
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

                    # 一部の FundManager 実装は place_order が真/偽を返さず None を返す場合がある。
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
                    if isinstance(state, dict) and state.get('watch_reference') is not None:
                        try:
                            watch_ref = float(state.get('watch_reference'))
                        except Exception:
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

                do_buy_by_pct = False
                try:
                    if latest_price_now is not None and watch_ref is not None:
                        threshold_buy = watch_ref * (1.0 - float(TRADE_TRIGGER_PCT) / 100.0)
                        do_buy_by_pct = float(latest_price_now) <= float(threshold_buy)
                        print(f"DEBUG: watch_ref={watch_ref}, threshold_buy={threshold_buy}, latest={latest_price_now}, do_buy_by_pct={do_buy_by_pct}")
                except Exception:
                    do_buy_by_pct = False

                # optionally still allow breakout buys if configured
                allow_buy_by_breakout = False
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
                    if indicators is None:
                        indicators = compute_indicators(exchange, pair, timeframe='1h', limit=500)
                    write_indicators_csv(indicators if indicators is not None else {}, pair, signal=sig)
                except Exception:
                    pass

                if not do_buy:
                    print(f"🚫 買い条件未達（watch_ref={watch_ref}, latest={latest_price_now}, buy_pct={TRADE_TRIGGER_PCT}, breakout_allowed={allow_buy_by_breakout}）→ 予約を返金して終了")
                    if reserved:
                        with FileLock(LOCKFILE, timeout=LOCK_TIMEOUT):
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
                    with FileLock(LOCKFILE, timeout=LOCK_TIMEOUT):
                        if hasattr(fund_manager, 'release'):
                            fund_manager.release(reserved_budget)
                        elif hasattr(fund_manager, 'add_funds'):
                            fund_manager.add_funds(reserved_budget)
                        reserved = False
                return
            if latest_price_now is None:
                print("⚠️ 注文直前に価格が取得できませんでした。予約を取り消します。")
                # 返金
                if reserved:
                    with FileLock(LOCKFILE, timeout=LOCK_TIMEOUT):
                        if hasattr(fund_manager, 'release'):
                            fund_manager.release(reserved_budget)
                        elif hasattr(fund_manager, 'add_funds'):
                            fund_manager.add_funds(reserved_budget)
                return

            # 最新価格取得後にスリッページ最終判定
            if 'reservation_price' in globals() and reservation_price is not None:
                if is_slippage_too_large(reservation_price, latest_price_now):
                    print(f"Slippage too large (ref={reservation_price}, latest={latest_price_now}) -> cancelling & refund")
                    if reserved:
                        with FileLock(LOCKFILE, timeout=LOCK_TIMEOUT):
                            if hasattr(fund_manager, 'release'):
                                fund_manager.release(reserved_budget)
                            elif hasattr(fund_manager, 'add_funds'):
                                fund_manager.add_funds(reserved_budget)
                    return

            final_qty, final_cost, final_fee = compute_qty_for_budget_with_fee(
                reserved_budget, float(latest_price_now), min_btc=MIN_ORDER_BTC, step=MIN_ORDER_BTC,
                available_jpy=available, balance_buffer=float(BALANCE_BUFFER)
            )

            # 最小数量チェック
            if final_qty <= 0:
                print(f"ℹ️ 注文直前で数量が最小取引単位を下回りましたまたは手数料で合計が超過しました。予約を取り消します。")
                if reserved:
                    with FileLock(LOCKFILE, timeout=LOCK_TIMEOUT):
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
                    with FileLock(LOCKFILE, timeout=LOCK_TIMEOUT):
                        if hasattr(fund_manager, 'release'):
                            fund_manager.release(reserved_budget)
                        elif hasattr(fund_manager, 'add_funds'):
                            fund_manager.add_funds(reserved_budget)
                        reserved = False
                except Exception as e2:
                    print(f"⚠️ 予約取り消しに失敗しました: {e2}")
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
                        if 'price' in order and order.get('price') is not None:
                            try:
                                entry_price = float(order.get('price'))
                            except Exception:
                                entry_price = None
                        elif 'cost' in order and order.get('amount'):
                            try:
                                entry_price = float(order.get('cost')) / float(order.get('amount'))
                            except Exception:
                                entry_price = None
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
                    dbg_final_qty = float(final_qty) if 'final_qty' in locals() else (float(order.get('amount', 0)) if isinstance(order, dict) and order.get('amount') is not None else 0.0)
                except Exception:
                    dbg_final_qty = 0.0
                try:
                    dbg_final_cost = float(final_cost) if 'final_cost' in locals() else (float(order.get('cost')) if isinstance(order, dict) and order.get('cost') is not None else None)
                except Exception:
                    dbg_final_cost = None
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
                    with FileLock(LOCKFILE, timeout=LOCK_TIMEOUT):
                        if hasattr(fund_manager, 'release'):
                            fund_manager.release(reserved_budget)
                        elif hasattr(fund_manager, 'add_funds'):
                            fund_manager.add_funds(reserved_budget)
                        reserved = False
                except Exception as e:
                    print(f"⚠️ 予約取り消し（返金）に失敗しました: {e}")

    except Exception as e:
        # 途中例外発生時は予約取り消しを試みる
        if reserved:
            try:
                with FileLock(LOCKFILE, timeout=LOCK_TIMEOUT):
                    if hasattr(fund_manager, 'release'):
                        fund_manager.release(reserved_budget)
                    elif hasattr(fund_manager, 'add_funds'):
                        # refund reserved_budget on unexpected exception
                        fund_manager.add_funds(reserved_budget)
                    reserved = False
            except Exception as e2:
                print(f"⚠️ 予約取り消しに失敗しました: {e2}")
        print(f"⚠️ 注文に失敗しました: {e}")

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
    print("🔁 自動売買Botを継続運用モードで起動します")
    # DEBUG: main entry
    try:
        print(f"DEBUG: __main__ start - DRY_RUN={DRY_RUN}")
    except Exception:
        pass
    # CLI helper: run small adapter test and exit
    try:
        if len(sys.argv) > 1 and sys.argv[1] == 'test_adapter':
            test_fund_adapter()
            sys.exit(0)
    except Exception:
        pass
    exchange = connect_to_bitbank()
    # 初期資金は env で設定可能（なければ 20000 円）
    initial_fund = float(os.getenv('INITIAL_FUND', '20000'))
    fund_manager = FundManager(initial_fund=initial_fund, state_file=os.getenv('FUND_STATE_FILE', 'funds_state.json'))
    # Ensure fund state is usable for local DRY_RUN/tests
    _ensure_fund_manager_has_funds(fund_manager, initial_amount=initial_fund)
    # 毎ループで自動投入する少額（例: 毎時間100円ずつ入金する）を環境変数で指定
    deposit_amount = float(os.getenv('DEPOSIT_AMOUNT', '0'))
    # 自動トップアップの閾値（この金額を下回ったら自動入金を行う）
    # 例: MIN_BALANCE_THRESHOLD=5000
    min_balance_threshold = float(os.getenv('MIN_BALANCE_THRESHOLD', os.getenv('BALANCE_TOPUP_THRESHOLD', '5000')))
    if not exchange:
        print("API接続に失敗したためBotを終了します。")
    else:
        while True:
            # 事前トップアップ（安全な段階的入金）: 残高が閾値未満なら段階的に入金
            try:
                if deposit_amount and deposit_amount > 0:
                    try:
                        current = float(fund_manager.available_fund())
                    except Exception:
                        current = None

                    # 目標残高: 環境変数 TOPUP_TARGET があればそれを使い、無ければ閾値の2倍を目標にする
                    try:
                        topup_target = float(os.getenv('TOPUP_TARGET', str(min_balance_threshold * 2)))
                    except Exception:
                        topup_target = min_balance_threshold * 2

                    if current is None:
                        # available_fund が使えない場合は通常の自動入金を行う
                        fund_manager.add_funds(deposit_amount)
                        print(f"💳 自動入金(保険): {deposit_amount:.2f} 円 → 残高: {fund_manager.available_fund():.2f} 円")
                    else:
                        if current < min_balance_threshold:
                            # 実際に入金する額は、目標との差分と deposit_amount の小さい方
                            to_add = min(deposit_amount, max(0.0, topup_target - current))
                            if to_add > 0:
                                fund_manager.add_funds(to_add)
                                print(f"💳 閾値以下のため自動入金: {to_add:.2f} 円 → 残高: {fund_manager.available_fund():.2f} 円 (閾値: {min_balance_threshold:.2f}, 目標: {topup_target:.2f})")
                            else:
                                print(f"ℹ️ 自動入金は不要（目標残高に達しています）: 現在 {current:.2f} 円, 目標 {topup_target:.2f} 円")

            except Exception as e:
                print(f"⚠️ 自動入金処理中にエラーが発生しました: {e}")

            run_bot(exchange, fund_manager)

            # 従来の毎ループ入金（あえて残す。ENVで無効化可）
            try:
                post_deposit = float(os.getenv('POST_LOOP_DEPOSIT', '0'))
            except Exception:
                post_deposit = 0
            if post_deposit and post_deposit > 0:
                try:
                    fund_manager.add_funds(post_deposit)
                    print(f"💳 ループ終了後の自動入金: {post_deposit:.2f} 円 → 残高: {fund_manager.available_fund():.2f} 円")
                except Exception as e:
                    print(f"⚠️ ループ後自動入金に失敗しました: {e}")

            time.sleep(3600)  # 1時間待機


# === DI対応版のエントリーポイント ===
def run_bot_di(dry_run=False, exchange_override=None):
    """
    Bot のメインエントリーポイント（DI対応）
    
    Args:
        dry_run (bool): True の場合、実際の取引を行わずログ出力のみ
        exchange_override: テスト用の Exchange オブジェクト（None の場合は実際の取引所に接続）
    
    Returns:
        dict: 実行結果の辞書
    """
    # 環境変数チェック（dry_run の場合は必須チェックを緩和する）
    # DRY_RUN 実行時は外部サービス（SMTP/APIキー等）を必須にしない
    env_dry_run = os.getenv("DRY_RUN", "").lower() in ["1", "true", "yes", "on"]
    actual_dry_run = dry_run or env_dry_run

    if not actual_dry_run:
        required_env_vars = ["SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS", "TO_EMAIL", "API_KEY", "SECRET_KEY"]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"以下の環境変数が .env に設定されていません: {', '.join(missing_vars)}")

    # DRY_RUN フラグの確認
    env_dry_run = os.getenv("DRY_RUN", "").lower() in ["1", "true", "yes", "on"]
    actual_dry_run = dry_run or env_dry_run
    
    print(f"🚀 Bot開始 (DRY_RUN: {actual_dry_run})")
    
    # Exchange の準備
    if exchange_override:
        exchange = exchange_override
    elif actual_dry_run:
        exchange = ExchangeStub()
        print("🔧 DRY_RUN モード: ExchangeStub を使用")
    else:
        exchange = connect_to_bitbank()
        if not exchange:
            return {"status": "error", "message": "取引所接続に失敗"}
    
    # FundManager の準備
    initial_fund = float(os.getenv('INITIAL_FUND', '20000'))
    # Create raw FundManager instance, ensure it has funds when appropriate, then adapt
    _raw_fm = FundManager(initial_fund=initial_fund, state_file=os.getenv('FUND_STATE_FILE', 'funds_state.json'))
    _ensure_fund_manager_has_funds(_raw_fm, initial_amount=initial_fund)
    fund_manager = _adapt_fund_manager_instance(_raw_fm)
    
    try:
        run_bot(exchange, fund_manager)
        return {"status": "success", "message": "Bot実行完了"}
    except Exception as e:
        return {"status": "error", "message": f"Bot実行中にエラー: {e}"}


def test_fund_adapter():
    """Quick smoke test for FundAdapter/_adapt_fund_manager_instance.
    Prints expected behavior for reserve/confirm/release in DRY_RUN and live modes.
    """
    print("--- FundAdapter smoke test ---")
    # live-like stub
    fm = FundManager(initial_fund=2000, state_file=os.getenv('FUND_STATE_FILE', 'funds_state.json'))
    adapter = _adapt_fund_manager_instance(fm)
    print("initial available (live stub):", adapter.available_fund())
    cost = 500
    ok = adapter.reserve(cost) if hasattr(adapter, 'reserve') else adapter.place_order(cost)
    print(f"reserve/place_order({cost}) ->", ok)
    print("available after reserve:", adapter.available_fund())
    # attempt release (refund)
    if hasattr(adapter, 'release'):
        adapter.release(cost)
        print("after release available:", adapter.available_fund())
    else:
        # fallback: add_funds used as refund
        adapter.add_funds(cost)
        print("after add_funds available:", adapter.available_fund())

    # Dry-run adapter
    dry_adapter = _adapt_fund_manager_instance(None)
    # ensure it simulates local funds
    try:
        da = FundAdapter(fund_manager=None, initial_fund=1000.0, dry_run=True)
        print("dry initial available:", da.available_fund())
        ok2 = da.reserve(300)
        print("dry reserve(300) ->", ok2, "available->", da.available_fund())
        da.confirm(300)
        print("dry confirm(300) -> available->", da.available_fund())
    except Exception as e:
        print("dry adapter test failed:", e)

>>>>>>> 74f1ab306ca4f7cbafdafeccf820148ccd40d52d
