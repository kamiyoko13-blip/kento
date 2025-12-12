<<<<<<< HEAD
"""FundManager: シンプルな資金管理クラス

このモジュールは `ninibo1127.py` と互換性のある資金管理を提供します。
提供するメソッド:
  - available_fund() -> float
  - place_order(cost) -> bool            # 旧 API（即時差し引き）
  - add_funds(amount)
  - reserve(cost) -> bool                # 予約 API
  - confirm(cost)                        # 予約確定（消費）
  - release(cost)                        # 予約取消（返金）

内部的にはスレッドロックと簡易的な JSON 永続化を行います。
実運用では正しい会計・監査システムを用いてください。
"""

from typing import Optional, Callable
import threading
import json
from pathlib import Path


class FundManager:
    """資金管理クラス。state は簡易 JSON で永続化されます。"""

    def __init__(self, initial_fund: float = 0.0, state_file: Optional[str] = None):
        self._lock = threading.Lock()
        self._state_file = Path(state_file) if state_file else Path('funds_state.json')
        self._available = 0.0
        self._reserved = 0.0
        self._load_or_init(float(initial_fund or 0.0))

    def _load_or_init(self, initial: float) -> None:
        try:
            if self._state_file.exists():
                raw = json.loads(self._state_file.read_text(encoding='utf-8'))
                self._available = float(raw.get('available', initial))
                self._reserved = float(raw.get('reserved', 0.0))
            else:
                self._available = float(initial)
                self._reserved = 0.0
                self._persist()
        except Exception:
            # フォールバック
            self._available = float(initial)
            self._reserved = 0.0
            try:
                self._persist()
            except Exception:
                pass

    def _persist(self) -> None:
        try:
            obj = {'available': float(self._available), 'reserved': float(self._reserved)}
            self._state_file.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception:
            pass

    # --- 基本 API ---
    def available_fund(self) -> float:
        with self._lock:
            try:
                return float(max(0.0, self._available - self._reserved))
            except Exception:
                return 0.0

    def add_funds(self, amount: float) -> None:
        try:
            a = float(amount or 0.0)
        except Exception:
            return
        if a <= 0:
            return
        with self._lock:
            self._available = float(self._available) + a
            self._persist()

    # Legacy wrapper
    def add_fund(self, amount: float) -> None:
        return self.add_funds(amount)

    # --- 予約 API ---
    def reserve(self, cost: float) -> bool:
        try:
            c = float(cost or 0.0)
        except Exception:
            return False
        if c <= 0:
            return False
        with self._lock:
            unreserved = max(0.0, self._available - self._reserved)
            if unreserved < c:
                return False
            self._reserved = float(self._reserved) + c
            self._persist()
            return True

    def confirm(self, cost: float) -> None:
        try:
            c = float(cost or 0.0)
        except Exception:
            return
        if c <= 0:
            return
        with self._lock:
            consume = min(self._reserved, c)
            self._reserved = max(0.0, self._reserved - consume)
            # 予約分からの消費が足りない場合は available から差し引く
            remain = max(0.0, c - consume)
            if remain > 0:
                self._available = max(0.0, self._available - remain)
            self._persist()

    def release(self, cost: float) -> None:
        try:
            c = float(cost or 0.0)
        except Exception:
            return
        if c <= 0:
            return
        with self._lock:
            dec = min(self._reserved, c)
            self._reserved = max(0.0, self._reserved - dec)
            self._persist()

    # --- 互換性のための旧 API ---
    def place_order(self, cost: float) -> bool:
        """Legacy: attempt to deduct cost immediately from available funds.

        This behaves as an immediate consumption of funds (no reservation).
        Returns True if deduction succeeded (sufficient unreserved funds), False otherwise.
        """
        try:
            c = float(cost or 0.0)
        except Exception:
            return False
        if c <= 0:
            return False
        with self._lock:
            unreserved = max(0.0, self._available - self._reserved)
            if unreserved < c:
                return False
            # deduct immediately from available (consume funds)
            self._available = max(0.0, float(self._available) - c)
            # persist new state
            self._persist()
            return True

    def __repr__(self) -> str:
        with self._lock:
            return f"<FundManager available={self._available:.2f} reserved={self._reserved:.2f} file={str(self._state_file)}>"


def check_and_notify(fund: FundManager, threshold: float, notifier: Optional[Callable[[str, str], bool]] = None) -> bool:
    """資金が閾値を下回ったら通知。

    notifier(subject, body) を渡せばそれを使う。無ければ標準出力に警告を出す。
    """
    try:
        available = fund.available_fund()
    except Exception:
        return False
    if available < float(threshold):
        subject = "⚠️ 資金警告"
        body = f"資金が少なくなっています：残り {available:,.0f} 円"
        if notifier:
            try:
                return bool(notifier(subject, body))
            except Exception:
                return False
        else:
            print(subject)
            print(body)
            return True
    return False


if __name__ == '__main__':
    fm = FundManager(initial_fund=10000)
    print('initial:', fm)
    print('avail:', fm.available_fund())
    ok = fm.reserve(2000)
    print('reserve 2000 ->', ok, 'avail_after:', fm.available_fund(), fm)
    fm.confirm(2000)
    print('confirm 2000 ->', fm)
    fm.add_funds(500)
    print('add 500 ->', fm)
    ok2 = fm.place_order(300)
=======
"""FundManager: シンプルな資金管理クラス

このモジュールは `ninibo1127.py` と互換性のある資金管理を提供します。
提供するメソッド:
  - available_fund() -> float
  - place_order(cost) -> bool            # 旧 API（即時差し引き）
  - add_funds(amount)
  - reserve(cost) -> bool                # 予約 API
  - confirm(cost)                        # 予約確定（消費）
  - release(cost)                        # 予約取消（返金）

内部的にはスレッドロックと簡易的な JSON 永続化を行います。
実運用では正しい会計・監査システムを用いてください。
"""

from typing import Optional, Callable
import threading
import json
from pathlib import Path


class FundManager:
    """資金管理クラス。state は簡易 JSON で永続化されます。"""

    def __init__(self, initial_fund: float = 0.0, state_file: Optional[str] = None):
        self._lock = threading.Lock()
        self._state_file = Path(state_file) if state_file else Path('funds_state.json')
        self._available = 0.0
        self._reserved = 0.0
        self._load_or_init(float(initial_fund or 0.0))

    def _load_or_init(self, initial: float) -> None:
        try:
            if self._state_file.exists():
                raw = json.loads(self._state_file.read_text(encoding='utf-8'))
                self._available = float(raw.get('available', initial))
                self._reserved = float(raw.get('reserved', 0.0))
            else:
                self._available = float(initial)
                self._reserved = 0.0
                self._persist()
        except Exception:
            # フォールバック
            self._available = float(initial)
            self._reserved = 0.0
            try:
                self._persist()
            except Exception:
                pass

    def _persist(self) -> None:
        try:
            obj = {'available': float(self._available), 'reserved': float(self._reserved)}
            self._state_file.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception:
            pass

    # --- 基本 API ---
    def available_fund(self) -> float:
        with self._lock:
            try:
                return float(max(0.0, self._available - self._reserved))
            except Exception:
                return 0.0

    def add_funds(self, amount: float) -> None:
        try:
            a = float(amount or 0.0)
        except Exception:
            return
        if a <= 0:
            return
        with self._lock:
            self._available = float(self._available) + a
            self._persist()

    # Legacy wrapper
    def add_fund(self, amount: float) -> None:
        return self.add_funds(amount)

    # --- 予約 API ---
    def reserve(self, cost: float) -> bool:
        try:
            c = float(cost or 0.0)
        except Exception:
            return False
        if c <= 0:
            return False
        with self._lock:
            unreserved = max(0.0, self._available - self._reserved)
            if unreserved < c:
                return False
            self._reserved = float(self._reserved) + c
            self._persist()
            return True

    def confirm(self, cost: float) -> None:
        try:
            c = float(cost or 0.0)
        except Exception:
            return
        if c <= 0:
            return
        with self._lock:
            consume = min(self._reserved, c)
            self._reserved = max(0.0, self._reserved - consume)
            # 予約分からの消費が足りない場合は available から差し引く
            remain = max(0.0, c - consume)
            if remain > 0:
                self._available = max(0.0, self._available - remain)
            self._persist()

    def release(self, cost: float) -> None:
        try:
            c = float(cost or 0.0)
        except Exception:
            return
        if c <= 0:
            return
        with self._lock:
            dec = min(self._reserved, c)
            self._reserved = max(0.0, self._reserved - dec)
            self._persist()

    # --- 互換性のための旧 API ---
    def place_order(self, cost: float) -> bool:
        """Legacy: attempt to deduct cost immediately from available funds.

        This behaves as an immediate consumption of funds (no reservation).
        Returns True if deduction succeeded (sufficient unreserved funds), False otherwise.
        """
        try:
            c = float(cost or 0.0)
        except Exception:
            return False
        if c <= 0:
            return False
        with self._lock:
            unreserved = max(0.0, self._available - self._reserved)
            if unreserved < c:
                return False
            # deduct immediately from available (consume funds)
            self._available = max(0.0, float(self._available) - c)
            # persist new state
            self._persist()
            return True

    def __repr__(self) -> str:
        with self._lock:
            return f"<FundManager available={self._available:.2f} reserved={self._reserved:.2f} file={str(self._state_file)}>"


def check_and_notify(fund: FundManager, threshold: float, notifier: Optional[Callable[[str, str], bool]] = None) -> bool:
    """資金が閾値を下回ったら通知。

    notifier(subject, body) を渡せばそれを使う。無ければ標準出力に警告を出す。
    """
    try:
        available = fund.available_fund()
    except Exception:
        return False
    if available < float(threshold):
        subject = "⚠️ 資金警告"
        body = f"資金が少なくなっています：残り {available:,.0f} 円"
        if notifier:
            try:
                return bool(notifier(subject, body))
            except Exception:
                return False
        else:
            print(subject)
            print(body)
            return True
    return False


if __name__ == '__main__':
    fm = FundManager(initial_fund=10000)
    print('initial:', fm)
    print('avail:', fm.available_fund())
    ok = fm.reserve(2000)
    print('reserve 2000 ->', ok, 'avail_after:', fm.available_fund(), fm)
    fm.confirm(2000)
    print('confirm 2000 ->', fm)
    fm.add_funds(500)
    print('add 500 ->', fm)
    ok2 = fm.place_order(300)
>>>>>>> 74f1ab306ca4f7cbafdafeccf820148ccd40d52d
    print('place_order 300 ->', ok2, fm)