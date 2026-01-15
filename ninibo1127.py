# --- メインループ（Botの実行部分） ---
def run_bot(exchange, fund_manager, dry_run=False):
    # 板情報取得
    try:
        orderbook = exchange.fetch_order_book('BTC/JPY')
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        current_price = get_latest_price(exchange, 'BTC/JPY')
        # 厚い買い板・売り板判定
        bids_near = [bid for bid in bids if abs(bid[0] - current_price) < current_price * 0.01]
        asks_near = [ask for ask in asks if abs(ask[0] - current_price) < current_price * 0.01]
        avg_bid_size = sum([b[1] for b in bids_near]) / len(bids_near) if bids_near else 0
        avg_ask_size = sum([a[1] for a in asks_near]) / len(asks_near) if asks_near else 0

        # インジケータ取得
        indicators = compute_indicators(exchange, pair='BTC/JPY', timeframe='1h', limit=1000)
        rsi = indicators.get('rsi_14')
        bb_lower = None
        bb_upper = None
        try:
            import numpy as np
            closes = [float(pos['price']) for pos in positions] if positions else [current_price]
            period = 14
            if len(closes) >= period:
                sma = np.mean(closes[-period:])
                std = np.std(closes[-period:])
                bb_lower = sma - 2 * std
                bb_upper = sma + 2 * std
        except Exception:
            bb_lower = None
            bb_upper = None

        # 買い判定: RSI25～28かつBB-2σ下限タッチ/はみ出し、かつ現値の0.1～0.3%下に明らかに厚い買い板
        buy_board = [bid for bid in bids if (current_price * 0.997 <= bid[0] <= current_price * 0.999) and bid[1] > avg_bid_size * 2]
        if (rsi is not None and 25 <= rsi <= 28) and (bb_lower is not None and current_price <= bb_lower) and buy_board:
            try:
                smtp_host = os.getenv('SMTP_HOST')
                smtp_port = int(os.getenv('SMTP_PORT', '587'))
                smtp_user = os.getenv('SMTP_USER')
                smtp_password = os.getenv('SMTP_PASS')
                email_to = os.getenv('TO_EMAIL')
                if smtp_host and email_to:
                    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    subject = f"【買いシグナル】RSI25-28+BB-2σ下限+厚い買い板 {now}"
                    message = f"【板情報】\n時刻: {now}\n現在価格: {current_price} 円\nRSI: {rsi}\nBB下限: {bb_lower}\n厚い買い板: {buy_board}\n条件揃いで買い推奨。"
                    send_notification(smtp_host, smtp_port, smtp_user, smtp_password, email_to, subject, message)
            except Exception as e:
                print(f"⚠️ 買いシグナルメール送信エラー: {e}")

        # 売り判定: RSI65～75かつBB+2σタッチ/はみ出し、かつ現値の0.1～0.3%上に明らかに厚い売り板
        sell_board = [ask for ask in asks if (current_price * 1.001 <= ask[0] <= current_price * 1.003) and ask[1] > avg_ask_size * 2]
        if (rsi is not None and 65 <= rsi <= 75) and (bb_upper is not None and current_price >= bb_upper) and sell_board:
            try:
                smtp_host = os.getenv('SMTP_HOST')
                smtp_port = int(os.getenv('SMTP_PORT', '587'))
                smtp_user = os.getenv('SMTP_USER')
                smtp_password = os.getenv('SMTP_PASS')
                email_to = os.getenv('TO_EMAIL')
                if smtp_host and email_to:
                    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    subject = f"【売りシグナル】RSI65-75+BB+2σ上限+厚い売り板 {now}"
                    message = f"【板情報】\n時刻: {now}\n現在価格: {current_price} 円\nRSI: {rsi}\nBB上限: {bb_upper}\n厚い売り板: {sell_board}\n条件揃いで売り推奨。"
                    send_notification(smtp_host, smtp_port, smtp_user, smtp_password, email_to, subject, message)
            except Exception as e:
                print(f"⚠️ 売りシグナルメール送信エラー: {e}")
        # 売り板が厚い場合（平均の2倍以上）
        if asks_near and any(a[1] > avg_ask_size * 2 for a in asks_near):
            thick_ask_price = max([a[0] for a in asks_near if a[1] > avg_ask_size * 2])
            custom_take_profit = thick_ask_price
        else:
            custom_take_profit = None
        # 板が薄い場合（買い板・売り板とも平均の半分以下）
        if (avg_bid_size < 0.5 * (sum([b[1] for b in bids]) / len(bids) if bids else 1)) and (avg_ask_size < 0.5 * (sum([a[1] for a in asks]) / len(asks) if asks else 1)):
            nampin_interval = 0.20
        else:
            nampin_interval = 0.10
    except Exception as e:
        print(f"⚠️ 板情報取得・判定エラー: {e}")
        custom_take_profit = None
        nampin_interval = 0.10

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
            # dict形式（新）とリスト形式（旧）両対応
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
                # --- API残高取得とDEBUG出力 ---
                try:
                    api_balance = get_account_balance(exchange)
                    print(f"[DEBUG] API balance取得結果: {api_balance}", flush=True)
                    btc_api_balance = api_balance.get('free', {}).get('BTC', 0)
                    print(f"[DEBUG] API BTC残高: {btc_api_balance}", flush=True)
                except Exception as e:
                    print(f"[ERROR] API残高取得例外: {e}", flush=True)
                # 現在価格取得
                current_price = get_latest_price(exchange, 'BTC/JPY')
                logging.info(f"現在価格: {current_price}")
                print(f"[DEBUG] 現在価格: {current_price}", flush=True)
                # 売買判定
                btc_balance = sum([float(pos.get('amount', 0)) for pos in positions])
                # trade_decisionでpositions[0]['price']を参照できるようbuiltinsにセット
                import builtins
                builtins.positions = positions
                # --- インジケータ取得 ---
                indicators = compute_indicators(exchange, pair=PAIR, timeframe='1h', limit=1000)
                rsi = indicators.get('rsi_14')
                # BB下限はSMA-2σ相当を仮定（compute_indicatorsで未算出ならNone）
                bb_lower = None
                # BB下限を計算（SMA-2σ）
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
                print(f"[DEBUG] trade_decision result: {td}", flush=True)
                # --- メール通知ロジック ---
                smtp_host = os.getenv('SMTP_HOST')
                smtp_port = int(os.getenv('SMTP_PORT', '587'))
                smtp_user = os.getenv('SMTP_USER')
                smtp_password = os.getenv('SMTP_PASS')
                email_to = os.getenv('TO_EMAIL')
                now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                reason = ""
                if td.get('action') == 'sell':
                    print("[DEBUG] 売り判定: 条件成立", flush=True)
                    reason = "利確条件成立（買値より10%以上上昇）"
                    if smtp_host and email_to:
                        subject = f"【売りシグナル】BTC売却推奨 {now}"
                        message = f"【売りシグナル】\n時刻: {now}\n現在価格: {current_price} 円\nRSI: {rsi}\nBB下限: {bb_lower}\n根拠: {reason}"
                        send_notification(smtp_host, smtp_port, smtp_user, smtp_password, email_to, subject, message)
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
                    print("[DEBUG] 買い判定: 条件成立", flush=True)
                    if rsi is not None and rsi <= 30:
                        reason = f"RSIが30以下（現在値: {rsi}）"
                    elif bb_lower is not None and current_price <= bb_lower:
                        reason = f"価格がBB下限（{bb_lower}）以下"
                    else:
                        reason = "買い条件成立"
                    if smtp_host and email_to:
                        subject = f"【買いシグナル】BTC購入推奨 {now}"
                        message = f"【買いシグナル】\n時刻: {now}\n現在価格: {current_price} 円\nRSI: {rsi}\nBB下限: {bb_lower}\n根拠: {reason}"
                        send_notification(smtp_host, smtp_port, smtp_user, smtp_password, email_to, subject, message)
                else:
                    print("[DEBUG] 売買判定: ホールド", flush=True)
            except Exception as e:
                logging.error(f"positionsファイル読み込み例外: {e}")
                print(f"[ERROR] positionsファイル読み込み例外: {e}", flush=True)
            logging.info("ループ末尾: sleep前")
            print("[DEBUG] ループ末尾: sleep前", flush=True)
            time.sleep(10)
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
                    smtp_port = int(os.getenv('SMTP_PORT', '587'))
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
    time.sleep(10)  # 10秒待機
    return "run_bot executed"
def get_last_buy_price(state):
    """
    state(dict)からlast_buy_priceを取得
    """
    return state.get("last_buy_price")

def set_last_buy_price(state, price):
    """
    state(dict)にlast_buy_priceを保存
    """
    state["last_buy_price"] = float(price) if price is not None else None

# --- DI対応のDRY_RUNラッパー関数 ---
def run_bot_di(dry_run=True):
    import os
    os.environ['DRY_RUN'] = '1' if dry_run else '0'
    # 必要に応じてfund_managerやexchangeの初期化をここで行う
    try:
        # run_bot関数が存在する場合は呼び出し
        if 'run_bot' in globals():
            result = run_bot(None, None, dry_run=dry_run)
        else:
            result = {'status': 'run_bot未定義'}
        return result
    except Exception as e:
        return {'error': str(e)}
# --- bitbank接続ユーティリティ ---
def connect_to_bitbank():
    import ccxt
    import os
    # 先にDRY_RUN判定
    dry_run = str(os.getenv('DRY_RUN', '')).lower() in ('1', 'true', 'yes', 'on')
    if dry_run:
        print("[DEBUG] connect_to_bitbank: DRY_RUNモードなのでAPIキー不要")
        return None  # またはダミーのexchangeオブジェクト
    api_key = os.getenv("API_KEY")
    secret_key = os.getenv("SECRET_KEY")
    if not api_key or not secret_key:
        raise ValueError("API_KEYまたはSECRET_KEYが環境変数に設定されていません")
    import ccxt
    return ccxt.bitbank({
        'apiKey': api_key,
        'secret': secret_key,
    })
DRY_RUN_PRICE = 18000000  # DRY_RUN時のデフォルト価格（現実的な相場に合わせて変更）
# === RSIにボリンジャーバンドを重ねて表示 ===
import matplotlib.pyplot as plt
import numpy as np
import datetime
JST = datetime.timezone(datetime.timedelta(hours=9))

def plot_rsi_with_bbands(rsi_values, period=14, num_std=2):
    """
    RSI値のリストに対して、ボリンジャーバンドを重ねてグラフ表示する。
    :param rsi_values: RSI値のリスト
    :param period: ボリンジャーバンドの移動平均期間
    :param num_std: 標準偏差の倍率（通常2）
    """
    rsi_values = np.array(rsi_values)
    sma = np.convolve(rsi_values, np.ones(period)/period, mode='valid')
    std = np.array([np.std(rsi_values[i-period:i]) if i >= period else np.nan for i in range(1, len(rsi_values)+1)])
    upper_band = sma + num_std * std[period-1:]
    lower_band = sma - num_std * std[period-1:]

    plt.figure(figsize=(12,5))
    plt.plot(rsi_values, label='RSI', color='blue')
    plt.plot(range(period-1, len(rsi_values)), upper_band, label=f'Upper Band (+{num_std}σ)', color='red', linestyle='--')
    plt.plot(range(period-1, len(rsi_values)), lower_band, label=f'Lower Band (-{num_std}σ)', color='green', linestyle='--')
    plt.plot(range(period-1, len(rsi_values)), sma, label='SMA', color='orange', linestyle=':')
    plt.title(f'RSI({period}) with Bollinger Bands')
    plt.xlabel('Index')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid(True)
    plt.show()
# --- 通知ユーティリティ ---
def send_notification(smtp_host, smtp_port, smtp_user, smtp_password, email_to, subject, message):
    print(f"[NOTIFY] {subject}\n{message}")
    import smtplib
    from email.mime.text import MIMEText
    from email.utils import formatdate
    try:
        msg = MIMEText(message, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = email_to
        msg["Date"] = formatdate()
        if int(smtp_port) == 465:
            # SSL/TLS専用ポートの場合
            with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
                server.login(smtp_user, smtp_password)
                server.sendmail(smtp_user, [email_to], msg.as_string())
        else:
            # 通常（STARTTLSなど）
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.sendmail(smtp_user, [email_to], msg.as_string())
    except Exception as e:
        print(f"[ERROR] メール送信失敗: {e}")
# --- 価格取得ユーティリティ ---
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
        time.sleep(1)
    print("[ERROR] 価格取得に3回失敗しました。Noneを返します。")
    return None

# --- 売買判定ロジック ---
def trade_decision(current_price, btc_balance=0.0027, buy_btc=0.02, last_buy_price=None, rsi=None, bb_lower=None):
    """
    current_price: 現在のBTC/JPY価格
    btc_balance: 現在のBTC総保有量
    buy_btc: 売買対象のBTC量（デフォルト0.02BTC）
    last_buy_price: 直近の買値（売却判定に使用）
    rsi: 最新のRSI値
    bb_lower: ボリンジャーバンド下限
    """
    # デバッグ用: 各値を出力
    print(f"[DEBUG] trade_decision: current_price={current_price}, btc_balance={btc_balance}, last_buy_price={last_buy_price}, rsi={rsi}, bb_lower={bb_lower}")
    # 売り判定: 保有BTCがあり、現在価格が最初の買値の10%上昇（利確）
    # ポジションの最初の買値を基準にする
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
    if btc_balance > 0 and first_buy_price is not None and current_price > first_buy_price * 1.10:
        return {'action': 'sell', 'amount': btc_balance, 'price': current_price, 'sell_condition': True}
    # 買い判定: BTC未保有、RSI<=30 または BB下限タッチ
    if btc_balance == 0 and ((rsi is not None and rsi <= 30) or (bb_lower is not None and current_price <= bb_lower)):
        return {'action': 'buy', 'amount': buy_btc, 'price': current_price, 'buy_condition': True}
    # 何もしない
    return {'action': 'hold', 'amount': 0.0, 'price': current_price, 'buy_condition': False, 'sell_condition': False}

# --- BTC残高を売買結果で更新する ---
def sell_all_positions(positions, exchange, pair):
    """
    保有ポジション全量を売却する（ダミー実装）
    positions: 保有ポジションリスト
    exchange: ccxtの取引所オブジェクト
    pair: 通貨ペア（例: 'BTC/JPY'）
    """
    results = []
    for pos in positions:
        amount = pos.get('amount', 0)
        try:
            # 成行売り注文を発行
            order = exchange.create_market_sell_order(pair, amount)
            print(f"[DEBUG] 売却APIレスポンス: {order}", flush=True)
            with open("sell_log.txt", "a", encoding="utf-8") as logf:
                print(f"[DEBUG] 売却APIレスポンス: {order}", file=logf)
            results.append({'amount': amount, 'order': order, 'status': 'sold'})
        except Exception as e:
            print(f"[ERROR] 売却APIエラー: {e}", flush=True)
            with open("sell_log.txt", "a", encoding="utf-8") as logf:
                print(f"[ERROR] 売却APIエラー: {e}", file=logf)
            results.append({'amount': amount, 'error': str(e), 'status': 'error'})
    return results
def update_btc_balance(btc_balance, trade_result):
    """
    btc_balance: 現在のBTC残高
    trade_result: nの戻り値（dict）
    """
    action = trade_result.get('action')
    amount = trade_result.get('amount', 0.0)
    if action == 'sell':
        btc_balance -= amount
    elif action == 'buy':
        btc_balance += amount
    return btc_balance
import os
from dotenv import load_dotenv
load_dotenv()
def get_dry_run_flag():
    return str(os.getenv('DRY_RUN', '')).lower() in ('1', 'true', 'yes', 'on')
import logging




if __name__ == "__main__":
    print("[DEBUG] ファイル先頭到達")
    print("mainブロック実行開始")
    print("[DEBUG] mainブロックtry直前")
    try:
        # ExchangeとFundManagerの初期化
        print("[DEBUG] Exchange初期化直前")
        exchange = connect_to_bitbank()
        print("[DEBUG] FundManager初期化直前")
        fund_manager = None  # 必要ならFundManagerクラスをここで初期化
        print("[DEBUG] run_bot呼び出し直前")
        run_bot(exchange, fund_manager)
        print("[DEBUG] mainブロックtry直後（run_bot呼び出し後）")
    except Exception as e:
        print(f"[ERROR] mainブロック例外: {e}")
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
            except Exception:
                pass

        def get_positions_reserved(self, positions_file='positions_state.json'):
            """
            positions_state.json から全ポジションの合計コスト(予約額)を計算
            """
            try:
                pf = Path(positions_file)
                if not pf.exists():
                    return 0.0
                with pf.open(encoding='utf-8') as f:
                    data = json.load(f)
                # Fix: if data is a dict with 'positions', use that
                if isinstance(data, dict) and 'positions' in data:
                    positions = data['positions']
                else:
                    positions = data
                if not isinstance(positions, list):
                    return 0.0
                total = 0.0
                for pos in positions:
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
                # positions_state.jsonの合計予約額を取得
                reserved_from_positions = self.get_positions_reserved()
                # DRY_RUN時のみ予約額のデバッグ出力を行う
                try:
                    if str(os.getenv('DRY_RUN', '')).lower() in ('1', 'true', 'yes', 'on'):
                        print(f"予約フェーズ: 予約額（JPY）={reserved_from_positions}")
                except Exception:
                    pass
                # 注文後に500円以上残る場合のみ許可
                if self._available - c < 500:
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
                reserved_from_positions = self.get_positions_reserved()
                # DRY_RUN時のみ予約額のデバッグ出力を行う
                try:
                    if str(os.getenv('DRY_RUN', '')).lower() in ('1', 'true', 'yes', 'on'):
                        print(f"予約フェーズ: 予約額（JPY）={reserved_from_positions}")
                except Exception:
                    pass
                if self._available - c < 500:
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
    pass
    pass  # ← ここに必要な処理があれば記述
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
            # 注文後に500円以上残る場合のみ許可
            if self.available_fund() - c < 500:
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
import time
import sys


# log_info, log_warn, log_debug, log_error の重複定義を防ぐ
# すでにファイル先頭で定義済みなので、以降の重複定義は削除
# --- 未定義変数のダミー定義（未定義エラー防止用） ---

# --- bitbank 残高取得サンプル ---
import json
import time
import hmac
import hashlib
import requests

# config.jsonからAPIキー・シークレットを読み込む
def load_api_keys(config_path='config.json'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config['api_key'], config['api_secret']

# bitbankの残高取得APIを呼び出す
def get_bitbank_balance(api_key, api_secret):
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

# --- connect_to_bitbank: Bitbank用の簡易接続関数（未定義エラー対策のダミー実装） ---
    api_key = os.getenv("API_KEY")
    secret_key = os.getenv("SECRET_KEY")
    return ccxt.bitbank({
        'apiKey': api_key or "",
        'secret': secret_key or "",
    })

# --- 残高取得ユーティリティ ---
from typing import Dict, Any

def get_account_balance(exchange) -> dict[str, dict[str, Any]]:
    """
    Returns:
        dict[str, dict[str, Any]]: { 'total': {...}, 'free': {...}, 'used': {...} }
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


def compute_dynamic_threshold(exchange, pair='BTC/JPY', days=30, buffer_jpy=500, buffer_pct=0.01):
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


def compute_indicators(exchange, pair='BTC/JPY', timeframe='1h', limit=1000):
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
        indicators['rsi_14'] = compute_rsi(closes, period=14, exchange=exchange, pair=pair)
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



def get_last_buy_time(state):
    return state.get("last_buy_time")


def set_last_buy_time(state, ts=None):
    state["last_buy_time"] = ts or int(time.time())


def record_position(state, side, price, qty):
    print("DEBUG: record_position called", side, price, qty)
    state.setdefault("positions", [])
    state["positions"].append({
        "side": side,
        "price": float(price),
        "qty": float(qty),
        "time": int(time())
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

    # --- RSI単独による売買判定（30以下で買い、70以上で売り）---
    if latest_data['rsi'] <= 30:
        signal = 'buy_entry'
        message = f"✅ RSI買い注文: RSI={latest_data['rsi']:.2f} (30以下)"
        return signal, message
    elif latest_data['rsi'] >= 70:
        signal = 'sell_all'
        message = f"❌ RSI売り注文: RSI={latest_data['rsi']:.2f} (70以上)"
        return signal, message

    # --- RSI＋ボリンジャーバンドによる売買判定 ---
    rsi_series = df['rsi'].values
    period = 14
    num_std = 2
    if len(rsi_series) >= period:
        import numpy as np
        sma = np.convolve(rsi_series, np.ones(period)/period, mode='valid')
        std = np.array([np.std(rsi_series[i-period:i]) if i >= period else np.nan for i in range(1, len(rsi_series)+1)])
        upper_band = sma + num_std * std[period-1:]
        lower_band = sma - num_std * std[period-1:]

        rsi_now = rsi_series[-1]
        # 最新のバンド値
        upper = upper_band[-1]
        lower = lower_band[-1]
        # 下抜けで買い
        if rsi_now < lower:
            signal = 'buy_entry'
            message = f"✅ RSIボリンジャー下抜け買い: RSI={rsi_now:.2f} < 下バンド{lower:.2f}"
            return signal, message
        # 上抜けで売り
        elif rsi_now > upper:
            signal = 'sell_all'
            message = f"❌ RSIボリンジャー上抜け売り: RSI={rsi_now:.2f} > 上バンド{upper:.2f}"
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
    exchange = None
    if 'exchange' not in locals() or exchange is None:
        exchange = connect_to_bitbank()
        # 板情報取得
        try:
            orderbook = exchange.fetch_order_book('BTC/JPY')
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            current_price = get_latest_price(exchange, 'BTC/JPY')
            # 厚い買い板・売り板判定
            bids_near = [bid for bid in bids if abs(bid[0] - current_price) < current_price * 0.01]
            asks_near = [ask for ask in asks if abs(ask[0] - current_price) < current_price * 0.01]
            avg_bid_size = sum([b[1] for b in bids_near]) / len(bids_near) if bids_near else 0
            avg_ask_size = sum([a[1] for a in asks_near]) / len(asks_near) if asks_near else 0
            # 買い板が厚い場合（平均の2倍以上）
            if bids_near and any(b[1] > avg_bid_size * 2 for b in bids_near):
                try:
                    smtp_host = os.getenv('SMTP_HOST')
                    smtp_port = int(os.getenv('SMTP_PORT', '587'))
                    smtp_user = os.getenv('SMTP_USER')
                    smtp_password = os.getenv('SMTP_PASS')
                    email_to = os.getenv('TO_EMAIL')
                    if smtp_host and email_to:
                        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        subject = f"厚い買い板付近:資金投入推奨 {now}"
                        message = f"【板情報】\n時刻: {now}\n現在価格: {current_price} 円\n厚い買い板付近です。資金投入を推奨します。"
                        send_notification(smtp_host, smtp_port, smtp_user, smtp_password, email_to, subject, message)
                except Exception as e:
                    print(f"⚠️ 板資金投入通知メール送信エラー: {e}")
            # 売り板が厚い場合（平均の2倍以上）
            if asks_near and any(a[1] > avg_ask_size * 2 for a in asks_near):
                thick_ask_price = max([a[0] for a in asks_near if a[1] > avg_ask_size * 2])
                custom_take_profit = thick_ask_price
            else:
                custom_take_profit = None
            # 板が薄い場合（買い板・売り板とも平均の半分以下）
            if (avg_bid_size < 0.5 * (sum([b[1] for b in bids]) / len(bids) if bids else 1)) and (avg_ask_size < 0.5 * (sum([a[1] for a in asks]) / len(asks) if asks else 1)):
                nampin_interval = 0.20
            else:
                nampin_interval = 0.10
        except Exception as e:
            print(f"⚠️ 板情報取得・判定エラー: {e}")
            custom_take_profit = None
            nampin_interval = 0.10

def run_bot(exchange, fund_manager, dry_run=False):
    # 板情報取得
    try:
        orderbook = exchange.fetch_order_book('BTC/JPY')
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        current_price = get_latest_price(exchange, 'BTC/JPY')
        # 厚い買い板・売り板判定
        bids_near = [bid for bid in bids if abs(bid[0] - current_price) < current_price * 0.01]
        asks_near = [ask for ask in asks if abs(ask[0] - current_price) < current_price * 0.01]
        avg_bid_size = sum([b[1] for b in bids_near]) / len(bids_near) if bids_near else 0
        avg_ask_size = sum([a[1] for a in asks_near]) / len(asks_near) if asks_near else 0
        # 買い板が厚い場合（平均の2倍以上）
        if bids_near and any(b[1] > avg_bid_size * 2 for b in bids_near):
            try:
                smtp_host = os.getenv('SMTP_HOST')
                smtp_port = int(os.getenv('SMTP_PORT', '587'))
                smtp_user = os.getenv('SMTP_USER')
                smtp_password = os.getenv('SMTP_PASS')
                email_to = os.getenv('TO_EMAIL')
                if smtp_host and email_to:
                    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    subject = f"厚い買い板付近:資金投入推奨 {now}"
                    message = f"【板情報】\n時刻: {now}\n現在価格: {current_price} 円\n厚い買い板付近です。資金投入を推奨します。"
                    send_notification(smtp_host, smtp_port, smtp_user, smtp_password, email_to, subject, message)
            except Exception as e:
                print(f"⚠️ 板資金投入通知メール送信エラー: {e}")
        # 売り板が厚い場合（平均の2倍以上）
        if asks_near and any(a[1] > avg_ask_size * 2 for a in asks_near):
            thick_ask_price = max([a[0] for a in asks_near if a[1] > avg_ask_size * 2])
            custom_take_profit = thick_ask_price
        else:
            custom_take_profit = None
        # 板が薄い場合（買い板・売り板とも平均の半分以下）
        if (avg_bid_size < 0.5 * (sum([b[1] for b in bids]) / len(bids) if bids else 1)) and (avg_ask_size < 0.5 * (sum([a[1] for a in asks]) / len(asks) if asks else 1)):
            nampin_interval = 0.20
        else:
            nampin_interval = 0.10
    except Exception as e:
        print(f"⚠️ 板情報取得・判定エラー: {e}")
        custom_take_profit = None
        nampin_interval = 0.10

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
            # dict形式（新）とリスト形式（旧）両対応
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
                print(f"[DEBUG] API balance取得結果: {api_balance}", flush=True)
                btc_api_balance = api_balance.get('free', {}).get('BTC', 0)
                print(f"[DEBUG] API BTC残高: {btc_api_balance}", flush=True)
                # 現在価格取得
                current_price = get_latest_price(exchange, 'BTC/JPY')
                logging.info(f"現在価格: {current_price}")
                print(f"[DEBUG] 現在価格: {current_price}", flush=True)
                # 売買判定
                btc_balance = sum([float(pos.get('amount', 0)) for pos in positions])
                td = trade_decision(current_price, btc_balance, MIN_ORDER_BTC, last_buy_price)
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
                else:
                    print("[DEBUG] 売り判定: 条件不成立", flush=True)
            except Exception as e:
                logging.error(f"positionsファイル読み込み例外: {e}")
                print(f"[ERROR] positionsファイル読み込み例外: {e}", flush=True)
            logging.info("ループ末尾: sleep前")
            print("[DEBUG] ループ末尾: sleep前", flush=True)
            time.sleep(10)
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
                    smtp_port = int(os.getenv('SMTP_PORT', '587'))
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
    time.sleep(10)  # 10秒待機
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





