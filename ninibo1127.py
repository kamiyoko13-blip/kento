
# --- bitbank用ccxtラッパー ---

import os
import ccxt
# 表示用ライブラリ
from tabulate import tabulate
from colorama import Fore, Style, init as colorama_init
colorama_init(autoreset=True)

# 表形式＋色付きで出力する関数
def print_table(data, headers=None, color=None):
    table = tabulate(data, headers=headers, tablefmt="fancy_grid", floatfmt=".4f")
    if color:
        print(color + table + Style.RESET_ALL)
    else:
        print(table)

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
import os
import datetime

def get_latest_price(exchange, pair='BTC/JPY'):
    import time
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            ticker = exchange.fetch_ticker(pair)
            if 'last' in ticker:
                # 価格情報を表形式で表示
                print_table([
                    ["現在価格", ticker['last']],
                    ["高値", ticker.get('high', '-')],
                    ["安値", ticker.get('low', '-')],
                    ["出来高", ticker.get('baseVolume', '-')]
                ], headers=["項目", "値"], color=Fore.YELLOW)
                return float(ticker['last'])
            else:
                print(Fore.RED + f"[RETRY] 価格情報に'last'がありません (attempt {attempt})" + Style.RESET_ALL)
        except Exception as e:
            print(f"[RETRY] 価格取得失敗 (attempt {attempt}): {e}")
        time.sleep(1)
    print("[ERROR] 価格取得に3回失敗しました。Noneを返します。")
    return None
def send_notification(*args, **kwargs):
    pass
def set_last_buy_price(state, price):
    state['last_buy_price'] = price
def get_account_balance(exchange):
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

    # --- RSIが35未満になったらフラグを立て、次の足で買うロジック ---
    # フラグ保存用（グローバル変数やファイル保存も可。ここでは簡易的にグローバル変数を使用）
    global rsi_buy_flag
    try:
        rsi_buy_flag
    except NameError:
        rsi_buy_flag = False

    # RSIが35未満ならフラグを立てる
    if rsi is not None and rsi < 35:
        rsi_buy_flag = True
        return {'action': 'hold', 'amount': 0.0, 'price': current_price, 'buy_condition': False, 'sell_condition': False}

    # フラグが立っていて、次の足で条件を満たしていれば買い
    if btc_balance == 0 and rsi_buy_flag:
        rsi_buy_flag = False  # フラグリセット
        return {'action': 'buy', 'amount': min_order_btc, 'price': current_price, 'buy_condition': True}

    return {'action': 'hold', 'amount': 0.0, 'price': current_price, 'buy_condition': False, 'sell_condition': False}
def execute_order(*args, **kwargs):
    # 本番API呼び出し
    exchange = args[0] if len(args) > 0 else kwargs.get('exchange')
    pair = args[1] if len(args) > 1 else kwargs.get('pair', 'BTC/JPY')
    side = args[2] if len(args) > 2 else kwargs.get('side', 'buy')
    amount = args[3] if len(args) > 3 else kwargs.get('amount')
    price = args[4] if len(args) > 4 else kwargs.get('price')
    try:
        if side == 'buy':
            order = exchange.create_market_buy_order(pair, amount)
        else:
            order = exchange.create_market_sell_order(pair, amount)
        # 表形式で表示
        order_disp = [
            ["注文ID", order.get('id', order.get('order_id', 'N/A'))],
            ["通貨ペア", order.get('symbol', pair)],
            ["注文タイプ", order.get('type', 'market')],
            ["売買", order.get('side', side)],
            ["価格", order.get('price', '成行')],
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
    sell_amount = round(total_btc * 0.8, 8)
    if total_btc > 0 and sell_amount > 0:
        try:
            order = exchange.create_market_sell_order(pair, sell_amount)
            # 表形式で表示
            order_disp = [
                ["注文ID", order.get('id', order.get('order_id', 'N/A'))],
                ["通貨ペア", order.get('symbol', pair)],
                ["注文タイプ", order.get('type', 'market')],
                ["売買", order.get('side', 'sell')],
                ["価格", order.get('price', '成行')],
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
def run_bot(exchange, fund_manager, dry_run=False):
    import datetime
    # 現在時刻から夜間判定（例: 22時～翌6時を夜間とする）
    now = datetime.datetime.now()
    is_night = now.hour >= 22 or now.hour < 6

    # シグナル判定（例: generate_signals(df) で取得）
    # dfは直近のOHLCVデータ。既存のロジックに合わせて適宜修正してください。
    # 例: df = get_ohlcv(exchange, PAIR, timeframe='5m', limit=200)
    try:
        df = get_ohlcv(exchange, 'BTC/JPY', timeframe='5m', limit=200)
        signal, message = generate_signals(df)
    except Exception:
        signal, message = None, None
    buy_signal = signal == 'buy_entry'
    sell_signal = signal == 'sell_all'
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
        log_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'action': action,
            'price': price,
            'amount': amount,
            'status': status,
            'reason': reason
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

    while True:
        # --- 強いトレンド判定（例: 短期SMA > 長期SMA で強い上昇トレンド） ---
        indicators_trend = compute_indicators(exchange, pair='BTC/JPY', timeframe='1h', limit=200)
        sma_short = indicators_trend.get('sma_short_50', 0)
        sma_long = indicators_trend.get('sma_long_200', 0)
        is_strong_trend = sma_short > sma_long if sma_short and sma_long else False
        # 1時間足で逆張り可否を判定（逆張りが効きやすい状況のみエントリー許可）
        indicators_1h = compute_indicators(exchange, pair='BTC/JPY', timeframe='1h', limit=100)
        rsi_1h_list = indicators_1h.get('rsi_list', None)
        rsi_1h = indicators_1h.get('rsi_14')
        can_counter_trade = False
        prev_rsi_1h = None
        latest_rsi_1h = None
        if not rsi_1h_list:
            print(f"[WARN] 1時間足のRSIリストが取得できません: {rsi_1h_list}", flush=True)
            time.sleep(10)
            continue
        if len(rsi_1h_list) < 2:
            print(f"[WARN] 1時間足のRSIリスト要素不足: {rsi_1h_list}", flush=True)
            time.sleep(10)
            continue
        prev_rsi_1h = rsi_1h_list[-2]
        latest_rsi_1h = rsi_1h_list[-1]
        # Noneが混じる場合は逆張り判定をスキップ
        if prev_rsi_1h is not None and latest_rsi_1h is not None:
            # 1時間足RSIが横ばい・反転兆候（レンジ・逆張り向き）
            if (latest_rsi_1h <= 40 and latest_rsi_1h > prev_rsi_1h) or (latest_rsi_1h >= 60 and latest_rsi_1h < prev_rsi_1h):
                can_counter_trade = True
        if not can_counter_trade:
            print(f"[INFO] 1時間足で逆張り不可: RSI(前): {prev_rsi_1h}, RSI(最新): {latest_rsi_1h}", flush=True)
            time.sleep(10)
            continue
        else:
            print(f"[DEBUG] 1時間足逆張り判定OK: RSI(前): {prev_rsi_1h}, RSI(最新): {latest_rsi_1h}")

        # --- 板情報・インジケータ取得 ---
        try:
            orderbook = exchange.fetch_order_book('BTC/JPY')
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            current_price = get_latest_price(exchange, 'BTC/JPY')
            bids_near = [bid for bid in bids if abs(bid[0] - current_price) < current_price * 0.01]
            asks_near = [ask for ask in asks if abs(ask[0] - current_price) < current_price * 0.01]
            avg_bid_size = sum([b[1] for b in bids_near]) / len(bids_near) if bids_near else 0
            avg_ask_size = sum([a[1] for a in asks_near]) / len(asks_near) if asks_near else 0

            indicators = compute_indicators(exchange, pair='BTC/JPY', timeframe='5m', limit=1000)
            rsi_list = indicators.get('rsi_list', None)
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

            # --- RSI反発ロジック ---
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
                    email_to = os.getenv('TO_EMAIL')
                    if smtp_host and email_to:
                        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        subject = f"【買いシグナル】RSI反発+BB-2σ下限 {now}"
                        message = f"【シグナル】\n時刻: {now}\n現在価格: {current_price} 円\nRSI(前): {prev_rsi}\nRSI(最新): {latest_rsi}\n条件揃いで買い推奨。"
                        send_notification(smtp_host, smtp_port, smtp_user, smtp_password, email_to, subject, message)
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
                closes_5m = [float(r[4]) for r in exchange.fetch_ohlcv('BTC/JPY', timeframe='5m', limit=20)]
                trailing_high = max(closes_5m) if closes_5m else None
            except Exception:
                trailing_high = None
            trailing_sell = False
            trailing_reason = ""
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
                    email_to = os.getenv('TO_EMAIL')
                    if smtp_host and email_to:
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
                        send_notification(smtp_host, smtp_port, smtp_user, smtp_password, email_to, subject, message)
                except Exception as e:
                    print(f"⚠️ 売りシグナルメール送信エラー: {e}")
            # 厚い板ロジック削除
            # 板が薄い場合（買い板・売り板とも平均の半分以下）
            nampin_interval = 0.10
            if (avg_bid_size < 0.5 * (sum([b[1] for b in bids]) / len(bids) if bids else 1)) and (avg_ask_size < 0.5 * (sum([a[1] for a in asks]) / len(asks) if asks else 1)):
                nampin_interval = 0.20
        except Exception as e:
            print(f"⚠️ 板情報取得・判定エラー: {e}")
        time.sleep(10)

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
            email_to = os.getenv('TO_EMAIL')
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            reason = ""
            import logging
            if td.get('action') == 'sell':
                reason = "利確条件成立（買値より10%以上上昇）"
                if smtp_host and email_to:
                    subject = f"【売りシグナル】BTC売却推奨 {now}"
                    message = f"【売りシグナル】\n時刻: {now}\n現在価格: {current_price} 円\nRSI: {rsi}\nBB下限: {bb_lower}\n根拠: {reason}"
                    notify_key = "sell"
                    if should_notify_once(notify_key):
                        send_notification(smtp_host, smtp_port, smtp_user, smtp_password, email_to, subject, message)
                try:
                    sell_results = sell_all_positions(positions, exchange, PAIR)
                    positions = []
                    set_last_buy_price(state, None)
                    save_data = {
                        "positions": positions,
                        "last_buy_price": get_last_buy_price(state)
                    }
                    with open(positions_file, 'w', encoding='utf-8') as f:
                        json.dump(save_data, f, ensure_ascii=False, indent=2)
                    log_trade('sell', current_price, btc_balance, 'success', reason)
                except Exception as e:
                    print(f"[ERROR] 売却処理例外: {e}", flush=True)
                    log_trade('sell', current_price, btc_balance, 'error', str(e))
            elif td.get('action') == 'buy':
                if rsi is not None and rsi <= 35:
                    reason = f"RSIが35以下（現在値: {rsi}）"
                else:
                    reason = "買い条件成立"
                # 資金管理: 余力チェック
                if btc_api_balance < MIN_ORDER_BTC:
                    print(f"[WARN] BTC残高不足: {btc_api_balance} < {MIN_ORDER_BTC}")
                    log_trade('buy', current_price, MIN_ORDER_BTC, 'skipped', 'BTC残高不足')
                elif not positions:  # ポジションが空のときだけ
                    notify_key = "buy"
                    if should_notify_once(notify_key):
                        # 本当に注文を出す
                        order = execute_order(exchange, PAIR, 'buy', MIN_ORDER_BTC, current_price)
                        # 注文が成功した場合のみpositionsを更新
                        if order and order.get('id') or order.get('order_id'):
                            positions.append({'price': current_price, 'amount': MIN_ORDER_BTC, 'timestamp': time.time()})
                            set_last_buy_price(state, current_price)
                            # ポジション情報保存
                            try:
                                save_data = {
                                    "positions": positions,
                                    "last_buy_price": get_last_buy_price(state)
                                }
                                with open(positions_file, 'w', encoding='utf-8') as f:
                                    json.dump(save_data, f, ensure_ascii=False, indent=2)
                            except Exception as e:
                                print(f"ポジション保存エラー: {e}")
                            if smtp_host and email_to:
                                subject = f"【買いシグナル】BTC購入推奨 {now}"
                                message = f"【買いシグナル】\n時刻: {now}\n現在価格: {current_price} 円\nRSI: {rsi}\n根拠: {reason}"
                                send_notification(smtp_host, smtp_port, smtp_user, smtp_password, email_to, subject, message)
                            log_trade('buy', current_price, MIN_ORDER_BTC, 'success', reason)
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
            time.sleep(60)  # 1分待機
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
            limit_factor = 1.004 if is_night else 1.002  # 深夜は0.4%上、それ以外は0.2%上
            limit_price = current_price * limit_factor
            order = execute_order(exchange, PAIR, 'sell', MIN_ORDER_BTC, limit_price)
            print(f"[INFO] 指値売り注文発行: {limit_price}")
            for _ in range(10):
                time.sleep(60)
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
            print("[INFO] 強いトレンド中のため逆張り買いを回避")

def compute_indicators(exchange, pair='BTC/JPY', timeframe='1h', limit=1000):
    # Fetch OHLCV and compute a set of indicators. Returns dict of values (may contain None).
    try:
        # OHLCVデータ取得（本番用: 取引所APIから取得）
        raw = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
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
            'sell_pressure': 0,
            'pressure_ratio': None,
            'signal': 'NEUTRAL'
        }

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

def compute_rsi(closes, period=14, exchange=None, pair=None):
    if len(closes) < period:
        return None
    gains = [max(0, closes[i] - closes[i-1]) for i in range(1, len(closes))]
    losses = [max(0, closes[i-1] - closes[i]) for i in range(1, len(closes))]
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_latest_price(exchange, pair):
    return 5000000.0

def send_notification(smtp_host, smtp_port, smtp_user, smtp_password, email_to, subject, message):
    print(f"[通知] {subject}: {message}")

def set_last_buy_price(state, price):
    state['last_buy_price'] = price

def get_last_buy_price(state):
    return state.get('last_buy_price', None)

def trade_decision(current_price, btc_balance, min_order_btc, last_buy_price, rsi, bb_lower):
    return {'action': 'hold'}

def sell_all_positions(positions, exchange, pair):
    return {'result': 'sold'}

def execute_order(exchange, pair, side, amount, price):
    return {'order_id': 'dummy'}

class DummyFundManager:
    def available_fund(self):
        return 1000000
    def place_order(self, cost):
        return True

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
    api_key = os.getenv("BITBANK_API_KEY")
    secret_key = os.getenv("BITBANK_API_SECRET")
    if not api_key or not secret_key:
        raise ValueError("BITBANK_API_KEYまたはBITBANK_API_SECRETが環境変数に設定されていません")
    import ccxt
    return ccxt.bitbank({
        'apiKey': api_key,
        'secret': secret_key,
        'enableRateLimit': True,
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
    # rsi_values: RSI値のリスト
    # period: ボリンジャーバンドの移動平均期間
    # num_std: 標準偏差の倍率(通常2)
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
def trade_decision(current_price, btc_balance=0.0027, buy_btc=None, last_buy_price=None, rsi=None, bb_lower=None):
    """
    # current_price: 現在のBTC/JPY価格
    # btc_balance: 現在のBTC総保有量
    # buy_btc: 売買対象のBTC量(全保有BTCの80%を推奨)
    # last_buy_price: 直近の買値(売却判定に使用)
    # rsi: 最新のRSI値
    # bb_lower: ボリンジャーバンド下限
    """
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
    """
    # 保有BTCの80%を売却する
    # positions: 保有ポジションリスト
    # exchange: ccxtの取引所オブジェクト
    # pair: 通貨ペア(例: 'BTC/JPY')
    """
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
    """
    # btc_balance: 現在のBTC残高
    # trade_result: nの戻り値(dict)
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
    print("[DEBUG] ファイル先頭到達")
    print("mainブロック実行開始")
    print("[INFO] .envからBITBANK_API_KEY/BITBANK_API_SECRETを自動ロードします")
    try:
        print("[DEBUG] Exchange初期化直前")
        exchange = create_bitbank_exchange()  # bitbank用ccxtラッパーを必ず利用
        print("[DEBUG] FundManager初期化直前")
        fund_manager = None  # 必要ならFundManagerクラスをここで初期化
        print("[DEBUG] run_bot呼び出し直前")
        run_bot(exchange, fund_manager)
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
                print(f"⚠️ FundManagerStub初期化エラー: {e}")

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
                print(f"⚠️ positions_state.json予約額取得エラー: {e}")
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
                try:
                    if str(os.getenv('DRY_RUN', '')).lower() in ('1', 'true', 'yes', 'on'):
                        print(f"予約フェーズ: 予約額（JPY）={reserved_from_positions}")
                except Exception as e:
                    print(f"⚠️ reserveデバッグ出力エラー: {e}")
                if self._available - c < 500:
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


def compute_indicators(exchange, pair='BTC/JPY', timeframe='5m', limit=1000):
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
                # 指定価格で指値注文を出す
                order = exchange.create_order(pair, 'limit', 'buy', amount, price)
                log_order("💰 買い", pair, amount, price)
            else:
                # 価格が指定されていなければ注文しない
                log_info("⚠️ 価格未指定のため買い注文をスキップ")
                return None

        elif order_type == 'sell':
            if price:
                order = exchange.create_order(pair, 'limit', 'sell', amount, price)
                log_order("💸 売り", pair, amount, price)
            else:
                # 価格が指定されていなければ注文しない
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





