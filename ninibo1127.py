# python 3.14環境で動作確認済み (仮想環境venv314を使用)
# === 必要なライブラリを1回ずつインポート（心臓部の準備） ===
try:
    from fund_manager import FundManager  # ←★資金管理クラスのインポート（存在する場合）
except Exception:
    # fund_manager が見つからない環境でも動作するよう最小限のスタブを定義します
    class FundManager:
        def __init__(self, *args, **kwargs):
            pass

import os
import time
import datetime
import math
import pandas as pd
from zoneinfo import ZoneInfo  # 標準ライブラリのタイムゾーン処理

import ccxt  # pyright: ignore[reportMissingImports]
from dotenv import load_dotenv

# === 日本時間のタイムゾーン定義 ===
JST = ZoneInfo('Asia/Tokyo')

# === .envファイルの読み込み（同じフォルダにある場合） ===

load_dotenv(dotenv_path=".env")

# === 環境変数の取得 ===
smtp_user = os.getenv("SMTP_USER")
smtp_password = os.getenv("SMTP_PASSWORD")
email_to = os.getenv("EMAIL_TO")
smtp_server = os.getenv("SMTP_SERVER")
if smtp_server is None:
    raise ValueError("SMTP_SERVER が .env に設定されていません")

subject = os.getenv("SUBJECT", "📬 通知")

# === SMTP_PORT の安全な読み込み ===
port_str = os.getenv("SMTP_PORT")
if port_str is None:
    raise ValueError("SMTP_PORT が .env に設定されていません")
smtp_port = int(port_str)

# === メール送信関数 ===
def send_notification(smtp_server, smtp_port, smtp_user, smtp_password, to, subject, body):
    from email.mime.text import MIMEText
    import smtplib

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        print("✅ メール送信成功")
    except Exception as e:
        print(f"❌ メール送信失敗: {e}")


# 取引所の設定を取得
exchange_name = os.getenv("EXCHANGE", "bitbank")


# === メイン処理開始（Botの心臓が動き出す） ===
if __name__ == "__main__":
    print("Bot起動中...")
    # run_botの定義後に呼び出すように移動しました

# 1. 初期設定と認証 (APIキーの読み込みはここにあります)

# .envファイルからAPIキーを読み込みます（config.envから統合済み）

load_dotenv(dotenv_path='.env') 
api_key = os.getenv("API_KEY")
secret_key = os.getenv("SECRET_KEY")

print(f"✅ APIキーが読み込まれましたか: {bool(api_key)}")

# bitbankの取引所インスタンスを作成（認証情報込みで修正）
# 🚨 bitbank = ccxt.bitbank() の行を認証情報付きに置き換えています
bitbank = ccxt.bitbank({
    'apiKey': api_key,
    'secret': secret_key,
})

SYMBOL = 'BTC/JPY'

try:
    # 接続テストとして残高を取得 (認証が必要な操作)
    print("\n🚀 Bitbankへの認証接続をテスト中...")
    balance = bitbank.fetch_balance()
    
    jpy_balance = balance['total'].get('JPY')
    btc_balance = balance['total'].get('BTC')
    
    print("---------------------------------------")
    print("✅ 接続・認証に成功しました！")
    print(f"   現在の残高: {jpy_balance} JPY / {btc_balance} BTC")
    print("---------------------------------------")

except Exception as e:
    print(f"❌ Bitbankへの接続に失敗しました: {e}")
    exit(1)  # 必要ならプログラムを終了
    
    # ==========================================================
    # 1. メインロジック (1分ごとの価格監視ループ)
    # ==========================================================
    
    print("\n--- 🛒 ボットのメインロジックを開始します (Ctrl+Cで停止) ---")
    
    while True:
        try:
            # 現在のTicker（価格情報）を取得
            ticker = bitbank.fetch_ticker(SYMBOL)
            last_price = ticker['last']
            
            # JSTでログ出力

            now = datetime.datetime.now(JST)
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp} JST] {SYMBOL} 現在の価格: {last_price} JPY")

             # 🚨 売買ロジックをここに追加

        except Exception as e:
            print(f"❌ 価格取得エラーが発生しました: {e}")  # ← インデントが浅すぎてエラー

        # 60秒待機
        time.sleep(60)


    balance = exchange.fetch_balance()
except ccxt.AuthenticationError as e:
    print("認証エラー:", e)

    print("\n❌ 認証エラー: APIキーまたはIPアドレス制限を確認してください。")
    print(f"   詳細: {e}")
except Exception as e:
    print(f"\n❌ 予期せぬエラーが発生しました: {type(e).__name__}: {e}")


# ==========================================================
# 🔑 2. グローバルキー読み込みと定義 (修正点: 最上部に移動)
# ==========================================================
#.env# config.envからAPIキーを読み込みます

load_dotenv(dotenv_path='.env') 
API_KEY = os.getenv("API_KEY") # グローバル定数として定義
SECRET_KEY = os.getenv("SECRET_KEY") # グローバル定数として定義

# 日本標準時 (JST) のタイムゾーンオブジェクトを作成
JST = ZoneInfo('Asia/Tokyo')

# === 1. 取引所への接続 ===
# 修正点: グローバルキーを使用するため引数を削除し、冗長なコードを削除
def connect_to_bitbank():
    """bitbankに接続します。グローバルで読み込んだAPIキーを使用します。"""
    try:
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
# 修正点: グローバルキーを使用するため、api_keyとsecret_keyの引数を削除
def run_bot(pair='BTC/JPY', interval_seconds=3600):
    """
    自動売買Botのメイン実行ループです。(ccxt使用)
    """
    # 接続関数にキーを渡す処理を削除（connect_to_bitbankでグローバルキーを使用するため）
    exchange = connect_to_bitbank()
    fund_manager = FundManager(initial_fund=20000)  # 初期資金2万円
    if not exchange:
        print("API接続に失敗したためBotを停止します。")
        return

    print(f"Botを {pair} で実行します。データ取得間隔: {interval_seconds}秒 (1時間)")

    # --- 🔑【初期設定】注文数量の計算と最低注文単位のチェック ---

    # 1. 予算と最低取引単位の設定
    JAPANESE_YEN_BUDGET = 10000 # 1回の注文に使う日本円の予算: 10,000円
    MIN_ORDER_BTC = 0.0001 # bitbank BTC/JPYの最低注文量 # bitbank BTC/JPYの最小注文量。ccxtの仕様と一致しているか確認すること。

    print(f"💰 1回あたりの注文予算: {JAPANESE_YEN_BUDGET} 円")
    print(f"📉 最低注文数量: {MIN_ORDER_BTC} BTC")

    # 2. 最新の市場価格を取得
    try:
        # 認証不要の public API を使用し、最新価格を取得
        ticker = ccxt.bitbank().fetch_ticker(pair)
        latest_price= ticker['last']
        print(f"💵 最新の市場価格: {latest_price} 円")
 
    # 注文数量を計算（bitbankの最小注文単位に合わせて丸める）
    # 3. 注文数量を計算 (予算 ÷ 価格)
        buy_amount_raw = JAPANESE_YEN_BUDGET / latest_price
        decimals = int(-math.log10(MIN_ORDER_BTC)) if MIN_ORDER_BTC < 1 else 0
        buy_amount = math.floor(buy_amount_raw * (10**decimals)) / (10**decimals)

    # 注文前に使う
        if buy_amount >= MIN_ORDER_BTC:
          order_cost = buy_amount * latest_price
          print(f"✅ 注文可能: {buy_amount} BTC (約 {order_cost:.2f} 円)")
          (f"💰 残高: {fund_manager.available_fund():.2f} 円")

    except Exception as e:
        print(f"エラー: 最新価格の取得に失敗しました。Botを停止します: {e}")
        return      
    # 例: 0.005 BTC などの有効桁数で切り捨てます。
    # 最小取引単位の桁数に合わせて切り捨てる (0.001の場合は小数点以下3桁)
    # math.floorで、小数点以下4桁目までで切り捨てを実行します。  print(f"エラー: 最新価格の取得に失敗しました。Botを停止します: {e}")
    
    # 4. 注文数量の計算と丸め処理
    # MIN_ORDER_BTC=0.001 の場合、小数点以下3桁に丸める
    decimals = int(-math.log10(MIN_ORDER_BTC)) if MIN_ORDER_BTC < 1 else 0
    buy_amount = math.floor(buy_amount_raw * (10**decimals)) / (10**decimals)
    
    if decimals == 0 and MIN_ORDER_BTC == 0.0001:
        # bitbank BTC/JPYの最小注文数量は0.0001BTC、注文単位は0.0001BTCです。
        # 0.0001 BTC 単位に丸めるのがより正確です。
        buy_amount = math.floor(buy_amount_raw * 10000) / 10000
        MIN_ORDER_BTC = 0.0001
        print("ℹ️ 最小注文数量を 0.0001 BTC に修正し、注文数量を調整しました。")

    print(f"🧮 注文数量: {buy_amount} BTC")

    # 5. 最低注文数量のチェックと注文の実行
    if  buy_amount >= MIN_ORDER_BTC:
        order_cost = buy_amount * latest_price
        print(f"✅ 注文可能: {buy_amount} BTC (約 {order_cost:.2f} 円) は最低注文量を満たしています。")
    # 注: 実際に取引を発行する場合は execute_order を呼ぶか、明示的に order を作成してください。
    # ここでは例として FundManager による資金管理処理を試行し、例外をキャッチします。
    try:
        # 実注文を行う場合のサンプル（コメントアウト）:
        # order = execute_order(exchange, pair, 'buy', buy_amount)
        # 今回はシミュレーション用のダミー注文情報を作成します
        order = {'id': 'simulated_order', 'amount': buy_amount, 'cost': order_cost}

        # FundManager に残高消費を通知（実装に依存）
        if hasattr(fund_manager, "place_order"):
            fund_manager.place_order(order_cost)

        print(f"💰 注文後の残高: {fund_manager.available_fund():.2f} 円")
    
        print("✅ 注文が正常に完了しました。")
    except Exception as e:
        print(f"⚠️ 注文に失敗しました: {e}")

    # 注文後に共通で実行したい処理（成功でも失敗でも）
    required_cost = buy_amount * latest_price
    try: 
        if hasattr(fund_manager, "available_fund"):
            available = fund_manager.available_fund() if hasattr(fund_manager, "available_fund") else None
            if available is not None:
                print(f"🚫  残高不足のため注文をスキップします（必要: {required_cost:.2f} 円, 残高: {available:.2f} 円）")
        else:
            print(f"🚫 残高不足のため注文をスキップします（必要: {required_cost:.2f} 円）")
    except Exception:
        print(f"🚫  残高不足のため注文をスキップします（必要: {required_cost:.2f} 円）")

# 必要ならここでループ化や継続処理を追加できます。現状は初期チェック後に終了します。
    return


# Botを実行
if __name__ == "__main__":
    print("🔁 自動売買Botを継続運用モードで起動します")
    while True:
        run_bot('BTC/JPY', 3600)
        time.sleep(3600)  # 1時間待機
