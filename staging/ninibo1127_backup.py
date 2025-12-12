<<<<<<< HEAD
from dotenv import load_dotenv  
import os
import ccxt
import time # 価格監視ループに必要

# ==========================================================
# 1. 初期設定と認証 (APIキーの読み込みはここにあります)
# ==========================================================

# config.envからAPIキーを読み込みます
load_dotenv(dotenv_path='config.env') 
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
    
    # ==========================================================
    # 2. メインロジック (1分ごとの価格監視ループ)
    # ==========================================================
    
    print("\n--- 🛒 ボットのメインロジックを開始します (Ctrl+Cで停止) ---")
    
    while True:
        try:
            # 現在のTicker（価格情報）を取得
            ticker = bitbank.fetch_ticker(SYMBOL)
            last_price = ticker['last']
            
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {SYMBOL} 現在の価格: {last_price} JPY")
            
            # 🚨 ここに売買の判定ロジックを追加 🚨
            # あなたの複雑なロジックはここに移植します。
            
        except Exception as e:
            print(f"❌ 価格取得エラーが発生しました: {e}")
            
        # 60秒待機してからループの最初に戻る
        time.sleep(60)
        
except ccxt.base.errors.AuthenticationError as e:
    print("\n❌ 認証エラー: APIキーまたはIPアドレス制限を確認してください。")
    print(f"   詳細: {e}")
except Exception as e:
    print(f"\n❌ 予期せぬエラーが発生しました: {type(e).__name__}: {e}")
import ccxt
import pandas as pd
import time
import os
import datetime
import pytz
import math # mathモジュールを追加
from dotenv import load_dotenv # 最上部でインポート

# ==========================================================
# 🔑 1. グローバルキー読み込みと定義 (修正点: 最上部に移動)
# ==========================================================
# config.envからAPIキーを読み込みます
load_dotenv(dotenv_path='config.env') 
API_KEY = os.getenv("API_KEY") # グローバル定数として定義
SECRET_KEY = os.getenv("SECRET_KEY") # グローバル定数として定義

# 日本標準時 (JST) のタイムゾーンオブジェクトを作成
JST = pytz.timezone('Asia/Tokyo')

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
     
        # ccxtを使ってbitbankに接続
        exchange = ccxt.bitbank({
            'apiKey': api_key,
            'secret': secret_key,
        })
        print("✅ bitbankにccxtで認証接続しました。")
        return exchange

    except Exception as e:
        print(f"❌ bitbankへの接続中にエラーが発生しました: {e}")
        return None

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

    # 🔑 トレンドフィルター
    is_uptrend = latest_data['mid_mavg'] > latest_data['long_mavg']
    mid_mavg_is_rising = latest_data['mid_mavg'] > previous_data['mid_mavg']

    # --- 買いシグナル 1：新規エントリー (ゴールデンクロス) ---
    if (previous_data['short_mavg'] <= previous_data['mid_mavg'] and
        latest_data['short_mavg'] > latest_data['mid_mavg'] and
        is_uptrend and mid_mavg_is_rising):
        signal = 'buy_entry' # 新規エントリーシグナル
        print("✅ 新規エントリーシグナル (GC 25/75、トレンド確認) が発生しました。")

    # --- 買いシグナル 2：買い増し (押し目) ---
    # 注: GC後、ポジション保有中に価格がMA25を上回っている（押し目買い）でトレンド上昇中
    elif latest_data['close'] > latest_data['short_mavg'] and is_uptrend:
        signal = 'buy_add' # 買い増しシグナル
        print("📈 買い増しシグナル (押し目買い) が発生しました。")

    # --- 売りシグナル：全決済 (トレンド終了) ---
    # MA75がMA200を下回った、またはMA75が下向きに転じた
    elif not is_uptrend or latest_data['mid_mavg'] < previous_data['mid_mavg']:
        signal = 'sell_all'
        print("❌ 全決済シグナル (長期トレンド終了/反転) が発生しました。")

    return signal

# === 4. 注文の実行 ===
def execute_order(exchange, pair, order_type, amount, price=None):
    """
    bitbankに注文を出します。(ccxt使用)
    """
    try:
        # ccxtのcreate_orderメソッドを使用
        if order_type == 'buy':
            type = 'limit' if price else 'market' # 指値か成行かを決定
            order = exchange.create_order(pair, type, 'buy', amount, price)
            print(f"💰 {type} 買い注文を発注しました: {amount:.4f} {pair.split('/')[0]} {'@ ' + str(price) if price else ''}")
        elif order_type == 'sell':
            type = 'limit' if price else 'market' # 指値か成行かを決定
            order = exchange.create_order(pair, type, 'sell', amount, price)
            print(f"💸 {type} 売り注文を発注しました: {amount:.4f} {pair.split('/')[0]} {'@ ' + str(price) if price else ''}")
        else:
            print(f"無効な注文タイプです: {order_type}")
            return None

        if order:
            print("注文成功:", order['id']) # IDのみ表示に修正
            return order
        else:
            print("注文に失敗しました:", order)
            return None

    except Exception as e:
        print(f"❌ 注文実行中にエラーが発生しました: {e}")
        return None

# === 5. メインループ（Botの実行部分） ===
# 修正点: グローバルキーを使用するため、api_keyとsecret_keyの引数を削除
def run_bot(pair='BTC/JPY', interval_seconds=3600):
    """
    自動売買Botのメイン実行ループです。(ccxt使用)
    """
    # 接続関数にキーを渡す処理を削除（connect_to_bitbankでグローバルキーを使用するため）
    exchange = connect_to_bitbank()
    if not exchange:
        print("API接続に失敗したためBotを停止します。")
        return

    print(f"Botを {pair} で実行します。データ取得間隔: {interval_seconds}秒 (1時間)")

    # --- 🔑【初期設定】注文数量の計算と最低注文単位のチェック ---

    # 1. 予算と最低取引単位の設定
    JAPANESE_YEN_BUDGET = 10000 # 1回の注文に使う日本円の予算: 10,000円
    MIN_ORDER_BTC = 0.001 # bitbank BTC/JPYの最低注文量 (0.0001 BTC ではありません。ccxtの仕様に合わせるか、bitbankの実際の最小値を確認してください。)
    print(f"💰 1回あたりの注文予算: {JAPANESE_YEN_BUDGET} 円")
    print(f"📉 最低注文数量: {MIN_ORDER_BTC} BTC")

    # 2. 最新の市場価格を取得
    try:
        # 認証不要の public API を使用し、最新価格を取得
        ticker = ccxt.bitbank().fetch_ticker(pair)
        latest_price = ticker['last']
        print(f"💵 最新の市場価格: {latest_price} 円")
    except Exception as e:
        print(f"エラー: 最新価格の取得に失敗しました。Botを停止します: {e}")
        return

    # 3. 注文数量を計算 (予算 ÷ 価格)
    buy_amount_per_order_raw = JAPANESE_YEN_BUDGET / latest_price
    # bitbankの取引単位は0.0001 BTC（小数点以下4桁）です。
    # 例: 0.005 BTC などの有効桁数で切り捨てます。
    # 最小取引単位の桁数に合わせて切り捨てる (0.001の場合は小数点以下3桁)
    # math.floorで、小数点以下4桁目までで切り捨てを実行します。
    # MIN_ORDER_BTC=0.001 の場合、小数点以下3桁に丸める
    decimals = int(-math.log10(MIN_ORDER_BTC)) if MIN_ORDER_BTC < 1 else 0
    buy_amount_per_order = math.floor(buy_amount_per_order_raw * (10**decimals)) / (10**decimals)
    
    if decimals == 0 and MIN_ORDER_BTC == 0.001:
        # bitbank BTC/JPYの最小注文数量は0.0001BTC、注文単位は0.0001BTCです。
        # 0.0001 BTC 単位に丸めるのがより正確です。
        buy_amount_per_order = math.floor(buy_amount_per_order_raw * 10000) / 10000
        MIN_ORDER_BTC = 0.0001
        print("ℹ️ 最小注文数量を 0.0001 BTC に修正し、注文数量を調整しました。")


    # 4. 【安全チェック】予算が最低注文量を下回ったか確認
    if buy_amount_per_order < MIN_ORDER_BTC:
          print("---------------------------------------------------------")
          print("🚨 致命的な警告: 予算見直しが必要です！")
          print(f"現在の予算({JAPANESE_YEN_BUDGET}円)では、最低注文量({MIN_ORDER_BTC} BTC)を")
          print(f"満たせません。現在の計算量: {buy_amount_per_order:.4f} BTC")
          print("`JAPANESE_YEN_BUDGET`を増額するか、`MIN_ORDER_BTC`を確認してください。Botを停止します。")
          print("---------------------------------------------------------")
          return

    print(f"🛒 1回あたりの注文数量: {buy_amount_per_order:.4f} BTC")
    # -----------------------------------------------------

    timeframe = '1h'
    data_limit = 250

    while True:
        try:
            print("\n--- Bot loop iteration started. ---")

            # --- 🔑【追加】ポジションの正確な取得 ---
            # ループの最初にAPIから残高を取得し、Botの記憶を現実に合わせる
            try:
                balance = exchange.fetch_balance()
                # BTCの保有量を正確に取得 (キーは 'BTC' の total)
                current_position_amount_raw = balance['total'].get('BTC', 0.0)
                # 小数点以下4桁に丸める（bitbankの仕様に合わせる）
                current_position_amount = math.floor(current_position_amount_raw * 10000) / 10000
                print(f"✅ APIから取得した正確な保有数量: {current_position_amount:.4f} BTC")
            except Exception as e:
                print(f"⚠️ エラー: 残高の取得に失敗しました。取引をスキップします: {e}")
                time.sleep(interval_seconds)
                continue
            # ------------------------------------

            # 現在の日付と時刻を取得 (JST)
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            now_jst = now_utc.astimezone(JST)
            weekday = now_jst.weekday() # 月曜日が0、日曜日が6

            # 土曜日 (5) または日曜日 (6) でない場合はスキップ
            if weekday < 5:
                print(f"現在時刻 (JST): {now_jst.strftime('%Y-%m-%d %H:%M:%S')} - 平日なので取引をスキップします。")
                time.sleep(interval_seconds)
                continue

            print(f"現在時刻 (JST): {now_jst.strftime('%Y-%m-%d %H:%M:%S')} - 週末なので取引を実行します。")

            # 2. 価格データの取得: 1時間足データを250本取得
            ohlcv_df = get_ohlcv(exchange, pair, timeframe=timeframe, limit=data_limit)

            if ohlcv_df is not None and not ohlcv_df.empty:
                # 3. 売買シグナルの判定（25/75/200 MA）
                signal = generate_signals(ohlcv_df)

                # --- 🔑 損切りロジックの判定と実行 ---
                if current_position_amount > 0.0:
                    latest_close = ohlcv_df.iloc[-1]['close']
                    mid_mavg = ohlcv_df.iloc[-1]['mid_mavg'] # 中期MA(75)を損切りラインとして使用

                    # 損切りシグナル判定: 終値が中期MAを下回った
                    if latest_close < mid_mavg:
                        print(f"🚨 損切りシグナル！終値({latest_close})が中期MA({mid_mavg})を下回りました。")

                        # 損切りを実行（保有数量を全量売り）
                        # 売り数量も小数点以下4桁に丸める必要があります
                        sell_amount = current_position_amount
                        order_result = execute_order(exchange, pair, 'sell', sell_amount, price=None)

                        if order_result:
                            print("🔥 買いポジションを損切りにより全量解消しました。")
                        else:
                            print("⚠️ 損切り注文に失敗しました。")
                        
                        # 損切り後に他の取引を行わないように次のループへ
                        time.sleep(interval_seconds)
                        continue


                # --- 4. 注文の実行 (エントリー/買い増し/全決済) ---

                # 買いシグナル 1：新規エントリー (buy_entry)
                if signal == 'buy_entry' and current_position_amount < MIN_ORDER_BTC: # MIN_ORDER_BTC未満なら新規エントリーと見なす
                    order_result = execute_order(exchange, pair, 'buy', buy_amount_per_order, price=None)

                # 買いシグナル 2：買い増し (buy_add)
                elif signal == 'buy_add' and current_position_amount >= MIN_ORDER_BTC:
                    order_result = execute_order(exchange, pair, 'buy', buy_amount_per_order, price=None)

                # 売りシグナル：全決済 (sell_all)
                elif signal == 'sell_all' and current_position_amount >= MIN_ORDER_BTC:
                    # 全量売り（利確）
                    sell_amount = current_position_amount
                    order_result = execute_order(exchange, pair, 'sell', sell_amount, price=None)

                # ロジックの実行状況を表示
                print(f"現在のシグナル: {signal}, API取得の正確な保有数量: {current_position_amount:.4f} BTC")

            else:
                print("データ取得に失敗したため、次のループに進みます。")

        except Exception as e:
            print(f"Bot実行中に予期せぬエラーが発生しました: {e}")

        # 指定された間隔で待機
        print(f"次回の実行まで {interval_seconds}秒待機します...")
        time.sleep(interval_seconds)

# Botを実行
if __name__ == "__main__":
    # 修正点: run_botの引数からAPIキーを削除
    run_bot('BTC/JPY', 3600) 







=======
from dotenv import load_dotenv  
import os
import ccxt
import time # 価格監視ループに必要

# ==========================================================
# 1. 初期設定と認証 (APIキーの読み込みはここにあります)
# ==========================================================

# config.envからAPIキーを読み込みます
load_dotenv(dotenv_path='config.env') 
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
    
    # ==========================================================
    # 2. メインロジック (1分ごとの価格監視ループ)
    # ==========================================================
    
    print("\n--- 🛒 ボットのメインロジックを開始します (Ctrl+Cで停止) ---")
    
    while True:
        try:
            # 現在のTicker（価格情報）を取得
            ticker = bitbank.fetch_ticker(SYMBOL)
            last_price = ticker['last']
            
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {SYMBOL} 現在の価格: {last_price} JPY")
            
            # 🚨 ここに売買の判定ロジックを追加 🚨
            # あなたの複雑なロジックはここに移植します。
            
        except Exception as e:
            print(f"❌ 価格取得エラーが発生しました: {e}")
            
        # 60秒待機してからループの最初に戻る
        time.sleep(60)
        
except ccxt.base.errors.AuthenticationError as e:
    print("\n❌ 認証エラー: APIキーまたはIPアドレス制限を確認してください。")
    print(f"   詳細: {e}")
except Exception as e:
    print(f"\n❌ 予期せぬエラーが発生しました: {type(e).__name__}: {e}")
import ccxt
import pandas as pd
import time
import os
import datetime
import pytz
import math # mathモジュールを追加
from dotenv import load_dotenv # 最上部でインポート

# ==========================================================
# 🔑 1. グローバルキー読み込みと定義 (修正点: 最上部に移動)
# ==========================================================
# config.envからAPIキーを読み込みます
load_dotenv(dotenv_path='config.env') 
API_KEY = os.getenv("API_KEY") # グローバル定数として定義
SECRET_KEY = os.getenv("SECRET_KEY") # グローバル定数として定義

# 日本標準時 (JST) のタイムゾーンオブジェクトを作成
JST = pytz.timezone('Asia/Tokyo')

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
     
        # ccxtを使ってbitbankに接続
        exchange = ccxt.bitbank({
            'apiKey': api_key,
            'secret': secret_key,
        })
        print("✅ bitbankにccxtで認証接続しました。")
        return exchange

    except Exception as e:
        print(f"❌ bitbankへの接続中にエラーが発生しました: {e}")
        return None

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

    # 🔑 トレンドフィルター
    is_uptrend = latest_data['mid_mavg'] > latest_data['long_mavg']
    mid_mavg_is_rising = latest_data['mid_mavg'] > previous_data['mid_mavg']

    # --- 買いシグナル 1：新規エントリー (ゴールデンクロス) ---
    if (previous_data['short_mavg'] <= previous_data['mid_mavg'] and
        latest_data['short_mavg'] > latest_data['mid_mavg'] and
        is_uptrend and mid_mavg_is_rising):
        signal = 'buy_entry' # 新規エントリーシグナル
        print("✅ 新規エントリーシグナル (GC 25/75、トレンド確認) が発生しました。")

    # --- 買いシグナル 2：買い増し (押し目) ---
    # 注: GC後、ポジション保有中に価格がMA25を上回っている（押し目買い）でトレンド上昇中
    elif latest_data['close'] > latest_data['short_mavg'] and is_uptrend:
        signal = 'buy_add' # 買い増しシグナル
        print("📈 買い増しシグナル (押し目買い) が発生しました。")

    # --- 売りシグナル：全決済 (トレンド終了) ---
    # MA75がMA200を下回った、またはMA75が下向きに転じた
    elif not is_uptrend or latest_data['mid_mavg'] < previous_data['mid_mavg']:
        signal = 'sell_all'
        print("❌ 全決済シグナル (長期トレンド終了/反転) が発生しました。")

    return signal

# === 4. 注文の実行 ===
def execute_order(exchange, pair, order_type, amount, price=None):
    """
    bitbankに注文を出します。(ccxt使用)
    """
    try:
        # ccxtのcreate_orderメソッドを使用
        if order_type == 'buy':
            type = 'limit' if price else 'market' # 指値か成行かを決定
            order = exchange.create_order(pair, type, 'buy', amount, price)
            print(f"💰 {type} 買い注文を発注しました: {amount:.4f} {pair.split('/')[0]} {'@ ' + str(price) if price else ''}")
        elif order_type == 'sell':
            type = 'limit' if price else 'market' # 指値か成行かを決定
            order = exchange.create_order(pair, type, 'sell', amount, price)
            print(f"💸 {type} 売り注文を発注しました: {amount:.4f} {pair.split('/')[0]} {'@ ' + str(price) if price else ''}")
        else:
            print(f"無効な注文タイプです: {order_type}")
            return None

        if order:
            print("注文成功:", order['id']) # IDのみ表示に修正
            return order
        else:
            print("注文に失敗しました:", order)
            return None

    except Exception as e:
        print(f"❌ 注文実行中にエラーが発生しました: {e}")
        return None

# === 5. メインループ（Botの実行部分） ===
# 修正点: グローバルキーを使用するため、api_keyとsecret_keyの引数を削除
def run_bot(pair='BTC/JPY', interval_seconds=3600):
    """
    自動売買Botのメイン実行ループです。(ccxt使用)
    """
    # 接続関数にキーを渡す処理を削除（connect_to_bitbankでグローバルキーを使用するため）
    exchange = connect_to_bitbank()
    if not exchange:
        print("API接続に失敗したためBotを停止します。")
        return

    print(f"Botを {pair} で実行します。データ取得間隔: {interval_seconds}秒 (1時間)")

    # --- 🔑【初期設定】注文数量の計算と最低注文単位のチェック ---

    # 1. 予算と最低取引単位の設定
    JAPANESE_YEN_BUDGET = 10000 # 1回の注文に使う日本円の予算: 10,000円
    MIN_ORDER_BTC = 0.001 # bitbank BTC/JPYの最低注文量 (0.0001 BTC ではありません。ccxtの仕様に合わせるか、bitbankの実際の最小値を確認してください。)
    print(f"💰 1回あたりの注文予算: {JAPANESE_YEN_BUDGET} 円")
    print(f"📉 最低注文数量: {MIN_ORDER_BTC} BTC")

    # 2. 最新の市場価格を取得
    try:
        # 認証不要の public API を使用し、最新価格を取得
        ticker = ccxt.bitbank().fetch_ticker(pair)
        latest_price = ticker['last']
        print(f"💵 最新の市場価格: {latest_price} 円")
    except Exception as e:
        print(f"エラー: 最新価格の取得に失敗しました。Botを停止します: {e}")
        return

    # 3. 注文数量を計算 (予算 ÷ 価格)
    buy_amount_per_order_raw = JAPANESE_YEN_BUDGET / latest_price
    # bitbankの取引単位は0.0001 BTC（小数点以下4桁）です。
    # 例: 0.005 BTC などの有効桁数で切り捨てます。
    # 最小取引単位の桁数に合わせて切り捨てる (0.001の場合は小数点以下3桁)
    # math.floorで、小数点以下4桁目までで切り捨てを実行します。
    # MIN_ORDER_BTC=0.001 の場合、小数点以下3桁に丸める
    decimals = int(-math.log10(MIN_ORDER_BTC)) if MIN_ORDER_BTC < 1 else 0
    buy_amount_per_order = math.floor(buy_amount_per_order_raw * (10**decimals)) / (10**decimals)
    
    if decimals == 0 and MIN_ORDER_BTC == 0.001:
        # bitbank BTC/JPYの最小注文数量は0.0001BTC、注文単位は0.0001BTCです。
        # 0.0001 BTC 単位に丸めるのがより正確です。
        buy_amount_per_order = math.floor(buy_amount_per_order_raw * 10000) / 10000
        MIN_ORDER_BTC = 0.0001
        print("ℹ️ 最小注文数量を 0.0001 BTC に修正し、注文数量を調整しました。")


    # 4. 【安全チェック】予算が最低注文量を下回ったか確認
    if buy_amount_per_order < MIN_ORDER_BTC:
          print("---------------------------------------------------------")
          print("🚨 致命的な警告: 予算見直しが必要です！")
          print(f"現在の予算({JAPANESE_YEN_BUDGET}円)では、最低注文量({MIN_ORDER_BTC} BTC)を")
          print(f"満たせません。現在の計算量: {buy_amount_per_order:.4f} BTC")
          print("`JAPANESE_YEN_BUDGET`を増額するか、`MIN_ORDER_BTC`を確認してください。Botを停止します。")
          print("---------------------------------------------------------")
          return

    print(f"🛒 1回あたりの注文数量: {buy_amount_per_order:.4f} BTC")
    # -----------------------------------------------------

    timeframe = '1h'
    data_limit = 250

    while True:
        try:
            print("\n--- Bot loop iteration started. ---")

            # --- 🔑【追加】ポジションの正確な取得 ---
            # ループの最初にAPIから残高を取得し、Botの記憶を現実に合わせる
            try:
                balance = exchange.fetch_balance()
                # BTCの保有量を正確に取得 (キーは 'BTC' の total)
                current_position_amount_raw = balance['total'].get('BTC', 0.0)
                # 小数点以下4桁に丸める（bitbankの仕様に合わせる）
                current_position_amount = math.floor(current_position_amount_raw * 10000) / 10000
                print(f"✅ APIから取得した正確な保有数量: {current_position_amount:.4f} BTC")
            except Exception as e:
                print(f"⚠️ エラー: 残高の取得に失敗しました。取引をスキップします: {e}")
                time.sleep(interval_seconds)
                continue
            # ------------------------------------

            # 現在の日付と時刻を取得 (JST)
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            now_jst = now_utc.astimezone(JST)
            weekday = now_jst.weekday() # 月曜日が0、日曜日が6

            # 土曜日 (5) または日曜日 (6) でない場合はスキップ
            if weekday < 5:
                print(f"現在時刻 (JST): {now_jst.strftime('%Y-%m-%d %H:%M:%S')} - 平日なので取引をスキップします。")
                time.sleep(interval_seconds)
                continue

            print(f"現在時刻 (JST): {now_jst.strftime('%Y-%m-%d %H:%M:%S')} - 週末なので取引を実行します。")

            # 2. 価格データの取得: 1時間足データを250本取得
            ohlcv_df = get_ohlcv(exchange, pair, timeframe=timeframe, limit=data_limit)

            if ohlcv_df is not None and not ohlcv_df.empty:
                # 3. 売買シグナルの判定（25/75/200 MA）
                signal = generate_signals(ohlcv_df)

                # --- 🔑 損切りロジックの判定と実行 ---
                if current_position_amount > 0.0:
                    latest_close = ohlcv_df.iloc[-1]['close']
                    mid_mavg = ohlcv_df.iloc[-1]['mid_mavg'] # 中期MA(75)を損切りラインとして使用

                    # 損切りシグナル判定: 終値が中期MAを下回った
                    if latest_close < mid_mavg:
                        print(f"🚨 損切りシグナル！終値({latest_close})が中期MA({mid_mavg})を下回りました。")

                        # 損切りを実行（保有数量を全量売り）
                        # 売り数量も小数点以下4桁に丸める必要があります
                        sell_amount = current_position_amount
                        order_result = execute_order(exchange, pair, 'sell', sell_amount, price=None)

                        if order_result:
                            print("🔥 買いポジションを損切りにより全量解消しました。")
                        else:
                            print("⚠️ 損切り注文に失敗しました。")
                        
                        # 損切り後に他の取引を行わないように次のループへ
                        time.sleep(interval_seconds)
                        continue


                # --- 4. 注文の実行 (エントリー/買い増し/全決済) ---

                # 買いシグナル 1：新規エントリー (buy_entry)
                if signal == 'buy_entry' and current_position_amount < MIN_ORDER_BTC: # MIN_ORDER_BTC未満なら新規エントリーと見なす
                    order_result = execute_order(exchange, pair, 'buy', buy_amount_per_order, price=None)

                # 買いシグナル 2：買い増し (buy_add)
                elif signal == 'buy_add' and current_position_amount >= MIN_ORDER_BTC:
                    order_result = execute_order(exchange, pair, 'buy', buy_amount_per_order, price=None)

                # 売りシグナル：全決済 (sell_all)
                elif signal == 'sell_all' and current_position_amount >= MIN_ORDER_BTC:
                    # 全量売り（利確）
                    sell_amount = current_position_amount
                    order_result = execute_order(exchange, pair, 'sell', sell_amount, price=None)

                # ロジックの実行状況を表示
                print(f"現在のシグナル: {signal}, API取得の正確な保有数量: {current_position_amount:.4f} BTC")

            else:
                print("データ取得に失敗したため、次のループに進みます。")

        except Exception as e:
            print(f"Bot実行中に予期せぬエラーが発生しました: {e}")

        # 指定された間隔で待機
        print(f"次回の実行まで {interval_seconds}秒待機します...")
        time.sleep(interval_seconds)

# Botを実行
if __name__ == "__main__":
    # 修正点: run_botの引数からAPIキーを削除
    run_bot('BTC/JPY', 3600) 







>>>>>>> 74f1ab306ca4f7cbafdafeccf820148ccd40d52d
