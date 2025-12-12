# BTC/JPY 20%変動頻度分析スクリプト
# 必要パッケージ: ccxt, pandas
# 事前に pip install ccxt pandas でインストールしてください

symbol = 'BTC/JPY'
timeframe = '1d'  # 日足
data_limit = 1000  # 取得する最大本数（bitbankは最大1000本）
# df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

import ccxt
import pandas as pd
import datetime


# 1. 取引所インスタンス作成（binance）
exchange = ccxt.binance()

symbol = 'BTC/USDT'
timeframe = '1d'  # 日足
data_limit = 1000  # binanceは一度に1000本まで

# 2. 過去データ取得（binanceは1000本ずつなのでループで全期間取得）
def fetch_ohlcv_all(exchange, symbol, timeframe, since=None, limit=1000):
    all_ohlcv = []
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not ohlcv:
            break
        all_ohlcv += ohlcv
        if len(ohlcv) < limit:
            break
        since = ohlcv[-1][0] + 1  # 次の取得開始時刻
    return all_ohlcv

print('データ取得中...')
all_ohlcv = fetch_ohlcv_all(exchange, symbol, timeframe)
print(f'取得本数: {len(all_ohlcv)}')

# 3. DataFrame化
columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
df = pd.DataFrame(all_ohlcv, columns=columns)
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

# 4. 20%変動頻度の計算
pct = 0.2
count_up = 0
count_down = 0
for i in range(1, len(df)):
    prev_close = df.iloc[i-1]['close']
    high = df.iloc[i]['high']
    low = df.iloc[i]['low']
    if (high - prev_close) / prev_close >= pct:
        count_up += 1
    if (prev_close - low) / prev_close >= pct:
        count_down += 1

print(f'20%以上上昇した日数: {count_up}')
print(f'20%以上下落した日数: {count_down}')
print(f'全体の日数: {len(df)-1}')
print(f'上昇頻度: {count_up/(len(df)-1)*100:.2f}%')
print(f'下落頻度: {count_down/(len(df)-1)*100:.2f}%')

# 5. 直近1年分の頻度も参考表示
df1y = df[df['datetime'] > (df['datetime'].max() - pd.Timedelta(days=365))]
count_up_1y = 0
count_down_1y = 0
for i in range(1, len(df1y)):
    prev_close = df1y.iloc[i-1]['close']
    high = df1y.iloc[i]['high']
    low = df1y.iloc[i]['low']
    if (high - prev_close) / prev_close >= pct:
        count_up_1y += 1
    if (prev_close - low) / prev_close >= pct:
        count_down_1y += 1
print('\n【直近1年】')
print(f'20%以上上昇した日数: {count_up_1y}')
print(f'20%以上下落した日数: {count_down_1y}')
print(f'全体の日数: {len(df1y)-1}')
print(f'上昇頻度: {count_up_1y/(len(df1y)-1)*100:.2f}%')
print(f'下落頻度: {count_down_1y/(len(df1y)-1)*100:.2f}%')
