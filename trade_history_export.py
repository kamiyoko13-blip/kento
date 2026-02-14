import ccxt
import os
import datetime

# .envからAPIキー等を取得
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('API_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')

# ccxtでbitbankインスタンス作成
def get_bitbank():
    return ccxt.bitbank({
        'apiKey': API_KEY,
        'secret': SECRET_KEY,
        'enableRateLimit': True,
    })

def fetch_trade_history(pair='BTC/JPY', limit=20):
    exchange = get_bitbank()
    # 取引履歴取得
    trades = exchange.fetch_my_trades(pair, limit=limit)
    return trades

def save_trade_history(filename='trade_history.json', pair='BTC/JPY', limit=20):
    trades = fetch_trade_history(pair, limit)
    # 日時を見やすく整形
    for t in trades:
        t['datetime'] = datetime.datetime.fromtimestamp(t['timestamp']/1000).strftime('%Y-%m-%d %H:%M:%S')
    import json
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(trades, f, ensure_ascii=False, indent=2)
    print(f'取引履歴を{filename}に保存しました')

if __name__ == '__main__':
    save_trade_history()
