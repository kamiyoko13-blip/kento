from ninibo1127 import create_bitbank_exchange, run_bot
from funds import FundManager
import os

if __name__ == '__main__':
    # 仮想環境が有効なシェルで実行してください
    exchange = create_bitbank_exchange()
    initial = float(os.getenv('INITIAL_FUND', '20000'))
    fm = FundManager(initial_fund=initial)
    dry_run = str(os.getenv('DRY_RUN', '0')).lower() in ('1', 'true', 'yes', 'on')
    run_bot(exchange, fm, dry_run=dry_run)
