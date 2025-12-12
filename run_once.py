<<<<<<< HEAD
from ninibo1127 import connect_to_bitbank, FundManager, run_bot
import os

if __name__ == '__main__':
    # 仮想環境が有効なシェルで実行してください
    exchange = connect_to_bitbank()
    initial = float(os.getenv('INITIAL_FUND', '20000'))
    fm = FundManager(initial_fund=initial)
    # run_bot は1回だけ実行するので安全に呼び出せます
    run_bot(exchange, fm)
=======
from ninibo1127 import connect_to_bitbank, FundManager, run_bot
import os

if __name__ == '__main__':
    # 仮想環境が有効なシェルで実行してください
    exchange = connect_to_bitbank()
    initial = float(os.getenv('INITIAL_FUND', '20000'))
    fm = FundManager(initial_fund=initial)
    # run_bot は1回だけ実行するので安全に呼び出せます
    run_bot(exchange, fm)
>>>>>>> 74f1ab306ca4f7cbafdafeccf820148ccd40d52d
