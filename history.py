import logging
from logging.handlers import TimedRotatingFileHandler

# ログローテーション（毎日分割・最大7日保存）を管理する関数

def setup_trade_history_logger(log_filename='trade_history.log', backup_count=7):
    handler = TimedRotatingFileHandler(
        filename=log_filename,
        when='midnight',
        interval=1,
        backupCount=backup_count,
        encoding='utf-8'
    )
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger = logging.getLogger('trade_history')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger

# 例：履歴記録用の関数

def log_trade_history(message, level='info'):
    logger = logging.getLogger('trade_history')
    if level == 'info':
        logger.info(message)
    elif level == 'error':
        logger.error(message)
    elif level == 'debug':
        logger.debug(message)
    elif level == 'warning':
        logger.warning(message)
    else:
        logger.info(message)

# 初期化（他ファイルから呼び出し）
# setup_trade_history_logger() を最初に呼び出してください。
