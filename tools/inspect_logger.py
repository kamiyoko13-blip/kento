<<<<<<< HEAD
import os, importlib

# Ensure environment for logger
os.environ['DRY_RUN'] = '1'
os.environ['LOG_DIR'] = r'C:\Users\81806\Desktop\notify_project\logs'

m = importlib.import_module('ninibo1127')
print('LOG_DIR env:', os.environ.get('LOG_DIR'))
print('module LOG_FILE_PATH:', getattr(m, 'LOG_FILE_PATH', None))
logger = getattr(m, 'logger', None)
print('has_logger:', logger is not None)
if logger is not None:
    print('handlers count:', len(logger.handlers))
    for h in logger.handlers:
        try:
            print(' -', type(h).__name__, getattr(h, 'baseFilename', None))
        except Exception as e:
            print(' - handler info error', e)

# Emit a test log line
try:
    m.log_info('INSPECTOR: test log line')
except Exception as e:
    print('failed to call m.log_info:', e)

print('log file exists:', os.path.exists(os.path.join(os.environ['LOG_DIR'], 'notify_bot.log')))
=======
import os, importlib

# Ensure environment for logger
os.environ['DRY_RUN'] = '1'
os.environ['LOG_DIR'] = r'C:\Users\81806\Desktop\notify_project\logs'

m = importlib.import_module('ninibo1127')
print('LOG_DIR env:', os.environ.get('LOG_DIR'))
print('module LOG_FILE_PATH:', getattr(m, 'LOG_FILE_PATH', None))
logger = getattr(m, 'logger', None)
print('has_logger:', logger is not None)
if logger is not None:
    print('handlers count:', len(logger.handlers))
    for h in logger.handlers:
        try:
            print(' -', type(h).__name__, getattr(h, 'baseFilename', None))
        except Exception as e:
            print(' - handler info error', e)

# Emit a test log line
try:
    m.log_info('INSPECTOR: test log line')
except Exception as e:
    print('failed to call m.log_info:', e)

print('log file exists:', os.path.exists(os.path.join(os.environ['LOG_DIR'], 'notify_bot.log')))
>>>>>>> 74f1ab306ca4f7cbafdafeccf820148ccd40d52d
