#!/usr/bin/env python3
# Safe DRY_RUN runner for notify_project
# Usage: python run_dry.py

import os, sys, traceback

# Import-only validation: never start the live bot or place orders from this script.
os.environ['DRY_RUN'] = '1'
os.environ['LOG_DIR'] = os.environ.get('LOG_DIR', os.path.join(os.path.dirname(__file__), 'logs'))

# Reload module if already loaded
if 'ninibo1127' in sys.modules:
    del sys.modules['ninibo1127']

try:
    try:
        import ninibo1127 as m
    except ModuleNotFoundError:
        sys.path.insert(0, os.path.dirname(__file__))
        import ninibo1127 as m
except Exception as e:
    print('ERROR importing ninibo1127:', repr(e))
    traceback.print_exc()
    sys.exit(2)

if not callable(getattr(m, 'run_bot', None)):
    print('ERROR: canonical run_bot is missing')
    sys.exit(3)

print('Import check passed. The bot loop was not started; no orders were sent.')

