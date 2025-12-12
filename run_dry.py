<<<<<<< HEAD
#!/usr/bin/env python3
# Safe DRY_RUN runner for notify_project
# Usage: python run_dry.py

import os, sys, json, traceback

# Ensure dry-run and logs directory
os.environ['DRY_RUN'] = os.environ.get('DRY_RUN', '1')
os.environ['LOG_DIR'] = os.environ.get('LOG_DIR', os.path.join(os.path.dirname(__file__), 'logs'))

# Reload module if already loaded
if 'ninibo1127' in sys.modules:
    del sys.modules['ninibo1127']

try:
    import ninibo1127 as m
except Exception as e:
    print('ERROR importing ninibo1127:', repr(e))
    traceback.print_exc()
    sys.exit(2)

print('run_bot_di exists:', getattr(m, 'run_bot_di', None) is not None)
try:
    result = m.run_bot_di(dry_run=True)
    try:
        print('RESULT_JSON:', json.dumps(result, ensure_ascii=False))
    except Exception:
        print('RESULT (repr):', repr(result))
except Exception as e:
    print('EXCEPTION during run_bot_di:', repr(e))
    traceback.print_exc()
    sys.exit(3)

print('done')
=======
#!/usr/bin/env python3
# Safe DRY_RUN runner for notify_project
# Usage: python run_dry.py

import os, sys, json, traceback

# Ensure dry-run and logs directory
os.environ['DRY_RUN'] = os.environ.get('DRY_RUN', '1')
os.environ['LOG_DIR'] = os.environ.get('LOG_DIR', os.path.join(os.path.dirname(__file__), 'logs'))

# Reload module if already loaded
if 'ninibo1127' in sys.modules:
    del sys.modules['ninibo1127']

try:
    import ninibo1127 as m
except Exception as e:
    print('ERROR importing ninibo1127:', repr(e))
    traceback.print_exc()
    sys.exit(2)

print('run_bot_di exists:', getattr(m, 'run_bot_di', None) is not None)
try:
    result = m.run_bot_di(dry_run=True)
    try:
        print('RESULT_JSON:', json.dumps(result, ensure_ascii=False))
    except Exception:
        print('RESULT (repr):', repr(result))
except Exception as e:
    print('EXCEPTION during run_bot_di:', repr(e))
    traceback.print_exc()
    sys.exit(3)

print('done')
>>>>>>> 74f1ab306ca4f7cbafdafeccf820148ccd40d52d
