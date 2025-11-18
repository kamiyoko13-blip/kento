import importlib, traceback, sys

try:
    importlib.import_module('ninibo1127')
    print('OK: import succeeded')
except Exception:
    traceback.print_exc()
    sys.exit(1)
