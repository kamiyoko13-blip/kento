import json, os, sys
p = 'funds_state_test.json' if os.path.exists('funds_state_test.json') else ('funds_state.json' if os.path.exists('funds_state.json') else None)
if not p:
    print('no funds_state file found')
    sys.exit(0)
with open(p,'r',encoding='utf-8') as f:
    s=json.load(f)
<<<<<<< HEAD
print(json.dumps(s, ensure_ascii=True, indent=2))
=======
print(json.dumps(s, ensure_ascii=True, indent=2))
>>>>>>> 74f1ab306ca4f7cbafdafeccf820148ccd40d52d
