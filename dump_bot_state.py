import json, os, sys
p='bot_state.json'
if not os.path.exists(p):
    print('bot_state.json not found')
    sys.exit(0)
with open(p,'r',encoding='utf-8') as f:
    s=json.load(f)
out = {
  'watch_reference': s.get('watch_reference'),
  'last_position': s.get('last_position'),
  'positions': s.get('positions'),
  'last_buy_time': s.get('last_buy_time') if 'last_buy_time' in s else None
}
<<<<<<< HEAD
print(json.dumps(out, ensure_ascii=True, indent=2))
=======
print(json.dumps(out, ensure_ascii=True, indent=2))
>>>>>>> 74f1ab306ca4f7cbafdafeccf820148ccd40d52d
