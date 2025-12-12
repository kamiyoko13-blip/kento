<<<<<<< HEAD
#!/usr/bin/env python3
# サービス稼働確認スクリプト
# 使い方:
# sudo /home/ninitan/notify_project/venv/bin/python3 monitor.py

import subprocess
import sys

def get_main_pid():
    try:
        out = subprocess.check_output(['systemctl', 'show', '-p', 'MainPID', 'ninibo.service', '--value'])
        pid = out.decode().strip()
        return pid
    except Exception as e:
        print('systemctl コマンド実行エラー:', e)
        return None

def dump_env(pid):
    try:
        # /proc/<pid>/environ を直接読み、NULL区切りを改行に変換して必要なキーだけ表示します。
        env_path = f"/proc/{pid}/environ"
        with open(env_path, 'rb') as f:
            raw = f.read()
        parts = raw.split(b"\x00")
        for p in parts:
            try:
                s = p.decode('utf-8')
            except Exception:
                continue
            if any(k in s for k in ("API_KEY","DRY_RUN","INITIAL_FUND","DEPOSIT_AMOUNT")):
                # 値は表示しないようにマスク
                name = s.split('=',1)[0]
                print(f"{name}=SET")
    except Exception as e:
        print('環境の読み取りでエラー:', e)

if __name__ == '__main__':
    pid = get_main_pid()
    if not pid or pid == '0':
        print('ninibo.service は動作していません。')
        sys.exit(2)
    print('ninibo.service 動作中, PID =', pid)
    print('\n==== マスク済みの環境変数スナップショット ====')
    dump_env(pid)
    print('\nプロセスの稼働状況を確認しました。')
=======
#!/usr/bin/env python3
# サービス稼働確認スクリプト
# 使い方:
# sudo /home/ninitan/notify_project/venv/bin/python3 monitor.py

import subprocess
import sys

def get_main_pid():
    try:
        out = subprocess.check_output(['systemctl', 'show', '-p', 'MainPID', 'ninibo.service', '--value'])
        pid = out.decode().strip()
        return pid
    except Exception as e:
        print('systemctl コマンド実行エラー:', e)
        return None

def dump_env(pid):
    try:
        # /proc/<pid>/environ を直接読み、NULL区切りを改行に変換して必要なキーだけ表示します。
        env_path = f"/proc/{pid}/environ"
        with open(env_path, 'rb') as f:
            raw = f.read()
        parts = raw.split(b"\x00")
        for p in parts:
            try:
                s = p.decode('utf-8')
            except Exception:
                continue
            if any(k in s for k in ("API_KEY","DRY_RUN","INITIAL_FUND","DEPOSIT_AMOUNT")):
                # 値は表示しないようにマスク
                name = s.split('=',1)[0]
                print(f"{name}=SET")
    except Exception as e:
        print('環境の読み取りでエラー:', e)

if __name__ == '__main__':
    pid = get_main_pid()
    if not pid or pid == '0':
        print('ninibo.service は動作していません。')
        sys.exit(2)
    print('ninibo.service 動作中, PID =', pid)
    print('\n==== マスク済みの環境変数スナップショット ====')
    dump_env(pid)
    print('\nプロセスの稼働状況を確認しました。')
>>>>>>> 74f1ab306ca4f7cbafdafeccf820148ccd40d52d
