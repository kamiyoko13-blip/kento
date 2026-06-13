#!/usr/bin/env bash
set -e
cd /home/ninitan/notify_project
mkdir -p logs
pkill -f '/home/ninitan/notify_project/ninibo1127.py' || true
nohup /home/ninitan/notify_project/venv/bin/python /home/ninitan/notify_project/ninibo1127.py >> /home/ninitan/notify_project/logs/buy_watch.log 2>&1 < /dev/null &
sleep 1
pgrep -af ninibo1127.py || true
tail -n 30 /home/ninitan/notify_project/logs/buy_watch.log || true
