<<<<<<< HEAD
# NiNiBo プロジェクト — ブートストラップ手順

このファイルは、サーバ起動後や作業再開時に行う「立ち上げ（bootstrap）」の手順をまとめたものです。
必要なコマンドはコピーして 
Tera Term / SSH で実行できます。

## 前提
- プロジェクトのパス: `/home/ninitan/notify_project`（サーバ上）
- ローカル開発は `c:\Users\81806\Desktop\notify_project` に作業コピーあり
- 仮想環境は `venv` に作成
- systemd ユニット: `ninibo.service`（ボット）, `healthcheck.timer`（監視）

---

## 1) ログインと準備
```bash
# SSH でログイン
# 作業ディレクトリへ移動
cd /home/ninitan/notify_project
# 仮想環境を有効化
source venv/bin/activate
python3 --version
```

※ venv が無い場合:
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
# 必要なパッケージがあればインストール
# pip install -r requirements.txt
```

## 2) .env の確認
```bash
# .env 中身確認（機密注意）
sed -n '1,200p' /home/ninitan/notify_project/.env
# 権限確認
ls -l /home/ninitan/notify_project/.env
# 必要なら権限を調整
sudo chown ninitan:ninitan /home/ninitan/notify_project/.env
sudo chmod 600 /home/ninitan/notify_project/.env
```
- DRY_RUN=1 のまま運用しているか確認してください（テスト運用中は DRY_RUN=1 を推奨）。

## 3) systemd の有効化・起動
```bash
# ユニット状態確認
sudo systemctl status ninibo.service -l
sudo systemctl status healthcheck.timer -l

# 起動していなければ有効化して起動
sudo systemctl enable --now ninibo.service
sudo systemctl enable --now healthcheck.timer

# ユニット編集後は再読み込み
sudo systemctl daemon-reload
```

## 4) 動作確認（すぐに実行）
```bash
# サービス状態とログ
sudo systemctl status ninibo.service -l
sudo journalctl -u ninibo.service -n 200 --no-pager -o short-iso -l

# healthcheck の状態とログ
sudo systemctl status healthcheck.timer -l
sudo journalctl -u healthcheck.service -n 200 --no-pager -o short-iso -l

# healthcheck を手動実行してメール送信テスト
/home/ninitan/notify_project/venv/bin/python3 /home/ninitan/notify_project/healthcheck.py

# DRY_RUN の動作確認（run_once がある場合）
/home/ninitan/notify_project/venv/bin/python3 /home/ninitan/notify_project/run_once.py
```

## 5) 24–72 時間の観察
- DRY_RUN=1 のままログと healthcheck の通知を観察してください。
- 異常（例: エラーログ、通知未着、例外）があれば `journalctl` と `healthcheck` の出力を確認し、スクリーンショットやログを保存してから対処します。

## 6) 本番切替（DRY_RUN=0） — 十分観察後に実行
```bash
# バックアップ
cp /home/ninitan/notify_project/.env /home/ninitan/notify_project/.env.bak.$(date +%s)

# 原子的に DRY_RUN=0 に置換
awk 'BEGIN{f=0} /^DRY_RUN=/ { print "DRY_RUN=0"; f=1; next } { print } END { if(!f) print "DRY_RUN=0" }' /home/ninitan/notify_project/.env > /tmp/.env.new.$$ \
  && chmod 600 /tmp/.env.new.$$ \
  && sudo mv /tmp/.env.new.$$ /home/ninitan/notify_project/.env \
  && sudo chown ninitan:ninitan /home/ninitan/notify_project/.env \
  && sudo systemctl restart ninibo.service
```

## 7) 切り戻し（rollback）
```bash
# バックアップファイルがあれば戻す
sudo mv /home/ninitan/notify_project/.env.bak.<timestamp> /home/ninitan/notify_project/.env
sudo chown ninitan:ninitan /home/ninitan/notify_project/.env
sudo chmod 600 /home/ninitan/notify_project/.env
sudo systemctl restart ninibo.service
```

## 8) よく使うトラブルシュートコマンド
```bash
# unit の ExecStart を確認
sudo systemctl show -p ExecStart --value ninibo.service

# 直近ログを追う
sudo journalctl -u ninibo.service -f

# healthcheck を手動で実行して詳細メッセージを確認
/home/ninitan/notify_project/venv/bin/python3 /home/ninitan/notify_project/healthcheck.py
```

---

何か特別に追加してほしい手順（例: Slack 通知手順、ログローテート設定、cron スケジュールの例）があれば教えてください。明日からの立ち上げはこれでスムーズに進められるはずです。
=======
# NiNiBo プロジェクト — ブートストラップ手順

このファイルは、サーバ起動後や作業再開時に行う「立ち上げ（bootstrap）」の手順をまとめたものです。
必要なコマンドはコピーして 
Tera Term / SSH で実行できます。

## 前提
- プロジェクトのパス: `/home/ninitan/notify_project`（サーバ上）
- ローカル開発は `c:\Users\81806\Desktop\notify_project` に作業コピーあり
- 仮想環境は `venv` に作成
- systemd ユニット: `ninibo.service`（ボット）, `healthcheck.timer`（監視）

---

## 1) ログインと準備
```bash
# SSH でログイン
# 作業ディレクトリへ移動
cd /home/ninitan/notify_project
# 仮想環境を有効化
source venv/bin/activate
python3 --version
```

※ venv が無い場合:
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
# 必要なパッケージがあればインストール
# pip install -r requirements.txt
```

## 2) .env の確認
```bash
# .env 中身確認（機密注意）
sed -n '1,200p' /home/ninitan/notify_project/.env
# 権限確認
ls -l /home/ninitan/notify_project/.env
# 必要なら権限を調整
sudo chown ninitan:ninitan /home/ninitan/notify_project/.env
sudo chmod 600 /home/ninitan/notify_project/.env
```
- DRY_RUN=1 のまま運用しているか確認してください（テスト運用中は DRY_RUN=1 を推奨）。

## 3) systemd の有効化・起動
```bash
# ユニット状態確認
sudo systemctl status ninibo.service -l
sudo systemctl status healthcheck.timer -l

# 起動していなければ有効化して起動
sudo systemctl enable --now ninibo.service
sudo systemctl enable --now healthcheck.timer

# ユニット編集後は再読み込み
sudo systemctl daemon-reload
```

## 4) 動作確認（すぐに実行）
```bash
# サービス状態とログ
sudo systemctl status ninibo.service -l
sudo journalctl -u ninibo.service -n 200 --no-pager -o short-iso -l

# healthcheck の状態とログ
sudo systemctl status healthcheck.timer -l
sudo journalctl -u healthcheck.service -n 200 --no-pager -o short-iso -l

# healthcheck を手動実行してメール送信テスト
/home/ninitan/notify_project/venv/bin/python3 /home/ninitan/notify_project/healthcheck.py

# DRY_RUN の動作確認（run_once がある場合）
/home/ninitan/notify_project/venv/bin/python3 /home/ninitan/notify_project/run_once.py
```

## 5) 24–72 時間の観察
- DRY_RUN=1 のままログと healthcheck の通知を観察してください。
- 異常（例: エラーログ、通知未着、例外）があれば `journalctl` と `healthcheck` の出力を確認し、スクリーンショットやログを保存してから対処します。

## 6) 本番切替（DRY_RUN=0） — 十分観察後に実行
```bash
# バックアップ
cp /home/ninitan/notify_project/.env /home/ninitan/notify_project/.env.bak.$(date +%s)

# 原子的に DRY_RUN=0 に置換
awk 'BEGIN{f=0} /^DRY_RUN=/ { print "DRY_RUN=0"; f=1; next } { print } END { if(!f) print "DRY_RUN=0" }' /home/ninitan/notify_project/.env > /tmp/.env.new.$$ \
  && chmod 600 /tmp/.env.new.$$ \
  && sudo mv /tmp/.env.new.$$ /home/ninitan/notify_project/.env \
  && sudo chown ninitan:ninitan /home/ninitan/notify_project/.env \
  && sudo systemctl restart ninibo.service
```

## 7) 切り戻し（rollback）
```bash
# バックアップファイルがあれば戻す
sudo mv /home/ninitan/notify_project/.env.bak.<timestamp> /home/ninitan/notify_project/.env
sudo chown ninitan:ninitan /home/ninitan/notify_project/.env
sudo chmod 600 /home/ninitan/notify_project/.env
sudo systemctl restart ninibo.service
```

## 8) よく使うトラブルシュートコマンド
```bash
# unit の ExecStart を確認
sudo systemctl show -p ExecStart --value ninibo.service

# 直近ログを追う
sudo journalctl -u ninibo.service -f

# healthcheck を手動で実行して詳細メッセージを確認
/home/ninitan/notify_project/venv/bin/python3 /home/ninitan/notify_project/healthcheck.py
```

---

何か特別に追加してほしい手順（例: Slack 通知手順、ログローテート設定、cron スケジュールの例）があれば教えてください。明日からの立ち上げはこれでスムーズに進められるはずです。
>>>>>>> 74f1ab306ca4f7cbafdafeccf820148ccd40d52d
