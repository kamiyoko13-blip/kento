



# DRY_RUN とテスト手順

このファイルは `ninibo1127.py` を安全にテストするための手順をまとめたものです。

前提
- 仮想環境を有効にしていること（例: `source .venv/bin/activate`）
- `requirements.txt` を使って依存をインストール済みであること

依存インストール（未実行の場合）
```bash
python -m pip install -r requirements.txt
```
基本的な DRY_RUN 実行手順
```bash
# 仮想環境内で
export DRY_RUN=1
export JAPANESE_YEN_BUDGET=20000
# 自動リサイズを有効にする場合（任意）
# export AUTO_RESIZE=1
# export AUTO_RESIZE_MAX_MULTIPLIER=1.5
python ninibo1127.py
# 実行後に環境変数を外す
unset DRY_RUN
```

よく使う環境変数
- `DRY_RUN`: 1 にすると実際の注文は行われず ExchangeStub を使ってシミュレーションします。
- `DRY_RUN_PRICE`: DRY_RUN 時の価格（デフォルト 5000000）
- `JAPANESE_YEN_BUDGET`: 1回あたりの注文予算（円）
- `MIN_ORDER_BTC`: 最小注文数量（例: 0.0001）
- `AUTO_RESIZE`: 1 にすると予約額が最小数量未満のときに拡張を試みます（デフォルト OFF）
- `AUTO_RESIZE_MAX_MULTIPLIER`: AUTO_RESIZE 時の最大拡張倍率（デフォルト 1.5）
- `FEE_RATE`: 手数料率（例: 0.001 = 0.1%）。デフォルト 0.001
- `FEE_FIXED_JPY`: 固定手数料（円）。デフォルト 0
- `TRADE_ONLY_WEEKENDS`: 1 にすると土日のみ取引を行います（ローカル or TRADE_TIMEZONE 指定のタイムゾーン）
- `TRADE_TIMEZONE`: 取引日判定に使うタイムゾーン（例: Asia/Tokyo）

期待ログの一例
```
🔧 DRY_RUN enabled — using ExchangeStub (no network calls).
💵 最新の市場価格: 5000000.0 円
✅ 注文可能: 0.0002 BTC (約 20000.00 円, 手数料: 20.00 円)
DEBUG: 予約前 available=20000.0, reserved_budget=20000.00
DEBUG: 予約後 available=0.0
💰 (DRY) 買い注文: 0.0002 BTC （成行）
ℹ️ DRY_RUN: 注文は実行されませんでした（シミュレーション）
✅ 注文が正常に完了しました。
```

トラブルシュート
- メール送信で DNS/接続エラーが出る場合は DRY_RUN を有効にして実行して影響を切り分けてください。
- `AUTO_RESIZE` の挙動を確認するには `JAPANESE_YEN_BUDGET` を小さめに設定して実験してください。

安全上の注意
- 本番環境で `DRY_RUN` を外す前に、十分にテストしてください。
- 取引所の最小数量や手数料ルールは頻繁に変わる可能性があるため、実行前に取引所ドキュメントで確認してください。
=======
# DRY_RUN とテスト手順

このファイルは `ninibo1127.py` を安全にテストするための手順をまとめたものです。

前提
- 仮想環境を有効にしていること（例: `source .venv/bin/activate`）
- `requirements.txt` を使って依存をインストール済みであること

依存インストール（未実行の場合）
```bash
python -m pip install -r requirements.txt
```
>>>>>>> fff5f634858926693a8e82d30e8e22d7b439fdc2
# DRY_RUN とテスト手順

このファイルは `ninibo1127.py` を安全にテストするための手順をまとめたものです。

前提
- 仮想環境を有効にしていること（例: `source .venv/bin/activate`）
- `requirements.txt` を使って依存をインストール済みであること

依存インストール（未実行の場合）
```bash
python -m pip install -r requirements.txt
```

基本的な DRY_RUN 実行手順
```bash
# 仮想環境内で
export DRY_RUN=1
export JAPANESE_YEN_BUDGET=20000
# 自動リサイズを有効にする場合（任意）
# export AUTO_RESIZE=1
# export AUTO_RESIZE_MAX_MULTIPLIER=1.5
python ninibo1127.py
# 実行後に環境変数を外す
unset DRY_RUN
```

よく使う環境変数
- `DRY_RUN`: 1 にすると実際の注文は行われず ExchangeStub を使ってシミュレーションします。
- `DRY_RUN_PRICE`: DRY_RUN 時の価格（デフォルト 5000000）
- `JAPANESE_YEN_BUDGET`: 1回あたりの注文予算（円）
- `MIN_ORDER_BTC`: 最小注文数量（例: 0.0001）
- `AUTO_RESIZE`: 1 にすると予約額が最小数量未満のときに拡張を試みます（デフォルト OFF）
- `AUTO_RESIZE_MAX_MULTIPLIER`: AUTO_RESIZE 時の最大拡張倍率（デフォルト 1.5）
- `FEE_RATE`: 手数料率（例: 0.001 = 0.1%）。デフォルト 0.001
- `FEE_FIXED_JPY`: 固定手数料（円）。デフォルト 0
- `TRADE_ONLY_WEEKENDS`: 1 にすると土日のみ取引を行います（ローカル or TRADE_TIMEZONE 指定のタイムゾーン）
- `TRADE_TIMEZONE`: 取引日判定に使うタイムゾーン（例: Asia/Tokyo）

期待ログの一例
```
🔧 DRY_RUN enabled — using ExchangeStub (no network calls).
💵 最新の市場価格: 5000000.0 円
✅ 注文可能: 0.0002 BTC (約 20000.00 円, 手数料: 20.00 円)
DEBUG: 予約前 available=20000.0, reserved_budget=20000.00
DEBUG: 予約後 available=0.0
💰 (DRY) 買い注文: 0.0002 BTC （成行）
ℹ️ DRY_RUN: 注文は実行されませんでした（シミュレーション）
✅ 注文が正常に完了しました。
```

トラブルシュート
- メール送信で DNS/接続エラーが出る場合は DRY_RUN を有効にして実行して影響を切り分けてください。
- `AUTO_RESIZE` の挙動を確認するには `JAPANESE_YEN_BUDGET` を小さめに設定して実験してください。

安全上の注意
- 本番環境で `DRY_RUN` を外す前に、十分にテストしてください。
- 取引所の最小数量や手数料ルールは頻繁に変わる可能性があるため、実行前に取引所ドキュメントで確認してください。
<<<<<<< HEAD
=======
# DRY_RUN とテスト手順

このファイルは `ninibo1127.py` を安全にテストするための手順をまとめたものです。

前提
- 仮想環境を有効にしていること（例: `source .venv/bin/activate`）
- `requirements.txt` を使って依存をインストール済みであること

依存インストール（未実行の場合）
```bash
python -m pip install -r requirements.txt
```

基本的な DRY_RUN 実行手順
```bash
# 仮想環境内で
export DRY_RUN=1
export JAPANESE_YEN_BUDGET=20000
# 自動リサイズを有効にする場合（任意）
# export AUTO_RESIZE=1
# export AUTO_RESIZE_MAX_MULTIPLIER=1.5
python ninibo1127.py
# 実行後に環境変数を外す
unset DRY_RUN
```

よく使う環境変数
- `DRY_RUN`: 1 にすると実際の注文は行われず ExchangeStub を使ってシミュレーションします。
- `DRY_RUN_PRICE`: DRY_RUN 時の価格（デフォルト 5000000）
- `JAPANESE_YEN_BUDGET`: 1回あたりの注文予算（円）
- `MIN_ORDER_BTC`: 最小注文数量（例: 0.0001）
- `AUTO_RESIZE`: 1 にすると予約額が最小数量未満のときに拡張を試みます（デフォルト OFF）
- `AUTO_RESIZE_MAX_MULTIPLIER`: AUTO_RESIZE 時の最大拡張倍率（デフォルト 1.5）
- `FEE_RATE`: 手数料率（例: 0.001 = 0.1%）。デフォルト 0.001
- `FEE_FIXED_JPY`: 固定手数料（円）。デフォルト 0
- `TRADE_ONLY_WEEKENDS`: 1 にすると土日のみ取引を行います（ローカル or TRADE_TIMEZONE 指定のタイムゾーン）
- `TRADE_TIMEZONE`: 取引日判定に使うタイムゾーン（例: Asia/Tokyo）

期待ログの一例
```
🔧 DRY_RUN enabled — using ExchangeStub (no network calls).
💵 最新の市場価格: 5000000.0 円
✅ 注文可能: 0.0002 BTC (約 20000.00 円, 手数料: 20.00 円)
DEBUG: 予約前 available=20000.0, reserved_budget=20000.00
DEBUG: 予約後 available=0.0
💰 (DRY) 買い注文: 0.0002 BTC （成行）
ℹ️ DRY_RUN: 注文は実行されませんでした（シミュレーション）
✅ 注文が正常に完了しました。
```

トラブルシュート
- メール送信で DNS/接続エラーが出る場合は DRY_RUN を有効にして実行して影響を切り分けてください。
- `AUTO_RESIZE` の挙動を確認するには `JAPANESE_YEN_BUDGET` を小さめに設定して実験してください。

安全上の注意
- 本番環境で `DRY_RUN` を外す前に、十分にテストしてください。
- 取引所の最小数量や手数料ルールは頻繁に変わる可能性があるため、実行前に取引所ドキュメントで確認してください。
>>>>>>> 74f1ab306ca4f7cbafdafeccf820148ccd40d52d
=======
```bash
# 仮想環境内で
export DRY_RUN=1
export JAPANESE_YEN_BUDGET=20000
# 自動リサイズを有効にする場合（任意）
# export AUTO_RESIZE=1
# export AUTO_RESIZE_MAX_MULTIPLIER=1.5
python ninibo1127.py
# 実行後に環境変数を外す
unset DRY_RUN
```

よく使う環境変数
- `DRY_RUN`: 1 にすると実際の注文は行われず ExchangeStub を使ってシミュレーションします。
- `DRY_RUN_PRICE`: DRY_RUN 時の価格（デフォルト 5000000）
- `JAPANESE_YEN_BUDGET`: 1回あたりの注文予算（円）
- `MIN_ORDER_BTC`: 最小注文数量（例: 0.0001）
- `AUTO_RESIZE`: 1 にすると予約額が最小数量未満のときに拡張を試みます（デフォルト OFF）
- `AUTO_RESIZE_MAX_MULTIPLIER`: AUTO_RESIZE 時の最大拡張倍率（デフォルト 1.5）
- `FEE_RATE`: 手数料率（例: 0.001 = 0.1%）。デフォルト 0.001
- `FEE_FIXED_JPY`: 固定手数料（円）。デフォルト 0
- `TRADE_ONLY_WEEKENDS`: 1 にすると土日のみ取引を行います（ローカル or TRADE_TIMEZONE 指定のタイムゾーン）
- `TRADE_TIMEZONE`: 取引日判定に使うタイムゾーン（例: Asia/Tokyo）

期待ログの一例
```
🔧 DRY_RUN enabled — using ExchangeStub (no network calls).
💵 最新の市場価格: 5000000.0 円
✅ 注文可能: 0.0002 BTC (約 20000.00 円, 手数料: 20.00 円)
DEBUG: 予約前 available=20000.0, reserved_budget=20000.00
DEBUG: 予約後 available=0.0
💰 (DRY) 買い注文: 0.0002 BTC （成行）
ℹ️ DRY_RUN: 注文は実行されませんでした（シミュレーション）
✅ 注文が正常に完了しました。
```

トラブルシュート
- メール送信で DNS/接続エラーが出る場合は DRY_RUN を有効にして実行して影響を切り分けてください。
- `AUTO_RESIZE` の挙動を確認するには `JAPANESE_YEN_BUDGET` を小さめに設定して実験してください。

安全上の注意
- 本番環境で `DRY_RUN` を外す前に、十分にテストしてください。
- 取引所の最小数量や手数料ルールは頻繁に変わる可能性があるため、実行前に取引所ドキュメントで確認してください。
>>>>>>> 74f1ab306ca4f7cbafdafeccf820148ccd40d52d
>>>>>>> fff5f634858926693a8e82d30e8e22d7b439fdc2
