<<<<<<< HEAD
# notify_project — 小額自動売買ボット (簡易ドキュメント)

このリポジトリには、少額運用向けに安全性を高めた自動売買ロジックが含まれます。
ローカルの Linux サーバ（systemd で動かす環境）で DRY_RUN を使ってまず検証してください。

## 主要な環境変数（推奨値）
- JAPANESE_YEN_BUDGET=2000        # 1回あたりの上限予算（ただし残高割合やバッファで制限されます）
- MIN_ORDER_BTC=0.0001
- MAX_RISK_PERCENT=0.05           # 残高に対する1回の最大リスク割合（例: 0.05 = 5%）
- BALANCE_BUFFER=1000             # 注文後に常に残しておく最低バッファ（JPY）
- DEPOSIT_AMOUNT=100             # 自動入金（1ループあたり最大）
- MIN_BALANCE_THRESHOLD=5000     # これを下回ったら自動入金の検討
- TOPUP_TARGET=10000             # 自動入金する際の目標残高
- DRY_RUN=1                      # テストモード。実注文を行いません。
- DRY_RUN_PRICE=16000000         # DRY_RUN 時の模擬価格（必要なら変更）

## サーバ上でのテスト手順（推奨: Linux の bash）
1. 仮想環境をアクティベートして、必要ライブラリがあるか確認。
2. 環境変数を設定して DRY_RUN を有効にし、単発で実行します。

bash (Linux) の例:

```bash
export DRY_RUN=1
export DRY_RUN_PRICE=16000000
export JAPANESE_YEN_BUDGET=2000
export MAX_RISK_PERCENT=0.05
export BALANCE_BUFFER=1000
python run_once.py
```

PowerShell の例（Windows 上での簡易テスト、ただし文字化けに注意）:

```powershell
$env:DRY_RUN = '1'; $env:DRY_RUN_PRICE = '16000000'; $env:JAPANESE_YEN_BUDGET='2000'; python .\run_once.py
```

> 注: Windows の端末（CP932）だと日本語や絵文字出力でエンコードエラーが出ることがあります。可能ならサーバの Linux 上で実行してください。

## 期待ログ（DRY_RUN）
- DRY_RUN モード起動メッセージ
- 予約フェーズ: 予約額（JPY）決定のログ
- 注文直前の価格再取得と数量再計算のログ
- (DRY) 注文シミュレーションのログ
- 注文成功・残高表示

問題があれば journalctl の抜粋（systemd 実行時）や、`run_once.py` 実行ログをこちらに貼ってください。解析して次の修正案を出します。
=======
# notify_project — 小額自動売買ボット (簡易ドキュメント)

このリポジトリには、少額運用向けに安全性を高めた自動売買ロジックが含まれます。
ローカルの Linux サーバ（systemd で動かす環境）で DRY_RUN を使ってまず検証してください。

## 主要な環境変数（推奨値）
- JAPANESE_YEN_BUDGET=2000        # 1回あたりの上限予算（ただし残高割合やバッファで制限されます）
- MIN_ORDER_BTC=0.0001
- MAX_RISK_PERCENT=0.05           # 残高に対する1回の最大リスク割合（例: 0.05 = 5%）
- BALANCE_BUFFER=1000             # 注文後に常に残しておく最低バッファ（JPY）
- DEPOSIT_AMOUNT=100             # 自動入金（1ループあたり最大）
- MIN_BALANCE_THRESHOLD=5000     # これを下回ったら自動入金の検討
- TOPUP_TARGET=10000             # 自動入金する際の目標残高
- DRY_RUN=1                      # テストモード。実注文を行いません。
- DRY_RUN_PRICE=16000000         # DRY_RUN 時の模擬価格（必要なら変更）

## サーバ上でのテスト手順（推奨: Linux の bash）
1. 仮想環境をアクティベートして、必要ライブラリがあるか確認。
2. 環境変数を設定して DRY_RUN を有効にし、単発で実行します。

bash (Linux) の例:

```bash
export DRY_RUN=1
export DRY_RUN_PRICE=16000000
export JAPANESE_YEN_BUDGET=2000
export MAX_RISK_PERCENT=0.05
export BALANCE_BUFFER=1000
python run_once.py
```

PowerShell の例（Windows 上での簡易テスト、ただし文字化けに注意）:

```powershell
$env:DRY_RUN = '1'; $env:DRY_RUN_PRICE = '16000000'; $env:JAPANESE_YEN_BUDGET='2000'; python .\run_once.py
```

> 注: Windows の端末（CP932）だと日本語や絵文字出力でエンコードエラーが出ることがあります。可能ならサーバの Linux 上で実行してください。

## 期待ログ（DRY_RUN）
- DRY_RUN モード起動メッセージ
- 予約フェーズ: 予約額（JPY）決定のログ
- 注文直前の価格再取得と数量再計算のログ
- (DRY) 注文シミュレーションのログ
- 注文成功・残高表示

問題があれば journalctl の抜粋（systemd 実行時）や、`run_once.py` 実行ログをこちらに貼ってください。解析して次の修正案を出します。
>>>>>>> 74f1ab306ca4f7cbafdafeccf820148ccd40d52d
