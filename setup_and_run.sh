#!/bin/bash
# Anaconda/Miniconda環境でninibo1127.pyを自動セットアップ＆実行

# 環境名
ENV_NAME="botenv"

# Pythonバージョン
PY_VER="3.10"

# 1. conda環境作成（既存ならスキップ）
if ! conda info --envs | grep -q "$ENV_NAME"; then
  conda create -n $ENV_NAME python=$PY_VER -y
fi

# 2. 環境アクティベート
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# 3. 必要パッケージインストール
conda install -c conda-forge ta-lib pandas numpy matplotlib colorama tabulate -y
pip install ccxt websocket-client

# 4. ninibo1127.py 実行
python ninibo1127.py
