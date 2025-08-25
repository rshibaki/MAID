#!/bin/sh
#$ -S /bin/sh
#$ -ac d=nvcr-pytorch-2309,d_shm=64G
#$ -j y

. /fefs/opt/dgx/env_set/nvcr-pytorch-2309-py3.sh

mkdir -p ~/.raiden/nvcr-pytorch-2309
export PATH="${HOME}/.raiden/nvcr-pytorch-2309/bin:$PATH"
export LD_LIBRARY_PATH="${HOME}/.raiden/nvcr-pytorch-2309/lib:$LD_LIBRARY_PATH"
export LDFLAGS=-L/usr/local/nvidia/lib64
export PYTHONPATH="${HOME}/.raiden/nvcr-pytorch-2309/lib/python3.10/site-packages"
export PYTHONUSERBASE="${HOME}/.raiden/nvcr-pytorch-2309"
export PREFIX="${HOME}/.raiden/nvcr-pytorch-2309"

export MY_PROXY_URL="http://10.1.10.1:8080/"
export HTTP_PROXY=$MY_PROXY_URL
export HTTPS_PROXY=$MY_PROXY_URL
export FTP_PROXY=$MY_PROXY_URL
export http_proxy=$MY_PROXY_URL
export https_proxy=$MY_PROXY_URL
export ftp_proxy=$MY_PROXY_URL



ENV_FILE="$HOME/mykeys.env"
if [ -f "$ENV_FILE" ]; then
  set -a          # 読み込んだ変数をすべて export するモードに
  . "$ENV_FILE"   # KEY=VALUE 形式を展開して環境に反映
  set +a
fi


pip install -r requirements.txt
cd /hss/gMAI/RShibaki_tmp/MAID


python3 run_llm.py --model qwen2.5 --prompt c1 --engine hf

