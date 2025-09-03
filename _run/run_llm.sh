#!/bin/sh
#$ -S /bin/sh
#$ -cwd
##$ -N maid-qwen25
# --- ジョブクラス：4GPU / 高速共有ストレージ可 / 72h 上限 ---
#    * g1/g4/g8 でGPU枚数、".72h" で最大経過時間72hに変更
##$ -jc gs-container_g4.72h
##$ -ac d=nvcr-pytorch-2309,d_shm=64G
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

# ===== Hugging Face キャッシュは /hss に置くのが速い（gs-container_* で利用可）=====
export HF_HOME="${HOME}/.cache/hf_cache"
export TRANSFORMERS_CACHE="${HOME}/.cache/transformers"
export HF_DATASETS_CACHE="${HOME}/.cache/hf_datasets"
export DIFFUSERS_CACHE="${HOME}/.cache/diffusers"
# export HF_HOME="/hss/gMAI/RShibaki_tmp/hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=1

cd /hss/gMAI/RShibaki_tmp/MAID

python -m pip install --upgrade --user pip setuptools wheel packaging

# （重要）pyairports を直URLで先に入れる
#   ※ PyPI 検索で見つからないため。直URLなら入ります。
pip install --user \
  https://files.pythonhosted.org/packages/6e/75/b424aebc9f2fc5db319d5df5fff62fa19254c8ef974c254588d48c480df2/pyairports-2.1.1-py3-none-any.whl

# outlines のバージョン制約ファイルを一時作成（vLLM要件を満たす 0.0.46 を固定）
echo "outlines==0.0.46" > /tmp/constraints.txt

pip install --user -r requirements.txt

python - <<'PY'
import huggingface_hub, vllm, outlines, pkgutil
print("OK: huggingface_hub/vllm/outlines importable")
print("has pyairports?", pkgutil.find_loader("pyairports") is not None)
PY

#python3 generate_prompts.py
python3 run_llm.py --model qwen2.5 --prompt s4 --engine hf --d t

