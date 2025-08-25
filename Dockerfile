FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    git build-essential cmake python3 python3-pip python3-dev libopenblas-dev \
 && rm -rf /var/lib/apt/lists/*

# pip 基盤更新
RUN python3 -m pip install --upgrade pip setuptools wheel packaging

# CUDA 12.1 用の PyTorch（flash-attn用に先入れ）
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

# ★ pyairports を先に“直URL”で入れておく（Python3.10で使える最終版）
#   参考: 公式ホイール (py3-none-any)
RUN pip3 install --no-cache-dir \
  https://files.pythonhosted.org/packages/6e/75/b424aebc9f2fc5db319d5df5fff62fa19254c8ef974c254588d48c480df2/pyairports-2.1.1-py3-none-any.whl

# ★ vLLM→outlines の依存バックトラック抑止（制約ファイルで固定）
RUN printf "outlines==0.0.46\n" > /tmp/constraints.txt

# プロジェクト依存
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt -c /tmp/constraints.txt

WORKDIR /workspace
COPY . /workspace

ENV HF_HOME=/workspace/hf_cache

# --- Jupyter（必要なら変更可） ---
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
