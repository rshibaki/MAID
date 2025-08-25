FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# --- Python 3.10（デフォルト） + 基本ツール ---
RUN apt-get update && apt-get install -y \
    git build-essential cmake \
    python3 python3-dev python3-venv python3-pip \
    libopenblas-dev curl && \
    rm -rf /var/lib/apt/lists/*

# --- pip 等のビルド基盤も固定（再現性UP） ---
RUN pip3 install --no-cache-dir --upgrade \
    pip==24.2 setuptools==70.3.0 wheel==0.44.0

# --- ★ PyTorch (cu121) を厳密固定：2.4.1 系セット ---
RUN pip3 install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

# --- 依存ライブラリは requirements.txt から ---
WORKDIR /workspace
ENV HF_HOME=/workspace/hf_cache
RUN mkdir -p /workspace/hf_cache

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# --- プロジェクト一式 ---
COPY . .

# --- Jupyter（必要なら変更可） ---
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
