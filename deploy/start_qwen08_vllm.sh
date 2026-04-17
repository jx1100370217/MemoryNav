#!/bin/bash
# 启动 Qwen3.5-0.8B vLLM 服务 (意图分类 + 路径叙述用)
# 用法: bash deploy/start_qwen08_vllm.sh [GPU_ID] [PORT]
# 默认: GPU=0, PORT=8198
# 模型本体 1.72GB, util=0.10 给到 ~4.6GB, 足够 KV cache

set -e

GPU_ID=${1:-0}
PORT=${2:-8198}
MODEL_PATH="$HOME/Disk/models/Qwen3.5-0.8B"
CONDA_ENV="qwen3"

echo "[vLLM-0.8B] Starting Qwen3.5-0.8B on GPU ${GPU_ID}, port ${PORT}..."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

CUDA_VISIBLE_DEVICES=${GPU_ID} vllm serve "${MODEL_PATH}" \
    --port ${PORT} \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.10 \
    --enable-prefix-caching \
    --served-model-name qwen3.5-0.8b \
    --trust-remote-code \
    --no-enable-log-requests
