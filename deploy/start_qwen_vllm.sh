#!/bin/bash
# 启动 Qwen3.5-9B vLLM 服务 (语义增补用)
# 用法: bash deploy/start_qwen_vllm.sh [GPU_ID] [PORT]
# 默认: GPU=1, PORT=8199

set -e

GPU_ID=${1:-1}
PORT=${2:-8199}
MODEL_PATH="$HOME/Disk/models/Qwen3.5-9B"
CONDA_ENV="qwen3"

echo "[vLLM] Starting Qwen3.5-9B on GPU ${GPU_ID}, port ${PORT}..."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

# 确保使用 conda 环境的 libstdc++ (解决 CXXABI_1.3.15 问题)
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

# 使用 vLLM 自带的 attention 后端，不依赖 flash_attn 包
export VLLM_ATTENTION_BACKEND=FLASHINFER

CUDA_VISIBLE_DEVICES=${GPU_ID} vllm serve "${MODEL_PATH}" \
    --port ${PORT} \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --served-model-name qwen3.5-9b \
    --trust-remote-code \
    --no-enable-log-requests
