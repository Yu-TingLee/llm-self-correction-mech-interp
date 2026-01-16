#!/usr/bin/env bash
set -e
# source venv/bin/activate

DEFAULT_CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES="${1:-$DEFAULT_CUDA_VISIBLE_DEVICES}"
export CUDA_VISIBLE_DEVICES
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

NCCL_P2P_DISABLE=1
NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE
export NCCL_IB_DISABLE

MODELS=(
  "Qwen/Qwen2.5-3B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "LiquidAI/LFM2-2.6B-Exp"
  "HuggingFaceH4/zephyr-7b-alpha"
  "HuggingFaceH4/zephyr-7b-beta"
)

for m in "${MODELS[@]}"; do
  python3 steering3_cossim.py --model_name "$m"
done
