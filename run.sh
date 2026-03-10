#!/usr/bin/env bash
set -euo pipefail

export HF_TOKEN="${HF_TOKEN:-}"

# Paths
DATA_DIR="./data"
DATA_PROCESSED_DIR="./data_processed"
STEERING_TAG="steering_d1_t100"
ISC_STRENGTH="strong"

# GPU
CUDA_DEVICE="${1:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Models
MODELS=(
  "Qwen/Qwen2.5-3B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "LiquidAI/LFM2-2.6B-Exp"
  "HuggingFaceH4/zephyr-7b-beta"
  "HuggingFaceH4/zephyr-7b-alpha"
)

MODEL_BASENAMES=(
  "Qwen2.5-3B-Instruct"
  "Qwen2.5-7B-Instruct"
  "Mistral-7B-Instruct-v0.3"
  "LFM2-2.6B-Exp"
  "zephyr-7b-beta"
  "zephyr-7b-alpha"
)

STRENGTHS=("weak" "strong")

# ISC
python isc0_create_splits.py --data_dir "${DATA_DIR}"

for m in "${MODELS[@]}"; do
  echo "model: ${m}"
  python isc1_text_exp.py --model_name "${m}" --data_dir "${DATA_DIR}"
done

for s in "${STRENGTHS[@]}"; do
  for m in "${MODELS[@]}"; do
    python isc2_plotting.py --model_name "${m}" --strength "${s}"
  done
done

# Steering
python steering0_preprocess.py --data_dir "${DATA_DIR}" --output_dir "${DATA_PROCESSED_DIR}"

for m in "${MODELS[@]}"; do
  echo "model: ${m}"
  python steering1_build.py --model_dir "${m}" --data_tags 1 \
    --data_ratio 1 --data_processed_dir "${DATA_PROCESSED_DIR}"
done

for m in "${MODELS[@]}"; do
  echo "model: ${m}"
  python steering2_injection.py --model_dir "${m}" --steering_tag "${STEERING_TAG}" --data_dir "${DATA_DIR}"
done

for b in "${MODEL_BASENAMES[@]}"; do
  python steering2_plotting_inj.py "${b}" --alpha-abs 1 2 3 4 --steering_tag "${STEERING_TAG}"
done

for m in "${MODELS[@]}"; do
  echo "model: ${m}"
  python steering3_cossim.py --model_name "${m}" --steering_tag "${STEERING_TAG}" --isc_strength "${ISC_STRENGTH}"
done

echo ""
echo "All experiments complete."
