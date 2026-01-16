#!/usr/bin/env bash
set -e

MODELS=(
  "Qwen/Qwen2.5-3B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "LiquidAI/LFM2-2.6B-Exp"
  "HuggingFaceH4/zephyr-7b-beta"
  "HuggingFaceH4/zephyr-7b-alpha"
)

for m in "${MODELS[@]}"; do
  python isc1_text_exp.py --model_name "$m"
done