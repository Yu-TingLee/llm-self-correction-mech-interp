#!/usr/bin/env bash
set -e

MODEL_BASENAME=(
  "Qwen2.5-3B-Instruct"
  "Qwen2.5-7B-Instruct"
  "Mistral-7B-Instruct-v0.3"
  "zephyr-7b-beta"
  "zephyr-7b-alpha"
  "LFM2-2.6B-Exp"
)

for m in "${MODEL_BASENAME[@]}"; do
  python steering2_plotting_inj.py "$m" --alpha-abs 1 2 3 4
done
