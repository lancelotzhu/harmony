#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

MODEL="${MODEL:-Qwen/Qwen3.5-0.8B-Base}"
DATA_DIR="${DATA_DIR:-data/processed}"
ADAPTER_PATH="${ADAPTER_PATH:-models/adapters/qwen3.5-0.8b}"
ITERS="${ITERS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_LAYERS="${NUM_LAYERS:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"

if [[ ! -s "$DATA_DIR/train.jsonl" ]]; then
  echo "Error: $DATA_DIR/train.jsonl is missing or empty. Run prepare_data.py first." >&2
  exit 1
fi

mkdir -p "$(dirname "$ADAPTER_PATH")"

python -m mlx_lm lora \
  --model "$MODEL" \
  --train \
  --data "$DATA_DIR" \
  --mask-prompt \
  --fine-tune-type lora \
  --adapter-path "$ADAPTER_PATH" \
  --num-layers "$NUM_LAYERS" \
  --batch-size "$BATCH_SIZE" \
  --iters "$ITERS" \
  --learning-rate "$LEARNING_RATE"
