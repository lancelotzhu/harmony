#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

MODEL="${MODEL:-Qwen/Qwen3.5-0.8B-Base}"
ADAPTER_PATH="${ADAPTER_PATH:-models/adapters/qwen3.5-0.8b}"
FUSED_PATH="${FUSED_PATH:-models/fused/qwen3.5-0.8b-lora}"
MLX_PATH="${MLX_PATH:-models/mlx/qwen3.5-0.8b-lora-4bit}"
Q_BITS="${Q_BITS:-4}"

if [[ ! -s "$ADAPTER_PATH/adapters.safetensors" ]]; then
  echo "Error: no trained adapter found at $ADAPTER_PATH" >&2
  exit 1
fi
if [[ -e "$FUSED_PATH" || -e "$MLX_PATH" ]]; then
  echo "Error: refusing to overwrite an existing export directory." >&2
  echo "FUSED_PATH=$FUSED_PATH" >&2
  echo "MLX_PATH=$MLX_PATH" >&2
  echo "Move or remove the existing target directories, or choose new paths." >&2
  exit 1
fi

mkdir -p "$(dirname "$FUSED_PATH")" "$(dirname "$MLX_PATH")"

python -m mlx_lm fuse \
  --model "$MODEL" \
  --adapter-path "$ADAPTER_PATH" \
  --save-path "$FUSED_PATH"

python -m mlx_lm convert \
  --hf-path "$FUSED_PATH" \
  --mlx-path "$MLX_PATH" \
  --quantize \
  --q-bits "$Q_BITS"

echo "Quantized MLX model exported to: $MLX_PATH"
