#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-$([[ -x "$PROJECT_DIR/.venv/bin/python" ]] && echo "$PROJECT_DIR/.venv/bin/python" || echo python)}"
MODEL="${MODEL:-$(ls -d "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/* | head -1)}"
ADAPTER_DIR="${ADAPTER_DIR:-models/adapters/qwen3.5-0.8b-policy-v4-5k}"
HOLDOUT="${HOLDOUT:-eval/holdout-v4.jsonl}"
TEST_SET="${TEST_SET:-eval/policy-test-v4.jsonl}"
TAG="${TAG:-$(basename "$ADAPTER_DIR")}"
LOG_DIR="${LOG_DIR:-models}"

if [[ ! -d "$ADAPTER_DIR" ]]; then
  echo "Error: adapter directory not found: $ADAPTER_DIR" >&2
  exit 1
fi
for data in "$HOLDOUT" "$TEST_SET"; do
  if [[ ! -s "$data" ]]; then
    echo "Error: eval set is missing or empty: $data" >&2
    exit 1
  fi
done
if pgrep -f "mlx_lm lora" > /dev/null; then
  echo "Error: a training run is using the GPU; evaluating now will OOM both." >&2
  echo "Wait for it to finish, or stop it first." >&2
  exit 1
fi

summary="$LOG_DIR/eval-$TAG-summary.log"
: > "$summary"
# The adapter is only valid against the prompt it was trained on, and a
# mismatched eval set reads as a pure leak, so record the inputs in the log.
{
  echo "$TAG eval summary"
  echo "model:    $MODEL"
  echo "adapters: $ADAPTER_DIR"
  echo "holdout:  $HOLDOUT"
  echo "test set: $TEST_SET"
} >> "$summary"

shopt -s nullglob
checkpoints=("$ADAPTER_DIR"/[0-9]*_adapters.safetensors)
shopt -u nullglob
if (( ${#checkpoints[@]} == 0 )); then
  echo "Error: no numbered checkpoints in $ADAPTER_DIR" >&2
  exit 1
fi

for ckpt in "${checkpoints[@]}"; do
  step=$(basename "$ckpt" | sed -E 's/^0*([0-9]+)_adapters\.safetensors$/\1/')
  log="$LOG_DIR/eval-$TAG-holdout-$step.log"
  "$PYTHON" evaluate.py --model "$MODEL" --adapter-path "$ckpt" --data "$HOLDOUT" > "$log" 2>&1
  {
    echo "== holdout $step =="
    grep -E '^(correct|漏放|误拦|无法解析)' "$log"
  } >> "$summary"
done

# The 282-row test set is ~20x the hold-out cost, so only the final checkpoint.
final="${checkpoints[${#checkpoints[@]}-1]}"  # macOS bash 3.2 has no negative subscripts
step=$(basename "$final" | sed -E 's/^0*([0-9]+)_adapters\.safetensors$/\1/')
log="$LOG_DIR/eval-$TAG-test-$step.log"
"$PYTHON" evaluate.py --model "$MODEL" --adapter-path "$final" --data "$TEST_SET" > "$log" 2>&1
{
  echo "== $(basename "$TEST_SET" .jsonl) $step =="
  grep -E '^(correct|漏放|误拦|无法解析)' "$log"
} >> "$summary"

cat "$summary"
