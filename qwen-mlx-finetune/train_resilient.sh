#!/bin/bash
# Run a long LoRA training to completion across Metal OOM crashes.
#
# Peak GPU memory is dominated by the LM head: logits are computed for every
# position against a ~152k vocab, so batch 2 x ~990 tokens sits right at the
# 24.96 GB working-set limit and OOMs on the longest batch.  Batches are built
# from length-sorted indices, so that longest batch is hit once per epoch at a
# point fixed by the seed.  Two mitigations:
#
#   * --batch-size 1 with --grad-accumulation-steps 2 halves the peak while
#     staying mathematically identical to batch 2 (same samples, same number of
#     optimizer updates).  mlx_lm counts --iters in micro-batches, so the CLI
#     gets 2x the step count; STEP units below are always batch-2-equivalent
#     optimizer updates, matching the step numbering of earlier runs.
#   * Each retry uses a fresh seed.  Reusing one replays the same batch order
#     and so re-hits the batch that just crashed.
#
# Checkpoints are renamed to cumulative steps in one flat directory so
# eval_sweep.sh sees a single ordered sweep.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-$([[ -x "$PROJECT_DIR/.venv/bin/python" ]] && echo "$PROJECT_DIR/.venv/bin/python" || echo python)}"
MODEL="${MODEL:-$(ls -d "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/* | head -1)}"
DATA_DIR="${DATA_DIR:-data/policy-v4}"
ADAPTER_DIR="${ADAPTER_DIR:-models/adapters/qwen3.5-0.8b-policy-v4-5k}"
TOTAL_STEPS="${TOTAL_STEPS:-5000}"
MICRO_BATCH="${MICRO_BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
NUM_LAYERS="${NUM_LAYERS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
SAVE_EVERY="${SAVE_EVERY:-250}"
STEPS_PER_EVAL="${STEPS_PER_EVAL:-200}"
VAL_BATCHES="${VAL_BATCHES:-25}"
LOG="${LOG:-models/train-policy-v4-5k.log}"
MAX_SEGMENTS="${MAX_SEGMENTS:-40}"

if [[ ! -s "$DATA_DIR/train.jsonl" ]]; then
  echo "Error: $DATA_DIR/train.jsonl is missing or empty." >&2
  exit 1
fi
mkdir -p "$ADAPTER_DIR"

# A multi-hour run outlives any caffeinate the calling tool holds (the Claude
# CLI takes one with -t 300), and `pmset -g` on this machine reports
# `sleep 1` -- one idle minute and the GPU work stops.  Hold our own assertion
# for exactly as long as this script lives.  Note this cannot defeat clamshell
# sleep: closing the lid still suspends the run.
if command -v caffeinate > /dev/null; then
  caffeinate -i -m -s -w $$ &
  echo "caffeinate holding wake assertion for pid $$"
fi

latest_step() {
  local newest
  newest=$(ls "$ADAPTER_DIR"/[0-9]*_adapters.safetensors 2>/dev/null | sort | tail -1 || true)
  if [[ -z "$newest" ]]; then echo 0; else
    basename "$newest" | sed -E 's/^0*([0-9]+)_adapters\.safetensors$/\1/'
  fi
}

done_steps=$(latest_step)
stalls=0
for (( seg = 1; seg <= MAX_SEGMENTS; seg++ )); do
  (( done_steps >= TOTAL_STEPS )) && break
  remaining=$(( TOTAL_STEPS - done_steps ))
  seed=$(( RANDOM ))

  stage=$(mktemp -d "${TMPDIR:-/tmp}/lora-seg.XXXXXX")
  resume_args=()
  if (( done_steps > 0 )); then
    resume_args=(--resume-adapter-file "$ADAPTER_DIR/$(printf '%07d' "$done_steps")_adapters.safetensors")
  fi

  echo "=== segment $seg: resuming at $done_steps/$TOTAL_STEPS steps, $remaining to go (seed $seed) ===" | tee -a "$LOG"
  set +e
  "$PYTHON" -m mlx_lm lora \
    --model "$MODEL" --train --data "$DATA_DIR" --mask-prompt \
    --fine-tune-type lora --adapter-path "$stage" \
    --num-layers "$NUM_LAYERS" --batch-size "$MICRO_BATCH" \
    --grad-accumulation-steps "$GRAD_ACCUM" \
    --iters $(( remaining * GRAD_ACCUM )) --learning-rate "$LEARNING_RATE" \
    --save-every $(( SAVE_EVERY * GRAD_ACCUM )) \
    --steps-per-eval $(( STEPS_PER_EVAL * GRAD_ACCUM )) \
    --val-batches "$VAL_BATCHES" --grad-checkpoint --seed "$seed" \
    ${resume_args[@]+"${resume_args[@]}"} >> "$LOG" 2>&1  # bash 3.2: empty array is "unbound" under set -u
  status=$?
  set -e

  # Harvest whatever the segment saved, converting micro-batch counts back to
  # optimizer steps and renaming to cumulative numbering.
  segment_best=0
  shopt -s nullglob
  for ckpt in "$stage"/[0-9]*_adapters.safetensors; do
    micro=$(basename "$ckpt" | sed -E 's/^0*([0-9]+)_adapters\.safetensors$/\1/')
    steps=$(( micro / GRAD_ACCUM ))
    cp "$ckpt" "$ADAPTER_DIR/$(printf '%07d' $(( done_steps + steps )))_adapters.safetensors"
    (( steps > segment_best )) && segment_best=$steps
  done
  [[ -f "$stage/adapter_config.json" ]] && cp "$stage/adapter_config.json" "$ADAPTER_DIR/"
  if (( status == 0 && segment_best < remaining )) && [[ -f "$stage/adapters.safetensors" ]]; then
    # Finished cleanly between saves: the final weights live here only.
    cp "$stage/adapters.safetensors" "$ADAPTER_DIR/$(printf '%07d' "$TOTAL_STEPS")_adapters.safetensors"
    segment_best=$remaining
  fi
  shopt -u nullglob
  rm -rf "$stage"

  done_steps=$(( done_steps + segment_best ))
  echo "=== segment $seg ended (exit $status), cumulative steps: $done_steps ===" | tee -a "$LOG"

  if (( segment_best == 0 )); then
    stalls=$(( stalls + 1 ))
    if (( stalls >= 5 )); then
      echo "Error: 5 consecutive segments made no progress; giving up." | tee -a "$LOG" >&2
      exit 1
    fi
    sleep 30
  else
    stalls=0
  fi
done

if (( done_steps < TOTAL_STEPS )); then
  echo "Error: stopped at $done_steps/$TOTAL_STEPS steps." | tee -a "$LOG" >&2
  exit 1
fi

cp "$ADAPTER_DIR/$(printf '%07d' "$TOTAL_STEPS")_adapters.safetensors" "$ADAPTER_DIR/adapters.safetensors"
echo "Training complete: $done_steps steps in $ADAPTER_DIR" | tee -a "$LOG"
