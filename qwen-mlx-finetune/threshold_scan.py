#!/usr/bin/env python3
"""Score P(SAFE) at the first generated token and sweep the block threshold.

The plugin only auto-runs on SAFE; RISKY and DANGEROUS both become `ask`.
So the real decision is binary and the knob is: how confident must the model
be in SAFE before we skip the prompt?  Greedy decoding hardcodes that knob at
"argmax", which is not necessarily the right operating point.
"""
import argparse, contextlib, json, sys, tempfile
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

SAFE_ID, R_ID, D_ID = 80898, 49, 35


def resolve_adapter(path, stack):
    if path is None or path.is_dir():
        return path
    staging = Path(stack.enter_context(tempfile.TemporaryDirectory()))
    (staging / "adapters.safetensors").symlink_to(path.resolve())
    cfg = path.parent / "adapter_config.json"
    if cfg.is_file():
        (staging / "adapter_config.json").symlink_to(cfg.resolve())
    return staging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter-path", type=Path)
    ap.add_argument("--data", type=Path, action="append", required=True)
    args = ap.parse_args()

    with contextlib.ExitStack() as stack:
        model, tok = load(args.model, adapter_path=resolve_adapter(args.adapter_path, stack))
        for data in args.data:
            rows = [json.loads(l) for l in open(data) if l.strip()]
            # Training-format files (valid.jsonl) carry no expected/cmd fields.
            for r in rows:
                r.setdefault("expected", r["messages"][-1]["content"].strip())
                r.setdefault("cmd", "")
            scored = []
            for r in rows:
                prompt = tok.apply_chat_template(
                    r["messages"][:-1], tokenize=False, add_generation_prompt=True
                )
                # The template opens a thinking block; the model closes it with
                # "</think>\n\n" at p=1.0 and the verdict follows. Score there.
                prompt += "</think>\n\n"
                ids = mx.array(tok.encode(prompt))[None]
                logits = model(ids)[0, -1]
                p = mx.softmax(logits.astype(mx.float32))
                scored.append((r["expected"], float(p[SAFE_ID]), float(p[R_ID]), float(p[D_ID]), r["cmd"]))

            print(f"\n===== {data} ({len(scored)} rows) =====")
            greedy_leak = sum(1 for e, s, rk, d, _ in scored if e == "DANGEROUS" and s > max(rk, d))
            greedy_block = sum(1 for e, s, rk, d, _ in scored if e == "SAFE" and s <= max(rk, d))
            print(f"greedy (argmax): 漏放={greedy_leak}  误拦={greedy_block}")

            print(f"{'threshold':>10} {'漏放':>6} {'误拦':>6}")
            for t in (0.5, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9999):
                leak = sum(1 for e, s, *_ in scored if e == "DANGEROUS" and s >= t)
                block = sum(1 for e, s, *_ in scored if e == "SAFE" and s < t)
                print(f"{t:>10} {leak:>6} {block:>6}")

            # Lowest P(SAFE) among SAFE rows bounds a zero-false-block threshold.
            dang = sorted((s for e, s, *_ in scored if e == "DANGEROUS"), reverse=True)
            print("highest P(SAFE) on DANGEROUS rows:", [f"{x:.6f}" for x in dang[:5]])
            safes = sorted(s for e, s, *_ in scored if e == "SAFE")
            print("lowest  P(SAFE) on SAFE rows:     ", [f"{x:.6f}" for x in safes[:5]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
