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
import numpy as np
from mlx_lm import load

SAFE_ID, R_ID, D_ID = 80898, 49, 35


def letter_token_ids(tok):
    """每个选项字母的单 token 写法，裸的和带前导空格的。

    和 oMLX `OptionLogitsCapture` 同一口径：两种写法都要，后面用 logaddexp 合并。
    只收单 token 的——多 token 的字母 oMLX 根本读不到，那种情况它会抛
    `DecisionsRequestError`。
    """
    out = []
    for letter in "ABC":
        ids = []
        for form in (letter, " " + letter):
            encoded = tok.encode(form, add_special_tokens=False)
            if len(encoded) == 1 and encoded[0] not in ids:
                ids.append(encoded[0])
        if not ids:
            raise SystemExit(f"字母 {letter} 不是单 token，oMLX 读不到它的 logprob")
        out.append(ids)
    return out


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
    ap.add_argument("--format", choices=("word", "letter"), default="word",
                    help="letter=按 /v1/decisions 的形状打分并只在字母间归一化")
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
            letter_ids = letter_token_ids(tok) if args.format == "letter" else None
            for r in rows:
                if args.format == "letter":
                    """复刻 oMLX 的打分，而不是另算一套。

                    `/v1/decisions` 的做法是：prompt 以半截的 `Answer:` 结尾，
                    取字母 token 的 logprob（裸写法和带前导空格两种用 logaddexp
                    合并），**只在字母之间归一化**。

                    这几步必须一模一样，否则本地扫出来的阈值和线上标定的是两套数，
                    而阈值就活在小数点后那几位里。
                    """
                    from prepare_authoritative import make_letter_sample

                    messages = make_letter_sample(r["cmd"], "SAFE")["messages"]
                    messages[-1]["content"] = "Answer:"
                    prompt = tok.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False,
                        continue_final_message=True, enable_thinking=False,
                    )
                    logits = model(mx.array(tok.encode(prompt))[None])[0, -1].astype(mx.float32)
                    logprobs = np.asarray((logits - mx.logsumexp(logits)).tolist(), dtype=np.float64)
                    scores = np.array([np.logaddexp.reduce(logprobs[ids]) for ids in letter_ids])
                    weights = np.exp(scores - scores.max())
                    probs = weights / weights.sum()
                    scored.append((r["expected"], float(probs[0]), float(probs[1]), float(probs[2]), r["cmd"]))
                    continue
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
