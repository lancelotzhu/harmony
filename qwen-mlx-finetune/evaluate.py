#!/usr/bin/env python3
"""Score a fine-tuned reviewer against a labelled JSONL set.

Each line must contain the training `messages` plus `expected` and `cmd`:

    {"messages": [...], "expected": "SAFE" | "DANGEROUS", "cmd": "..."}

The user message is sent exactly as the plugin sends it (a single user turn,
greedy decoding), and the answer is parsed the same way as `parseVerdict`:
take the most severe verdict word found.
"""

import argparse
import contextlib
import json
import re
import sys
import tempfile
from pathlib import Path

VERDICTS = ("SAFE", "RISKY", "DANGEROUS")


def resolve_adapter(path: Path | None, stack: contextlib.ExitStack) -> Path | None:
    """`mlx_lm` only loads `<dir>/adapters.safetensors`.

    Training writes `0000500_adapters.safetensors`, so accept that file
    directly by staging it in a temporary directory next to its config.
    """
    if path is None or path.is_dir():
        return path
    if path.suffix != ".safetensors" or not path.is_file():
        raise SystemExit(f"adapter path is not a directory or .safetensors file: {path}")

    staging = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="mlx-adapter-")))
    (staging / "adapters.safetensors").symlink_to(path.resolve())
    config = path.parent / "adapter_config.json"
    if config.is_file():
        (staging / "adapter_config.json").symlink_to(config.resolve())
    return staging


def parse_verdict(content: str) -> str | None:
    found = [v for v in VERDICTS if v in content.upper()]
    if not found:
        return None
    return max(found, key=VERDICTS.index)


COMMAND_BLOCK = re.compile(r"<command>\n(.*)\n</command>", re.S)


def parse_letter(content: str | None) -> str | None:
    """字母格式的回答取第一个独立的 A/B/C，映射回判定词。

    `parse_verdict` 在这里用不了：字母模型吐的是 `Answer: C`，里面没有
    "DANGEROUS" 这个词。反过来也要小心——不能直接找任意 A/B/C 字符，
    `Answer` 这个词里就有 A，所以先把已知前缀剥掉再按词边界匹配。
    """
    from policy_prompt import CHOICES

    if not content:
        return None
    body = content.strip()
    if body.upper().startswith("ANSWER:"):
        body = body[len("ANSWER:"):].strip()
    match = re.search(r"\b([ABC])\b", body.upper())
    if not match:
        return None
    return CHOICES[ord(match.group(1)) - ord("A")]


def read_rows(path: Path) -> list[dict]:
    """评估集带 `expected` / `cmd`；训练格式的文件（train/valid.jsonl）没有。

    选 checkpoint 必须在验证集上做——拿测试集选型就是用尺子调参。而验证集是
    训练格式，所以这里把 `expected` 和 `cmd` 从 messages 里还原出来，省得为了
    选型再维护一份平行的评估集。
    """
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if "expected" not in row:
                row["expected"] = row["messages"][-1]["content"].strip()
            if "cmd" not in row:
                match = COMMAND_BLOCK.search(row["messages"][0]["content"])
                row["cmd"] = match.group(1) if match else "(无法还原命令)"
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Base model path or HF repo")
    parser.add_argument("--adapter-path", type=Path, help="Optional LoRA adapter")
    parser.add_argument("--data", type=Path, required=True, help="Labelled JSONL")
    # The chat template opens a thinking block, so the model spends
    # `</think>` + `\n\n` before the verdict and `DANGEROUS` costs 4 more
    # tokens: a budget of 6 leaves exactly zero margin, and one stray
    # whitespace token truncates it to `DANGER` and fails to parse.
    parser.add_argument(
        "--format",
        choices=("word", "letter"),
        default="word",
        help="word=模型吐 SAFE/RISKY/DANGEROUS（现行）；"
             "letter=吐 A/B/C，按 /v1/decisions 的形状构造 prompt",
    )
    parser.add_argument("--max-tokens", type=int, default=10)
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Pass enable_thinking=False to the chat template (Qwen instruct models)",
    )
    args = parser.parse_args()

    template_kwargs = {"enable_thinking": False} if args.no_thinking else {}

    from mlx_lm import generate, load

    rows = read_rows(args.data)

    with contextlib.ExitStack() as stack:
        adapter_path = resolve_adapter(args.adapter_path, stack)
        model, tokenizer = load(args.model, adapter_path=adapter_path)

        total = correct = leaked = overblocked = unparsed = 0
        failures = []
        for row in rows:
            expected = row["expected"]
            if args.format == "letter":
                # 字母格式：prompt 以半截的 `Answer:` 结尾，模型接着写字母。
                # 必须用 continue_final_message，不能用 add_generation_prompt——
                # 后者会另起一轮，`Answer:` 前缀就没了，和训练时对不上。
                from prepare_authoritative import make_letter_sample

                messages = make_letter_sample(row["cmd"], "SAFE")["messages"]
                messages[-1]["content"] = "Answer:"
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    continue_final_message=True,
                    **template_kwargs,
                )
            else:
                prompt = tokenizer.apply_chat_template(
                    row["messages"][:-1],
                    tokenize=False,
                    add_generation_prompt=True,
                    **template_kwargs,
                )
            answer = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=args.max_tokens,
                verbose=False,
            )
            got = parse_letter(answer) if args.format == "letter" else parse_verdict(answer)
            total += 1
            if got == expected:
                correct += 1
            elif got is None:
                unparsed += 1
                failures.append((row["cmd"], expected, answer.strip()))
            else:
                # Anything the plugin does not treat as SAFE is still blocked, so
                # only SAFE-when-DANGEROUS is a leak; the reverse is a false block.
                if expected == "DANGEROUS":
                    leaked += 1
                else:
                    overblocked += 1
                failures.append((row["cmd"], expected, got))

    print(f"model:   {args.model}")
    if args.adapter_path:
        print(f"adapter: {args.adapter_path}")
    print(f"samples: {total}")
    print(f"correct: {correct}/{total} ({correct / total:.1%})")
    print(f"漏放 (expected DANGEROUS, got SAFE): {leaked}")
    print(f"误拦 (expected SAFE, got blocked):   {overblocked}")
    print(f"无法解析:                            {unparsed}")
    for cmd, expected, got in failures:
        print(f"  ✗ [{expected}→{got}] {cmd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
