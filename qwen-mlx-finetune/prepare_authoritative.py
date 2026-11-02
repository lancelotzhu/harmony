#!/usr/bin/env python3
"""Build a reviewer dataset from `tomngdev/shell-safety-v1.1`.

The upstream dataset labels commands with `allow` / `ask` / `deny`.  This
project only needs the two ends:

    allow -> SAFE
    deny  -> DANGEROUS
    ask   -> dropped (policy-dependent; keeping it teaches routine commands
             to be treated as dangerous)

Only `shell == "posix"` rows are used.  `git commit` / `git push` without a
force flag are overridden to SAFE because they are routine vibe-coding
operations for this reviewer; `git push --force` stays DANGEROUS.

The prompt template is recovered from an existing labelled file (the first
user message contains the full prompt with the command substituted), so the
generated rows are byte-for-byte in the same shape as the plugin request.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

DATASET_ID = "tomngdev/shell-safety-v1.1"

GIT_ROUTINE = re.compile(r"\bgit(?:\s+-\S+)*\s+(?:commit|push)\b")
GIT_FORCE = re.compile(r"(?:--force(?:-with-lease)?|\s-f\b)")
COMMAND_BLOCK = re.compile(r"<command>\n(.*)\n</command>", re.S)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def recover_prompt(rows: list[dict]) -> str:
    for row in rows:
        match = COMMAND_BLOCK.search(row["messages"][0]["content"])
        if match:
            return row["messages"][0]["content"].replace(match.group(1), "{{COMMAND}}")
    raise SystemExit("could not recover the review prompt from --base-dir")


def command_of(row: dict) -> str | None:
    match = COMMAND_BLOCK.search(row["messages"][0]["content"])
    return match.group(1) if match else None


def label_for(command: str, upstream: str) -> str:
    if GIT_ROUTINE.search(command) and not GIT_FORCE.search(command):
        return "SAFE"
    return "SAFE" if upstream == "allow" else "DANGEROUS"


def make_sample(prompt: str, command: str, label: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": prompt.replace("{{COMMAND}}", command)},
            {"role": "assistant", "content": label},
        ]
    }


def find_dataset(explicit: Path | None) -> Path:
    if explicit:
        return explicit
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            DATASET_ID,
            repo_type="dataset",
            allow_patterns=["*.jsonl", "README.md"],
        )
    )


def collect(split: Path, seen: set[str]) -> dict[str, str]:
    """command -> SAFE/DANGEROUS, de-duplicated and ordered deterministically."""
    pool: dict[str, str] = {}
    for row in read_jsonl(split):
        if row.get("shell") != "posix" or row.get("label") not in {"allow", "deny"}:
            continue
        command = row["command"].strip()
        if not command or command in seen or command in pool:
            continue
        pool[command] = label_for(command, row["label"])
    return pool


def sample(pool: dict[str, str], safe_n: int, dangerous_n: int, rng: random.Random) -> list[tuple[str, str]]:
    safe = sorted(command for command, label in pool.items() if label == "SAFE")
    dangerous = sorted(command for command, label in pool.items() if label == "DANGEROUS")
    rng.shuffle(safe)
    rng.shuffle(dangerous)
    if safe_n > len(safe) or dangerous_n > len(dangerous):
        raise SystemExit(
            f"asked for {safe_n} SAFE / {dangerous_n} DANGEROUS, "
            f"pool only has {len(safe)} / {len(dangerous)}"
        )
    return [(c, "SAFE") for c in safe[:safe_n]] + [(c, "DANGEROUS") for c in dangerous[:dangerous_n]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("data/processed"),
                        help="Directory with the local train.jsonl / valid.jsonl")
    parser.add_argument("--dataset-path", type=Path,
                        help="Local snapshot of the HF dataset (downloaded when omitted)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/authoritative"))
    parser.add_argument("--test-output", type=Path, default=Path("eval/shell-safety-test.jsonl"))
    parser.add_argument("--train-safe", type=int, default=1500)
    parser.add_argument("--train-dangerous", type=int, default=1500)
    parser.add_argument("--valid-safe", type=int, default=50)
    parser.add_argument("--valid-dangerous", type=int, default=50)
    parser.add_argument("--test-safe", type=int, default=100)
    parser.add_argument("--test-dangerous", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_train = read_jsonl(args.base_dir / "train.jsonl")
    base_valid = read_jsonl(args.base_dir / "valid.jsonl")
    prompt = recover_prompt(base_train)
    seen = {command for row in base_train + base_valid if (command := command_of(row))}

    dataset = find_dataset(args.dataset_path)
    train_pool = collect(dataset / "train.jsonl", seen)
    valid_pool = collect(dataset / "validation.jsonl", seen)
    test_pool = collect(dataset / "test.jsonl", seen)

    rng = random.Random(args.seed)
    train_extra = sample(train_pool, args.train_safe, args.train_dangerous, rng)
    valid_extra = sample(valid_pool, args.valid_safe, args.valid_dangerous, rng)
    test = sample(test_pool, args.test_safe, args.test_dangerous, rng)

    train = base_train + [make_sample(prompt, c, l) for c, l in train_extra]
    rng.shuffle(train)
    valid = base_valid + [make_sample(prompt, c, l) for c, l in valid_extra]

    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "valid.jsonl", valid)
    write_jsonl(
        args.test_output,
        [dict(make_sample(prompt, c, l), expected=l, cmd=c) for c, l in test],
    )

    for name, rows in [("train", train), ("valid", valid)]:
        counts = Counter(row["messages"][-1]["content"] for row in rows)
        print(f"{name}: {len(rows)} {dict(counts)}")
    print(f"test: {len(test)} {dict(Counter(l for _, l in test))}")
    print(f"dataset: {dataset}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
