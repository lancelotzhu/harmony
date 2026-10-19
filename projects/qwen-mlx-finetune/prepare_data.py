#!/usr/bin/env python3
"""Validate OpenCode-style chat JSONL and split it for MLX-LM."""

import argparse
import json
import random
import sys
from pathlib import Path


SUPPORTED_ROLES = {"system", "user", "assistant"}


def validate_sample(sample: object, line_number: int) -> dict:
    prefix = f"line {line_number}"
    if not isinstance(sample, dict) or set(sample) != {"messages"}:
        raise ValueError(f"{prefix}: expected an object with only a 'messages' field")

    messages = sample["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{prefix}: 'messages' must be a non-empty array")

    roles = []
    for index, message in enumerate(messages, start=1):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError(
                f"{prefix}, message {index}: expected only 'role' and 'content'"
            )
        role = message["role"]
        if not isinstance(role, str) or role not in SUPPORTED_ROLES:
            raise ValueError(
                f"{prefix}, message {index}: unsupported role {role!r}"
            )
        if not isinstance(message["content"], str) or not message["content"].strip():
            raise ValueError(
                f"{prefix}, message {index}: 'content' must be a non-empty string"
            )
        roles.append(role)

    if "user" not in roles or roles[-1] != "assistant":
        raise ValueError(
            f"{prefix}: sample must contain a user message and end with assistant"
        )
    return sample


def read_samples(input_path: Path) -> list[dict]:
    samples = []
    with input_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"line {line_number}: invalid JSON ({error.msg})"
                ) from error
            samples.append(validate_sample(value, line_number))

    if not samples:
        raise ValueError(f"No training samples found in {input_path}")
    return samples


def write_jsonl(path: Path, samples: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for sample in samples:
            output.write(json.dumps(sample, ensure_ascii=False) + "\n")


def split_samples(
    samples: list[dict], validation_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 2 or validation_fraction == 0:
        return shuffled, []

    validation_count = min(
        len(shuffled) - 1,
        max(1, round(len(shuffled) * validation_fraction)),
    )
    return shuffled[validation_count:], shuffled[:validation_count]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory (default: data/processed)",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Fraction for validation data, from 0 (inclusive) to 1 (exclusive)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    args = parser.parse_args()

    if not 0 <= args.validation_fraction < 1:
        parser.error("--validation-fraction must be in [0, 1)")

    try:
        samples = read_samples(args.input)
        train, valid = split_samples(
            samples, args.validation_fraction, args.seed
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.output_dir / "train.jsonl", train)
        valid_path = args.output_dir / "valid.jsonl"
        if valid:
            write_jsonl(valid_path, valid)
        elif valid_path.exists():
            valid_path.unlink()
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Prepared {len(train)} training samples in {args.output_dir / 'train.jsonl'}")
    if valid:
        print(f"Prepared {len(valid)} validation samples in {valid_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
