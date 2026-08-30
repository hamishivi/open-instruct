#!/usr/bin/env python3
"""Prepare accurate on-policy generative-value traces for raw-prompt SFT."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

from open_instruct.value_model_utils import select_gen_value_sft_traces


def read_jsonl(paths: list[pathlib.Path]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected an object at {path}:{line_number}, got {type(row).__name__}.")
                examples.append(row)
    return examples


def outcome_counts(examples: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "correct": sum(float(example["outcome"]) > 0.5 for example in examples),
        "incorrect": sum(float(example["outcome"]) <= 0.5 for example in examples),
    }


def position_counts(examples: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in examples:
        position = str(example.get("state_kind") or "unknown")
        trajectory_fraction = example.get("trajectory_fraction")
        if position != "final_action" and isinstance(trajectory_fraction, int | float):
            fraction = float(trajectory_fraction)
            position = "early" if fraction <= 0.375 else "middle" if fraction <= 0.625 else "late"
        counts[position] = counts.get(position, 0) + 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=pathlib.Path, help="Reservoir JSONL file(s).")
    parser.add_argument("--output", required=True, type=pathlib.Path, help="Filtered SFT JSONL output.")
    parser.add_argument(
        "--max_squared_error",
        type=float,
        default=0.04,
        help="Retain traces whose parsed scalar prediction is within sqrt(threshold) of the outcome.",
    )
    parser.add_argument("--min_critic_version", type=int, default=0)
    parser.add_argument("--max_examples_per_outcome", type=int)
    parser.add_argument("--no_balance_outcomes", action="store_true")
    parser.add_argument("--no_balance_positions", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = read_jsonl(args.inputs)
    selected = select_gen_value_sft_traces(
        examples,
        max_squared_error=args.max_squared_error,
        min_critic_version=args.min_critic_version,
        max_examples_per_outcome=args.max_examples_per_outcome,
        balance_outcomes=not args.no_balance_outcomes,
        balance_positions=not args.no_balance_positions,
        seed=args.seed,
    )
    if not selected:
        raise RuntimeError("No traces passed the SFT selection criteria.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        for example in selected:
            output_file.write(json.dumps(example, ensure_ascii=False) + "\n")
    temporary_path.replace(args.output)

    summary = {
        "input_examples": len(examples),
        "selected_examples": len(selected),
        "selected_by_outcome": outcome_counts(selected),
        "selected_by_position": position_counts(selected),
        "max_squared_error": args.max_squared_error,
        "min_critic_version": args.min_critic_version,
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
