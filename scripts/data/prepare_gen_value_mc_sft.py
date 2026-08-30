#!/usr/bin/env python3
"""Convert exact-prefix Monte Carlo values into direct generative-value SFT targets.

The standard generative-critic SFT objective supervises every token in a long
teacher critique, so the scalar judgment can receive little effective weight.
This ablation retains the exact online critic prompt but uses a concise
``<answer>N</answer>`` completion, making every supervised completion token part
of the value judgment. The target is computed from fresh continuations rather
than the outcome of the sampled trajectory that supplied the prefix.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from collections.abc import Sequence
from typing import Any

from open_instruct import value_model_utils


def optional_sequence_as_list(value: Any) -> list[Any]:
    """Normalize list-like parquet cells without ambiguous array truth tests."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return list(value)


def quantize_mc_value(value: float, *, score_max: int = 10) -> int:
    """Map a probability to the nearest integer score with half-up rounding."""
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"Monte Carlo value must be finite and in [0, 1], got {value}.")
    if score_max <= 0:
        raise ValueError(f"score_max must be positive, got {score_max}.")
    return min(score_max, max(0, math.floor(value * score_max + 0.5)))


def build_mc_sft_examples(
    rows: Sequence[dict[str, Any]],
    *,
    tokenizer: Any,
    min_continuations: int,
    score_max: int = 10,
    gen_value_conditioning: str = "none",
) -> list[dict[str, Any]]:
    """Build prompt/completion examples from value-estimation parquet rows."""
    if min_continuations <= 0:
        raise ValueError(f"min_continuations must be positive, got {min_continuations}.")
    if gen_value_conditioning not in {"none", "gt"}:
        raise ValueError(
            "Direct MC value SFT supports only unconditioned or reference-answer-conditioned prompts; "
            f"got {gen_value_conditioning!r}."
        )

    examples: list[dict[str, Any]] = []
    prompts_seen: set[str] = set()
    for row_index, row in enumerate(rows):
        probe_positions = [int(position) for position in optional_sequence_as_list(row.get("probe_positions"))]
        mc_values = [float(value) for value in optional_sequence_as_list(row.get("mc_values"))]
        if len(probe_positions) != len(mc_values):
            raise ValueError(
                f"Row {row_index} has {len(probe_positions)} probe positions but {len(mc_values)} MC values."
            )
        num_continuations = int(row.get("num_continuations", 0))
        if num_continuations < min_continuations:
            raise ValueError(
                f"Row {row_index} has only {num_continuations} continuations; at least {min_continuations} required."
            )
        rollout_tokens = [int(token) for token in optional_sequence_as_list(row.get("rollout_tokens"))]
        response_token_limit = int(row.get("response_token_limit", 8192))
        problem = str(row.get("problem", row.get("prompt", "")))
        if not problem:
            raise ValueError(f"Row {row_index} has no problem text.")
        ground_truth = str(row.get("ground_truth", ""))
        if gen_value_conditioning == "gt" and not ground_truth:
            raise ValueError(f"Row {row_index} has no ground truth for reference-answer conditioning.")

        for probe_position, mc_value in zip(probe_positions, mc_values):
            if not 0 <= probe_position <= len(rollout_tokens):
                raise ValueError(
                    f"Row {row_index} probe position {probe_position} is outside rollout length {len(rollout_tokens)}."
                )
            # Match the online critic prompt exactly. Special response tokens can
            # encode meaningful causal boundaries and must not disappear during
            # offline SFT conversion.
            partial_response = tokenizer.decode(rollout_tokens[:probe_position], skip_special_tokens=False)
            prompt = value_model_utils.build_generative_value_prompt(
                partial_response,
                conditioning=gen_value_conditioning,
                ground_truth=ground_truth,
                problem=problem,
                actor_model_name=str(row.get("actor_model_name", "")) or None,
                actor_success_rate=(
                    float(row["actor_success_rate"]) if row.get("actor_success_rate") is not None else None
                ),
                response_tokens_used=probe_position,
                response_token_limit=response_token_limit,
            )
            if prompt in prompts_seen:
                raise ValueError(
                    f"Duplicate exact critic prompt generated for row {row_index}, probe {probe_position}."
                )
            prompts_seen.add(prompt)
            score = quantize_mc_value(mc_value, score_max=score_max)
            state_kind = "final_action" if probe_position == len(rollout_tokens) - 1 else "segment_start"
            # Match online critic metadata: zero at the first action and one at
            # the causal state immediately before the sampled final action.
            trajectory_fraction = probe_position / max(len(rollout_tokens) - 1, 1)
            examples.append(
                {
                    "prompt": prompt,
                    "generation": f" <answer>{score}</answer>",
                    "target": mc_value,
                    "outcome": mc_value,
                    "prediction": score / score_max,
                    "squared_error": (score / score_max - mc_value) ** 2,
                    "state_kind": state_kind,
                    "response_tokens_used": probe_position,
                    "response_token_limit": response_token_limit,
                    "rollout_length": len(rollout_tokens),
                    "trajectory_fraction": trajectory_fraction,
                    "num_continuations": num_continuations,
                    "problem": problem,
                    "ground_truth": ground_truth,
                    "gen_value_conditioning": gen_value_conditioning,
                    "direct_mc_score_supervision": True,
                }
            )
    return examples


def repeat_examples_for_horizon(
    examples: Sequence[dict[str, Any]],
    *,
    final_action_repeat: int,
    late_state_repeat: int,
    late_state_fraction: float,
) -> list[dict[str, Any]]:
    """Deterministically upweight final-action and late-trajectory states.

    Final-action states take precedence over the late-state multiplier so the
    two repeat factors do not multiply unexpectedly.
    """
    if final_action_repeat <= 0 or late_state_repeat <= 0:
        raise ValueError("final_action_repeat and late_state_repeat must be positive integers.")
    if not 0.0 <= late_state_fraction <= 1.0:
        raise ValueError(f"late_state_fraction must be in [0, 1], got {late_state_fraction}.")

    repeated: list[dict[str, Any]] = []
    for example in examples:
        if example.get("state_kind") == "final_action":
            repeat_count = final_action_repeat
        elif float(example.get("trajectory_fraction", 0.0)) >= late_state_fraction:
            repeat_count = late_state_repeat
        else:
            repeat_count = 1
        for repeat_index in range(repeat_count):
            repeated.append(
                {
                    **example,
                    "horizon_repeat_count": repeat_count,
                    "horizon_repeat_index": repeat_index,
                }
            )
    return repeated


def read_excluded_problems(path: pathlib.Path | None) -> set[str]:
    if path is None:
        return set()
    import pandas as pd  # noqa: PLC0415

    frame = pd.read_parquet(path, columns=["problem"])
    return {problem for problem in frame["problem"].tolist() if isinstance(problem, str) and problem}


def write_jsonl(path: pathlib.Path, examples: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        for example in examples:
            output_file.write(json.dumps(example, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_parquet", type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--tokenizer_name_or_path", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--exclude_problem_dataset_path", type=pathlib.Path)
    parser.add_argument("--min_continuations", type=int, default=16)
    parser.add_argument("--min_examples", type=int, default=256)
    parser.add_argument("--score_max", type=int, default=10)
    parser.add_argument("--gen_value_conditioning", choices=("none", "gt"), default="none")
    parser.add_argument("--final_action_repeat", type=int, default=1)
    parser.add_argument("--late_state_repeat", type=int, default=1)
    parser.add_argument("--late_state_fraction", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    import pandas as pd  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    args = parse_args()
    frame = pd.read_parquet(args.input_parquet)
    rows = frame.to_dict(orient="records")
    excluded_problems = read_excluded_problems(args.exclude_problem_dataset_path)
    overlaps = sorted({str(row.get("problem", "")) for row in rows} & excluded_problems)
    if overlaps:
        raise ValueError(f"MC SFT input overlaps {len(overlaps)} held-out calibration problems.")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    raw_examples = build_mc_sft_examples(
        rows,
        tokenizer=tokenizer,
        min_continuations=args.min_continuations,
        score_max=args.score_max,
        gen_value_conditioning=args.gen_value_conditioning,
    )
    if len(raw_examples) < args.min_examples:
        raise ValueError(f"MC SFT dataset has {len(raw_examples)} examples; at least {args.min_examples} required.")
    examples = repeat_examples_for_horizon(
        raw_examples,
        final_action_repeat=args.final_action_repeat,
        late_state_repeat=args.late_state_repeat,
        late_state_fraction=args.late_state_fraction,
    )
    write_jsonl(args.output, examples)
    score_counts: dict[int, int] = {}
    for example in examples:
        score = quantize_mc_value(float(example["target"]), score_max=args.score_max)
        score_counts[score] = score_counts.get(score, 0) + 1
    print(
        json.dumps(
            {
                "raw_examples": len(raw_examples),
                "training_examples": len(examples),
                "input_rows": len(rows),
                "output": str(args.output),
                "score_counts": score_counts,
                "unique_problems": len({example["problem"] for example in examples}),
                "gen_value_conditioning": args.gen_value_conditioning,
                "final_action_examples": sum(example["state_kind"] == "final_action" for example in examples),
                "late_state_examples": sum(
                    example["state_kind"] != "final_action"
                    and float(example["trajectory_fraction"]) >= args.late_state_fraction
                    for example in examples
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
