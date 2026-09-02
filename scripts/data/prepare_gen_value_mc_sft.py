#!/usr/bin/env python3
"""Convert exact-prefix Monte Carlo values into direct generative-value SFT targets.

The standard generative-critic SFT objective supervises every token in a long
teacher critique, so the scalar judgment can receive little effective weight.
This dataset retains the exact online critic prompt but uses a concise
``<answer>N</answer>`` completion, making every supervised completion token part
of the value judgment. The target is computed from fresh continuations rather
than the outcome of the sampled trajectory that supplied the prefix.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
from collections.abc import Sequence
from typing import Any

from open_instruct import value_estimation, value_model_utils


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
    prompt_to_example_index: dict[str, int] = {}
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
        critic_problem = value_model_utils.decode_generative_value_problem(
            tokenizer, optional_sequence_as_list(row.get("prompt_token_ids")) or None, fallback_problem=problem
        )
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
                problem=critic_problem,
                actor_model_name=str(row.get("actor_model_name", "")) or None,
                actor_success_rate=(
                    float(row["actor_success_rate"]) if row.get("actor_success_rate") is not None else None
                ),
                response_tokens_used=probe_position,
                response_token_limit=response_token_limit,
            )
            score = quantize_mc_value(mc_value, score_max=score_max)
            state_kind = "final_action" if probe_position == len(rollout_tokens) - 1 else "segment_start"
            # Match online critic metadata: zero at the first action and one at
            # the causal state immediately before the sampled final action.
            trajectory_fraction = probe_position / max(len(rollout_tokens) - 1, 1)
            example = {
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
                "source_rollout_lengths": [len(rollout_tokens)],
                "trajectory_fraction": trajectory_fraction,
                "source_trajectory_fractions": [trajectory_fraction],
                "num_continuations": num_continuations,
                "mc_source_count": 1,
                "problem": problem,
                "critic_problem": critic_problem,
                "ground_truth": ground_truth,
                "gen_value_conditioning": gen_value_conditioning,
                "direct_mc_score_supervision": True,
            }
            existing_index = prompt_to_example_index.get(prompt)
            if existing_index is None:
                prompt_to_example_index[prompt] = len(examples)
                examples.append(example)
                continue

            # Correct and incorrect sampled trajectories for one problem share
            # their initial causal state and can share a few later prefixes.
            # Their MC continuations are independent estimates of the same
            # state value, so pool successes rather than emitting contradictory
            # labels or discarding either estimate.
            existing = examples[existing_index]
            consistency_fields = (
                "problem",
                "critic_problem",
                "ground_truth",
                "gen_value_conditioning",
                "response_tokens_used",
                "response_token_limit",
            )
            inconsistent_fields = [field for field in consistency_fields if existing.get(field) != example.get(field)]
            if inconsistent_fields:
                raise ValueError(
                    f"Exact critic prompt collision for row {row_index}, probe {probe_position} has inconsistent "
                    f"metadata fields: {inconsistent_fields}."
                )
            existing_continuations = int(existing["num_continuations"])
            pooled_continuations = existing_continuations + num_continuations
            pooled_value = (
                float(existing["target"]) * existing_continuations + mc_value * num_continuations
            ) / pooled_continuations
            pooled_score = quantize_mc_value(pooled_value, score_max=score_max)
            existing.update(
                {
                    "generation": f" <answer>{pooled_score}</answer>",
                    "target": pooled_value,
                    "outcome": pooled_value,
                    "prediction": pooled_score / score_max,
                    "squared_error": (pooled_score / score_max - pooled_value) ** 2,
                    "state_kind": (
                        "final_action"
                        if "final_action" in {str(existing["state_kind"]), state_kind}
                        else "segment_start"
                    ),
                    "rollout_length": max(int(existing["rollout_length"]), len(rollout_tokens)),
                    "source_rollout_lengths": [
                        *optional_sequence_as_list(existing.get("source_rollout_lengths")),
                        len(rollout_tokens),
                    ],
                    "trajectory_fraction": max(float(existing["trajectory_fraction"]), trajectory_fraction),
                    "source_trajectory_fractions": [
                        *optional_sequence_as_list(existing.get("source_trajectory_fractions")),
                        trajectory_fraction,
                    ],
                    "num_continuations": pooled_continuations,
                    "mc_source_count": int(existing["mc_source_count"]) + 1,
                }
            )
    return examples


def repeat_examples_for_horizon(
    examples: Sequence[dict[str, Any]], *, final_action_repeat: int, late_state_repeat: int, late_state_fraction: float
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
            repeated.append({**example, "horizon_repeat_count": repeat_count, "horizon_repeat_index": repeat_index})
    return repeated


def _target_position_cell(example: dict[str, Any]) -> tuple[str, str] | None:
    if example.get("state_kind") == "final_action":
        return None
    trajectory_fraction = float(example.get("trajectory_fraction", 0.0))
    if not math.isfinite(trajectory_fraction) or not 0.0 <= trajectory_fraction <= 1.0:
        raise ValueError(f"Trajectory fraction must be finite and in [0, 1], got {trajectory_fraction}.")
    trajectory_band = "early" if trajectory_fraction < 1 / 3 else "middle" if trajectory_fraction < 2 / 3 else "late"
    target = float(example["target"])
    if not math.isfinite(target) or not 0.0 <= target <= 1.0:
        raise ValueError(f"Monte Carlo value must be finite and in [0, 1], got {target}.")
    target_bin = "0-.25" if target <= 0.25 else ".25-.5" if target <= 0.5 else ".5-.75" if target <= 0.75 else ".75-1"
    return trajectory_band, target_bin


def balance_examples_by_target_and_position(examples: Sequence[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    """Balance exact-MC target bins per trajectory band without changing labels."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a nonnegative integer, got {seed!r}.")
    target_bins = ("0-.25", ".25-.5", ".5-.75", ".75-1")
    cells: dict[tuple[str, str], list[int]] = {}
    retained_indices = {index for index, example in enumerate(examples) if example.get("state_kind") == "final_action"}
    for index, example in enumerate(examples):
        cell = _target_position_cell(example)
        if cell is not None:
            cells.setdefault(cell, []).append(index)

    rng = random.Random(seed)
    cell_sizes: dict[str, int] = {}
    for trajectory_band in ("early", "middle", "late"):
        band_cells = [cells.get((trajectory_band, target_bin), []) for target_bin in target_bins]
        if not any(band_cells):
            continue
        missing_bins = [target_bin for target_bin, indices in zip(target_bins, band_cells, strict=True) if not indices]
        if missing_bins:
            raise ValueError(
                f"Cannot target-balance {trajectory_band} states because target bins are empty: {missing_bins}."
            )
        cell_size = min(len(indices) for indices in band_cells)
        cell_sizes[trajectory_band] = cell_size
        for indices in band_cells:
            retained_indices.update(rng.sample(indices, cell_size))

    balanced = []
    for index, example in enumerate(examples):
        if index not in retained_indices:
            continue
        cell = _target_position_cell(example)
        balanced.append(
            {
                **example,
                "target_position_balanced": True,
                "target_position_balance_band": "final_action" if cell is None else cell[0],
                "target_position_balance_bin": "final_action" if cell is None else cell[1],
                "target_position_balance_cell_size": 1 if cell is None else cell_sizes[cell[0]],
            }
        )
    return balanced


def summarize_trajectory_coverage(examples: Sequence[dict[str, Any]]) -> dict[str, int | float]:
    """Count unique MC states in the trajectory bands used by held-out scoring.

    Coverage is measured before any replay multiplier so a large late/final
    repeat count cannot make a poorly targeted source dataset look adequate.
    """
    counts = {"early": 0, "middle": 0, "late_nonterminal": 0, "final_action": 0}
    for example in examples:
        if example.get("state_kind") == "final_action":
            counts["final_action"] += 1
            continue
        trajectory_fraction = float(example.get("trajectory_fraction", 0.0))
        if not math.isfinite(trajectory_fraction) or not 0.0 <= trajectory_fraction <= 1.0:
            raise ValueError(f"Trajectory fraction must be finite and in [0, 1], got {trajectory_fraction}.")
        if trajectory_fraction < 0.25:
            counts["early"] += 1
        elif trajectory_fraction < 0.75:
            counts["middle"] += 1
        else:
            counts["late_nonterminal"] += 1
    early_middle_examples = counts["early"] + counts["middle"]
    return {
        **counts,
        "early_middle_examples": early_middle_examples,
        "early_middle_fraction": early_middle_examples / len(examples) if examples else 0.0,
    }


def require_trajectory_coverage(
    examples: Sequence[dict[str, Any]], *, min_early_middle_fraction: float
) -> dict[str, int | float]:
    """Fail closed when the raw dataset misses its intended nonterminal bands."""
    if not 0.0 <= min_early_middle_fraction <= 1.0:
        raise ValueError(f"min_early_middle_fraction must be in [0, 1], got {min_early_middle_fraction}.")
    coverage = summarize_trajectory_coverage(examples)
    observed_fraction = float(coverage["early_middle_fraction"])
    if observed_fraction < min_early_middle_fraction:
        raise ValueError(
            "MC SFT trajectory coverage is too late-heavy: "
            f"{coverage['early_middle_examples']}/{len(examples)} "
            f"({observed_fraction:.3f}) unique states are early/middle, below the required "
            f"{min_early_middle_fraction:.3f}."
        )
    return coverage


def summarize_prefix_length_coverage(
    examples: Sequence[dict[str, Any]], *, long_prefix_token_threshold: int
) -> dict[str, int | float]:
    """Measure absolute consumed-token coverage before replay multipliers.

    Trajectory-relative bands alone can call a 600-token prefix "late" even
    when the online actor routinely reasons for several thousand tokens. This
    complementary view detects that short-context mismatch directly.
    """
    if isinstance(long_prefix_token_threshold, bool) or not isinstance(long_prefix_token_threshold, int):
        raise ValueError(
            f"long_prefix_token_threshold must be a nonnegative integer, got {long_prefix_token_threshold!r}."
        )
    if long_prefix_token_threshold < 0:
        raise ValueError(
            f"long_prefix_token_threshold must be a nonnegative integer, got {long_prefix_token_threshold}."
        )
    prefix_lengths = [int(example.get("response_tokens_used", 0)) for example in examples]
    if any(length < 0 for length in prefix_lengths):
        raise ValueError(f"response_tokens_used must be nonnegative, got {prefix_lengths}.")
    long_prefix_examples = sum(length >= long_prefix_token_threshold for length in prefix_lengths)
    return {
        "examples": len(prefix_lengths),
        "zero_prefix_examples": sum(length == 0 for length in prefix_lengths),
        "at_least_1024_tokens": sum(length >= 1024 for length in prefix_lengths),
        "at_least_2048_tokens": sum(length >= 2048 for length in prefix_lengths),
        "at_least_3072_tokens": sum(length >= 3072 for length in prefix_lengths),
        "long_prefix_token_threshold": long_prefix_token_threshold,
        "long_prefix_examples": long_prefix_examples,
        "long_prefix_fraction": long_prefix_examples / len(prefix_lengths) if prefix_lengths else 0.0,
    }


def require_prefix_length_coverage(
    examples: Sequence[dict[str, Any]], *, long_prefix_token_threshold: int, min_long_prefix_fraction: float
) -> dict[str, int | float]:
    """Fail closed when a purported long-context corpus remains too short."""
    if not 0.0 <= min_long_prefix_fraction <= 1.0:
        raise ValueError(f"min_long_prefix_fraction must be in [0, 1], got {min_long_prefix_fraction}.")
    coverage = summarize_prefix_length_coverage(
        examples, long_prefix_token_threshold=long_prefix_token_threshold
    )
    observed_fraction = float(coverage["long_prefix_fraction"])
    if observed_fraction < min_long_prefix_fraction:
        raise ValueError(
            "MC SFT prefix coverage is too short-context: "
            f"{coverage['long_prefix_examples']}/{len(examples)} ({observed_fraction:.3f}) unique states use at least "
            f"{long_prefix_token_threshold} response tokens, below the required {min_long_prefix_fraction:.3f}."
        )
    return coverage


def read_excluded_problems(path: pathlib.Path | None) -> set[str]:
    if path is None:
        return set()
    import pandas as pd  # noqa: PLC0415

    frame = pd.read_parquet(path, columns=["problem"])
    return {problem for problem in frame["problem"].tolist() if isinstance(problem, str) and problem}


def normalized_problem_overlaps(rows: Sequence[dict[str, Any]], excluded_problems: set[str]) -> list[str]:
    """Return formatting-insensitive problem identities shared with a holdout."""
    normalized_inputs = {
        value_estimation.normalize_problem_identity(str(row.get("problem", row.get("prompt", ""))))
        for row in rows
    }
    normalized_excluded = {
        value_estimation.normalize_problem_identity(problem) for problem in excluded_problems
    }
    return sorted(identity for identity in normalized_inputs & normalized_excluded if identity)


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
    parser.add_argument(
        "--min_early_middle_fraction",
        type=float,
        default=0.0,
        help=(
            "Minimum fraction of unique, pre-replay states with trajectory_fraction < 0.75. "
            "Use this to fail closed when an intended prefix-calibration dataset is late/final-heavy."
        ),
    )
    parser.add_argument(
        "--long_prefix_token_threshold",
        type=int,
        default=2048,
        help="Consumed response-token threshold used to define a long-prefix state.",
    )
    parser.add_argument(
        "--min_long_prefix_fraction",
        type=float,
        default=0.0,
        help=(
            "Minimum fraction of unique, pre-replay states at or beyond --long_prefix_token_threshold. "
            "Use this to fail closed when a current-policy calibration corpus is still too short-context."
        ),
    )
    parser.add_argument("--score_max", type=int, default=10)
    parser.add_argument("--gen_value_conditioning", choices=("none", "gt"), default="none")
    parser.add_argument("--final_action_repeat", type=int, default=1)
    parser.add_argument("--late_state_repeat", type=int, default=1)
    parser.add_argument("--late_state_fraction", type=float, default=0.75)
    parser.add_argument(
        "--balance_target_position",
        action="store_true",
        help=(
            "Deterministically downsample each nonterminal trajectory band to equal exact-MC target-bin "
            "counts. Labels, prompts, and the training objective are unchanged."
        ),
    )
    parser.add_argument("--balance_seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    import pandas as pd  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    args = parse_args()
    frame = pd.read_parquet(args.input_parquet)
    rows = frame.to_dict(orient="records")
    excluded_problems = read_excluded_problems(args.exclude_problem_dataset_path)
    overlaps = normalized_problem_overlaps(rows, excluded_problems)
    if overlaps:
        raise ValueError(
            f"MC SFT input overlaps {len(overlaps)} held-out calibration problem identities after whitespace "
            "normalization."
        )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    source_examples = build_mc_sft_examples(
        rows,
        tokenizer=tokenizer,
        min_continuations=args.min_continuations,
        score_max=args.score_max,
        gen_value_conditioning=args.gen_value_conditioning,
    )
    source_trajectory_coverage = summarize_trajectory_coverage(source_examples)
    source_prefix_length_coverage = summarize_prefix_length_coverage(
        source_examples, long_prefix_token_threshold=args.long_prefix_token_threshold
    )
    raw_examples = (
        balance_examples_by_target_and_position(source_examples, seed=args.balance_seed)
        if args.balance_target_position
        else source_examples
    )
    if len(raw_examples) < args.min_examples:
        raise ValueError(f"MC SFT dataset has {len(raw_examples)} examples; at least {args.min_examples} required.")
    trajectory_coverage = require_trajectory_coverage(
        raw_examples, min_early_middle_fraction=args.min_early_middle_fraction
    )
    prefix_length_coverage = require_prefix_length_coverage(
        raw_examples,
        long_prefix_token_threshold=args.long_prefix_token_threshold,
        min_long_prefix_fraction=args.min_long_prefix_fraction,
    )
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
                "source_examples": len(source_examples),
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
                "balance_target_position": args.balance_target_position,
                "balance_seed": args.balance_seed,
                "source_trajectory_coverage": source_trajectory_coverage,
                "raw_trajectory_coverage": trajectory_coverage,
                "source_prefix_length_coverage": source_prefix_length_coverage,
                "raw_prefix_length_coverage": prefix_length_coverage,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
