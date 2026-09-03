#!/usr/bin/env python3
"""Build a prompt-diverse generative-critic panel from a W&B AIME table."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
from collections import defaultdict
from typing import Any

from transformers import AutoTokenizer

from open_instruct import value_model_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb_table_json", required=True, type=pathlib.Path)
    parser.add_argument("--tokenizer_name_or_path", required=True)
    parser.add_argument("--output_jsonl", required=True, type=pathlib.Path)
    parser.add_argument("--actor_model_name")
    parser.add_argument("--actor_success_rate", type=float)
    parser.add_argument("--conditioning", choices=sorted(value_model_utils.GEN_VALUE_CONDITIONING_TYPES), default="none")
    parser.add_argument("--response_token_limit", type=int, default=8192)
    parser.add_argument("--max_trajectories_per_problem", type=int, default=2)
    parser.add_argument("--correct_score_threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _read_wandb_table(table: dict[str, Any]) -> list[dict[str, Any]]:
    columns = table.get("columns")
    data = table.get("data")
    if not isinstance(columns, list) or not all(isinstance(column, str) for column in columns):
        raise ValueError("W&B table must contain a string 'columns' list.")
    if not isinstance(data, list):
        raise ValueError("W&B table must contain a 'data' list.")
    required = {"prompt", "response", "scores", "ground_truth"}
    missing = required.difference(columns)
    if missing:
        raise ValueError(f"W&B table is missing required columns: {sorted(missing)}.")

    rows: list[dict[str, Any]] = []
    for row_index, values in enumerate(data):
        if not isinstance(values, list) or len(values) != len(columns):
            raise ValueError(
                f"W&B table row {row_index} has {len(values) if isinstance(values, list) else 'invalid'} "
                f"values for {len(columns)} columns."
            )
        row = dict(zip(columns, values, strict=True))
        if not isinstance(row["prompt"], str) or not row["prompt"]:
            raise ValueError(f"W&B table row {row_index} has an invalid prompt.")
        if not isinstance(row["response"], str):
            raise ValueError(f"W&B table row {row_index} has an invalid response.")
        if not isinstance(row["scores"], int | float):
            raise ValueError(f"W&B table row {row_index} has a nonnumeric score.")
        rows.append(row)
    if not rows:
        raise ValueError("W&B table contains no completion rows.")
    return rows


def _decode_actor_prompt(tokenizer: Any, prompt: str) -> str:
    """Match the online critic's decode-with-special-tokens-removed problem text."""
    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return tokenizer.decode(prompt_token_ids, skip_special_tokens=True)


def _select_balanced_trajectories(
    rows: list[dict[str, Any]], count: int, score_threshold: float, rng: random.Random
) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError(f"max_trajectories_per_problem must be positive, got {count}.")
    correct = [row for row in rows if float(row["scores"]) > score_threshold]
    incorrect = [row for row in rows if float(row["scores"]) <= score_threshold]
    rng.shuffle(correct)
    rng.shuffle(incorrect)

    if count == 1:
        candidates = correct + incorrect
        rng.shuffle(candidates)
        return candidates[:1]

    selected: list[dict[str, Any]] = []
    if correct:
        selected.append(correct.pop())
    if incorrect and len(selected) < count:
        selected.append(incorrect.pop())
    # The first two slots represent outcome classes, not arbitrary siblings. Only
    # an explicitly larger budget should add duplicate outcomes for one problem.
    if count > 2:
        leftovers = correct + incorrect
        rng.shuffle(leftovers)
        selected.extend(leftovers[: count - len(selected)])
    return selected


def build_aime_validation_examples(
    table: dict[str, Any],
    tokenizer: Any,
    *,
    conditioning: str = "none",
    actor_model_name: str | None = None,
    actor_success_rate: float | None = None,
    response_token_limit: int = 8192,
    max_trajectories_per_problem: int = 2,
    correct_score_threshold: float = 0.5,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Create fixed initial, prefix, and final states across every AIME prompt.

    The panel is evaluation-only: it is derived from already-generated AIME
    completions and is never inserted into the critic optimizer. For each problem,
    one correct and one incorrect trajectory are retained when both exist. This
    gives prompt diversity priority over the number of nearly duplicate siblings.
    """
    if conditioning not in value_model_utils.GEN_VALUE_CONDITIONING_TYPES:
        raise ValueError(f"Unknown generative-value conditioning mode: {conditioning!r}.")
    if response_token_limit <= 0:
        raise ValueError(f"response_token_limit must be positive, got {response_token_limit}.")
    if actor_success_rate is not None and not 0.0 <= actor_success_rate <= 1.0:
        raise ValueError(f"actor_success_rate must be in [0, 1], got {actor_success_rate}.")

    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_wandb_table(table):
        grouped_rows[row["prompt"]].append(row)

    rng = random.Random(seed)
    examples: list[dict[str, Any]] = []
    for raw_prompt, problem_rows in grouped_rows.items():
        problem_id = hashlib.sha256(raw_prompt.encode("utf-8")).hexdigest()[:16]
        problem = _decode_actor_prompt(tokenizer, raw_prompt)
        outcomes = [float(row["scores"]) > correct_score_threshold for row in problem_rows]
        ground_truth = str(problem_rows[0]["ground_truth"])
        common_prompt_args = {
            "conditioning": conditioning,
            "ground_truth": ground_truth,
            "problem": problem,
            "actor_model_name": actor_model_name,
            "actor_success_rate": actor_success_rate,
            "response_token_limit": response_token_limit,
        }
        examples.append(
            {
                "prompt": value_model_utils.build_generative_value_prompt(
                    partial_response="", response_tokens_used=0, **common_prompt_args
                ),
                "target": sum(outcomes) / len(outcomes),
                "target_source": "sibling_empirical_return",
                "kind": "initial",
                "response_tokens_used": 0,
                "response_token_limit": response_token_limit,
                "problem_id": problem_id,
                "problem_samples": len(problem_rows),
            }
        )

        selected_rows = _select_balanced_trajectories(
            problem_rows, max_trajectories_per_problem, correct_score_threshold, rng
        )
        for sample_index, row in enumerate(selected_rows):
            outcome = float(float(row["scores"]) > correct_score_threshold)
            response_token_ids = tokenizer.encode(row["response"], add_special_tokens=False)[
                :response_token_limit
            ]
            if not response_token_ids:
                continue
            # Online GenAC's final-action state is causal: it is immediately before
            # the last sampled response token, not after observing termination. Match
            # that exact definition here so the sampled terminal return remains an
            # unbiased target for V(s) under the actor's final-action distribution.
            final_response_tokens_used = max(len(response_token_ids) - 1, 0)
            state_specs: list[tuple[int, float, str]] = []
            seen_prefix_lengths: set[int] = set()
            for fraction in (0.25, 0.5, 0.75):
                prefix_length = max(
                    0,
                    min(final_response_tokens_used, round(final_response_tokens_used * fraction)),
                )
                if prefix_length in seen_prefix_lengths:
                    continue
                seen_prefix_lengths.add(prefix_length)
                state_specs.append((prefix_length, fraction, "segment_start"))
            state_specs.append((final_response_tokens_used, 1.0, "final_action"))

            for response_tokens_used, trajectory_fraction, kind in state_specs:
                partial_response = tokenizer.decode(
                    response_token_ids[:response_tokens_used], skip_special_tokens=False
                )
                examples.append(
                    {
                        "prompt": value_model_utils.build_generative_value_prompt(
                            partial_response=partial_response,
                            response_tokens_used=response_tokens_used,
                            **common_prompt_args,
                        ),
                        "target": outcome,
                        "target_source": "single_sample_return",
                        "kind": kind,
                        "response_tokens_used": response_tokens_used,
                        "response_token_limit": response_token_limit,
                        "trajectory_fraction": trajectory_fraction,
                        "problem_id": problem_id,
                        "sample_index": sample_index,
                    }
                )
    return examples


def main() -> None:
    args = parse_args()
    if args.output_jsonl.exists():
        raise FileExistsError(f"Refusing to overwrite validation snapshot: {args.output_jsonl}.")
    with args.wandb_table_json.open(encoding="utf-8") as table_file:
        table = json.load(table_file)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    examples = build_aime_validation_examples(
        table,
        tokenizer,
        conditioning=args.conditioning,
        actor_model_name=args.actor_model_name,
        actor_success_rate=args.actor_success_rate,
        response_token_limit=args.response_token_limit,
        max_trajectories_per_problem=args.max_trajectories_per_problem,
        correct_score_threshold=args.correct_score_threshold,
        seed=args.seed,
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        for example in examples:
            output_file.write(json.dumps(example, ensure_ascii=False) + "\n")
    temporary_path.replace(args.output_jsonl)

    sampled = [example for example in examples if example["target_source"] == "single_sample_return"]
    summary = {
        "examples": len(examples),
        "problems": len({example["problem_id"] for example in examples}),
        "sampled_correct_states": sum(float(example["target"]) > 0.5 for example in sampled),
        "sampled_incorrect_states": sum(float(example["target"]) <= 0.5 for example in sampled),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
