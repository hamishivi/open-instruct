#!/usr/bin/env python3
"""Prepare and collect GPT-5 teacher traces for GenAC-style cold-start SFT.

The ``prepare`` command converts a frozen-actor critic trace reservoir into an
OpenAI Batch API JSONL file plus a metadata sidecar. It balances correct and
incorrect trajectories across their available positions, but deliberately does
not filter states by the source critic's accuracy: GPT-5 is the teacher.

The ``collect`` command joins a completed Batch API result file back to the raw
critic prompts, validates every response, parses its scalar estimate, and writes
the prompt/completion JSONL consumed by ``genac_math_value_trace_sft_h200.sh``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections.abc import Sequence
from typing import Any

from open_instruct.value_model_utils import (
    parse_generative_value_score,
    rescale_gen_value_score,
    select_gen_value_sft_traces,
)

DEFAULT_TEACHER_INSTRUCTIONS = """You are synthesizing a cold-start reasoning trace for a generative value model.
Evaluate the supplied critic prompt exactly as written. Analyze the mathematical partial response for concrete
progress, errors, and remaining recovery opportunities, then estimate the probability that the named active actor
will ultimately produce a verifier-correct answer within its remaining token budget. Do not assume knowledge of the
sampled trajectory's eventual outcome. Respond with concise value-focused reasoning followed by exactly one final
tag of the form <answer>N</answer>, where N is an integer from 0 to 10. Do not output anything after the tag."""
GROUND_TRUTH_CONDITIONING_MARKER = "\n\nThe correct answer is "


def read_jsonl(paths: Sequence[pathlib.Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected an object at {path}:{line_number}, got {type(row).__name__}.")
                rows.append(row)
    return rows


def write_jsonl(path: pathlib.Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def select_teacher_states(
    examples: Sequence[dict[str, Any]], *, min_critic_version: int, max_examples_per_outcome: int | None, seed: int
) -> list[dict[str, Any]]:
    """Select outcome/position-balanced raw states independently of critic quality."""
    candidates: list[dict[str, Any]] = []
    for example in examples:
        candidate = dict(example)
        # Fixed validation snapshots use target/kind/version, while training-trace
        # reservoirs use outcome/state_kind/source_critic_version. Normalize both
        # into the reservoir schema so an already-held-out state set can seed SFT
        # without being regenerated or moved into critic training.
        if candidate.get("outcome") is None and candidate.get("target") is not None:
            candidate["outcome"] = candidate["target"]
        if candidate.get("state_kind") is None and candidate.get("kind") is not None:
            candidate["state_kind"] = candidate["kind"]
        if candidate.get("source_critic_version") is None and candidate.get("version") is not None:
            candidate["source_critic_version"] = candidate["version"]
        outcome = candidate.get("outcome")
        # The generic trace selector's accuracy gate is intentionally neutralized
        # here. It still validates/deduplicates prompts and performs the exact same
        # outcome/position balancing used by the self-trace fallback.
        candidate["prediction"] = outcome
        candidate["squared_error"] = 0.0
        candidates.append(candidate)
    return select_gen_value_sft_traces(
        candidates,
        max_squared_error=0.0,
        min_critic_version=min_critic_version,
        max_examples_per_outcome=max_examples_per_outcome,
        balance_outcomes=True,
        balance_positions=True,
        seed=seed,
    )


def make_batch_request(
    custom_id: str, prompt: str, *, model: str, reasoning_effort: str, max_output_tokens: int
) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "instructions": DEFAULT_TEACHER_INSTRUCTIONS,
            "input": prompt,
            "reasoning": {"effort": reasoning_effort},
            "max_output_tokens": max_output_tokens,
            "store": False,
        },
    }


def extract_response_text(batch_result: dict[str, Any]) -> str:
    response = batch_result.get("response")
    if not isinstance(response, dict):
        raise ValueError(f"Batch result {batch_result.get('custom_id')!r} has no response object.")
    status_code = response.get("status_code")
    if status_code != 200:
        raise ValueError(
            f"Batch result {batch_result.get('custom_id')!r} returned status {status_code}: {batch_result.get('error')}"
        )
    body = response.get("body")
    if not isinstance(body, dict):
        raise ValueError(f"Batch result {batch_result.get('custom_id')!r} has no response body.")

    output_texts: list[str] = []
    for output_item in body.get("output", []):
        if not isinstance(output_item, dict) or output_item.get("type") != "message":
            continue
        for content_item in output_item.get("content", []):
            if isinstance(content_item, dict) and content_item.get("type") == "output_text":
                text = content_item.get("text")
                if isinstance(text, str) and text:
                    output_texts.append(text)
    if not output_texts:
        raise ValueError(f"Batch result {batch_result.get('custom_id')!r} contains no output_text.")
    return "\n".join(output_texts)


def prepare(args: argparse.Namespace) -> None:
    examples = read_jsonl(args.inputs)
    selected = select_teacher_states(
        examples,
        min_critic_version=args.min_critic_version,
        max_examples_per_outcome=args.max_examples_per_outcome,
        seed=args.seed,
    )
    if not selected:
        raise RuntimeError("No raw critic states passed the teacher-state selection criteria.")
    if not args.allow_ground_truth_conditioning:
        leaked_answer_prompts = [
            example
            for example in selected
            if GROUND_TRUTH_CONDITIONING_MARKER in str(example.get("prompt", ""))
        ]
        if leaked_answer_prompts:
            raise ValueError(
                f"Refusing to synthesize paper-style teacher traces from {len(leaked_answer_prompts)} "
                "answer-conditioned critic prompts. GenAC's critic prompt does not reveal the ground-truth answer. "
                "Collect states with --gen_value_conditioning=none, or pass "
                "--allow_ground_truth_conditioning only for an intentional answer-conditioned ablation."
            )

    metadata_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    for index, example in enumerate(selected):
        prompt = example.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"Selected teacher state {index} has no prompt.")
        custom_id = f"gen-value-{index:06d}"
        metadata_rows.append({"custom_id": custom_id, "source": example})
        batch_rows.append(
            make_batch_request(
                custom_id,
                prompt,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                max_output_tokens=args.max_output_tokens,
            )
        )

    write_jsonl(args.batch_output, batch_rows)
    write_jsonl(args.metadata_output, metadata_rows)
    print(
        json.dumps(
            {
                "batch_output": str(args.batch_output),
                "metadata_output": str(args.metadata_output),
                "model": args.model,
                "selected_examples": len(selected),
                "selected_correct": sum(float(example["outcome"]) > 0.5 for example in selected),
                "selected_incorrect": sum(float(example["outcome"]) <= 0.5 for example in selected),
            },
            indent=2,
            sort_keys=True,
        )
    )


def collect(args: argparse.Namespace) -> None:
    metadata_rows = read_jsonl([args.metadata])
    result_rows = read_jsonl(args.results)
    metadata_by_id = {row.get("custom_id"): row for row in metadata_rows}
    results_by_id = {row.get("custom_id"): row for row in result_rows}
    if len(metadata_by_id) != len(metadata_rows) or None in metadata_by_id:
        raise ValueError("Teacher metadata contains missing or duplicate custom IDs.")
    if len(results_by_id) != len(result_rows) or None in results_by_id:
        raise ValueError("Batch results contain missing or duplicate custom IDs.")
    missing = sorted(set(metadata_by_id) - set(results_by_id))
    unexpected = sorted(set(results_by_id) - set(metadata_by_id))
    if missing or unexpected:
        raise ValueError(f"Batch/metadata custom IDs differ: missing={missing[:5]}, unexpected={unexpected[:5]}.")

    selected: list[dict[str, Any]] = []
    filtered_for_error = 0
    for custom_id, metadata in metadata_by_id.items():
        source = metadata.get("source")
        if not isinstance(source, dict):
            raise ValueError(f"Metadata row {custom_id!r} has no source object.")
        generation = extract_response_text(results_by_id[custom_id])
        raw_prediction = parse_generative_value_score(generation, score_min=0.0, score_max=10.0)
        if raw_prediction is None:
            raise ValueError(f"Teacher response {custom_id!r} has no valid <answer> score: {generation!r}")
        prediction = rescale_gen_value_score(raw_prediction, score_min=0.0, score_max=10.0)
        outcome = float(source["outcome"])
        squared_error = (outcome - prediction) ** 2
        if args.max_teacher_squared_error is not None and squared_error > args.max_teacher_squared_error:
            filtered_for_error += 1
            continue
        selected.append(
            {
                **source,
                "generation": generation,
                "prediction": prediction,
                "squared_error": squared_error,
                "teacher_model": args.teacher_model,
                "teacher_prediction": prediction,
                "teacher_squared_error": squared_error,
            }
        )
    if not selected:
        raise RuntimeError("No GPT-5 teacher traces remained after collection/filtering.")
    write_jsonl(args.output, selected)
    print(
        json.dumps(
            {
                "filtered_for_error": filtered_for_error,
                "output": str(args.output),
                "selected_examples": len(selected),
                "teacher_model": args.teacher_model,
            },
            indent=2,
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Create Batch API input and metadata JSONL files.")
    prepare_parser.add_argument("inputs", nargs="+", type=pathlib.Path, help="Critic reservoir JSONL file(s).")
    prepare_parser.add_argument("--batch_output", required=True, type=pathlib.Path)
    prepare_parser.add_argument("--metadata_output", required=True, type=pathlib.Path)
    prepare_parser.add_argument("--model", default="gpt-5")
    prepare_parser.add_argument("--reasoning_effort", choices=("minimal", "low", "medium", "high"), default="medium")
    prepare_parser.add_argument("--max_output_tokens", type=int, default=1024)
    prepare_parser.add_argument("--min_critic_version", type=int, default=0)
    prepare_parser.add_argument("--max_examples_per_outcome", type=int, default=512)
    prepare_parser.add_argument("--seed", type=int, default=0)
    prepare_parser.add_argument(
        "--allow_ground_truth_conditioning",
        action="store_true",
        help="Allow answer-conditioned prompts for an explicit non-paper ablation.",
    )
    prepare_parser.set_defaults(function=prepare)

    collect_parser = subparsers.add_parser("collect", help="Create SFT JSONL from completed Batch API results.")
    collect_parser.add_argument("--metadata", required=True, type=pathlib.Path)
    collect_parser.add_argument("--results", required=True, nargs="+", type=pathlib.Path)
    collect_parser.add_argument("--output", required=True, type=pathlib.Path)
    collect_parser.add_argument("--teacher_model", default="gpt-5")
    collect_parser.add_argument("--max_teacher_squared_error", type=float)
    collect_parser.set_defaults(function=collect)

    args = parser.parse_args()
    if args.command == "prepare" and args.max_output_tokens <= 0:
        parser.error("--max_output_tokens must be positive.")
    if args.command == "prepare" and args.max_examples_per_outcome <= 0:
        parser.error("--max_examples_per_outcome must be positive.")
    if args.command == "collect" and args.max_teacher_squared_error is not None and args.max_teacher_squared_error < 0:
        parser.error("--max_teacher_squared_error must be nonnegative.")
    return args


def main() -> None:
    args = parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
