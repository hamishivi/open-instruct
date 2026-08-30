#!/usr/bin/env python3
"""Prepare, collect, and select teacher traces for GenAC-style cold-start SFT.

The ``prepare`` command converts a frozen-actor critic trace reservoir into an
OpenAI Batch API JSONL file plus a metadata sidecar. It balances correct and
incorrect trajectories across their available positions, but deliberately does
not filter states by the source critic's accuracy: GPT-5 is the teacher.

The ``collect`` command joins a completed Batch API result file back to the raw
critic prompts, validates every response, parses its scalar estimate, and writes
the prompt/completion JSONL consumed by ``genac_math_value_trace_sft_h200.sh``.

The ``consensus`` command keeps a concise primary teacher trace only when its
score agrees with independent judge teachers. A single sampled rollout outcome
is used only to balance trajectory coverage, never as a per-state calibration
gate: even immediately before the sampled final token, the actor can choose a
different action and continue when response budget remains.
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
PARTIAL_RESPONSE_MARKER = "\n\nPartial response:\n<rollout>"


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


def prompt_has_ground_truth_conditioning(prompt: str) -> bool:
    """Detect answer conditioning without inspecting the actor's response text.

    Incorrect actor trajectories can themselves contain phrases such as
    ``The correct answer is ...``. Those are useful negative states, not leaked
    critic conditioning, so only inspect the prompt prefix before ``<rollout>``.
    Older/nonstandard prompts without the delimiter retain the conservative
    whole-prompt check.
    """
    prefix, delimiter, _ = prompt.partition(PARTIAL_RESPONSE_MARKER)
    return GROUND_TRUTH_CONDITIONING_MARKER in (prefix if delimiter else prompt)


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
    custom_id: str,
    prompt: str,
    *,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    request_format: str = "responses",
    enable_thinking: bool = False,
) -> dict[str, Any]:
    if request_format == "chat_completions":
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": DEFAULT_TEACHER_INSTRUCTIONS},
                    {"role": "user", "content": prompt},
                ],
                # Qwen3's hidden-thinking mode can consume the entire 1,024-token
                # critic response budget before producing the required score, so
                # keep it off by default.  A larger offline teacher can opt in
                # with a larger output budget for a controlled teacher-quality
                # comparison; ``extract_response_text`` preserves both reasoning
                # and the final scored content when the server separates them.
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
                "max_tokens": max_output_tokens,
                "temperature": 0.6,
                "top_p": 0.95,
                "stream": False,
            },
        }
    if request_format != "responses":
        raise ValueError(f"Unsupported batch request format: {request_format!r}.")
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
    for choice in body.get("choices", []):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        reasoning = message.get("reasoning_content")
        content = message.get("content")
        if isinstance(reasoning, str) and reasoning:
            output_texts.append(reasoning.rstrip())
        if isinstance(content, str) and content:
            output_texts.append(content.lstrip())
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
            example for example in selected if prompt_has_ground_truth_conditioning(str(example.get("prompt", "")))
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
                request_format=getattr(args, "request_format", "responses"),
                enable_thinking=getattr(args, "enable_thinking", False),
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
                "request_format": getattr(args, "request_format", "responses"),
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
    invalid_score_ids: list[str] = []
    for custom_id, metadata in metadata_by_id.items():
        source = metadata.get("source")
        if not isinstance(source, dict):
            raise ValueError(f"Metadata row {custom_id!r} has no source object.")
        generation = extract_response_text(results_by_id[custom_id])
        raw_prediction = parse_generative_value_score(generation, score_min=0.0, score_max=10.0)
        if raw_prediction is None:
            if getattr(args, "skip_invalid_scores", False):
                invalid_score_ids.append(custom_id)
                continue
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
                "invalid_score_ids": invalid_score_ids,
                "invalid_scores": len(invalid_score_ids),
                "output": str(args.output),
                "selected_examples": len(selected),
                "teacher_model": args.teacher_model,
            },
            indent=2,
            sort_keys=True,
        )
    )


def audit(args: argparse.Namespace) -> None:
    """Fail closed unless an SFT JSONL is large, parseable, and unconditioned."""
    examples = read_jsonl(args.inputs)
    if len(examples) < args.min_examples:
        raise ValueError(f"Value SFT dataset has {len(examples)} traces; at least {args.min_examples} are required.")

    missing_prompt = 0
    missing_generation = 0
    invalid_score = 0
    leaked_answer_conditioning = 0
    prompts: list[str] = []
    for example in examples:
        prompt = example.get("prompt")
        generation = example.get("generation")
        if not isinstance(prompt, str) or not prompt:
            missing_prompt += 1
        else:
            prompts.append(prompt)
            if prompt_has_ground_truth_conditioning(prompt):
                leaked_answer_conditioning += 1
        if not isinstance(generation, str) or not generation:
            missing_generation += 1
        elif parse_generative_value_score(generation, score_min=0.0, score_max=10.0) is None:
            invalid_score += 1

    duplicate_prompts = len(prompts) - len(set(prompts))
    failures = {
        "missing_prompt": missing_prompt,
        "missing_generation": missing_generation,
        "invalid_score": invalid_score,
        "duplicate_prompts": duplicate_prompts,
    }
    if leaked_answer_conditioning and not args.allow_ground_truth_conditioning:
        failures["answer_conditioned_prompt"] = leaked_answer_conditioning
    nonzero_failures = {name: count for name, count in failures.items() if count}
    if nonzero_failures:
        raise ValueError(f"Value SFT audit failed: {nonzero_failures}.")

    print(
        json.dumps(
            {
                "answer_conditioned_prompts": leaked_answer_conditioning,
                "examples": len(examples),
                "inputs": [str(path) for path in args.inputs],
                "unique_prompts": len(set(prompts)),
            },
            indent=2,
            sort_keys=True,
        )
    )


def teacher_prediction(example: dict[str, Any], *, source: str, prompt: str) -> float:
    prediction = example.get("teacher_prediction", example.get("prediction"))
    if not isinstance(prediction, int | float):
        raise ValueError(f"Teacher row for prompt {prompt!r} in {source} has no numeric prediction.")
    prediction = float(prediction)
    if not 0.0 <= prediction <= 1.0:
        raise ValueError(f"Teacher prediction for prompt {prompt!r} in {source} is outside [0, 1]: {prediction}.")
    return prediction


def index_teacher_rows(examples: Sequence[dict[str, Any]], *, source: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row_number, example in enumerate(examples, start=1):
        prompt = example.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"Teacher row {row_number} in {source} has no prompt.")
        if prompt in indexed:
            raise ValueError(f"Teacher source {source} contains duplicate prompt {prompt!r}.")
        teacher_prediction(example, source=source, prompt=prompt)
        indexed[prompt] = example
    return indexed


def select_teacher_consensus(
    primary_examples: Sequence[dict[str, Any]],
    judge_example_groups: Sequence[Sequence[dict[str, Any]]],
    *,
    max_teacher_range: float,
    max_examples_per_outcome: int | None,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select balanced SFT traces using independent teacher agreement."""
    if not judge_example_groups:
        raise ValueError("Teacher consensus requires at least one independent judge.")
    if max_teacher_range < 0:
        raise ValueError(f"max_teacher_range must be nonnegative, got {max_teacher_range}.")

    primary_by_prompt = index_teacher_rows(primary_examples, source="primary")
    judges_by_prompt = [
        index_teacher_rows(examples, source=f"judge_{index}")
        for index, examples in enumerate(judge_example_groups, start=1)
    ]
    candidates: list[dict[str, Any]] = []
    stats = {
        "primary_examples": len(primary_examples),
        "missing_judge": 0,
        "teacher_disagreement": 0,
        "consensus_candidates": 0,
    }
    for prompt, primary in primary_by_prompt.items():
        judge_rows = [judge.get(prompt) for judge in judges_by_prompt]
        if any(row is None for row in judge_rows):
            stats["missing_judge"] += 1
            continue
        rows = [primary, *(row for row in judge_rows if row is not None)]
        predictions = [
            teacher_prediction(row, source=f"teacher_{index}", prompt=prompt) for index, row in enumerate(rows)
        ]
        prediction_range = max(predictions) - min(predictions)
        if prediction_range > max_teacher_range + 1e-12:
            stats["teacher_disagreement"] += 1
            continue

        outcome = primary.get("outcome", primary.get("target"))
        if not isinstance(outcome, int | float) or not 0.0 <= float(outcome) <= 1.0:
            raise ValueError(f"Primary teacher row for prompt {prompt!r} has no valid outcome.")
        state_kind = primary.get("state_kind", primary.get("kind"))
        primary_prediction = predictions[0]

        candidate = dict(primary)
        candidate.update(
            {
                "outcome": float(outcome),
                "state_kind": state_kind,
                "prediction": primary_prediction,
                # The generic selector performs prompt deduplication and balanced
                # position/outcome sampling after this independent quality gate.
                "squared_error": 0.0,
                "teacher_consensus_mean": sum(predictions) / len(predictions),
                "teacher_consensus_predictions": predictions,
                "teacher_consensus_range": prediction_range,
                "teacher_consensus_size": len(predictions),
            }
        )
        candidates.append(candidate)
    stats["consensus_candidates"] = len(candidates)

    selected = select_gen_value_sft_traces(
        candidates,
        max_squared_error=0.0,
        min_critic_version=0,
        max_examples_per_outcome=max_examples_per_outcome,
        balance_outcomes=True,
        balance_positions=True,
        seed=seed,
    )
    stats["selected_examples"] = len(selected)
    stats["selected_correct"] = sum(float(example["outcome"]) > 0.5 for example in selected)
    stats["selected_incorrect"] = len(selected) - stats["selected_correct"]
    return selected, stats


def consensus(args: argparse.Namespace) -> None:
    primary_examples = read_jsonl([args.primary])
    judge_example_groups = [read_jsonl([judge]) for judge in args.judges]
    selected, stats = select_teacher_consensus(
        primary_examples,
        judge_example_groups,
        max_teacher_range=args.max_teacher_range,
        max_examples_per_outcome=args.max_examples_per_outcome,
        seed=args.seed,
    )
    if not selected:
        raise RuntimeError("No balanced teacher traces remained after consensus selection.")
    write_jsonl(args.output, selected)
    print(json.dumps({**stats, "output": str(args.output)}, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Create Batch API input and metadata JSONL files.")
    prepare_parser.add_argument("inputs", nargs="+", type=pathlib.Path, help="Critic reservoir JSONL file(s).")
    prepare_parser.add_argument("--batch_output", required=True, type=pathlib.Path)
    prepare_parser.add_argument("--metadata_output", required=True, type=pathlib.Path)
    prepare_parser.add_argument("--model", default="gpt-5")
    prepare_parser.add_argument(
        "--request_format",
        choices=("responses", "chat_completions"),
        default="responses",
        help="Use Responses for OpenAI Batch or chat_completions for an OpenAI-compatible local batch runner.",
    )
    prepare_parser.add_argument("--reasoning_effort", choices=("minimal", "low", "medium", "high"), default="medium")
    prepare_parser.add_argument("--max_output_tokens", type=int, default=1024)
    prepare_parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable local Qwen3 hidden reasoning; use with chat_completions and a larger output budget.",
    )
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
    collect_parser.add_argument(
        "--skip_invalid_scores",
        action="store_true",
        help="Explicitly report and omit successful responses that never emitted a valid score tag.",
    )
    collect_parser.set_defaults(function=collect)

    audit_parser = subparsers.add_parser(
        "audit", help="Validate size, score format, prompt uniqueness, and conditioning of SFT JSONL files."
    )
    audit_parser.add_argument("inputs", nargs="+", type=pathlib.Path)
    audit_parser.add_argument("--min_examples", type=int, default=512)
    audit_parser.add_argument("--allow_ground_truth_conditioning", action="store_true")
    audit_parser.set_defaults(function=audit)

    consensus_parser = subparsers.add_parser(
        "consensus", help="Select a balanced primary-teacher SFT set using independent teacher agreement."
    )
    consensus_parser.add_argument("--primary", required=True, type=pathlib.Path)
    consensus_parser.add_argument("--judges", required=True, nargs="+", type=pathlib.Path)
    consensus_parser.add_argument("--output", required=True, type=pathlib.Path)
    consensus_parser.add_argument("--max_teacher_range", type=float, default=0.2)
    consensus_parser.add_argument("--max_examples_per_outcome", type=int, default=512)
    consensus_parser.add_argument("--seed", type=int, default=0)
    consensus_parser.set_defaults(function=consensus)

    args = parser.parse_args()
    if args.command == "prepare" and args.max_output_tokens <= 0:
        parser.error("--max_output_tokens must be positive.")
    if args.command == "prepare" and args.max_examples_per_outcome <= 0:
        parser.error("--max_examples_per_outcome must be positive.")
    if args.command == "collect" and args.max_teacher_squared_error is not None and args.max_teacher_squared_error < 0:
        parser.error("--max_teacher_squared_error must be nonnegative.")
    if args.command == "audit" and args.min_examples < 0:
        parser.error("--min_examples must be nonnegative.")
    if args.command == "consensus" and args.max_teacher_range < 0:
        parser.error("--max_teacher_range must be nonnegative.")
    if args.command == "consensus" and args.max_examples_per_outcome <= 0:
        parser.error("--max_examples_per_outcome must be positive.")
    return args


def main() -> None:
    args = parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
