#!/usr/bin/env python3
"""Score a fixed generative-value validation snapshot with a vLLM checkpoint."""

from __future__ import annotations

import argparse
import json
import pathlib

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from open_instruct import value_model_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_snapshot", required=True, type=pathlib.Path)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_jsonl", required=True, type=pathlib.Path)
    parser.add_argument("--metrics_json", type=pathlib.Path)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--soft_class_probabilities",
        action="store_true",
        help="Also score all exact <answer>0</answer> through <answer>10</answer> sequences.",
    )
    return parser.parse_args()


def score_soft_class_predictions(
    llm: LLM, prompts: list[str]
) -> tuple[list[float], list[list[float]], list[list[float]]]:
    """Score the normalized likelihood of every valid discrete critic class.

    This is a diagnostic-only continuous view of the existing 0-10 output
    distribution. It neither changes greedy generation nor the critic objective.
    Exact sequence likelihoods are required because Qwen tokenizes score 10 as
    two tokens while scores 0-9 each use one token.
    """
    tokenizer = llm.get_tokenizer()
    class_scores = [float(score) for score in range(11)]
    token_prompts: list[TokensPrompt] = []
    suffix_starts: list[int] = []
    for prompt in prompts:
        class_prefix = f"{prompt} <answer>"
        prefix_ids = tokenizer.encode(class_prefix, add_special_tokens=False)
        for score in range(11):
            full_ids = tokenizer.encode(f"{class_prefix}{score}</answer>", add_special_tokens=False)
            if full_ids[: len(prefix_ids)] != prefix_ids:
                raise RuntimeError(f"Score {score} does not preserve the tokenized generative-value class prefix.")
            token_prompts.append(TokensPrompt(prompt_token_ids=full_ids))
            suffix_starts.append(len(prefix_ids))

    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=0,
        detokenize=False,
    )
    outputs = llm.generate(token_prompts, sampling_params)
    if len(outputs) != len(token_prompts):
        raise RuntimeError(f"Soft class scorer returned {len(outputs)} outputs for {len(token_prompts)} prompts.")

    sequence_logprobs: list[float] = []
    for output, suffix_start in zip(outputs, suffix_starts, strict=True):
        if output.prompt_token_ids is None or output.prompt_logprobs is None:
            raise RuntimeError("vLLM did not return token ids and prompt log probabilities for soft class scoring.")
        if len(output.prompt_token_ids) != len(output.prompt_logprobs):
            raise RuntimeError(
                "vLLM returned misaligned soft-class token ids and prompt log probabilities "
                f"({len(output.prompt_token_ids)} != {len(output.prompt_logprobs)})."
            )
        sequence_logprob = 0.0
        for position in range(suffix_start, len(output.prompt_token_ids)):
            token_id = output.prompt_token_ids[position]
            position_logprobs = output.prompt_logprobs[position]
            if position_logprobs is None or token_id not in position_logprobs:
                raise RuntimeError(
                    f"vLLM omitted chosen token {token_id} from prompt log probabilities at position {position}."
                )
            sequence_logprob += float(position_logprobs[token_id].logprob)
        sequence_logprobs.append(sequence_logprob)

    soft_predictions: list[float] = []
    class_probabilities: list[list[float]] = []
    grouped_logprobs: list[list[float]] = []
    for start in range(0, len(sequence_logprobs), len(class_scores)):
        row_logprobs = sequence_logprobs[start : start + len(class_scores)]
        expected_score, probabilities = value_model_utils.expected_gen_value_score_from_logprobs(
            class_scores, row_logprobs
        )
        soft_predictions.append(value_model_utils.rescale_gen_value_score(expected_score, 0.0, 10.0))
        class_probabilities.append(probabilities)
        grouped_logprobs.append(row_logprobs)
    return soft_predictions, class_probabilities, grouped_logprobs


def main() -> None:
    args = parse_args()
    if args.output_jsonl.exists():
        raise FileExistsError(f"Refusing to overwrite validation output: {args.output_jsonl}.")
    metrics_path = args.metrics_json or args.output_jsonl.with_suffix(".metrics.json")
    if metrics_path.exists():
        raise FileExistsError(f"Refusing to overwrite validation metrics: {metrics_path}.")

    examples = value_model_utils.read_gen_value_validation_snapshot(args.input_snapshot)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
        enable_prefix_caching=args.soft_class_probabilities,
    )
    sampling_params = SamplingParams(
        n=1, temperature=0.0, max_tokens=args.max_new_tokens, stop=["</answer>"], include_stop_str_in_output=True
    )
    outputs = llm.generate([example["prompt"] for example in examples], sampling_params)
    generations = [output.outputs[0].text for output in outputs]
    parsed_scores = [
        value_model_utils.parse_generative_value_score(generation, score_min=0.0, score_max=10.0)
        for generation in generations
    ]
    predictions = [
        None if score is None else value_model_utils.rescale_gen_value_score(score, 0.0, 10.0)
        for score in parsed_scores
    ]
    metrics = value_model_utils.gen_value_validation_metrics(examples, predictions)
    soft_predictions: list[float] = []
    class_probabilities: list[list[float]] = []
    class_sequence_logprobs: list[list[float]] = []
    if args.soft_class_probabilities:
        soft_predictions, class_probabilities, class_sequence_logprobs = score_soft_class_predictions(
            llm, [example["prompt"] for example in examples]
        )
        soft_metrics = value_model_utils.gen_value_validation_metrics(examples, soft_predictions)
        metrics.update(
            {
                key.replace("gen_value/validation_", "gen_value/soft_validation_", 1): value
                for key, value in soft_metrics.items()
            }
        )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".tmp")
    with temporary_output.open("w", encoding="utf-8") as output_file:
        for index, (example, prediction, generation) in enumerate(zip(examples, predictions, generations, strict=True)):
            row = dict(example)
            row.update({"prediction": prediction, "generation": generation, "scored_model": args.model_path})
            if args.soft_class_probabilities:
                row.update(
                    {
                        "soft_prediction": soft_predictions[index],
                        "soft_class_scores": list(range(11)),
                        "soft_class_probabilities": class_probabilities[index],
                        "soft_class_sequence_logprobs": class_sequence_logprobs[index],
                    }
                )
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_output.replace(args.output_jsonl)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_metrics = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
    with temporary_metrics.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2, sort_keys=True)
        metrics_file.write("\n")
    temporary_metrics.replace(metrics_path)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
