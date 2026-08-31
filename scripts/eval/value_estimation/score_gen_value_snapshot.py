#!/usr/bin/env python3
"""Score a fixed generative-value validation snapshot with a vLLM checkpoint."""

from __future__ import annotations

import argparse
import json
import pathlib

from vllm import LLM, SamplingParams

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
    return parser.parse_args()


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

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".tmp")
    with temporary_output.open("w", encoding="utf-8") as output_file:
        for example, prediction, generation in zip(examples, predictions, generations, strict=True):
            row = dict(example)
            row.update({"prediction": prediction, "generation": generation, "scored_model": args.model_path})
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
