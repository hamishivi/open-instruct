"""Evaluate one frozen policy checkpoint on the local AIME protocol with vLLM.

This mirrors the in-training math evaluation while removing asynchronous weight
updates: every completion is sampled from the same immutable checkpoint.
"""

import argparse
import json
import pathlib
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from open_instruct import grpo_utils, logger_utils
from open_instruct.ground_truth_utils import FinalBoxedMathVerifier

logger = logger_utils.setup_logger(__name__)


@dataclass(frozen=True)
class EvalConfig:
    model_name_or_path: str
    output_dir: str
    run_name: str
    dataset_name: str = "mnoukhov/aime_2025_openinstruct"
    dataset_split: str = "train"
    num_samples: int = 8
    max_prompt_tokens: int = 2048
    max_new_tokens: int = 8192
    max_model_len: int = 10240
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int = 1
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    max_num_seqs: int = 256
    enable_prefix_caching: bool = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--dataset_name", default=EvalConfig.dataset_name)
    parser.add_argument("--dataset_split", default=EvalConfig.dataset_split)
    parser.add_argument("--num_samples", type=int, default=EvalConfig.num_samples)
    parser.add_argument("--max_prompt_tokens", type=int, default=EvalConfig.max_prompt_tokens)
    parser.add_argument("--max_new_tokens", type=int, default=EvalConfig.max_new_tokens)
    parser.add_argument("--max_model_len", type=int, default=EvalConfig.max_model_len)
    parser.add_argument("--temperature", type=float, default=EvalConfig.temperature)
    parser.add_argument("--top_p", type=float, default=EvalConfig.top_p)
    parser.add_argument("--seed", type=int, default=EvalConfig.seed)
    parser.add_argument("--tensor_parallel_size", type=int, default=EvalConfig.tensor_parallel_size)
    parser.add_argument("--gpu_memory_utilization", type=float, default=EvalConfig.gpu_memory_utilization)
    parser.add_argument("--max_num_seqs", type=int, default=EvalConfig.max_num_seqs)
    parser.add_argument(
        "--disable_prefix_caching",
        action="store_false",
        dest="enable_prefix_caching",
        help="Disable vLLM prefix caching (enabled by default).",
    )
    parser.set_defaults(enable_prefix_caching=EvalConfig.enable_prefix_caching)
    return parser


def validate_config(config: EvalConfig) -> None:
    if config.num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    if config.max_prompt_tokens < 1 or config.max_new_tokens < 1:
        raise ValueError("prompt and response token limits must be positive")
    if config.max_model_len < config.max_prompt_tokens + config.max_new_tokens:
        raise ValueError("max_model_len must cover max_prompt_tokens + max_new_tokens")
    if config.temperature <= 0:
        raise ValueError("temperature must be positive for stochastic pass@k evaluation")
    if not 0 < config.top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if config.max_num_seqs < 1:
        raise ValueError("max_num_seqs must be at least 1")


def evaluate(config: EvalConfig) -> dict[str, object]:
    validate_config(config)
    output_dir = pathlib.Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / f"{config.run_name}.jsonl"
    summary_path = output_dir / f"{config.run_name}.summary.json"
    if rows_path.exists() or summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing frozen evaluation for {config.run_name!r}")

    dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    if tokenizer.chat_template is None:
        raise ValueError(f"Checkpoint {config.model_name_or_path!r} does not contain a chat template")

    prompt_token_ids: list[list[int]] = []
    prompt_texts: list[str] = []
    for prompt_index, example in enumerate(dataset):
        token_ids = tokenizer.apply_chat_template(example["messages"], tokenize=True, add_generation_prompt=True)
        if len(token_ids) > config.max_prompt_tokens:
            raise ValueError(
                f"AIME prompt {prompt_index} has {len(token_ids)} tokens, exceeding {config.max_prompt_tokens}"
            )
        prompt_token_ids.append(token_ids)
        prompt_texts.append(tokenizer.decode(token_ids, skip_special_tokens=False))

    flat_prompts: list[list[int]] = []
    sampling_params: list[SamplingParams] = []
    metadata: list[tuple[int, int]] = []
    for prompt_index, token_ids in enumerate(prompt_token_ids):
        for sample_index in range(config.num_samples):
            flat_prompts.append(token_ids)
            metadata.append((prompt_index, sample_index))
            sampling_params.append(
                SamplingParams(
                    n=1,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    max_tokens=config.max_new_tokens,
                    seed=config.seed + sample_index,
                    logprobs=1,
                    include_stop_str_in_output=True,
                    skip_special_tokens=False,
                )
            )

    logger.info(
        "Evaluating %s frozen checkpoint on %d prompts x %d samples (%d requests)",
        config.model_name_or_path,
        len(prompt_token_ids),
        config.num_samples,
        len(flat_prompts),
    )
    llm = LLM(
        model=config.model_name_or_path,
        tokenizer=config.model_name_or_path,
        tensor_parallel_size=config.tensor_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
        max_num_seqs=config.max_num_seqs,
        enable_prefix_caching=config.enable_prefix_caching,
        generation_config="vllm",
    )
    started = time.perf_counter()
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_token_ids) for prompt_token_ids in flat_prompts],
        sampling_params,
    )
    generation_seconds = time.perf_counter() - started
    if len(outputs) != len(metadata):
        raise RuntimeError(f"vLLM returned {len(outputs)} requests for {len(metadata)} inputs")

    verifier = FinalBoxedMathVerifier()
    evaluation_rows: list[dict[str, object]] = []
    scores: list[float] = []
    response_lengths: list[int] = []
    finish_reasons: list[str] = []
    for request_output, (prompt_index, sample_index) in zip(outputs, metadata, strict=True):
        if len(request_output.outputs) != 1:
            raise RuntimeError(
                f"vLLM returned {len(request_output.outputs)} completions for prompt {prompt_index}, sample {sample_index}"
            )
        completion = request_output.outputs[0]
        response_token_ids = list(completion.token_ids)
        response = tokenizer.decode(response_token_ids, skip_special_tokens=True)
        raw_response = tokenizer.decode(response_token_ids, skip_special_tokens=False)
        example = dataset[prompt_index]
        score = verifier(
            tokenized_prediction=response_token_ids,
            prediction=response,
            label=example["ground_truth"],
            query=example["messages"][0]["content"],
        ).score
        scores.append(score)
        response_lengths.append(len(response_token_ids))
        finish_reasons.append(str(completion.finish_reason))
        evaluation_rows.append(
            {
                "prompt_index": prompt_index,
                "sample_index": sample_index,
                "seed": config.seed + sample_index,
                "prompt": prompt_texts[prompt_index],
                "ground_truth": example["ground_truth"],
                "response": raw_response,
                "response_token_ids": response_token_ids,
                "response_length": len(response_token_ids),
                "finish_reason": str(completion.finish_reason),
                "stop_reason": completion.stop_reason,
                "score": score,
            }
        )

    score_matrix = np.asarray(scores, dtype=np.float64).reshape(len(dataset), config.num_samples)
    pass_at_k = grpo_utils.compute_pass_at_k_metrics(score_matrix > 1.0 - 1e-8)
    total_response_tokens = int(sum(response_lengths))
    summary: dict[str, object] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": asdict(config),
        "num_prompts": len(dataset),
        "num_samples": config.num_samples,
        "num_completions": len(scores),
        "num_correct": int(sum(scores)),
        "eval/scores": float(np.mean(scores)),
        **pass_at_k,
        "eval/response_length_mean": float(np.mean(response_lengths)),
        "eval/response_length_min": int(min(response_lengths)),
        "eval/response_length_max": int(max(response_lengths)),
        "eval/stop_rate": float(np.mean([reason == "stop" for reason in finish_reasons])),
        "generation_seconds": generation_seconds,
        "response_tokens_per_second": total_response_tokens / generation_seconds,
        "total_response_tokens": total_response_tokens,
    }

    rows_tmp = rows_path.with_suffix(rows_path.suffix + ".tmp")
    with rows_tmp.open("w") as output_file:
        for row in evaluation_rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    rows_tmp.replace(rows_path)

    summary_tmp = summary_path.with_suffix(summary_path.suffix + ".tmp")
    summary_tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    summary_tmp.replace(summary_path)
    logger.info("Frozen AIME summary: %s", json.dumps(summary, sort_keys=True))
    return summary


def main() -> None:
    namespace = build_parser().parse_args()
    evaluate(EvalConfig(**vars(namespace)))


if __name__ == "__main__":
    main()
