# Copyright 2026 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Offline value-estimation harness.

Two entry points:

- ``make_dataset``: build a parquet of (prompt, ground_truth, rollout, probe_positions, mc_values)
  from DAPO math. 100 prompts each contribute one correct + one incorrect rollout; for each rollout,
  we probe at every 1000-th token and estimate the Monte-Carlo value as ``fraction_correct`` across
  32 continuations.
- ``score_dataset``: load a trained value model (scalar PPO or generative) and score the
  probes using whatever conditioning flags match its training-time conditioning.

A third helper, ``compare_runs``, ingests several ``score_dataset`` parquet outputs and emits a
consolidated comparison table.

All three are CLI-addressable. Shell wrappers live in ``scripts/eval/value_estimation/``.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import pathlib
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


# --------------------------------------------------------------------------------------------
# Common config
# --------------------------------------------------------------------------------------------
@dataclass
class MakeDatasetConfig:
    model_name_or_path: str
    output_path: str
    dataset_name: str = "hamishivi/DAPO-Math-17k-Processed_filtered"
    dataset_split: str = "train"
    exclude_problem_dataset_path: str | None = None
    """Optional parquet whose ``problem`` strings are excluded before sampling.

    This keeps offline value-training prefixes disjoint from a fixed Monte Carlo
    calibration set without relying on different random seeds to avoid overlap.
    """
    num_prompts_to_sample: int = 2000  # Sample this many, keep first 100 with 1 correct + 1 wrong.
    target_num_pairs: int = 100
    rollouts_per_prompt: int = 8
    continuations_per_probe: int = 32
    probe_interval: int = 1000
    min_probe_remaining_tokens: int = 64
    # Probe selection mode: "fixed" (every probe_interval tokens) or "sae" (SAE boundaries
    # from segment_rollout — tokens with prob < sae_threshold, downsampled to max_probes).
    probe_mode: str = "fixed"
    sae_threshold: float = 0.2
    max_probes: int = 16
    include_final_action_probe: bool = True
    """Also probe the state immediately before the sampled rollout's final token.

    This is not assigned the sampled rollout's binary outcome: fresh continuations
    may choose a different next token and recover, especially when much of the
    configured response budget remains.
    """
    # Chat template applied to each prompt before rollout. If set and registered in
    # dataset_transformation.CHAT_TEMPLATES, that template is used; otherwise the model's
    # built-in template is used. None skips templating (raw-string completion).
    chat_template_name: str | None = None
    max_prompt_length: int = 2048
    max_response_length: int = 8192
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int = 1
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    verifier_name: str = "math"
    keep_continuation_texts: bool = False


@dataclass
class ScoreDatasetConfig:
    input_dataset_path: str
    output_path: str
    value_model_path: str
    value_model_type: str = "scalar"  # one of: scalar, generative
    # Conditioning flags; must match what the value model was trained with.
    value_model_ground_truth_conditioning: bool = False
    gt_conditioning_template: str = "answer_prefix"
    rollout_context_num_siblings: int = -1
    gen_value_conditioning: str = "none"
    gen_value_score_min: float = 0.0
    gen_value_score_max: float = 10.0
    # Match the online critic budget.  GenAC critics are explicitly prompted to
    # reason before emitting <answer>...</answer>; an eight-token evaluator makes
    # a healthy reasoning critic look like a parse failure.
    gen_value_max_new_tokens: int = 1024
    gen_value_actor_model_name: str | None = None
    gen_value_actor_success_rate: float | None = None
    tokenizer_name_or_path: str | None = None
    run_name: str = "value_estimation_run"
    device: str = "cuda"
    batch_size: int = 4
    # vLLM options used only for generative value models.
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.85


@dataclass
class CompareRunsConfig:
    score_dataset_paths: list[str] = field(default_factory=list)
    output_markdown_path: str | None = None
    output_csv_path: str | None = None


@dataclass
class ConvertCheckpointConfig:
    checkpoint_dir: str  # directory containing value_model.bin
    output_dir: str
    # Path to a full HF model dir for config + tokenizer. Defaults to the parent of checkpoint_dir.
    base_model_path: str | None = None


# --------------------------------------------------------------------------------------------
# make_dataset
# --------------------------------------------------------------------------------------------
def _run_rollouts(
    prompts: list[str | list[int]],
    *,
    model_name_or_path: str,
    n: int,
    temperature: float,
    top_p: float,
    max_tokens: int | Sequence[int],
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    logprobs: bool = False,
) -> list[list[dict[str, Any]]]:
    """Run `n` rollouts per prompt through vLLM. Returns list-of-lists of dicts with keys
    ``token_ids`` (list[int]), ``text`` (str), ``logprobs`` (list[float] | None).
    """
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    llm = LLM(
        model=model_name_or_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    if isinstance(max_tokens, int):
        sampling: SamplingParams | list[SamplingParams] = SamplingParams(
            n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens, logprobs=1 if logprobs else None
        )
    else:
        if len(max_tokens) != len(prompts):
            raise ValueError(
                f"max_tokens must be one integer or one value per prompt ({len(max_tokens)} != {len(prompts)})."
            )
        sampling = [
            SamplingParams(
                n=n,
                temperature=temperature,
                top_p=top_p,
                max_tokens=int(prompt_max_tokens),
                logprobs=1 if logprobs else None,
            )
            for prompt_max_tokens in max_tokens
        ]
    raw = llm.generate(prompts, sampling)
    result: list[list[dict[str, Any]]] = []
    for out in raw:
        cands: list[dict[str, Any]] = []
        for c in out.outputs:
            lp = None
            if logprobs and getattr(c, "logprobs", None) is not None:
                lp = [next(iter(p.values())).logprob if p else 0.0 for p in c.logprobs]
            cands.append(
                {
                    "prompt_token_ids": list(out.prompt_token_ids),
                    "token_ids": list(c.token_ids),
                    "text": c.text,
                    "logprobs": lp,
                }
            )
        result.append(cands)
    # Cleanly shut down the LLM engine so the next call can re-init without leaks.
    with contextlib.suppress(Exception):
        del llm
    return result


def _actor_state_token_ids(prompt_token_ids: Sequence[int], rollout_token_ids: Sequence[int], probe: int) -> list[int]:
    """Return the exact actor state used to sample continuations at ``probe``.

    Reusing token IDs avoids the lossy token-to-character approximation that can
    otherwise move a probe into a different reasoning state after decode/re-encode.
    """
    if not 0 <= probe <= len(rollout_token_ids):
        raise ValueError(f"Probe {probe} must be in [0, {len(rollout_token_ids)}].")
    return [*prompt_token_ids, *rollout_token_ids[:probe]]


def _decode_full_continuation(
    tokenizer: Any, rollout_prefix_token_ids: Sequence[int], continuation_token_ids: Sequence[int]
) -> str:
    """Decode the complete candidate response that the verifier must grade."""
    return tokenizer.decode([*rollout_prefix_token_ids, *continuation_token_ids], skip_special_tokens=True)


def _fixed_probe_positions(
    rollout_length: int,
    response_token_limit: int,
    probe_interval: int,
    min_remaining_tokens: int,
    max_probes: int,
    include_final_action_probe: bool,
) -> list[int]:
    """Choose states using the configured response horizon, not sampled EOS.

    A rollout ending at token 1,000 does not imply only zero tokens remain: from
    the state before its sampled EOS, another continuation can decline to stop and
    use the rest of the 8,192-token budget. The old selector incorrectly filtered
    such states using ``rollout_length - probe``.
    """
    if rollout_length < 0 or response_token_limit <= 0 or rollout_length > response_token_limit:
        raise ValueError(
            "rollout_length must be nonnegative and no larger than the positive response_token_limit, got "
            f"{rollout_length} and {response_token_limit}."
        )
    if probe_interval <= 0 or min_remaining_tokens < 0 or max_probes <= 0:
        raise ValueError("probe_interval and max_probes must be positive; min_remaining_tokens must be nonnegative.")

    positions = [
        probe
        for probe in range(probe_interval, rollout_length, probe_interval)
        if response_token_limit - probe >= min_remaining_tokens
    ]
    if include_final_action_probe and rollout_length > 0:
        final_action_probe = rollout_length - 1
        # The final-action state needs only the sampled final token's remaining
        # budget. Keeping the generic intermediate-probe floor here discards
        # exactly the near-budget states where terminal calibration matters.
        positions.append(final_action_probe)
    positions = sorted(set(positions))
    if len(positions) <= max_probes:
        return positions
    # Retain broad horizon coverage and always preserve the latest selected state.
    selected_indices = np.linspace(0, len(positions) - 1, num=max_probes, dtype=int)
    return [positions[index] for index in sorted(set(selected_indices.tolist()))]


_VERIFIER_CACHE: dict[str, Any] = {}


def _verify(prediction: str, ground_truth: str, verifier_name: str) -> bool:
    from open_instruct.ground_truth_utils import MathVerifier, StringMatcherVerifier  # noqa: PLC0415

    key = (verifier_name or "math").lower()
    if key not in _VERIFIER_CACHE:
        _VERIFIER_CACHE[key] = MathVerifier() if key in {"math", "strict_math"} else StringMatcherVerifier()
    v = _VERIFIER_CACHE[key]
    return float(v(tokenized_prediction=[], prediction=prediction, label=ground_truth).score) >= 1.0


def _extract_problem(row: dict[str, Any]) -> str:
    """Return the unformatted user problem used to identify held-out states."""
    if "messages" in row and isinstance(row["messages"], list) and row["messages"]:
        return str(row["messages"][-1].get("content", ""))
    return str(row.get("prompt", ""))


def _sample_record_indices(
    dataset: Any, *, num_to_sample: int, seed: int, excluded_problems: set[str] | None = None
) -> list[int]:
    """Sample dataset rows after exact problem-level holdout exclusion."""
    excluded_problems = excluded_problems or set()
    eligible_indices = [
        index for index in range(len(dataset)) if _extract_problem(dataset[index]) not in excluded_problems
    ]
    if not eligible_indices:
        raise ValueError("No dataset rows remain after held-out problem exclusion.")
    rng = random.Random(seed)
    return rng.sample(eligible_indices, min(num_to_sample, len(eligible_indices)))


def make_dataset(cfg: MakeDatasetConfig) -> str:
    """Build the value-estimation dataset described in the plan."""
    import pandas as pd  # noqa: PLC0415
    from datasets import load_dataset  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    np.random.seed(cfg.seed)

    logger.info(f"Loading dataset {cfg.dataset_name} split={cfg.dataset_split}")
    ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split)
    excluded_problems: set[str] = set()
    if cfg.exclude_problem_dataset_path is not None:
        exclusion_path = pathlib.Path(cfg.exclude_problem_dataset_path)
        if not exclusion_path.is_file():
            raise FileNotFoundError(f"Held-out problem dataset does not exist: {exclusion_path}")
        exclusion_frame = pd.read_parquet(exclusion_path, columns=["problem"])
        excluded_problems = {
            problem for problem in exclusion_frame["problem"].tolist() if isinstance(problem, str) and problem
        }
        logger.info(f"Excluding {len(excluded_problems)} held-out problems from value-estimation sampling")
    indices = _sample_record_indices(
        ds, num_to_sample=cfg.num_prompts_to_sample, seed=cfg.seed, excluded_problems=excluded_problems
    )
    records = [ds[i] for i in indices]

    def _extract_prompt(row: dict) -> str:
        return _extract_problem(row)

    def _extract_gt(row: dict) -> str:
        gt = row.get("ground_truth") or row.get("gt") or row.get("answer") or ""
        if isinstance(gt, list):
            gt = gt[0] if gt else ""
        return str(gt)

    problems = [_extract_prompt(r) for r in records]
    prompts = list(problems)
    ground_truths = [_extract_gt(r) for r in records]
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)

    if cfg.chat_template_name is not None:
        from open_instruct.dataset_transformation import CHAT_TEMPLATES  # noqa: PLC0415

        if cfg.chat_template_name == "builtin":
            if not getattr(tokenizer, "chat_template", None):
                raise ValueError(
                    f"--chat_template_name=builtin but {cfg.model_name_or_path} has no built-in chat template."
                )
            source = "built-in"
        elif cfg.chat_template_name in CHAT_TEMPLATES:
            tokenizer.chat_template = CHAT_TEMPLATES[cfg.chat_template_name]
            source = "registered"
        else:
            raise ValueError(
                f"--chat_template_name={cfg.chat_template_name!r} is not registered in CHAT_TEMPLATES "
                f"and is not the 'builtin' sentinel. Known: {sorted(CHAT_TEMPLATES.keys())}."
            )
        logger.info(f"Applying chat template {cfg.chat_template_name!r} ({source})")
        prompts = [
            tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]

    logger.info(f"Running {cfg.rollouts_per_prompt} rollouts/prompt for {len(prompts)} prompts")
    rollouts = _run_rollouts(
        prompts,
        model_name_or_path=cfg.model_name_or_path,
        n=cfg.rollouts_per_prompt,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_response_length,
        tensor_parallel_size=cfg.tensor_parallel_size,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        logprobs=cfg.probe_mode == "sae",
    )

    # Filter to prompts with at least one correct and one incorrect rollout.
    kept = []
    evaluated_verdicts: list[bool] = []
    for i, cands in enumerate(rollouts):
        gt = ground_truths[i]
        verdicts = [_verify(c["text"], gt, cfg.verifier_name) for c in cands]
        evaluated_verdicts.extend(verdicts)
        has_correct = any(verdicts)
        has_incorrect = not all(verdicts)
        if has_correct and has_incorrect:
            kept.append((i, cands, verdicts))
            if len(kept) >= cfg.target_num_pairs:
                break
    logger.info(f"Kept {len(kept)} prompts with at least one correct + incorrect rollout")
    observed_actor_success_rate = sum(evaluated_verdicts) / max(len(evaluated_verdicts), 1)
    logger.info(f"Observed actor success rate across screened rollouts: {observed_actor_success_rate:.3f}")

    # Build rows: for each kept prompt, pick the first correct + first incorrect rollout.
    # The other rollouts become the sibling_rollouts pool for conditioning variants.
    rows: list[dict[str, Any]] = []
    continuation_prompts: list[list[int]] = []
    continuation_max_tokens: list[int] = []
    continuation_indices: list[tuple[int, int]] = []  # (row_idx, probe_idx)
    for orig_idx, cands, verdicts in kept:
        gt = ground_truths[orig_idx]
        prompt = prompts[orig_idx]
        first_correct = next(i for i, v in enumerate(verdicts) if v)
        first_incorrect = next(i for i, v in enumerate(verdicts) if not v)
        for rollout_idx in (first_correct, first_incorrect):
            main = cands[rollout_idx]
            siblings = [
                {"text": cands[k]["text"], "is_correct": verdicts[k]} for k in range(len(cands)) if k != rollout_idx
            ]
            tokens = main["token_ids"]
            length = len(tokens)
            if cfg.probe_mode == "sae":
                from open_instruct import value_model_utils  # noqa: PLC0415

                raw_boundaries = value_model_utils.segment_rollout(
                    response_tokens=tokens,
                    response_logprobs=main.get("logprobs"),
                    mode="sae",
                    sae_threshold=cfg.sae_threshold,
                    max_segments=cfg.max_probes,
                )
                probe_positions = [
                    t
                    for t in raw_boundaries
                    if 0 <= t < length and cfg.max_response_length - t >= cfg.min_probe_remaining_tokens
                ]
                if cfg.include_final_action_probe and length > 0:
                    final_action_probe = length - 1
                    # Always retain the causal state immediately before the
                    # sampled final action. It has at least one token of budget,
                    # even when ordinary probes require a larger continuation.
                    probe_positions.append(final_action_probe)
                probe_positions = sorted(set(probe_positions))
                if len(probe_positions) > cfg.max_probes:
                    selected_indices = np.linspace(0, len(probe_positions) - 1, num=cfg.max_probes, dtype=int)
                    probe_positions = [probe_positions[index] for index in sorted(set(selected_indices.tolist()))]
            else:
                if cfg.probe_mode != "fixed":
                    raise ValueError(f"Unknown probe_mode: {cfg.probe_mode!r}; expected 'fixed' or 'sae'.")
                probe_positions = _fixed_probe_positions(
                    rollout_length=length,
                    response_token_limit=cfg.max_response_length,
                    probe_interval=cfg.probe_interval,
                    min_remaining_tokens=cfg.min_probe_remaining_tokens,
                    max_probes=cfg.max_probes,
                    include_final_action_probe=cfg.include_final_action_probe,
                )
            row = {
                "prompt": prompt,
                "problem": problems[orig_idx],
                "prompt_token_ids": main["prompt_token_ids"],
                "ground_truth": gt,
                "verifier_name": cfg.verifier_name,
                "rollout_text": main["text"],
                "rollout_tokens": tokens,
                "rollout_is_correct": bool(verdicts[rollout_idx]),
                "sibling_rollouts": siblings,
                "probe_positions": probe_positions,
                "mc_values": [],  # filled in below
                "num_continuations": cfg.continuations_per_probe,
                "response_token_limit": cfg.max_response_length,
                "actor_model_name": cfg.model_name_or_path,
                "actor_success_rate": observed_actor_success_rate,
            }
            row_idx = len(rows)
            rows.append(row)
            for p_idx, t in enumerate(probe_positions):
                continuation_prompts.append(_actor_state_token_ids(main["prompt_token_ids"], main["token_ids"], t))
                continuation_max_tokens.append(cfg.max_response_length - t)
                continuation_indices.append((row_idx, p_idx))

    # Compute MC values per probe by generating 32 continuations in one big vLLM batch.
    if continuation_prompts:
        logger.info(
            f"Running {cfg.continuations_per_probe} continuations for "
            f"{len(continuation_prompts)} probes ({len(rows)} rollouts)"
        )
        conts = _run_rollouts(
            continuation_prompts,
            model_name_or_path=cfg.model_name_or_path,
            n=cfg.continuations_per_probe,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=continuation_max_tokens,
            tensor_parallel_size=cfg.tensor_parallel_size,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
        )
        for (row_idx, p_idx), cands in zip(continuation_indices, conts):
            gt = rows[row_idx]["ground_truth"]
            probe = rows[row_idx]["probe_positions"][p_idx]
            rollout_prefix = rows[row_idx]["rollout_tokens"][:probe]
            full_responses = [_decode_full_continuation(tokenizer, rollout_prefix, c["token_ids"]) for c in cands]
            verdicts = [_verify(response, gt, rows[row_idx]["verifier_name"]) for response in full_responses]
            mc = sum(1 for v in verdicts if v) / max(len(verdicts), 1)
            rows[row_idx]["mc_values"].append(float(mc))
            if cfg.keep_continuation_texts:
                rows[row_idx].setdefault("continuation_texts", []).append(full_responses)

    pathlib.Path(os.path.dirname(cfg.output_path) or ".").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(cfg.output_path, index=False)
    logger.info(f"Wrote {len(rows)} rows to {cfg.output_path}")
    return cfg.output_path


# --------------------------------------------------------------------------------------------
# score_dataset
# --------------------------------------------------------------------------------------------
def _average_ranks(values: Sequence[float]) -> np.ndarray:
    """Return one-based ranks with tied values assigned their average rank."""
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return ranks


def _pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if len(left_array) != len(right_array) or len(left_array) < 2:
        return float("nan")
    if np.ptp(left_array) == 0 or np.ptp(right_array) == 0:
        return float("nan")
    return float(np.corrcoef(left_array, right_array)[0, 1])


def _spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    return _pearson_correlation(_average_ranks(left), _average_ranks(right))


def _optional_sequence_as_list(value: Any) -> list[Any]:
    """Normalize optional parquet list columns without truth-testing NumPy arrays."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return list(value)
    if isinstance(value, float) and np.isnan(value):
        return []
    return [value]


def _prediction_group_metrics(
    predictions: Sequence[float | None], targets: Sequence[float], *, prefix: str
) -> dict[str, float]:
    """Summarize one semantically meaningful value-estimation slice.

    Parse failures count as unit error in ``penalized_mse``.  Parsed-only
    statistics remain separate so a model cannot appear calibrated merely by
    refusing to emit a valid value on difficult states.
    """
    if len(predictions) != len(targets):
        raise ValueError(f"Predictions and targets differ in length ({len(predictions)} != {len(targets)}).")
    if not predictions:
        return {}

    normalized_predictions = [
        None if prediction is None or not np.isfinite(float(prediction)) else float(prediction)
        for prediction in predictions
    ]
    parsed_indices = [index for index, prediction in enumerate(normalized_predictions) if prediction is not None]
    metrics = {
        f"{prefix}_examples": float(len(predictions)),
        f"{prefix}_parse_rate": len(parsed_indices) / len(predictions),
        f"{prefix}_target_mean": float(np.mean([float(target) for target in targets])),
        f"{prefix}_penalized_mse": float(
            np.mean(
                [
                    1.0 if prediction is None else (float(prediction) - float(target)) ** 2
                    for prediction, target in zip(normalized_predictions, targets)
                ]
            )
        ),
    }
    if parsed_indices:
        parsed_predictions = [float(normalized_predictions[index]) for index in parsed_indices]
        parsed_targets = [float(targets[index]) for index in parsed_indices]
        metrics[f"{prefix}_pred_mean"] = float(np.mean(parsed_predictions))
        metrics[f"{prefix}_parsed_target_mean"] = float(np.mean(parsed_targets))
        metrics[f"{prefix}_mse"] = float(
            np.mean([(prediction - target) ** 2 for prediction, target in zip(parsed_predictions, parsed_targets)])
        )
    return metrics


def score_dataset(cfg: ScoreDatasetConfig) -> str:
    import pandas as pd  # noqa: PLC0415

    # Accept either a local parquet path or a HuggingFace dataset name (org/repo).
    if os.path.exists(cfg.input_dataset_path):
        df = pd.read_parquet(cfg.input_dataset_path)
    else:
        from datasets import load_dataset as _load_dataset  # noqa: PLC0415

        logger.info(f"Local path not found; loading from HuggingFace: {cfg.input_dataset_path}")
        hf_ds = _load_dataset(cfg.input_dataset_path, split="test")
        df = hf_ds.to_pandas()
    logger.info(f"Loaded {len(df)} rows from {cfg.input_dataset_path}")

    # Conditioning warning: compare training_args.json if present.
    training_args_path = os.path.join(cfg.value_model_path, "..", "training_args.json")
    if os.path.exists(training_args_path):
        with open(training_args_path) as f:
            ta = json.load(f)
        if bool(ta.get("value_model_ground_truth_conditioning", False)) != cfg.value_model_ground_truth_conditioning:
            logger.warning(
                "Conditioning flag mismatch between checkpoint and score_dataset: "
                f"ckpt={ta.get('value_model_ground_truth_conditioning')}, "
                f"score={cfg.value_model_ground_truth_conditioning}."
            )
        if ta.get("gt_conditioning_template") != cfg.gt_conditioning_template:
            logger.warning(
                f"gt_conditioning_template mismatch: ckpt={ta.get('gt_conditioning_template')!r}, "
                f"score={cfg.gt_conditioning_template!r}."
            )

    preds_per_row: list[list[float | None]] = []
    all_preds: list[float | None] = []
    all_mc: list[float] = []
    all_horizon_fractions: list[float] = []
    grouped_predictions: dict[str, list[float | None]] = {
        "final_action_correct": [],
        "final_action_incorrect": [],
        "intermediate_correct": [],
        "intermediate_incorrect": [],
    }
    grouped_targets: dict[str, list[float]] = {group: [] for group in grouped_predictions}

    if cfg.value_model_type == "scalar":
        preds_per_row = _score_with_scalar_value(df, cfg)
    elif cfg.value_model_type == "generative":
        preds_per_row = _score_with_generative_value(df, cfg)
    else:
        raise ValueError(f"Unknown value_model_type: {cfg.value_model_type}")

    correct_preds: list[float] = []
    incorrect_preds: list[float] = []
    probe_rows = []
    for i, row in df.iterrows():
        is_correct = bool(row.get("rollout_is_correct"))
        rollout_length = len(row["rollout_tokens"])
        response_token_limit = int(row.get("response_token_limit", 8192))
        for pos, p, mc in zip(row["probe_positions"], preds_per_row[i], row["mc_values"]):
            prediction = None if p is None or not np.isfinite(float(p)) else float(p)
            state_kind = "final_action" if int(pos) == rollout_length - 1 else "intermediate"
            group = f"{state_kind}_{'correct' if is_correct else 'incorrect'}"
            all_preds.append(prediction)
            all_mc.append(float(mc))
            grouped_predictions[group].append(prediction)
            grouped_targets[group].append(float(mc))
            horizon_fraction = int(pos) / max(response_token_limit, 1)
            all_horizon_fractions.append(horizon_fraction)
            if prediction is not None:
                if is_correct:
                    correct_preds.append(prediction)
                else:
                    incorrect_preds.append(prediction)
            probe_rows.append(
                {
                    "run_name": cfg.run_name,
                    "rollout_idx": i,
                    "rollout_is_correct": is_correct,
                    "state_kind": state_kind,
                    "probe_position": int(pos),
                    "response_tokens_remaining": response_token_limit - int(pos),
                    "horizon_fraction": horizon_fraction,
                    "predicted_value": prediction,
                    "parsed": prediction is not None,
                    "mc_value": float(mc),
                }
            )

    # Metrics
    metrics: dict[str, float] = {}
    if all_preds:
        parsed_indices = [index for index, prediction in enumerate(all_preds) if prediction is not None]
        parsed_preds = [float(all_preds[index]) for index in parsed_indices]
        parsed_mc = [all_mc[index] for index in parsed_indices]
        metrics["examples"] = float(len(all_preds))
        metrics["parse_rate"] = len(parsed_indices) / len(all_preds)
        metrics["penalized_mse"] = float(
            np.mean(
                [
                    1.0 if prediction is None else (float(prediction) - target) ** 2
                    for prediction, target in zip(all_preds, all_mc)
                ]
            )
        )
        if parsed_preds:
            diffs = [prediction - target for prediction, target in zip(parsed_preds, parsed_mc)]
            metrics["mae"] = float(np.mean([abs(difference) for difference in diffs]))
            metrics["mse"] = float(np.mean([difference**2 for difference in diffs]))
        if len(parsed_preds) > 1:
            pearson = _pearson_correlation(parsed_preds, parsed_mc)
            spearman = _spearman_correlation(parsed_preds, parsed_mc)
            if np.isfinite(pearson):
                metrics["pearson"] = pearson
            if np.isfinite(spearman):
                metrics["spearman"] = spearman
        # Calibration bins (deciles of predicted values).
        order = np.argsort(parsed_preds)
        bin_size = max(1, len(order) // 10)
        for b in range(10):
            chunk = order[b * bin_size : (b + 1) * bin_size] if b < 9 else order[b * bin_size :]
            if len(chunk) == 0:
                continue
            metrics[f"calib_bin_{b}_pred_mean"] = float(np.mean([parsed_preds[j] for j in chunk]))
            metrics[f"calib_bin_{b}_mc_mean"] = float(np.mean([parsed_mc[j] for j in chunk]))

        horizon_buckets = {
            "early": [j for j, fraction in enumerate(all_horizon_fractions) if fraction < 0.25],
            "middle": [j for j, fraction in enumerate(all_horizon_fractions) if 0.25 <= fraction < 0.75],
            "late": [j for j, fraction in enumerate(all_horizon_fractions) if fraction >= 0.75],
        }
        for name, indices in horizon_buckets.items():
            if not indices:
                continue
            bucket_parsed_indices = [index for index in indices if all_preds[index] is not None]
            bucket_preds = [float(all_preds[index]) for index in bucket_parsed_indices]
            bucket_mc = [all_mc[index] for index in bucket_parsed_indices]
            metrics[f"horizon_{name}_examples"] = float(len(indices))
            metrics[f"horizon_{name}_parse_rate"] = len(bucket_parsed_indices) / len(indices)
            metrics[f"horizon_{name}_penalized_mse"] = float(
                np.mean(
                    [
                        1.0 if all_preds[index] is None else (float(all_preds[index]) - all_mc[index]) ** 2
                        for index in indices
                    ]
                )
            )
            if bucket_preds:
                metrics[f"horizon_{name}_pred_mean"] = float(np.mean(bucket_preds))
                metrics[f"horizon_{name}_mc_mean"] = float(np.mean(bucket_mc))
                metrics[f"horizon_{name}_mse"] = float(
                    np.mean([(prediction - target) ** 2 for prediction, target in zip(bucket_preds, bucket_mc)])
                )

    if correct_preds:
        metrics["correct_pred_mean"] = float(np.mean(correct_preds))
    if incorrect_preds:
        metrics["incorrect_pred_mean"] = float(np.mean(incorrect_preds))
    for group in grouped_predictions:
        metrics.update(_prediction_group_metrics(grouped_predictions[group], grouped_targets[group], prefix=group))

    # Write output parquet.
    import pandas as pd  # noqa: PLC0415

    df_out = df.copy()
    df_out["predicted_values"] = preds_per_row
    df_out["run_config"] = [json.dumps(dataclasses.asdict(cfg))] * len(df_out)
    pathlib.Path(os.path.dirname(cfg.output_path) or ".").mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(cfg.output_path, index=False)
    pd.DataFrame(probe_rows).to_csv(cfg.output_path + ".probes.csv", index=False)
    # Write a small JSON summary next to the parquet.
    summary_path = cfg.output_path + ".summary.json"
    summary = {"run_name": cfg.run_name, "value_model_type": cfg.value_model_type, "metrics": metrics}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote {len(df_out)} predictions to {cfg.output_path}; metrics: {metrics}")
    return cfg.output_path


def _score_with_scalar_value(df, cfg: ScoreDatasetConfig) -> list[list[float]]:
    """Score probes using a scalar value model loaded via HF."""
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415
    from safetensors.torch import load_file as _load_sf  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tok_path = cfg.tokenizer_name_or_path or cfg.value_model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    # lm_head.weight has shape (1, hidden_size) but config.vocab_size is the full vocabulary.
    # Load with ignore_mismatched_sizes so the embedding table loads correctly, then
    # replace lm_head and load its weight manually from the safetensors file.
    value_model = AutoModelForCausalLM.from_pretrained(
        cfg.value_model_path, torch_dtype=torch.bfloat16, ignore_mismatched_sizes=True
    )
    hidden_size = value_model.config.hidden_size
    value_model.lm_head = nn.Linear(hidden_size, 1, bias=False, dtype=torch.bfloat16)
    sf_path = os.path.join(cfg.value_model_path, "model.safetensors")
    if os.path.exists(sf_path):
        sd = _load_sf(sf_path)
        if "lm_head.weight" in sd:
            value_model.lm_head.weight.data.copy_(sd["lm_head.weight"].to(torch.bfloat16))
    value_model = value_model.to(cfg.device)
    value_model.eval()
    from open_instruct import value_model_utils  # noqa: PLC0415

    all_preds: list[list[float]] = []
    with torch.no_grad():
        for _, row in df.iterrows():
            prompt = row["prompt"]
            rollout_tokens = list(row["rollout_tokens"])
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            cond_ids: list[int] = []
            if cfg.value_model_ground_truth_conditioning:
                sibs = _optional_sequence_as_list(row.get("sibling_rollouts"))
                cond_text = value_model_utils.build_conditioning_text(
                    cfg.gt_conditioning_template, row["ground_truth"], siblings=sibs
                )
                cond_ids = tokenizer.encode(cond_text, add_special_tokens=False)
            is_postfix = value_model_utils.is_postfix_template(cfg.gt_conditioning_template)
            preds: list[float] = []
            for t in row["probe_positions"]:
                partial_ids = rollout_tokens[:t]
                all_ids = prompt_ids + cond_ids + partial_ids if is_postfix else cond_ids + prompt_ids + partial_ids
                input_ids = torch.tensor([all_ids[-16384:]], dtype=torch.long).to(cfg.device)
                out = value_model(input_ids=input_ids)
                logits = getattr(out, "logits", out)[:, -1]  # last-token logit
                v = float(logits.squeeze(-1).item())
                preds.append(v)
            all_preds.append(preds)
    return all_preds


def _score_with_generative_value(df, cfg: ScoreDatasetConfig) -> list[list[float | None]]:
    """Score probes using a generative value model served via vLLM."""
    from transformers import AutoTokenizer  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    from open_instruct import value_model_utils  # noqa: PLC0415

    tok_path = cfg.tokenizer_name_or_path or cfg.value_model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    llm = LLM(
        model=cfg.value_model_path,
        tensor_parallel_size=cfg.vllm_tensor_parallel_size,
        gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
    )
    sp = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=cfg.gen_value_max_new_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    prompts: list[str] = []
    positions: list[tuple[int, int]] = []
    for idx, row in df.iterrows():
        rollout_tokens = list(row["rollout_tokens"])
        for p_idx, t in enumerate(row["probe_positions"]):
            # Online generative-value scoring preserves special response tokens
            # when decoding causal prefixes; use the identical representation in
            # held-out scoring.
            partial = tokenizer.decode(rollout_tokens[:t], skip_special_tokens=False)
            prompt = value_model_utils.build_generative_value_prompt(
                partial,
                conditioning=cfg.gen_value_conditioning,
                ground_truth=row["ground_truth"],
                siblings=_optional_sequence_as_list(row.get("sibling_rollouts")),
                score_min=cfg.gen_value_score_min,
                score_max=cfg.gen_value_score_max,
                problem=row.get("problem", row.get("prompt", "")),
                actor_model_name=cfg.gen_value_actor_model_name or row.get("actor_model_name"),
                actor_success_rate=(
                    cfg.gen_value_actor_success_rate
                    if cfg.gen_value_actor_success_rate is not None
                    else row.get("actor_success_rate")
                ),
                response_tokens_used=int(t),
                response_token_limit=int(row.get("response_token_limit", 8192)),
            )
            prompts.append(prompt)
            positions.append((idx, p_idx))

    raw = llm.generate(prompts, sp)
    # Re-collate into per-row lists.
    all_preds: list[list[float | None]] = [[None for _ in row["probe_positions"]] for _, row in df.iterrows()]
    for out, (row_idx, p_idx) in zip(raw, positions):
        txt = out.outputs[0].text if hasattr(out, "outputs") else ""
        parsed = value_model_utils.parse_generative_value_score(
            txt, score_min=cfg.gen_value_score_min, score_max=cfg.gen_value_score_max
        )
        if parsed is not None:
            all_preds[row_idx][p_idx] = value_model_utils.rescale_gen_value_score(
                parsed, cfg.gen_value_score_min, cfg.gen_value_score_max
            )
    return all_preds


# --------------------------------------------------------------------------------------------
# compare_runs
# --------------------------------------------------------------------------------------------
def compare_runs(cfg: CompareRunsConfig) -> str | None:
    import pandas as pd  # noqa: PLC0415

    rows = []
    for p in cfg.score_dataset_paths:
        summary_path = p + ".summary.json"
        if not os.path.exists(summary_path):
            logger.warning(f"Summary missing for {p}, skipping")
            continue
        with open(summary_path) as f:
            s = json.load(f)
        row = {"run_name": s.get("run_name"), "value_model_type": s.get("value_model_type")}
        row.update(s.get("metrics", {}))
        rows.append(row)
    if not rows:
        logger.warning("No runs to compare; skipping")
        return None
    frame = pd.DataFrame(rows)
    md_path: str | None = None
    if cfg.output_csv_path:
        frame.to_csv(cfg.output_csv_path, index=False)
        logger.info(f"Wrote comparison CSV to {cfg.output_csv_path}")
    if cfg.output_markdown_path:
        md = frame.to_markdown(index=False)
        with open(cfg.output_markdown_path, "w") as f:
            f.write(md)
        md_path = cfg.output_markdown_path
        logger.info(f"Wrote comparison markdown to {cfg.output_markdown_path}")
    logger.info("\n" + frame.to_string())
    return md_path


# --------------------------------------------------------------------------------------------
# convert_checkpoint
# --------------------------------------------------------------------------------------------
def convert_checkpoint(cfg: ConvertCheckpointConfig) -> str:
    """Convert a ``value_model.bin`` checkpoint into a HF-loadable directory.

    The value model's ``lm_head`` has shape ``[1, hidden_size]`` rather than
    ``[vocab_size, hidden_size]``.  We set ``vocab_size=1`` and truncate
    ``embed_tokens.weight`` so that ``AutoModelForCausalLM.from_pretrained``
    can load the converted directory directly.
    """
    import shutil  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from safetensors.torch import save_file  # noqa: PLC0415

    ckpt_dir = pathlib.Path(cfg.checkpoint_dir)
    out_dir = pathlib.Path(cfg.output_dir)
    base_dir = pathlib.Path(cfg.base_model_path) if cfg.base_model_path else ckpt_dir.parent

    value_bin = ckpt_dir / "value_model.bin"
    if not value_bin.exists():
        raise FileNotFoundError(f"value_model.bin not found in {ckpt_dir}")

    config_src = base_dir / "config.json"
    if not config_src.exists():
        raise FileNotFoundError(f"config.json not found in {base_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    with open(config_src) as f:
        cfg_json = json.load(f)
    # Keep vocab_size intact so embed_tokens stays full-size at inference.
    # lm_head.weight has shape (1, hidden) in the checkpoint; the loader handles
    # the mismatch with ignore_mismatched_sizes and loads it manually.
    cfg_json["tie_word_embeddings"] = False
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg_json, f, indent=2)

    sd = torch.load(value_bin, map_location="cpu", weights_only=True)
    sd_mod = {k: v.bfloat16() for k, v in sd.items()}
    save_file(sd_mod, out_dir / "model.safetensors")
    weight_map = {k: "model.safetensors" for k in sd_mod}
    total_size = sum(v.numel() * 2 for v in sd_mod.values())
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(out_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"]:
        src = base_dir / fname
        if src.exists():
            shutil.copy(src, out_dir / fname)

    logger.info(f"Converted value checkpoint to {out_dir}")
    return str(out_dir)


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def _cfg_from_args(cfg_cls, args_ns) -> Any:
    kwargs = {}
    for f in dataclasses.fields(cfg_cls):
        if hasattr(args_ns, f.name):
            v = getattr(args_ns, f.name)
            if v is not None:
                kwargs[f.name] = v
    return cfg_cls(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_make = sub.add_parser("make_dataset", help="Build the value-estimation dataset.")
    for f in dataclasses.fields(MakeDatasetConfig):
        _add_field(p_make, f)

    p_score = sub.add_parser("score_dataset", help="Score the value-estimation dataset with a model.")
    for f in dataclasses.fields(ScoreDatasetConfig):
        _add_field(p_score, f)

    p_cmp = sub.add_parser("compare_runs", help="Aggregate score_dataset outputs into a table.")
    p_cmp.add_argument("--score_dataset_paths", nargs="+", required=True)
    p_cmp.add_argument("--output_markdown_path", default=None)
    p_cmp.add_argument("--output_csv_path", default=None)

    p_conv = sub.add_parser("convert_checkpoint", help="Convert value_model.bin to a HF-loadable directory.")
    for f in dataclasses.fields(ConvertCheckpointConfig):
        _add_field(p_conv, f)

    args = parser.parse_args()
    if args.cmd == "make_dataset":
        cfg = _cfg_from_args(MakeDatasetConfig, args)
        make_dataset(cfg)
    elif args.cmd == "score_dataset":
        cfg = _cfg_from_args(ScoreDatasetConfig, args)
        score_dataset(cfg)
    elif args.cmd == "compare_runs":
        cfg = CompareRunsConfig(
            score_dataset_paths=args.score_dataset_paths,
            output_markdown_path=args.output_markdown_path,
            output_csv_path=args.output_csv_path,
        )
        compare_runs(cfg)
    elif args.cmd == "convert_checkpoint":
        cfg = _cfg_from_args(ConvertCheckpointConfig, args)
        convert_checkpoint(cfg)


def _add_field(parser, f) -> None:
    kwargs: dict[str, Any] = {"dest": f.name}
    if f.default is not dataclasses.MISSING:
        kwargs["default"] = None  # use None so CLI absence means "use default"
    field_type = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", str(f.type))
    if field_type in {"bool", "bool | None"}:
        kwargs["action"] = "store_true"
    elif field_type in {"int", "int | None"}:
        kwargs["type"] = int
    elif field_type in {"float", "float | None"}:
        kwargs["type"] = float
    elif field_type in {"str", "str | None"}:
        kwargs["type"] = str
    else:
        kwargs["type"] = str  # fallback; works for list[str] via comma-separated
    parser.add_argument(f"--{f.name}", **kwargs)


if __name__ == "__main__":
    main()
