# Copyright 2026 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Helpers for the PPO / SAE value model used by grpo_fast.py.

The value model itself is built, optimized, and DeepSpeed-managed inside
`PolicyTrainerRayProcess.from_pretrained`; this module provides stateless helpers for:

- building value-conditioning strings from ground truths + sibling rollouts;
- running the value forward with or without between-prompt-and-response conditioning;
- extracting scalar per-token values from regression or classification heads.
"""

from __future__ import annotations

import json
import math
import pathlib
import random
import re
from collections.abc import Sequence
from typing import Any, overload

import torch
import torch.nn.functional as F

from open_instruct import logger_utils


def causal_value_mask(response_mask: torch.Tensor) -> torch.Tensor:
    """Align action-token membership with causal outputs at positions ``[:-1]``."""
    return response_mask[:, 1:].bool()


def validate_value_bounds(value_min: float, value_max: float) -> None:
    if value_max <= value_min:
        raise ValueError(f"value_max must be greater than value_min, got [{value_min}, {value_max}].")


def validate_terminal_rewards(rewards: torch.Tensor, dones: torch.Tensor, value_min: float, value_max: float) -> None:
    """Fail before critic training when observed outcomes exceed its declared support."""
    validate_value_bounds(value_min, value_max)
    if rewards.shape != dones.shape:
        raise ValueError(f"rewards and dones must have the same shape ({rewards.shape} != {dones.shape}).")
    terminal_rewards = rewards[dones.bool()].float()
    if terminal_rewards.numel() == 0:
        return
    if not bool(torch.isfinite(terminal_rewards).all()):
        raise ValueError("Observed a non-finite terminal reward while constructing value targets.")
    tolerance = 1e-5
    invalid = (terminal_rewards < value_min - tolerance) | (terminal_rewards > value_max + tolerance)
    if bool(invalid.any()):
        observed_min = float(terminal_rewards.min())
        observed_max = float(terminal_rewards.max())
        raise ValueError(
            "Observed terminal rewards outside the configured value range: "
            f"observed [{observed_min}, {observed_max}], configured [{value_min}, {value_max}]. "
            "Set --value_reward_min and --value_reward_max to the true reward support."
        )


def bounded_value_prediction(logits: torch.Tensor, value_min: float, value_max: float) -> torch.Tensor:
    """Map an unconstrained scalar head into the open reward interval with BPCO's scaled arctangent."""
    validate_value_bounds(value_min, value_max)
    unit_value = 0.5 + torch.atan(logits.float()) / math.pi
    return value_min + (value_max - value_min) * unit_value


@overload
def unit_value_to_reward(value: float, value_min: float, value_max: float) -> float: ...


@overload
def unit_value_to_reward(value: torch.Tensor, value_min: float, value_max: float) -> torch.Tensor: ...


def unit_value_to_reward(value: float | torch.Tensor, value_min: float, value_max: float) -> float | torch.Tensor:
    """Map a value on [0, 1] to the configured outcome-reward range."""
    validate_value_bounds(value_min, value_max)
    return value_min + (value_max - value_min) * value


def reward_to_unit_value(value: float, value_min: float, value_max: float) -> float:
    """Map an outcome reward to [0, 1], clipping only after applying the affine transform."""
    validate_value_bounds(value_min, value_max)
    return max(0.0, min(1.0, (value - value_min) / (value_max - value_min)))


def gen_value_sampled_version_metrics(rollouts: list[dict]) -> dict[str, float]:
    """Summarize the critic versions that produced one actor step's values.

    Critic training reports source versions only after the asynchronous queue is
    consumed. These actor-side metrics expose publication lag at the policy step
    where the values are actually used.
    """
    if not rollouts:
        return {}
    versions = [int(rollout["critic_version"]) for rollout in rollouts]
    if any(version < 0 for version in versions):
        raise ValueError(f"Generative-value critic versions must be nonnegative, got {versions}.")
    minimum = min(versions)
    maximum = max(versions)
    return {
        "gen_value/sampled_value_version_min": float(minimum),
        "gen_value/sampled_value_version_max": float(maximum),
        "gen_value/sampled_value_version_spread": float(maximum - minimum),
    }


def missing_value_fallback(value_min: float, value_max: float) -> float:
    """Return the reward-support value closest to zero for a missing prediction."""
    validate_value_bounds(value_min, value_max)
    return max(value_min, min(0.0, value_max))


def predict_values(
    logits: torch.Tensor,
    loss_type: str,
    *,
    bound_predictions: bool = False,
    value_min: float = 0.0,
    value_max: float = 1.0,
) -> torch.Tensor:
    """Convert value-head outputs to scalar predictions in the configured reward range."""
    if loss_type == "mse":
        values = logits.squeeze(-1).float()
        return bounded_value_prediction(values, value_min, value_max) if bound_predictions else values
    if loss_type == "classification":
        if logits.shape[-1] != 2:
            raise ValueError(f"Classification value head must have 2 outputs, got {logits.shape[-1]}.")
        probability = logits.float().softmax(dim=-1)[..., 1]
        return unit_value_to_reward(probability, value_min, value_max)
    raise ValueError(f"Unknown value loss type: {loss_type}")


def classification_value_loss(
    logits: torch.Tensor, returns: torch.Tensor, mask: torch.Tensor, value_min: float = 0.0, value_max: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-bin cross entropy for continuous targets on the configured reward support."""
    validate_value_bounds(value_min, value_max)
    tolerance = 1e-5
    invalid = (returns < value_min - tolerance) | (returns > value_max + tolerance)
    if bool((invalid & mask).any()):
        invalid_target = returns[invalid & mask][0].item()
        raise ValueError(f"Classification value target {invalid_target} is outside [{value_min}, {value_max}].")

    targets = ((returns.float() - value_min) / (value_max - value_min)).clamp(0.0, 1.0)
    target_distribution = torch.stack((1.0 - targets, targets), dim=-1)
    per_token = -(target_distribution * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1)
    clipfrac = torch.zeros((), dtype=torch.float32, device=logits.device)
    return per_token * mask.float(), clipfrac


def compute_value_loss(
    logits: torch.Tensor,
    returns: torch.Tensor,
    old_values: torch.Tensor | None,
    mask: torch.Tensor,
    loss_type: str,
    clip_range: float,
    *,
    bound_predictions: bool = False,
    value_min: float = 0.0,
    value_max: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the configured per-token value loss and clipping diagnostic."""
    if loss_type == "classification":
        return classification_value_loss(logits, returns, mask, value_min, value_max)
    if loss_type == "mse":
        values = predict_values(
            logits, loss_type, bound_predictions=bound_predictions, value_min=value_min, value_max=value_max
        )
        return value_clipped_mse_loss(values, returns, old_values, mask, clip_range)
    raise ValueError(f"Unknown value loss type: {loss_type}")


def accumulation_group_token_counts(masks: Sequence[torch.Tensor], accumulation_steps: int) -> torch.Tensor:
    """Return local valid-token counts for each gradient-accumulation group."""
    if accumulation_steps <= 0:
        raise ValueError(f"accumulation_steps must be positive, got {accumulation_steps}.")
    if not masks:
        return torch.empty(0, dtype=torch.float32)

    num_groups = math.ceil(len(masks) / accumulation_steps)
    counts = torch.zeros(num_groups, dtype=torch.float32, device=masks[0].device)
    for sample_idx, mask in enumerate(masks):
        counts[sample_idx // accumulation_steps] += mask.sum(dtype=torch.float32)
    return counts


def balanced_accumulation_group_ids(num_samples: int, num_groups: int) -> list[int]:
    """Assign contiguous samples to exactly ``num_groups`` balanced groups."""
    if num_samples < 0:
        raise ValueError(f"num_samples must be non-negative, got {num_samples}.")
    if num_groups <= 0:
        raise ValueError(f"num_groups must be positive, got {num_groups}.")
    if num_samples == 0:
        return []
    if num_groups > num_samples:
        raise ValueError(
            f"num_groups cannot exceed num_samples when every optimizer step needs data "
            f"({num_groups} > {num_samples})."
        )

    base_size, extra = divmod(num_samples, num_groups)
    group_ids: list[int] = []
    for group_idx in range(num_groups):
        group_size = base_size + int(group_idx < extra)
        group_ids.extend([group_idx] * group_size)
    return group_ids


def grouped_token_counts(masks: Sequence[torch.Tensor], group_ids: Sequence[int]) -> torch.Tensor:
    """Return local token counts for an explicit contiguous accumulation grouping."""
    if len(masks) != len(group_ids):
        raise ValueError(f"masks and group_ids must have the same length ({len(masks)} != {len(group_ids)}).")
    if not masks:
        return torch.empty(0, dtype=torch.float32)
    if not group_ids or min(group_ids) < 0:
        raise ValueError(f"group_ids must be non-negative, got {list(group_ids)}.")
    if list(group_ids) != sorted(group_ids):
        raise ValueError(f"group_ids must define contiguous ordered groups, got {list(group_ids)}.")
    if sorted(set(group_ids)) != list(range(max(group_ids) + 1)):
        raise ValueError(f"group_ids must not skip group numbers, got {list(group_ids)}.")

    counts = torch.zeros(max(group_ids) + 1, dtype=torch.float32, device=masks[0].device)
    for mask, group_idx in zip(masks, group_ids, strict=True):
        counts[group_idx] += mask.sum(dtype=torch.float32)
    return counts


def normalize_value_loss(
    per_token_loss: torch.Tensor,
    global_token_count: float | torch.Tensor,
    loss_coef: float,
    data_parallel_world_size: int,
) -> torch.Tensor:
    """Scale one local value-loss contribution into a global token mean.

    DeepSpeed averages gradients over data-parallel ranks, so the DP multiplier
    restores the sum of local numerators after division by the global token count.
    Contributions from every pack in the accumulation group must be backpropagated
    before stepping the optimizer.
    """
    if data_parallel_world_size <= 0:
        raise ValueError(f"data_parallel_world_size must be positive, got {data_parallel_world_size}.")
    if isinstance(global_token_count, torch.Tensor):
        denominator = global_token_count.to(device=per_token_loss.device, dtype=torch.float32).clamp(min=1)
    else:
        denominator = max(float(global_token_count), 1.0)
    return per_token_loss.sum() / denominator * loss_coef * data_parallel_world_size


def value_metric_sums(per_token_loss: torch.Tensor, clipfrac: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return loss numerator, clipped-token count, and token count for reduction."""
    token_count = mask.sum(dtype=torch.float64)
    return torch.stack(
        (
            per_token_loss.detach().sum(dtype=torch.float64),
            clipfrac.detach().to(torch.float64) * token_count,
            token_count,
        )
    )


def value_metrics_from_sums(metric_sums: torch.Tensor, loss_coef: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute globally token-weighted value loss and clipping fraction."""
    token_count = metric_sums[2].clamp(min=1.0)
    value_loss = metric_sums[0] / token_count * loss_coef
    clipfrac = metric_sums[1] / token_count
    return value_loss, clipfrac


def regression_metric_sums(returns: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    """Return sufficient statistics for exact distributed regression diagnostics."""
    if returns.shape != predictions.shape:
        raise ValueError(f"returns and predictions must have the same shape ({returns.shape} != {predictions.shape}).")
    returns = returns.detach().to(torch.float64).reshape(-1)
    predictions = predictions.detach().to(torch.float64).reshape(-1)
    residuals = returns - predictions
    return torch.stack(
        (
            torch.tensor(float(returns.numel()), dtype=torch.float64, device=returns.device),
            returns.sum(),
            returns.square().sum(),
            predictions.sum(),
            predictions.square().sum(),
            residuals.sum(),
            residuals.square().sum(),
        )
    )


def regression_metrics_from_sums(metric_sums: torch.Tensor) -> dict[str, float]:
    """Compute means, population standard deviations, and explained variance from global sums."""
    if metric_sums.numel() != 7:
        raise ValueError(f"Expected seven regression sufficient statistics, got {metric_sums.numel()}.")
    count = float(metric_sums[0])
    if count <= 0.0:
        return {}

    returns_mean = float(metric_sums[1]) / count
    predictions_mean = float(metric_sums[3]) / count
    residual_mean = float(metric_sums[5]) / count
    returns_variance = max(float(metric_sums[2]) / count - returns_mean**2, 0.0)
    predictions_variance = max(float(metric_sums[4]) / count - predictions_mean**2, 0.0)
    residual_variance = max(float(metric_sums[6]) / count - residual_mean**2, 0.0)
    return {
        "value/returns_mean": returns_mean,
        "value/returns_std": math.sqrt(returns_variance),
        "value/predictions_mean": predictions_mean,
        "value/predictions_std": math.sqrt(predictions_variance),
        "value/explained_variance": 1.0 - residual_variance / (returns_variance + 1e-8),
    }


def generative_value_reinforce_reward(outcome: float, prediction: float | None) -> tuple[float, float | None]:
    """Return the GenAC critic reward and parsed-prediction squared error.

    Malformed generations receive no REINFORCE signal. Their prediction error is undefined rather
    than being reported as though the critic had intentionally predicted zero.
    """
    if prediction is None:
        return 0.0, None
    squared_error = (outcome - prediction) ** 2
    return 1.0 - squared_error, squared_error


def generative_value_reinforce_weights(
    rewards: Sequence[float], baseline: str, outcomes: Sequence[float] | None = None
) -> list[float]:
    """Convert raw GenAC rewards into policy-gradient weights.

    ``leave_one_out`` subtracts the mean reward of the other critic samples in the
    batch.  That baseline is independent of the current sample's generation, so it
    preserves the expected policy gradient while turning malformed and inaccurate
    generations into an explicit negative signal.  The one-sample case falls back
    to the raw reward because no independent baseline is available.
    """
    raw_rewards = [float(reward) for reward in rewards]
    if baseline == "none" or len(raw_rewards) <= 1:
        return raw_rewards
    if baseline not in {"leave_one_out", "leave_one_out_by_outcome"}:
        raise ValueError(f"Unknown generative-value REINFORCE baseline: {baseline!r}.")

    groups: list[list[int]]
    if baseline == "leave_one_out":
        groups = [list(range(len(raw_rewards)))]
    else:
        if outcomes is None or len(outcomes) != len(raw_rewards):
            raise ValueError("leave_one_out_by_outcome requires one outcome for every reward.")
        groups_by_outcome: dict[bool, list[int]] = {False: [], True: []}
        for index, outcome in enumerate(outcomes):
            groups_by_outcome[float(outcome) > 0.5].append(index)
        groups = list(groups_by_outcome.values())

    weights = list(raw_rewards)
    for indices in groups:
        if len(indices) <= 1:
            continue
        reward_sum = sum(raw_rewards[index] for index in indices)
        denominator = len(indices) - 1
        for index in indices:
            reward = raw_rewards[index]
            weights[index] = reward - (reward_sum - reward) / denominator
    return weights


def replay_gen_value_final_actions(
    training_pairs: Sequence[dict[str, Any]], replay_weight: int
) -> list[dict[str, Any]]:
    """Repeat exact final-action states without changing other critic states."""
    if replay_weight < 1:
        raise ValueError(f"replay_weight must be at least 1, got {replay_weight}.")
    replayed: list[dict[str, Any]] = []
    for pair in training_pairs:
        repeats = replay_weight if pair.get("state_kind") == "final_action" else 1
        replayed.extend([pair] * repeats)
    return replayed


def unique_replayed_gen_value_pairs(training_pairs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove identity-preserving replay copies while retaining original order."""
    seen: set[int] = set()
    unique: list[dict[str, Any]] = []
    for pair in training_pairs:
        identity = id(pair)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(pair)
    return unique


def is_gen_value_near_horizon_incorrect(example: dict[str, Any]) -> bool:
    """Whether a failed critic state is close to the hard response-token limit."""
    target = example.get("target", example.get("outcome"))
    used = example.get("response_tokens_used")
    limit = example.get("response_token_limit")
    if target is None or float(target) > 0.5 or used is None or limit is None:
        return False
    limit = int(limit)
    if limit <= 0:
        return False
    threshold = max(512, math.ceil(0.1 * limit))
    return limit - int(used) <= threshold


def build_gen_value_validation_holdout(
    rollouts: list[dict[str, Any]], max_examples: int, seed: int = 0, prompt_holdout_fraction: float = 0.125
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Hold out fixed actor-prompt groups from the first on-policy batch.

    Initial states are grouped by their exact critic prompt and assigned the empirical
    success rate across sibling policy rollouts. Spread-out trajectory prefixes and final
    actions retain their binary observed outcome. Every state from a selected actor-prompt
    group is removed from the REINFORCE pairs returned to the caller, preventing prefix
    leakage into the repeated calibration diagnostic.
    """
    if not 0.0 < prompt_holdout_fraction <= 1.0:
        raise ValueError(f"prompt_holdout_fraction must be in (0, 1], got {prompt_holdout_fraction}.")
    all_pairs = [pair for rollout in rollouts for pair in rollout.get("pairs", [])]
    if max_examples <= 0 or not all_pairs:
        return [], all_pairs

    rollout_groups: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for rollout in rollouts:
        pairs = rollout.get("pairs", [])
        if not pairs:
            continue
        first = pairs[0]
        prompt_ids = tuple(first["request_output"].prompt_token_ids)
        rollout_groups.setdefault(prompt_ids, []).append(rollout)

    rng = random.Random(seed)
    grouped_rollouts = list(rollout_groups.items())
    holdout_group_count = min(
        len(grouped_rollouts), max_examples, max(1, math.ceil(prompt_holdout_fraction * len(grouped_rollouts)))
    )
    # Near-horizon failures are rare but are exactly the states that reveal whether
    # the critic can recognize a rollout that has exhausted its opportunity to
    # recover. Reserve up to half the held-out prompt-group budget for groups that
    # contain one, then fill the remainder randomly. Without this reservation a
    # seemingly healthy aggregate validation curve can entirely miss the collapse
    # mode that actor training is most likely to exploit.
    near_horizon_groups = [
        group
        for group in grouped_rollouts
        if any(is_gen_value_near_horizon_incorrect(pair) for rollout in group[1] for pair in rollout.get("pairs", []))
    ]
    rng.shuffle(near_horizon_groups)
    reserved_count = min(len(near_horizon_groups), max(1, math.ceil(holdout_group_count / 2)))
    reserved_group_ids = {id(group) for group in near_horizon_groups[:reserved_count]}
    remaining_groups = [group for group in grouped_rollouts if id(group) not in reserved_group_ids]
    rng.shuffle(remaining_groups)
    heldout_groups = near_horizon_groups[:reserved_count] + remaining_groups[: holdout_group_count - reserved_count]
    rng.shuffle(heldout_groups)
    heldout_rollout_ids = {id(rollout) for _, group in heldout_groups for rollout in group}
    validation_examples: list[dict[str, Any]] = []
    final_candidates: list[dict[str, Any]] = []
    prefix_candidates: list[dict[str, Any]] = []
    trajectory_fractions: dict[int, float] = {}
    for prompt_ids, group in heldout_groups:
        first_pairs = [rollout["pairs"][0] for rollout in group]
        outcomes = [float(pair["outcome"]) for pair in first_pairs]
        validation_examples.append(
            {
                "prompt_token_ids": list(prompt_ids),
                "target": sum(outcomes) / len(outcomes),
                "kind": "initial",
                "response_tokens_used": 0,
                "response_token_limit": first_pairs[0].get("response_token_limit"),
            }
        )
        for rollout in group:
            pairs = rollout["pairs"]
            final_action = next((pair for pair in reversed(pairs) if pair.get("state_kind") == "final_action"), None)
            if final_action is not None:
                final_candidates.append(final_action)
            elif len(pairs) > 1:
                # Backward-compatible fallback for rollouts captured before explicit
                # final-action states were added.
                final_action = pairs[-1]
                final_candidates.append(final_action)
            if final_action is None:
                continue
            trajectory_fractions[id(final_action)] = 1.0
            final_tokens_used = max(int(final_action.get("response_tokens_used") or 0), 1)
            eligible_prefixes = [
                pair for pair in pairs[1:] if pair is not final_action and pair.get("response_tokens_used") is not None
            ]
            selected_prefix_ids: set[int] = set()
            for target_fraction in (0.25, 0.5, 0.75):
                if not eligible_prefixes:
                    break
                prefix = min(
                    eligible_prefixes,
                    key=lambda pair: abs(int(pair["response_tokens_used"]) / final_tokens_used - target_fraction),
                )
                if id(prefix) in selected_prefix_ids:
                    continue
                selected_prefix_ids.add(id(prefix))
                prefix_candidates.append(prefix)
                trajectory_fractions[id(prefix)] = min(int(prefix["response_tokens_used"]) / final_tokens_used, 1.0)

    remaining = max_examples - len(validation_examples)

    def balanced_select(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
        correct = [pair for pair in candidates if float(pair["outcome"]) > 0.5]
        incorrect = [pair for pair in candidates if float(pair["outcome"]) <= 0.5]
        rng.shuffle(correct)
        rng.shuffle(incorrect)
        incorrect.sort(key=lambda pair: not is_gen_value_near_horizon_incorrect(pair))
        selected_correct = correct[: min(len(correct), count // 2)]
        selected_incorrect = incorrect[: min(len(incorrect), count - len(selected_correct))]
        selected = selected_correct + selected_incorrect
        leftovers = correct[len(selected_correct) :] + incorrect[len(selected_incorrect) :]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: count - len(selected)])
        return selected

    # Preserve final-action coverage first; use the rest of the fixed budget for
    # intermediate states spread across each held-out trajectory.
    selected = balanced_select(final_candidates, min(len(final_candidates), remaining))
    remaining -= len(selected)
    selected.extend(balanced_select(prefix_candidates, remaining))

    for pair in selected:
        validation_examples.append(
            {
                "prompt_token_ids": list(pair["request_output"].prompt_token_ids),
                "target": float(pair["outcome"]),
                "kind": pair.get("state_kind", "final_segment"),
                "response_tokens_used": pair.get("response_tokens_used"),
                "response_token_limit": pair.get("response_token_limit"),
                "trajectory_fraction": trajectory_fractions.get(id(pair)),
            }
        )

    training_pairs = [
        pair for rollout in rollouts if id(rollout) not in heldout_rollout_ids for pair in rollout.get("pairs", [])
    ]
    return validation_examples, training_pairs


def gen_value_validation_metrics(
    examples: list[dict[str, Any]], predictions: Sequence[float | None]
) -> dict[str, float]:
    """Compute held-out calibration metrics for fixed generative-critic states."""
    if len(examples) != len(predictions):
        raise ValueError(
            f"Validation examples and predictions differ in length ({len(examples)} != {len(predictions)})."
        )
    if not examples:
        return {}

    parsed = [
        (example, float(prediction)) for example, prediction in zip(examples, predictions) if prediction is not None
    ]
    metrics: dict[str, float] = {
        "gen_value/validation_examples": float(len(examples)),
        "gen_value/validation_parse_rate": len(parsed) / len(examples),
        # Parse failure is maximally bad here, rather than being mistaken for a prediction of zero.
        "gen_value/validation_penalized_mse": sum(
            1.0 if prediction is None else (float(example["target"]) - float(prediction)) ** 2
            for example, prediction in zip(examples, predictions)
        )
        / len(examples),
        "gen_value/validation_target_mean": sum(float(example["target"]) for example in examples) / len(examples),
    }
    if parsed:
        metrics["gen_value/validation_mse"] = sum(
            (float(example["target"]) - prediction) ** 2 for example, prediction in parsed
        ) / len(parsed)
        metrics["gen_value/validation_v_hat_mean"] = sum(prediction for _, prediction in parsed) / len(parsed)

    def add_group(prefix: str, rows: list[tuple[dict[str, Any], float]]) -> None:
        if not rows:
            return
        metrics[f"gen_value/validation_{prefix}_examples"] = float(len(rows))
        metrics[f"gen_value/validation_{prefix}_v_hat_mean"] = sum(prediction for _, prediction in rows) / len(rows)
        metrics[f"gen_value/validation_{prefix}_target_mean"] = sum(
            float(example["target"]) for example, _ in rows
        ) / len(rows)
        metrics[f"gen_value/validation_{prefix}_mse"] = sum(
            (float(example["target"]) - prediction) ** 2 for example, prediction in rows
        ) / len(rows)

    initial = [(example, prediction) for example, prediction in parsed if example["kind"] == "initial"]
    final_kinds = {"final_segment", "final_action"}
    final_correct = [
        (example, prediction)
        for example, prediction in parsed
        if example["kind"] in final_kinds and float(example["target"]) > 0.5
    ]
    final_incorrect = [
        (example, prediction)
        for example, prediction in parsed
        if example["kind"] in final_kinds and float(example["target"]) <= 0.5
    ]
    final_action_correct = [
        (example, prediction)
        for example, prediction in parsed
        if example["kind"] == "final_action" and float(example["target"]) > 0.5
    ]
    final_action_incorrect = [
        (example, prediction)
        for example, prediction in parsed
        if example["kind"] == "final_action" and float(example["target"]) <= 0.5
    ]
    prefixes = [(example, prediction) for example, prediction in parsed if example["kind"] == "segment_start"]
    prefix_correct = [(example, prediction) for example, prediction in prefixes if float(example["target"]) > 0.5]
    prefix_incorrect = [(example, prediction) for example, prediction in prefixes if float(example["target"]) <= 0.5]
    # A single prefix aggregate can hide the failure mode that matters most for
    # actor training: values that become less discriminative as a rollout gets
    # closer to its observed outcome.  The fixed holdout deliberately samples
    # prefixes near 25%, 50%, and 75% of each completed trajectory, so expose
    # those regions separately.  These are relative to the observed rollout
    # end, unlike ``near_horizon`` below, which measures proximity to the hard
    # response-token limit.
    prefix_position_bands = {"early": (0.0, 0.375), "middle": (0.375, 0.625), "late": (0.625, 1.0)}
    prefix_position_groups: dict[str, list[tuple[dict[str, Any], float]]] = {}
    for band, (lower, upper) in prefix_position_bands.items():
        band_rows = [
            (example, prediction)
            for example, prediction in prefixes
            if example.get("trajectory_fraction") is not None
            and lower <= float(example["trajectory_fraction"]) <= upper
        ]
        prefix_position_groups[f"prefix_{band}_correct"] = [
            (example, prediction) for example, prediction in band_rows if float(example["target"]) > 0.5
        ]
        prefix_position_groups[f"prefix_{band}_incorrect"] = [
            (example, prediction) for example, prediction in band_rows if float(example["target"]) <= 0.5
        ]
    near_horizon_incorrect = [
        (example, prediction)
        for example, prediction in parsed
        if example["kind"] != "initial" and is_gen_value_near_horizon_incorrect(example)
    ]

    add_group("initial", initial)
    add_group("final_correct", final_correct)
    add_group("final_incorrect", final_incorrect)
    add_group("final_action_correct", final_action_correct)
    add_group("final_action_incorrect", final_action_incorrect)
    add_group("prefix_correct", prefix_correct)
    add_group("prefix_incorrect", prefix_incorrect)
    for name, rows in prefix_position_groups.items():
        add_group(name, rows)
    add_group("near_horizon_incorrect", near_horizon_incorrect)
    if final_correct and final_incorrect:
        metrics["gen_value/validation_final_value_gap"] = (
            metrics["gen_value/validation_final_correct_v_hat_mean"]
            - metrics["gen_value/validation_final_incorrect_v_hat_mean"]
        )
    if prefix_correct and prefix_incorrect:
        metrics["gen_value/validation_prefix_value_gap"] = (
            metrics["gen_value/validation_prefix_correct_v_hat_mean"]
            - metrics["gen_value/validation_prefix_incorrect_v_hat_mean"]
        )
    for band in prefix_position_bands:
        correct = prefix_position_groups[f"prefix_{band}_correct"]
        incorrect = prefix_position_groups[f"prefix_{band}_incorrect"]
        if correct and incorrect:
            metrics[f"gen_value/validation_prefix_{band}_value_gap"] = (
                metrics[f"gen_value/validation_prefix_{band}_correct_v_hat_mean"]
                - metrics[f"gen_value/validation_prefix_{band}_incorrect_v_hat_mean"]
            )
    return metrics


def write_gen_value_validation_snapshot(
    output_dir: str,
    version: int,
    examples: list[dict[str, Any]],
    predictions: Sequence[float | None],
    prompts: Sequence[str],
    generations: Sequence[str],
) -> pathlib.Path:
    """Persist inspectable held-out critic predictions for one published version."""
    lengths = {len(examples), len(predictions), len(prompts), len(generations)}
    if len(lengths) != 1:
        raise ValueError(
            "Validation snapshot fields differ in length: "
            f"examples={len(examples)}, predictions={len(predictions)}, "
            f"prompts={len(prompts)}, generations={len(generations)}."
        )

    snapshot_dir = pathlib.Path(output_dir) / "gen_value_validation"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"version_{int(version):06d}.jsonl"
    temporary_path = snapshot_path.with_suffix(".jsonl.tmp")
    with temporary_path.open("w", encoding="utf-8") as snapshot_file:
        for example, prediction, prompt, generation in zip(examples, predictions, prompts, generations, strict=True):
            row = {key: value for key, value in example.items() if key != "prompt_token_ids"}
            row.update(
                {
                    "version": int(version),
                    "prediction": None if prediction is None else float(prediction),
                    "prompt": prompt,
                    "generation": generation,
                }
            )
            snapshot_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_path.replace(snapshot_path)
    return snapshot_path


def write_gen_value_training_trace_reservoir(
    output_dir: str, version: int, examples: Sequence[dict[str, Any]], seen_by_outcome: dict[str, int]
) -> pathlib.Path:
    """Persist a bounded sample of on-policy critic prompts and generations for inspection/SFT."""
    trace_dir = pathlib.Path(output_dir) / "gen_value_training_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / "reservoir.jsonl"
    temporary_path = trace_path.with_suffix(".jsonl.tmp")
    with temporary_path.open("w", encoding="utf-8") as trace_file:
        for example in examples:
            trace_file.write(json.dumps(example, ensure_ascii=False) + "\n")
    temporary_path.replace(trace_path)

    manifest_path = trace_dir / "manifest.json"
    manifest_tmp = manifest_path.with_suffix(".json.tmp")
    manifest = {
        "critic_version": int(version),
        "retained_examples": len(examples),
        "seen_by_outcome": {key: int(value) for key, value in seen_by_outcome.items()},
    }
    with manifest_tmp.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")
    manifest_tmp.replace(manifest_path)
    return trace_path


def select_gen_value_sft_traces(
    examples: Sequence[dict[str, Any]],
    max_squared_error: float,
    min_critic_version: int = 0,
    max_examples_per_outcome: int | None = None,
    balance_outcomes: bool = True,
    balance_positions: bool = True,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Select parsed, accurate critic traces for prompt-preserving SFT.

    Each retained prompt is used at most once so SFT cannot see contradictory
    completions for the same state. By default, sampling after the accuracy gate
    balances both outcomes and trajectory positions. This prevents final-action
    replay from crowding intermediate states out of the SFT set and explicitly
    retains the early-to-late coverage needed to learn value propagation.
    """
    if max_squared_error < 0:
        raise ValueError(f"max_squared_error must be nonnegative, got {max_squared_error}.")
    if min_critic_version < 0:
        raise ValueError(f"min_critic_version must be nonnegative, got {min_critic_version}.")
    if max_examples_per_outcome is not None and max_examples_per_outcome <= 0:
        raise ValueError(f"max_examples_per_outcome must be positive when set, got {max_examples_per_outcome}.")

    best_by_prompt: dict[str, dict[str, Any]] = {}
    for example in examples:
        prompt = example.get("prompt")
        generation = example.get("generation")
        prediction = example.get("prediction")
        squared_error = example.get("squared_error")
        outcome = example.get("outcome")
        critic_version = example.get("source_critic_version", 0)
        if not isinstance(prompt, str) or not prompt or not isinstance(generation, str) or not generation:
            continue
        if not isinstance(prediction, (float, int)) or not math.isfinite(float(prediction)):
            continue
        if not isinstance(squared_error, (float, int)) or not math.isfinite(float(squared_error)):
            continue
        if not isinstance(outcome, (float, int)) or not math.isfinite(float(outcome)):
            continue
        if not isinstance(critic_version, int) or critic_version < min_critic_version:
            continue
        if float(squared_error) > max_squared_error:
            continue
        previous = best_by_prompt.get(prompt)
        if previous is None or float(squared_error) < float(previous["squared_error"]):
            best_by_prompt[prompt] = dict(example)

    def position_bucket(example: dict[str, Any]) -> str:
        if example.get("state_kind") == "final_action":
            return "final_action"
        trajectory_fraction = example.get("trajectory_fraction")
        if isinstance(trajectory_fraction, int | float) and math.isfinite(float(trajectory_fraction)):
            fraction = float(trajectory_fraction)
            if fraction <= 0.375:
                return "early"
            if fraction <= 0.625:
                return "middle"
            return "late"
        # Reservoirs produced before trajectory fractions were recorded still
        # distinguish exact-final states from the coarser segment-start states.
        return str(example.get("state_kind") or "unknown")

    buckets: dict[str, dict[str, list[dict[str, Any]]]] = {"correct": {}, "incorrect": {}}
    for example in best_by_prompt.values():
        outcome_bucket = "correct" if float(example["outcome"]) > 0.5 else "incorrect"
        position = position_bucket(example) if balance_positions else "all"
        buckets[outcome_bucket].setdefault(position, []).append(example)

    rng = random.Random(seed)
    for outcome_buckets in buckets.values():
        for bucket in outcome_buckets.values():
            rng.shuffle(bucket)

    if balance_outcomes and all(buckets.values()):
        positions = sorted(set(buckets["correct"]) & set(buckets["incorrect"]))
        capacities = {
            position: min(len(buckets["correct"][position]), len(buckets["incorrect"][position]))
            for position in positions
        }
        per_outcome_limit = sum(capacities.values())
        if max_examples_per_outcome is not None:
            per_outcome_limit = min(per_outcome_limit, max_examples_per_outcome)

        # Allocate the same position quota to both outcomes. Round-robin allocation
        # gives every available position coverage before adding a second example.
        shuffled_positions = list(positions)
        rng.shuffle(shuffled_positions)
        quotas = dict.fromkeys(positions, 0)
        while sum(quotas.values()) < per_outcome_limit:
            made_progress = False
            for position in shuffled_positions:
                if quotas[position] >= capacities[position]:
                    continue
                quotas[position] += 1
                made_progress = True
                if sum(quotas.values()) == per_outcome_limit:
                    break
            if not made_progress:
                break
        selected = [
            example
            for outcome in ("correct", "incorrect")
            for position in positions
            for example in buckets[outcome][position][: quotas[position]]
        ]
    else:
        selected = []
        for outcome_buckets in buckets.values():
            positions = sorted(outcome_buckets)
            available = sum(len(outcome_buckets[position]) for position in positions)
            limit = available if max_examples_per_outcome is None else min(available, max_examples_per_outcome)
            outcome_selected: list[dict[str, Any]] = []
            next_index = dict.fromkeys(positions, 0)
            while len(outcome_selected) < limit:
                made_progress = False
                for position in positions:
                    bucket = outcome_buckets[position]
                    if next_index[position] < len(bucket):
                        outcome_selected.append(bucket[next_index[position]])
                        next_index[position] += 1
                        made_progress = True
                        if len(outcome_selected) == limit:
                            break
                if not made_progress:
                    break
            selected.extend(outcome_selected)
    rng.shuffle(selected)
    return selected


def pack_gen_value_examples(examples: list[dict[str, Any]], target_tokens: int) -> list[list[dict[str, Any]]]:
    """Pack critic examples in order up to the policy's token budget.

    An example longer than the target remains intact in its own pack. The critic's
    actual context limit is validated separately, so packing never truncates data.
    """
    if target_tokens <= 0:
        raise ValueError(f"Generative critic pack target must be > 0, got {target_tokens}.")

    packs: list[list[dict[str, Any]]] = []
    current_pack: list[dict[str, Any]] = []
    current_tokens = 0
    for example in examples:
        sequence_tokens = len(example["sequence_ids"])
        if current_pack and current_tokens + sequence_tokens > target_tokens:
            packs.append(current_pack)
            current_pack = []
            current_tokens = 0
        current_pack.append(example)
        current_tokens += sequence_tokens
    if current_pack:
        packs.append(current_pack)
    return packs


def flatten_gen_value_pack(
    examples: list[dict[str, Any]],
) -> tuple[list[int], list[int], list[int], list[int], list[float], list[float]]:
    """Flatten one critic pack and identify exactly the generated-token logits."""
    input_ids: list[int] = []
    position_ids: list[int] = []
    logit_positions: list[int] = []
    target_ids: list[int] = []
    rollout_logprobs: list[float] = []
    token_rewards: list[float] = []

    for example in examples:
        sequence_ids = example["sequence_ids"]
        generated_ids = example["generated_ids"]
        prompt_length = len(sequence_ids) - len(generated_ids)
        if prompt_length <= 0:
            raise ValueError("Gen-value REINFORCE requires at least one prompt token per completion.")

        sequence_offset = len(input_ids)
        input_ids.extend(sequence_ids)
        position_ids.extend(range(len(sequence_ids)))
        # The hidden state immediately before each generated token predicts that token.
        logit_positions.extend(range(sequence_offset + prompt_length - 1, sequence_offset + len(sequence_ids) - 1))
        target_ids.extend(generated_ids)
        rollout_logprobs.extend(example["rollout_logprobs"])
        token_rewards.extend([example["reward"]] * len(generated_ids))

    return input_ids, position_ids, logit_positions, target_ids, rollout_logprobs, token_rewards


logger = logger_utils.setup_logger(__name__)

_ROLLOUT_CONTEXT_MAX_TOKENS = 4096
_RUBRIC_CONDITIONING_MAX_TOKENS = 2048
_DEFAULT_ROLLOUT_CONTEXT_NUM_SIBLINGS = 4
_AUTO_ROLLOUT_CONTEXT_NUM_SIBLINGS = -1
_POSTFIX_TEMPLATES = {"expected_accuracy", "rollout_context", "correct_demo"}
_PREFIX_TEMPLATES = {"answer_prefix", "rubrics"}

# Templates that need a ground-truth string to be meaningful.
TEMPLATES_REQUIRING_GT: frozenset[str] = frozenset({"rollout_context", "correct_demo", "rubrics"})
# Templates that need sibling rollouts (decoded responses from the same prompt group).
TEMPLATES_REQUIRING_SIBLINGS: frozenset[str] = frozenset({"rollout_context", "correct_demo"})
# All valid gt_conditioning_template values.
ALL_GT_CONDITIONING_TEMPLATES: frozenset[str] = frozenset(_POSTFIX_TEMPLATES | _PREFIX_TEMPLATES)
# Valid values for --gen_value_conditioning.
GEN_VALUE_CONDITIONING_TYPES: frozenset[str] = frozenset({"none", "gt", "correct_demo", "rollout_context"})
TEMPLATES_USING_HINTS: frozenset[str] = frozenset({"answer_prefix"})


def segment_rollout(
    response_tokens: list[int],
    response_logprobs: list[float] | None,
    *,
    mode: str,
    sae_threshold: float = 0.2,
    fixed_chunk_size: int = 512,
    max_segments: int | None = None,
) -> list[int]:
    """Return boundary positions (response-token indices) at which the gen-value model is queried.

    In ``sae`` mode boundaries are tokens whose probability is below ``sae_threshold``.
    In ``fixed`` mode boundaries are emitted every ``fixed_chunk_size`` tokens.
    The final token is always a boundary.

    When ``max_segments`` is set and the raw boundary count exceeds it, boundaries are
    downsampled to ``max_segments`` by picking evenly spaced entries from the full list.
    """
    length = len(response_tokens)
    if length == 0:
        return []
    boundaries: list[int] = []
    if mode == "sae":
        if response_logprobs is None:
            raise ValueError("SAE segmentation requires response_logprobs.")
        log_threshold = math.log(max(sae_threshold, 1e-12))
        for t, lp in enumerate(response_logprobs):
            if lp < log_threshold:
                boundaries.append(t)
    else:  # fixed
        # Boundaries are inclusive. A chunk of size N therefore ends at N - 1,
        # not N; the old loop made the first chunk one token too long.
        t = fixed_chunk_size - 1
        while t < length:
            boundaries.append(t)
            t += fixed_chunk_size
    if not boundaries or boundaries[-1] != length - 1:
        boundaries.append(length - 1)
    if max_segments is not None and len(boundaries) > max_segments:
        if max_segments < 1:
            raise ValueError(f"max_segments must be >= 1, got {max_segments}")
        if max_segments == 1:
            boundaries = [length - 1]
        else:
            n = len(boundaries)
            step = (n - 1) / (max_segments - 1)
            kept = [boundaries[round(i * step)] for i in range(max_segments)]
            kept[-1] = length - 1
            boundaries = kept
    return boundaries


def add_observation_segment_boundaries(
    response_mask: Sequence[bool], segment_end_boundaries: Sequence[int]
) -> list[int]:
    """End a critic segment before every masked gap between policy actions.

    A masked gap is typically a tool/environment observation. The first action
    after that gap belongs to a new state and must therefore receive a value
    computed from a prefix that includes the observation.
    """
    response_positions = [idx for idx, is_response in enumerate(response_mask) if is_response]
    if not response_positions:
        if segment_end_boundaries:
            raise ValueError("Cannot define response segments without response tokens.")
        return []

    boundaries = set(segment_end_boundaries)
    for response_idx in range(1, len(response_positions)):
        if response_positions[response_idx] > response_positions[response_idx - 1] + 1:
            boundaries.add(response_idx - 1)
    boundaries.add(len(response_positions) - 1)
    return sorted(boundaries)


def causal_segment_start_prefix_token_ids(
    sequence_token_ids: Sequence[int], response_mask: Sequence[bool], segment_end_boundaries: Sequence[int]
) -> list[list[int]]:
    """Return the full trajectory prefix available before each response segment.

    ``segment_end_boundaries`` are inclusive indices in the compressed sequence of
    policy-produced response tokens. The returned prefixes instead slice the original
    uncompressed trajectory, so masked tool observations between policy actions remain
    visible to the critic. Prefix ``i`` ends immediately before the first policy action
    in segment ``i``; consequently no score can depend on an action it will baseline.
    """
    if len(sequence_token_ids) != len(response_mask):
        raise ValueError(
            "sequence_token_ids and response_mask must have the same length "
            f"({len(sequence_token_ids)} != {len(response_mask)})."
        )

    response_positions = [idx for idx, is_response in enumerate(response_mask) if is_response]
    if not response_positions:
        if segment_end_boundaries:
            raise ValueError("Cannot define response segments without response tokens.")
        return []

    boundaries = list(segment_end_boundaries)
    if not boundaries:
        raise ValueError("At least one segment boundary is required for a non-empty response.")
    if boundaries != sorted(set(boundaries)):
        raise ValueError(f"Segment boundaries must be strictly increasing, got {boundaries}.")
    if boundaries[-1] != len(response_positions) - 1:
        raise ValueError(
            "The final segment boundary must be the final response token "
            f"({boundaries[-1]} != {len(response_positions) - 1})."
        )
    if boundaries[0] < 0:
        raise ValueError(f"Segment boundaries must be non-negative, got {boundaries}.")

    segment_starts = [0, *(boundary + 1 for boundary in boundaries[:-1])]
    first_response_position = response_positions[0]
    prefixes: list[list[int]] = []
    for response_start in segment_starts:
        if response_start >= len(response_positions):
            raise ValueError(f"Segment start {response_start} exceeds response length {len(response_positions)}.")
        original_start_position = response_positions[response_start]
        prefixes.append(list(sequence_token_ids[first_response_position:original_start_position]))
    return prefixes


def causal_final_action_prefix_token_ids(
    sequence_token_ids: Sequence[int], response_mask: Sequence[bool]
) -> tuple[list[int], int]:
    """Return the trajectory state immediately before the final policy action.

    Unlike the start of the final *segment*, this state is at most one sampled
    action from the observed terminal boundary. Masked tool/environment tokens
    are retained, and prompt tokens are omitted because the original problem is
    supplied separately to the generative critic.
    """
    if len(sequence_token_ids) != len(response_mask):
        raise ValueError(
            "sequence_token_ids and response_mask must have the same length "
            f"({len(sequence_token_ids)} != {len(response_mask)})."
        )
    response_positions = [idx for idx, is_response in enumerate(response_mask) if is_response]
    if not response_positions:
        raise ValueError("Cannot build a final-action state without response tokens.")

    first_response_position = response_positions[0]
    final_response_position = response_positions[-1]
    return list(sequence_token_ids[first_response_position:final_response_position]), len(response_positions) - 1


def rescale_gen_value_score(parsed: float, score_min: float, score_max: float) -> float:
    """Rescale a raw gen-value score from [score_min, score_max] to [0, 1]."""
    return max(0.0, min(1.0, (parsed - score_min) / max(score_max - score_min, 1e-8)))


def is_postfix_template(template: str) -> bool:
    """Postfix templates are spliced BETWEEN prompt and response (per sub-sequence).

    Prefix templates are prepended to the entire packed sub-sequence.
    """
    return template in _POSTFIX_TEMPLATES


def resolve_num_siblings_to_sample(template: str, num_siblings_to_sample: int, num_samples_per_prompt: int) -> int:
    """Resolve the auto sibling count used by rollout-context-style templates.

    ``correct_demo`` needs access to every other rollout by default so it does not
    drop a successful sibling before choosing the reference demo.
    """
    if num_siblings_to_sample >= 0:
        return num_siblings_to_sample
    if num_siblings_to_sample != _AUTO_ROLLOUT_CONTEXT_NUM_SIBLINGS:
        raise ValueError(f"num_siblings_to_sample must be >= -1, got {num_siblings_to_sample}.")
    if template == "correct_demo":
        return max(0, num_samples_per_prompt - 1)
    return _DEFAULT_ROLLOUT_CONTEXT_NUM_SIBLINGS


def build_conditioning_text(
    template: str, ground_truth: str, siblings: Sequence[dict] | None = None, hint: str | None = None
) -> str:
    """Return the conditioning text to splice for a single sub-sequence.

    The text is inserted between the prompt and the response (postfix templates) or as a prefix to
    the whole sub-sequence (prefix templates). Callers are expected to tokenize the returned
    string and extend position ids accordingly.
    """
    if template == "answer_prefix":
        if hint is not None:
            return f"Here is a hint: {hint}.\n"
        return f"The correct answer is: {ground_truth}\n"
    if template == "expected_accuracy":
        return f"Given the answer is {ground_truth}, Let me compute the expected accuracy of the partial rollout: "
    if template == "rollout_context":
        return _build_rollout_context(ground_truth, siblings or [])
    if template == "correct_demo":
        return _build_correct_demo_context(ground_truth, siblings or [])
    if template == "rubrics":
        return _build_rubric_context(ground_truth)
    raise ValueError(f"Unknown gt_conditioning_template: {template!r}")


def _build_rollout_context(ground_truth: str, siblings: Sequence[dict]) -> str:
    header = "Here are some other attempts at this question:\n"
    suffix = f"Given the answer is {ground_truth}, compute the expected accuracy of the current attempt: "
    if not siblings:
        return header + suffix
    lines: list[str] = []
    budget = _ROLLOUT_CONTEXT_MAX_TOKENS
    sorted_siblings = sorted(siblings, key=lambda s: len(str(s.get("text", ""))))
    for k, s in enumerate(sorted_siblings):
        tag = "CORRECT" if s.get("is_correct") else "INCORRECT"
        text = str(s.get("text", ""))
        line = f"Attempt {k + 1} ({tag}):\n{text}\n"
        approx_tokens = max(1, len(line) // 4)
        if approx_tokens > budget:
            continue
        budget -= approx_tokens
        lines.append(line)
    return header + "".join(lines) + suffix


def _build_correct_demo_context(ground_truth: str, siblings: Sequence[dict]) -> str:
    """Pick ONE sibling (prefer a correct one); if none, return a blank reference."""
    chosen = None
    for s in siblings:
        if s.get("is_correct"):
            chosen = s
            break
    if chosen is None and siblings:
        chosen = siblings[0]
    tag = "CORRECT" if (chosen and chosen.get("is_correct")) else "INCORRECT"
    text = str(chosen.get("text", "")) if chosen else ""
    reference = f"Here is a reference attempt ({tag}):\n{text}\n" if chosen else ""
    return reference + f"Given the answer is {ground_truth}, compute the expected accuracy of the current attempt: "


def _build_rubric_context(ground_truth: str) -> str:
    """Format the rubrics from a JSON ground truth as a value-model conditioning prefix.

    The ``ground_truth`` is the JSON-encoded payload that ``apply_evolving_rubric_reward``
    keeps up-to-date, of the form::

        {"query": ..., "rubrics": [{"title": ..., "description": ..., "weight": +/-1}, ...],
         "rubrics_types": ["persistent", ..., "evolving", ...]}

    Both the static (persistent) rubrics shipped with the dataset and any active evolving
    rubrics generated during training appear in ``rubrics``; this helper just renders them
    as a positive/negative criteria prefix so the value model is conditioned on the same
    criteria the verifier will use to grade the response.

    Token budget is enforced approximately (4 chars ~= 1 token) so a long rubric set never
    crowds out the rollout in the value forward.
    """
    if not ground_truth:
        return ""
    try:
        gt_obj = json.loads(ground_truth)
    except (TypeError, ValueError, json.JSONDecodeError):
        logger.debug("rubric conditioning: ground_truth was not valid JSON; skipping conditioning")
        return ""
    if not isinstance(gt_obj, dict):
        return ""
    rubrics = gt_obj.get("rubrics") or []
    if not isinstance(rubrics, list) or not rubrics:
        return ""

    positive_lines: list[str] = []
    negative_lines: list[str] = []
    budget = _RUBRIC_CONDITIONING_MAX_TOKENS
    for rubric in rubrics:
        if not isinstance(rubric, dict):
            continue
        title = str(rubric.get("title", "")).strip()
        description = str(rubric.get("description", "")).strip()
        if not description and not title:
            continue
        try:
            weight = float(rubric.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        body = f"{title}: {description}" if title and description else (title or description)
        line = f"- {body}\n"
        approx_tokens = max(1, len(line) // 4)
        if approx_tokens > budget:
            continue
        budget -= approx_tokens
        if weight >= 0:
            positive_lines.append(line)
        else:
            negative_lines.append(line)

    if not positive_lines and not negative_lines:
        return ""

    parts: list[str] = ["The final response will be graded against the following criteria.\n"]
    if positive_lines:
        parts.append("Positive criteria (the response should satisfy these):\n")
        parts.extend(positive_lines)
    if negative_lines:
        parts.append("Negative criteria (the response should NOT satisfy these):\n")
        parts.extend(negative_lines)
    parts.append("\n")
    return "".join(parts)


_SCORE_RE = re.compile(r"<answer>\s*([-+]?[0-9]*\.?[0-9]+)\s*</answer>")


def build_generative_value_prompt(
    partial_response: str,
    conditioning: str,  # one of: "none", "gt", "correct_demo", "rollout_context"
    ground_truth: str = "",
    siblings: Sequence[dict] | None = None,
    score_min: float = 0.0,
    score_max: float = 10.0,
    problem: str = "",
    actor_model_name: str | None = None,
    actor_success_rate: float | None = None,
    response_tokens_used: int | None = None,
    response_token_limit: int | None = None,
) -> str:
    """Build the gen-value prompt.

    Mirrors the template in Figure 3 of GenAC (arXiv:2604.10701): the critic sees the
    original ``problem`` followed by a ``partial_response`` and is asked to reason
    briefly before emitting ``<answer>X</answer>`` with X in [score_min, score_max].
    Generation stops on ``</answer>``; scores are later rescaled to [0, 1].
    """
    conditioning_text = ""
    if conditioning == "gt" and ground_truth:
        conditioning_text = f"The correct answer is {ground_truth}. "
    elif conditioning == "correct_demo" and siblings:
        chosen = next((s for s in siblings if s.get("is_correct")), siblings[0])
        tag = "CORRECT" if chosen.get("is_correct") else "INCORRECT"
        conditioning_text = f"Here is a reference attempt ({tag}):\n{str(chosen.get('text', ''))}\n"
    elif conditioning == "rollout_context" and siblings:
        lines = []
        budget = 4096
        for k, s in enumerate(siblings):
            tag = "CORRECT" if s.get("is_correct") else "INCORRECT"
            line = f"Attempt {k + 1} ({tag}):\n{str(s.get('text', ''))}\n"
            approx = max(1, len(line) // 4)
            if approx > budget:
                continue
            budget -= approx
            lines.append(line)
        conditioning_text = "Here are some other attempts at this question:\n" + "".join(lines)

    state_context: list[str] = []
    if actor_model_name:
        state_context.append(f"The active actor is {actor_model_name}.")
    if actor_success_rate is not None:
        bounded_success_rate = max(0.0, min(1.0, float(actor_success_rate)))
        state_context.append(
            f"Its smoothed recent success rate on this task distribution is {bounded_success_rate:.1%}."
        )
    if response_token_limit is not None:
        if response_token_limit <= 0:
            raise ValueError(f"response_token_limit must be positive, got {response_token_limit}.")
        used = 0 if response_tokens_used is None else int(response_tokens_used)
        if not 0 <= used <= response_token_limit:
            raise ValueError(
                f"response_tokens_used must be in [0, {response_token_limit}], got {response_tokens_used}."
            )
        state_context.append(
            f"The response has used {used} of its {response_token_limit} token budget; "
            f"{response_token_limit - used} tokens remain."
        )

    state_context_block = "\n".join(state_context)
    if state_context_block:
        state_context_block = f"Actor and horizon context:\n{state_context_block}\n\n"

    # Instruction template mirrors Figure 3 of GenAC (arXiv:2604.10701), including
    # policy conditioning and the finite response horizon that defines the value function.
    instruction = (
        "You will be given a problem and a partial response. Your job is to predict the "
        f"expected value of the response on an integer scale from {int(score_min)} (very "
        f"unlikely to succeed) to {int(score_max)} ({int(score_max)} most likely).\n"
        "\n"
        "Instructions:\n"
        "1. Evaluate the difficulty of the problem.\n"
        "2. Skim through the partial solution and detect any progress, error, or confusion.\n"
        "3. Analyze the probability of success if the model finishes the solution.\n"
        f"4. Output your final answer as an integer between {int(score_min)} and "
        f"{int(score_max)} inclusive, wrapped in <answer>...</answer>."
    )
    problem_block = f"Problem:\n{problem}\n\n" if problem else ""
    conditioning_block = f"{conditioning_text}\n" if conditioning_text else ""
    return (
        f"{instruction}\n\n"
        f"{state_context_block}"
        f"{problem_block}"
        f"{conditioning_block}"
        f"Partial response:\n<rollout>{partial_response}</rollout>\nAnswer:"
    )


def parse_generative_value_score(text: str, score_min: float = 0.0, score_max: float = 10.0) -> float | None:
    """Extract the score from a `{score: X}` pattern."""
    m = _SCORE_RE.search(text)
    if m:
        try:
            v = float(m.group(1))
            return max(score_min, min(score_max, v))
        except ValueError:
            return None
    return None


def value_clipped_mse_loss(
    new_values: torch.Tensor,
    returns: torch.Tensor,
    old_values: torch.Tensor | None,
    mask: torch.Tensor,
    clip_range: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PPO2-style clipped value loss. Returns (per_token_loss, clipfrac)."""
    mask_f = mask.float()
    vf_losses1 = (new_values - returns).pow(2)
    if clip_range > 0 and old_values is not None:
        values_clipped = old_values + torch.clamp(new_values - old_values, -clip_range, clip_range)
        vf_losses2 = (values_clipped - returns).pow(2)
        per_token = torch.maximum(vf_losses1, vf_losses2)
        clipfrac = ((vf_losses2 > vf_losses1).float() * mask_f).sum() / mask_f.sum().clamp(min=1)
    else:
        per_token = vf_losses1
        clipfrac = torch.zeros((), dtype=torch.float32, device=new_values.device)
    return per_token * mask_f, clipfrac
