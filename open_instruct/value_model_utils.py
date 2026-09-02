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
import time
from collections.abc import Callable, Sequence
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


def gen_value_policy_guard_active(min_advantage_gap: float | None, observed_advantage_gap: float | None) -> bool:
    """Whether an unreliable critic signal should freeze this actor update.

    Missing gaps occur when a batch has only one outcome class, so they cannot
    support a correct-vs-incorrect comparison and do not activate the guard.
    Non-finite observed gaps are unsafe and do activate it.
    """
    if min_advantage_gap is None or observed_advantage_gap is None:
        return False
    if not math.isfinite(min_advantage_gap) or min_advantage_gap < 0.0:
        raise ValueError(f"min_advantage_gap must be finite and nonnegative when set, got {min_advantage_gap}.")
    return not math.isfinite(observed_advantage_gap) or observed_advantage_gap < min_advantage_gap


def gen_value_training_queue_capacity(world_size: int, max_async_steps: int, capacity_steps: int = 0) -> int:
    """Resolve the bounded queue capacity in learner shards.

    Each learner enqueues one shard per policy step. If the newest source step is
    ``N``, samples from ``N - max_async_steps`` are still admissible, so the queue
    normally needs room for ``max_async_steps + 1`` complete policy batches.
    ``capacity_steps`` can retain a longer bounded history without changing that
    admissibility rule. This is useful during frozen-policy critic pretraining,
    where many fresh batches share the same policy model version. Producers evict
    the oldest shard instead of blocking when the resolved bound is reached.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}.")
    if max_async_steps <= 0:
        raise ValueError(f"max_async_steps must be positive, got {max_async_steps}.")
    if capacity_steps < 0:
        raise ValueError(f"capacity_steps must be nonnegative, got {capacity_steps}.")
    if 0 < capacity_steps < max_async_steps + 1:
        raise ValueError(
            "capacity_steps must be zero or at least the inclusive freshness window "
            f"({max_async_steps + 1}), got {capacity_steps}."
        )
    retained_steps = capacity_steps if capacity_steps > 0 else max_async_steps + 1
    return world_size * retained_steps


def select_fresh_gen_value_rollouts(
    pending_rollouts: list[dict], batch_size: int, max_async_steps: int, newest_policy_model_version: int | None = None
) -> tuple[list[dict], list[dict], list[dict]]:
    """Select one latest-first critic batch and partition stale rollouts.

    Staleness is measured against the actual policy-weight version that generated
    each sample, not the outer policy training-step counter. This distinction is
    important during critic-only warmup: many fresh batches can share one frozen
    policy version. Samples exactly ``max_async_steps`` versions behind remain
    eligible; older samples are returned separately for explicit accounting.
    Within the admissible version window, newer policy batches win ties. When
    fewer than ``batch_size`` eligible samples are available, no partial optimizer
    batch is returned.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if max_async_steps <= 0:
        raise ValueError(f"max_async_steps must be positive, got {max_async_steps}.")
    if newest_policy_model_version is not None and newest_policy_model_version < 0:
        raise ValueError(
            f"newest_policy_model_version must be nonnegative when set, got {newest_policy_model_version}."
        )
    if not pending_rollouts:
        return [], [], []

    try:
        source_versions = [int(rollout["policy_model_version"]) for rollout in pending_rollouts]
    except KeyError as exc:
        raise ValueError("Every generative-critic rollout must include policy_model_version.") from exc
    try:
        source_steps = [int(rollout["policy_training_step"]) for rollout in pending_rollouts]
    except KeyError as exc:
        raise ValueError("Every generative-critic rollout must include policy_training_step.") from exc
    if any(source_version < 0 for source_version in source_versions):
        raise ValueError(f"policy_model_version must be nonnegative, got {source_versions}.")
    if any(source_step < 0 for source_step in source_steps):
        raise ValueError(f"policy_training_step must be nonnegative, got {source_steps}.")

    newest_source_version = max(source_versions)
    if newest_policy_model_version is not None:
        newest_source_version = max(newest_source_version, newest_policy_model_version)
    oldest_admissible_version = newest_source_version - max_async_steps
    stale_indices = {
        index for index, source_version in enumerate(source_versions) if source_version < oldest_admissible_version
    }
    eligible_indices = [index for index in range(len(pending_rollouts)) if index not in stale_indices]
    if len(eligible_indices) < batch_size:
        return (
            [],
            [pending_rollouts[index] for index in eligible_indices],
            [pending_rollouts[index] for index in range(len(pending_rollouts)) if index in stale_indices],
        )

    selected_indices = set(
        sorted(eligible_indices, key=lambda index: (-source_versions[index], -source_steps[index], index))[:batch_size]
    )
    return (
        [pending_rollouts[index] for index in range(len(pending_rollouts)) if index in selected_indices],
        [
            pending_rollouts[index]
            for index in range(len(pending_rollouts))
            if index not in stale_indices and index not in selected_indices
        ],
        [pending_rollouts[index] for index in range(len(pending_rollouts)) if index in stale_indices],
    )


def gen_value_source_step_is_admissible(
    policy_training_step: int, latest_trained_policy_step: int, max_async_steps: int
) -> bool:
    """Whether critic data is within the allowed source-policy training window."""
    if policy_training_step < 0:
        raise ValueError(f"policy_training_step must be nonnegative, got {policy_training_step}.")
    if latest_trained_policy_step < 0:
        raise ValueError(f"latest_trained_policy_step must be nonnegative, got {latest_trained_policy_step}.")
    if max_async_steps <= 0:
        raise ValueError(f"max_async_steps must be positive, got {max_async_steps}.")
    return policy_training_step - latest_trained_policy_step <= max_async_steps


def wait_for_gen_value_source_window(
    get_latest_trained_policy_step: Callable[[], int],
    policy_training_step: int,
    max_async_steps: int,
    timeout_s: float,
    poll_interval_s: float = 1.0,
) -> tuple[int, float]:
    """Wait until policy execution is within the critic's trained-source window.

    Queue freshness alone does not prevent policy execution from outrunning critic
    optimization: a latest-first queue can keep discarding old batches while the
    critic remains several policy steps behind. This barrier bounds that lag using
    the newest source-policy step that completed an optimizer update. It does not
    require every rollout from older steps to be retained or trained.
    """
    if timeout_s <= 0.0:
        raise ValueError(f"timeout_s must be positive, got {timeout_s}.")
    if poll_interval_s < 0.0:
        raise ValueError(f"poll_interval_s must be nonnegative, got {poll_interval_s}.")

    started_at = time.perf_counter()
    while True:
        latest_trained_policy_step = int(get_latest_trained_policy_step())
        if gen_value_source_step_is_admissible(policy_training_step, latest_trained_policy_step, max_async_steps):
            return latest_trained_policy_step, time.perf_counter() - started_at
        elapsed_s = time.perf_counter() - started_at
        if elapsed_s >= timeout_s:
            raise TimeoutError(
                "Timed out waiting for generative critic freshness: "
                f"policy_training_step={policy_training_step}, "
                f"latest_trained_policy_step={latest_trained_policy_step}, "
                f"max_async_steps={max_async_steps}, timeout_s={timeout_s}."
            )
        time.sleep(min(poll_interval_s, max(timeout_s - elapsed_s, 0.0)))


class GenValueTrainingProgressState:
    """Track when every critic rollout from a source-policy step is resolved."""

    def __init__(self, latest_trained_policy_step: int, policy_world_size: int) -> None:
        if latest_trained_policy_step < 0:
            raise ValueError(f"latest_trained_policy_step must be nonnegative, got {latest_trained_policy_step}.")
        if policy_world_size <= 0:
            raise ValueError(f"policy_world_size must be positive, got {policy_world_size}.")
        self._latest_processed_policy_step = int(latest_trained_policy_step)
        self._latest_trained_policy_step = int(latest_trained_policy_step)
        self._policy_world_size = int(policy_world_size)
        self._admitted_rollouts: dict[int, int] = {}
        self._trained_rollouts: dict[int, int] = {}
        self._discarded_rollouts: dict[int, int] = {}
        self._registered_ranks: dict[int, set[int]] = {}
        self._total_admitted_rollouts = 0
        self._total_trained_rollouts = 0
        self._total_discarded_rollouts = 0

    def get_latest_trained_policy_step(self) -> int:
        """Return the newest source-policy step used by a completed critic update."""
        return self._latest_trained_policy_step

    def get_latest_processed_policy_step(self) -> int:
        return self._latest_processed_policy_step

    def get_rollout_accounting(self) -> dict[str, int]:
        return {
            "latest_processed_policy_step": self._latest_processed_policy_step,
            "latest_trained_policy_step": self._latest_trained_policy_step,
            "admitted_rollouts": self._total_admitted_rollouts,
            "trained_rollouts": self._total_trained_rollouts,
            "discarded_rollouts": self._total_discarded_rollouts,
        }

    def register_admitted_policy_step(self, policy_training_step: int, policy_rank: int, num_rollouts: int) -> None:
        policy_training_step = int(policy_training_step)
        policy_rank = int(policy_rank)
        num_rollouts = int(num_rollouts)
        if policy_training_step <= self._latest_processed_policy_step:
            raise ValueError(
                "Cannot admit a generative-critic source step that is already complete: "
                f"latest={self._latest_processed_policy_step}, admitted={policy_training_step}."
            )
        if not 0 <= policy_rank < self._policy_world_size:
            raise ValueError(f"policy_rank must be in [0, {self._policy_world_size}), got {policy_rank}.")
        if num_rollouts < 0:
            raise ValueError(f"num_rollouts must be nonnegative, got {num_rollouts}.")
        registered_ranks = self._registered_ranks.setdefault(policy_training_step, set())
        if policy_rank in registered_ranks:
            raise ValueError(
                f"Policy rank {policy_rank} registered source step {policy_training_step} more than once."
            )
        registered_ranks.add(policy_rank)
        self._admitted_rollouts[policy_training_step] = (
            self._admitted_rollouts.get(policy_training_step, 0) + num_rollouts
        )
        self._total_admitted_rollouts += num_rollouts
        self._advance_completed_steps()

    def record_trained_policy_steps(self, policy_training_steps: list[int]) -> None:
        for policy_training_step in policy_training_steps:
            policy_training_step = int(policy_training_step)
            if policy_training_step <= self._latest_processed_policy_step:
                raise ValueError(
                    "Cannot record critic training for a source step that is already complete: "
                    f"latest={self._latest_processed_policy_step}, trained={policy_training_step}."
                )
            self._trained_rollouts[policy_training_step] = self._trained_rollouts.get(policy_training_step, 0) + 1
            self._total_trained_rollouts += 1
            self._latest_trained_policy_step = max(self._latest_trained_policy_step, policy_training_step)
        self._advance_completed_steps()

    def record_discarded_policy_steps(self, policy_training_steps: list[int]) -> None:
        """Resolve rollouts removed by latest-first queue eviction or stale filtering."""
        for policy_training_step in policy_training_steps:
            policy_training_step = int(policy_training_step)
            if policy_training_step <= self._latest_processed_policy_step:
                raise ValueError(
                    "Cannot record critic discard for a source step that is already complete: "
                    f"latest={self._latest_processed_policy_step}, discarded={policy_training_step}."
                )
            self._discarded_rollouts[policy_training_step] = self._discarded_rollouts.get(policy_training_step, 0) + 1
            self._total_discarded_rollouts += 1
        self._advance_completed_steps()

    def _advance_completed_steps(self) -> None:
        while True:
            next_step = self._latest_processed_policy_step + 1
            if len(self._registered_ranks.get(next_step, set())) < self._policy_world_size:
                return
            admitted = self._admitted_rollouts.get(next_step, 0)
            processed = self._trained_rollouts.get(next_step, 0) + self._discarded_rollouts.get(next_step, 0)
            if processed < admitted:
                return
            if processed > admitted:
                raise RuntimeError(
                    "Generative-critic rollout accounting exceeded admission: "
                    f"step={next_step}, admitted={admitted}, processed={processed}."
                )
            self._latest_processed_policy_step = next_step
            self._admitted_rollouts.pop(next_step, None)
            self._trained_rollouts.pop(next_step, None)
            self._discarded_rollouts.pop(next_step, None)
            self._registered_ranks.pop(next_step, None)


def update_gen_value_success_rate_ema(
    previous_rate: float | None, batch_success_rate: float, momentum: float
) -> float:
    """Update policy-conditioning success rate without a synthetic zero prior.

    The first critic batch is already an unbiased observation of the active
    policy. Initializing an EMA accumulator at zero makes early prompts claim a
    success rate smaller by ``1 - momentum`` and creates a train/serve mismatch
    with offline MC traces. Initialize directly from the first observation, then
    apply the configured EMA on later batches.
    """
    if not math.isfinite(batch_success_rate) or not 0.0 <= batch_success_rate <= 1.0:
        raise ValueError(f"batch_success_rate must be finite and in [0, 1], got {batch_success_rate}.")
    if not 0.0 <= momentum < 1.0:
        raise ValueError(f"momentum must be in [0, 1), got {momentum}.")
    if previous_rate is None:
        return float(batch_success_rate)
    if not math.isfinite(previous_rate) or not 0.0 <= previous_rate <= 1.0:
        raise ValueError(f"previous_rate must be finite and in [0, 1], got {previous_rate}.")
    return momentum * previous_rate + (1.0 - momentum) * batch_success_rate


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


def value_outcome_position_samples(
    predictions: torch.Tensor,
    returns: torch.Tensor,
    value_mask: torch.Tensor,
    correct_labels: torch.Tensor,
    percentile_bins: torch.Tensor,
    num_bins: int,
) -> dict[str, list[list[float]]]:
    """Split value predictions and return targets by outcome and trajectory position.

    The caller reduces the returned samples across data-parallel ranks. Keeping
    predictions and their training targets under the same labels distinguishes a
    changing on-policy target distribution from value-head miscalibration.
    """
    if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins <= 0:
        raise ValueError(f"num_bins must be a positive integer, got {num_bins!r}.")
    tensors = {
        "returns": returns,
        "value_mask": value_mask,
        "correct_labels": correct_labels,
        "percentile_bins": percentile_bins,
    }
    for name, tensor in tensors.items():
        if tensor.shape != predictions.shape:
            raise ValueError(
                f"{name} must have the same shape as predictions ({tensor.shape} != {predictions.shape})."
            )
    mask = value_mask.bool()
    bins = percentile_bins.long()
    invalid_bins = mask & ((bins < 0) | (bins >= num_bins))
    if bool(invalid_bins.any()):
        invalid = sorted(set(bins[invalid_bins].detach().cpu().tolist()))
        raise ValueError(f"Masked percentile bins must be in [0, {num_bins}), got {invalid}.")

    predictions = predictions.detach()
    returns = returns.detach()
    correct = correct_labels.bool()
    result = {
        "prediction_correct": [[] for _ in range(num_bins)],
        "prediction_incorrect": [[] for _ in range(num_bins)],
        "return_correct": [[] for _ in range(num_bins)],
        "return_incorrect": [[] for _ in range(num_bins)],
    }
    for bin_index in range(num_bins):
        in_bin = mask & (bins == bin_index)
        correct_in_bin = in_bin & correct
        incorrect_in_bin = in_bin & ~correct
        result["prediction_correct"][bin_index] = predictions[correct_in_bin].float().cpu().tolist()
        result["prediction_incorrect"][bin_index] = predictions[incorrect_in_bin].float().cpu().tolist()
        result["return_correct"][bin_index] = returns[correct_in_bin].float().cpu().tolist()
        result["return_incorrect"][bin_index] = returns[incorrect_in_bin].float().cpu().tolist()
    return result


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


def generative_value_reinforce_weights_with_replay(
    rewards: Sequence[float], baseline: str, sample_ids: Sequence[int], outcomes: Sequence[float] | None = None
) -> list[float]:
    """Compute baseline weights once per sampled completion, then replay them.

    Final-action replay intentionally repeats the same sampled critic completion.
    Including those copies in a leave-one-out baseline would make the baseline
    depend on the current action and would also let replay frequency distort the
    reference reward. Collapse copies by identity, compute the baseline on unique
    samples, and broadcast each resulting weight back to its replay copies.
    """
    if len(rewards) != len(sample_ids):
        raise ValueError("REINFORCE rewards and sample_ids must have the same length.")
    if outcomes is not None and len(outcomes) != len(rewards):
        raise ValueError("REINFORCE rewards and outcomes must have the same length.")

    unique_ids: list[int] = []
    unique_rewards: list[float] = []
    unique_outcomes: list[float] | None = [] if outcomes is not None else None
    identity_to_unique_index: dict[int, int] = {}
    for index, sample_id in enumerate(sample_ids):
        reward = float(rewards[index])
        outcome = float(outcomes[index]) if outcomes is not None else None
        unique_index = identity_to_unique_index.get(int(sample_id))
        if unique_index is None:
            identity_to_unique_index[int(sample_id)] = len(unique_ids)
            unique_ids.append(int(sample_id))
            unique_rewards.append(reward)
            if unique_outcomes is not None:
                assert outcome is not None
                unique_outcomes.append(outcome)
            continue
        if reward != unique_rewards[unique_index] or (
            unique_outcomes is not None and outcome != unique_outcomes[unique_index]
        ):
            raise ValueError("Replay copies with one sample_id must have identical rewards and outcomes.")

    unique_weights = generative_value_reinforce_weights(unique_rewards, baseline, unique_outcomes)
    return [unique_weights[identity_to_unique_index[int(sample_id)]] for sample_id in sample_ids]


def generative_value_reinforce_outcome_mass_metrics(
    weights: Sequence[float], outcomes: Sequence[float], generated_token_counts: Sequence[int]
) -> dict[str, float]:
    """Measure optimizer signal mass separately for successful and unsuccessful traces.

    The by-outcome leave-one-out baseline removes each class's reward offset, but it
    does not equalize the amount of gradient contributed by the two classes.  The
    actual loss repeats one scalar weight over every generated critic token, so
    example counts alone cannot reveal whether the majority outcome dominates an
    update.  These metrics include replay copies and generation lengths to match the
    optimizer inputs without changing their weights.
    """
    if len(weights) != len(outcomes) or len(weights) != len(generated_token_counts):
        raise ValueError(
            "REINFORCE weights, outcomes, and generated-token counts must have the same length "
            f"({len(weights)}, {len(outcomes)}, {len(generated_token_counts)})."
        )

    buckets = {
        "correct": {
            "examples": 0,
            "tokens": 0,
            "signed_weight": 0.0,
            "signed_token_weight": 0.0,
            "abs_weight": 0.0,
            "abs_token_weight": 0.0,
        },
        "incorrect": {
            "examples": 0,
            "tokens": 0,
            "signed_weight": 0.0,
            "signed_token_weight": 0.0,
            "abs_weight": 0.0,
            "abs_token_weight": 0.0,
        },
    }
    for weight, outcome, token_count in zip(weights, outcomes, generated_token_counts, strict=True):
        if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0:
            raise ValueError(f"Generated-token counts must be nonnegative integers, got {token_count!r}.")
        bucket = buckets["correct" if float(outcome) > 0.5 else "incorrect"]
        signed_weight = float(weight)
        absolute_weight = abs(signed_weight)
        bucket["examples"] += 1
        bucket["tokens"] += token_count
        bucket["signed_weight"] += signed_weight
        bucket["signed_token_weight"] += signed_weight * token_count
        bucket["abs_weight"] += absolute_weight
        bucket["abs_token_weight"] += absolute_weight * token_count

    metrics: dict[str, float] = {}
    for name, bucket in buckets.items():
        metrics[f"gen_value/reinforce_{name}_examples"] = float(bucket["examples"])
        metrics[f"gen_value/reinforce_{name}_tokens"] = float(bucket["tokens"])
        metrics[f"gen_value/reinforce_{name}_signed_weight_sum"] = float(bucket["signed_weight"])
        metrics[f"gen_value/reinforce_{name}_signed_token_weight_mass"] = float(bucket["signed_token_weight"])
        metrics[f"gen_value/reinforce_{name}_abs_weight_sum"] = float(bucket["abs_weight"])
        metrics[f"gen_value/reinforce_{name}_abs_token_weight_mass"] = float(bucket["abs_token_weight"])

    total_mass = sum(float(bucket["abs_token_weight"]) for bucket in buckets.values())
    if total_mass > 0.0:
        metrics["gen_value/reinforce_correct_abs_token_weight_mass_frac"] = (
            float(buckets["correct"]["abs_token_weight"]) / total_mass
        )
    return metrics


def generative_value_reinforce_state_kind_mass_metrics(
    weights: Sequence[float], state_kinds: Sequence[str], generated_token_counts: Sequence[int]
) -> dict[str, float]:
    """Measure how much critic gradient mass is spent on prefixes versus final actions.

    Final-action replay copies are intentionally present in these inputs. Reporting
    their logical token-weight mass makes the replay allocation visible even when
    exact copies are collapsed into one physical forward/backward example.
    """
    if len(weights) != len(state_kinds) or len(weights) != len(generated_token_counts):
        raise ValueError(
            "REINFORCE weights, state kinds, and generated-token counts must have the same length "
            f"({len(weights)}, {len(state_kinds)}, {len(generated_token_counts)})."
        )

    buckets = {
        "prefix": {
            "examples": 0,
            "tokens": 0,
            "signed_weight": 0.0,
            "signed_token_weight": 0.0,
            "abs_weight": 0.0,
            "abs_token_weight": 0.0,
        },
        "final_action": {
            "examples": 0,
            "tokens": 0,
            "signed_weight": 0.0,
            "signed_token_weight": 0.0,
            "abs_weight": 0.0,
            "abs_token_weight": 0.0,
        },
    }
    for weight, state_kind, token_count in zip(weights, state_kinds, generated_token_counts, strict=True):
        if not isinstance(state_kind, str) or not state_kind:
            raise ValueError(f"Generative-value state kinds must be non-empty strings, got {state_kind!r}.")
        if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0:
            raise ValueError(f"Generated-token counts must be nonnegative integers, got {token_count!r}.")
        bucket = buckets["final_action" if state_kind == "final_action" else "prefix"]
        signed_weight = float(weight)
        absolute_weight = abs(signed_weight)
        bucket["examples"] += 1
        bucket["tokens"] += token_count
        bucket["signed_weight"] += signed_weight
        bucket["signed_token_weight"] += signed_weight * token_count
        bucket["abs_weight"] += absolute_weight
        bucket["abs_token_weight"] += absolute_weight * token_count

    metrics: dict[str, float] = {}
    for name, bucket in buckets.items():
        metrics[f"gen_value/reinforce_{name}_examples"] = float(bucket["examples"])
        metrics[f"gen_value/reinforce_{name}_tokens"] = float(bucket["tokens"])
        metrics[f"gen_value/reinforce_{name}_signed_weight_sum"] = float(bucket["signed_weight"])
        metrics[f"gen_value/reinforce_{name}_signed_token_weight_mass"] = float(bucket["signed_token_weight"])
        metrics[f"gen_value/reinforce_{name}_abs_weight_sum"] = float(bucket["abs_weight"])
        metrics[f"gen_value/reinforce_{name}_abs_token_weight_mass"] = float(bucket["abs_token_weight"])

    total_mass = sum(float(bucket["abs_token_weight"]) for bucket in buckets.values())
    if total_mass > 0.0:
        metrics["gen_value/reinforce_final_action_abs_token_weight_mass_frac"] = (
            float(buckets["final_action"]["abs_token_weight"]) / total_mass
        )
    return metrics


def generative_value_prediction_outcome_metrics(
    predictions: Sequence[float | None], outcomes: Sequence[float]
) -> dict[str, float]:
    """Report calibration separately for parsed correct and incorrect targets.

    A global critic MSE can improve merely because the more common failed states
    become easier.  Splitting the actual optimizer examples by target outcome
    makes a shared downward prediction drift visible without changing the loss.
    Parse failures remain excluded from calibration statistics and are already
    represented by the optimizer parse-rate metric.
    """
    if len(predictions) != len(outcomes):
        raise ValueError(
            "Generative-value predictions and outcomes must have the same length "
            f"({len(predictions)} != {len(outcomes)})."
        )

    buckets: dict[str, list[tuple[float, float]]] = {"correct": [], "incorrect": []}
    for prediction, outcome in zip(predictions, outcomes, strict=True):
        target = float(outcome)
        if not math.isfinite(target) or not 0.0 <= target <= 1.0:
            raise ValueError(f"Generative-value outcome must be finite and in [0, 1], got {outcome}.")
        if prediction is None:
            continue
        prediction = float(prediction)
        if not math.isfinite(prediction) or not 0.0 <= prediction <= 1.0:
            raise ValueError(f"Generative-value prediction must be finite and in [0, 1], got {prediction}.")
        buckets["correct" if target > 0.5 else "incorrect"].append((target, prediction))

    metrics: dict[str, float] = {}
    for name, rows in buckets.items():
        if not rows:
            continue
        metrics[f"gen_value/optimization_{name}_parsed_examples"] = float(len(rows))
        metrics[f"gen_value/optimization_{name}_target_mean"] = sum(target for target, _ in rows) / len(rows)
        metrics[f"gen_value/optimization_{name}_v_hat_mean"] = sum(prediction for _, prediction in rows) / len(rows)
        metrics[f"gen_value/optimization_{name}_mse"] = sum(
            (prediction - target) ** 2 for target, prediction in rows
        ) / len(rows)
    return metrics


_GEN_VALUE_SAMPLE_ID_KEY = "_gen_value_sample_id"
_GEN_VALUE_OPTIMIZER_SELECTED_KEY = "_gen_value_optimizer_selected"
_GEN_VALUE_OPTIMIZER_INCLUSION_PROBABILITY_KEY = "_gen_value_optimizer_inclusion_probability"
_GEN_VALUE_REPLAY_MULTIPLICITY_KEY = "_gen_value_replay_multiplicity"


def _gen_value_optimizer_stratum(pair_group: Sequence[dict[str, Any]]) -> tuple[str, str]:
    """Return a stable state-length/outcome stratum for one shared-state group."""
    if not pair_group:
        raise ValueError("Cannot stratify an empty generative-value pair group.")

    state_kinds = {str(pair.get("state_kind", "segment_start")) for pair in pair_group}
    if len(state_kinds) != 1:
        raise ValueError(f"Token-identical generative-value states disagree on state kind: {state_kinds}.")
    state_kind = next(iter(state_kinds))
    if state_kind == "final_action":
        state_length_stratum = "final_action"
    else:
        response_positions = {pair.get("response_tokens_used") for pair in pair_group}
        if len(response_positions) != 1:
            raise ValueError(
                f"Token-identical generative-value states disagree on response_tokens_used: {response_positions}."
            )
        response_tokens_used = next(iter(response_positions))
        if isinstance(response_tokens_used, bool) or not isinstance(response_tokens_used, int):
            raise ValueError(
                "Length-outcome-stratified critic sampling requires integer response_tokens_used, got "
                f"{response_tokens_used!r}."
            )
        if response_tokens_used < 0:
            raise ValueError(f"response_tokens_used must be nonnegative, got {response_tokens_used}.")
        if response_tokens_used < 1024:
            state_length_stratum = "prefix_lt_1024"
        elif response_tokens_used < 2048:
            state_length_stratum = "prefix_1024_2047"
        elif response_tokens_used < 4096:
            state_length_stratum = "prefix_2048_4095"
        else:
            state_length_stratum = "prefix_ge_4096"

    outcomes = [float(pair["outcome"]) for pair in pair_group]
    if any(not math.isfinite(outcome) or not 0.0 <= outcome <= 1.0 for outcome in outcomes):
        raise ValueError(f"Generative-value outcomes must be finite and in [0, 1], got {outcomes}.")
    outcome_stratum = "correct" if sum(outcomes) / len(outcomes) > 0.5 else "incorrect"
    return state_length_stratum, outcome_stratum


def _allocate_gen_value_stratum_targets(
    stratum_sizes: dict[tuple[str, str], int], target: int
) -> dict[tuple[str, str], float]:
    """Water-fill an example target equally across nonempty strata without oversampling."""
    remaining_target = float(target)
    active = set(stratum_sizes)
    allocations: dict[tuple[str, str], float] = {}
    while active:
        equal_share = remaining_target / len(active)
        saturated = {stratum for stratum in active if stratum_sizes[stratum] <= equal_share}
        if not saturated:
            allocations.update({stratum: equal_share for stratum in active})
            break
        for stratum in saturated:
            allocation = float(stratum_sizes[stratum])
            allocations[stratum] = allocation
            remaining_target -= allocation
        active -= saturated
    return allocations


def mark_gen_value_training_pairs_for_optimizer(
    training_pairs: Sequence[dict[str, Any]],
    target_examples: int,
    rng: random.Random,
    *,
    sampling_strategy: str = "uniform",
) -> tuple[list[dict[str, Any]], float, int]:
    """Mark an unbiased critic minibatch while retaining shared-state groups.

    The asynchronous queue selects fresh *rollouts*. A complete policy-sized
    rollout batch can nevertheless contain thousands of segment-state critic
    examples, making one optimizer update substantially slower than a policy step.

    With uniform sampling, each token-identical prompt group is independently
    retained with probability ``target_examples / len(training_pairs)``. The
    optional ``length_outcome_stratified`` strategy water-fills the target equally
    across state-length/outcome strata, then samples whole prompt groups using the
    stratum-specific probabilities. This reduces the chance that rare long,
    successful states disappear from a critic update without changing the
    full-batch objective in expectation.

    If the Bernoulli draw is empty, one group is chosen uniformly. Every returned
    pair carries its exact inclusion probability, including that fallback, so the
    trainer can apply a per-example Horvitz-Thompson gradient scale.
    All pairs are returned: unselected pairs still define the original shared-state
    return targets and leave-one-out baseline, while only marked pairs participate
    in the expensive forward/backward pass. This makes the stochastic gradient an
    unbiased estimate of the existing full-batch token objective rather than a
    differently normalized ratio estimator.

    Returning every pair with inclusion probability one when the target covers the
    batch avoids perturbing historical runs. The caller owns ``rng`` so successive
    updates consume a reproducible random stream rather than repeatedly selecting
    the same subset.
    """
    if target_examples <= 0:
        raise ValueError(f"Generative-value training target examples must be positive, got {target_examples}.")
    if sampling_strategy not in {"uniform", "length_outcome_stratified"}:
        raise ValueError(
            "Generative-value optimizer sampling strategy must be 'uniform' or "
            f"'length_outcome_stratified', got {sampling_strategy!r}."
        )
    if len(training_pairs) <= target_examples:
        return (
            [
                dict(
                    pair,
                    **{_GEN_VALUE_OPTIMIZER_SELECTED_KEY: True, _GEN_VALUE_OPTIMIZER_INCLUSION_PROBABILITY_KEY: 1.0},
                )
                for pair in training_pairs
            ],
            1.0,
            len(training_pairs),
        )

    grouped_pairs: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for pair in training_pairs:
        prompt_ids = tuple(int(token_id) for token_id in pair["request_output"].prompt_token_ids)
        grouped_pairs.setdefault(prompt_ids, []).append(pair)

    if sampling_strategy == "uniform":
        base_probabilities = {prompt_ids: target_examples / len(training_pairs) for prompt_ids in grouped_pairs}
    else:
        grouped_strata = {
            prompt_ids: _gen_value_optimizer_stratum(pair_group) for prompt_ids, pair_group in grouped_pairs.items()
        }
        stratum_sizes: dict[tuple[str, str], int] = {}
        for prompt_ids, pair_group in grouped_pairs.items():
            stratum = grouped_strata[prompt_ids]
            stratum_sizes[stratum] = stratum_sizes.get(stratum, 0) + len(pair_group)
        allocations = _allocate_gen_value_stratum_targets(stratum_sizes, target_examples)
        base_probabilities = {
            prompt_ids: min(allocations[grouped_strata[prompt_ids]] / stratum_sizes[grouped_strata[prompt_ids]], 1.0)
            for prompt_ids in grouped_pairs
        }

    retained_groups = {
        prompt_ids for prompt_ids, probability in base_probabilities.items() if rng.random() < probability
    }
    empty_draw_probability = math.prod(1.0 - probability for probability in base_probabilities.values())
    fallback_probability = empty_draw_probability / len(grouped_pairs)
    inclusion_probabilities = {
        prompt_ids: probability + fallback_probability for prompt_ids, probability in base_probabilities.items()
    }
    if not retained_groups:
        retained_groups.add(rng.choice(list(grouped_pairs)))

    marked_pairs = []
    selected_examples = 0
    inclusion_probability_sum = 0.0
    for pair in training_pairs:
        prompt_ids = tuple(int(token_id) for token_id in pair["request_output"].prompt_token_ids)
        optimizer_selected = prompt_ids in retained_groups
        inclusion_probability = inclusion_probabilities[prompt_ids]
        marked_pairs.append(
            dict(
                pair,
                **{
                    _GEN_VALUE_OPTIMIZER_SELECTED_KEY: optimizer_selected,
                    _GEN_VALUE_OPTIMIZER_INCLUSION_PROBABILITY_KEY: inclusion_probability,
                },
            )
        )
        selected_examples += int(optimizer_selected)
        inclusion_probability_sum += inclusion_probability
    mean_inclusion_probability = inclusion_probability_sum / len(training_pairs)
    return marked_pairs, mean_inclusion_probability, selected_examples


def gen_value_optimizer_selected(pair: dict[str, Any]) -> bool:
    """Whether a critic pair was selected for the optimizer forward/backward pass."""
    selected = pair.get(_GEN_VALUE_OPTIMIZER_SELECTED_KEY, True)
    if not isinstance(selected, bool):
        raise ValueError(f"Generative-value optimizer selection flag must be boolean, got {selected!r}.")
    return selected


def gen_value_optimizer_inclusion_probability(pair: dict[str, Any], default: float = 1.0) -> float:
    """Return the exact probability that a critic pair entered the optimizer batch."""
    probability = pair.get(_GEN_VALUE_OPTIMIZER_INCLUSION_PROBABILITY_KEY, default)
    if isinstance(probability, bool) or not isinstance(probability, int | float):
        raise ValueError(f"Generative-value optimizer inclusion probability must be numeric, got {probability!r}.")
    probability = float(probability)
    if not math.isfinite(probability) or not 0.0 < probability <= 1.0:
        raise ValueError(
            f"Generative-value optimizer inclusion probability must be finite and in (0, 1], got {probability}."
        )
    return probability


def gen_value_pair_sample_id(pair: dict[str, Any]) -> int:
    """Return the stable per-batch identity of one critic sample.

    Ray does not preserve Python object aliasing when argument values cross the
    actor boundary, so ``id(pair)`` cannot identify replay copies in the trainer.
    Untagged pairs retain an object-identity fallback for direct utility callers.
    """
    if _GEN_VALUE_SAMPLE_ID_KEY not in pair:
        return id(pair)
    sample_id = pair[_GEN_VALUE_SAMPLE_ID_KEY]
    if isinstance(sample_id, bool) or not isinstance(sample_id, int) or sample_id < 0:
        raise ValueError(f"Generative-value sample ID must be a nonnegative integer, got {sample_id!r}.")
    return sample_id


def replay_gen_value_final_actions(
    training_pairs: Sequence[dict[str, Any]], replay_weight: int
) -> list[dict[str, Any]]:
    """Tag each critic sample, then repeat exact final-action states.

    The explicit tag survives Ray serialization even when repeated dictionary
    aliases do not, allowing the remote trainer to remove replay copies from
    calibration diagnostics and leave-one-out baseline estimation.
    """
    if replay_weight < 1:
        raise ValueError(f"replay_weight must be at least 1, got {replay_weight}.")
    replayed: list[dict[str, Any]] = []
    for sample_id, pair in enumerate(training_pairs):
        tagged_pair = dict(pair)
        tagged_pair[_GEN_VALUE_SAMPLE_ID_KEY] = sample_id
        repeats = replay_weight if pair.get("state_kind") == "final_action" else 1
        replayed.extend([tagged_pair] * repeats)
    return replayed


def unique_replayed_gen_value_pairs(training_pairs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove tagged replay copies while retaining original order."""
    seen: set[int] = set()
    unique: list[dict[str, Any]] = []
    for pair in training_pairs:
        sample_id = gen_value_pair_sample_id(pair)
        if sample_id in seen:
            continue
        seen.add(sample_id)
        unique.append(pair)
    return unique


def collapse_replayed_gen_value_optimizer_examples(examples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse exact replay copies while preserving their summed policy gradient.

    Final-action replay intentionally repeats one sampled critic completion. Sending
    those identical sequences through the model separately wastes prompt forward
    compute. Once the leave-one-out baseline has been computed on unique samples,
    copies have the same tokens, log probabilities, and scalar weight. Summing the
    copy weights onto one physical example therefore preserves the exact token-loss
    numerator. The caller must retain the original logical token denominator.
    """
    collapsed: list[dict[str, Any]] = []
    by_sample_id: dict[int, dict[str, Any]] = {}
    identity_fields = (
        "sequence_ids",
        "generated_ids",
        "rollout_logprobs",
        "outcome",
        "state_kind",
        "optimizer_selected",
        "optimizer_inclusion_probability",
        "parsed",
        "prediction",
        "squared_error",
    )
    for example in examples:
        sample_id = example.get("source_pair_id")
        if isinstance(sample_id, bool) or not isinstance(sample_id, int) or sample_id < 0:
            raise ValueError(
                f"Replay-collapsed optimizer examples require a nonnegative sample ID, got {sample_id!r}."
            )
        existing = by_sample_id.get(sample_id)
        if existing is None:
            collapsed_example = dict(example)
            collapsed_example[_GEN_VALUE_REPLAY_MULTIPLICITY_KEY] = 1
            collapsed.append(collapsed_example)
            by_sample_id[sample_id] = collapsed_example
            continue

        for field in identity_fields:
            if example.get(field) != existing.get(field):
                raise ValueError(
                    f"Replay copies with sample ID {sample_id} must have identical {field}, "
                    f"got {existing.get(field)!r} and {example.get(field)!r}."
                )
        multiplicity = gen_value_replay_multiplicity(existing)
        base_reward = float(existing["reward"]) / multiplicity
        if float(example["reward"]) != base_reward:
            raise ValueError(
                f"Replay copies with sample ID {sample_id} must have identical rewards, "
                f"got {base_reward} and {example['reward']}."
            )
        existing["reward"] = float(existing["reward"]) + float(example["reward"])
        existing[_GEN_VALUE_REPLAY_MULTIPLICITY_KEY] = multiplicity + 1
    return collapsed


def gen_value_replay_multiplicity(example: dict[str, Any]) -> int:
    """Return how many logical replay copies one physical optimizer example represents."""
    multiplicity = example.get(_GEN_VALUE_REPLAY_MULTIPLICITY_KEY, 1)
    if isinstance(multiplicity, bool) or not isinstance(multiplicity, int) or multiplicity < 1:
        raise ValueError(f"Generative-value replay multiplicity must be a positive integer, got {multiplicity!r}.")
    return multiplicity


def pool_gen_value_shared_state_returns(
    training_pairs: Sequence[dict[str, Any]],
) -> tuple[dict[int, float], dict[str, float]]:
    """Pool exact shared critic states to their empirical Monte Carlo return.

    Multiple policy continuations can begin from the same token-identical critic
    prompt. Training each copy against its individual Bernoulli outcome is
    unbiased, but needlessly high variance: the value of that shared state is the
    mean return across its sampled continuations. Explicitly tagged final-action
    replay copies are collapsed before pooling, then receive the target assigned
    to their original sample ID. No example is removed from the optimizer batch.
    """
    unique_pairs = unique_replayed_gen_value_pairs(training_pairs)
    grouped_pairs: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    sampled_outcomes: dict[int, float] = {}
    for pair in unique_pairs:
        outcome = pair.get("outcome")
        if outcome is None:
            raise ValueError("Shared-state return pooling requires a sampled outcome for every pair.")
        outcome = float(outcome)
        if not math.isfinite(outcome) or not 0.0 <= outcome <= 1.0:
            raise ValueError(f"Generative-value outcome must be finite and in [0, 1], got {outcome}.")
        sample_id = gen_value_pair_sample_id(pair)
        sampled_outcomes[sample_id] = outcome
        prompt_ids = tuple(int(token_id) for token_id in pair["request_output"].prompt_token_ids)
        grouped_pairs.setdefault(prompt_ids, []).append(pair)

    targets_by_sample_id: dict[int, float] = {}
    pooled_groups = 0
    pooled_examples = 0
    changed_examples = 0
    for pairs in grouped_pairs.values():
        target = sum(sampled_outcomes[gen_value_pair_sample_id(pair)] for pair in pairs) / len(pairs)
        if len(pairs) > 1:
            pooled_groups += 1
            pooled_examples += len(pairs)
        for pair in pairs:
            sample_id = gen_value_pair_sample_id(pair)
            targets_by_sample_id[sample_id] = target
            if target != sampled_outcomes[sample_id]:
                changed_examples += 1

    metrics = {
        "gen_value/shared_state_unique_examples": float(len(unique_pairs)),
        "gen_value/shared_state_groups": float(len(grouped_pairs)),
        "gen_value/shared_state_pooled_groups": float(pooled_groups),
        "gen_value/shared_state_pooled_examples": float(pooled_examples),
        "gen_value/shared_state_changed_examples": float(changed_examples),
    }
    return targets_by_sample_id, metrics


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
    actions retain their single sampled return. The latter are useful unbiased Bernoulli
    diagnostics in aggregate, but they are not low-variance state-value targets: precise
    per-state calibration requires multiple fresh continuations from each fixed prefix.
    Every state from a selected actor-prompt group is removed from the REINFORCE pairs
    returned to the caller, preventing prefix leakage into the repeated diagnostic.
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
    def group_outcomes(group: tuple[tuple[int, ...], list[dict[str, Any]]]) -> set[bool]:
        return {float(rollout["pairs"][0]["outcome"]) > 0.5 for rollout in group[1] if rollout.get("pairs")}

    near_horizon_groups = [
        group
        for group in grouped_rollouts
        if any(is_gen_value_near_horizon_incorrect(pair) for rollout in group[1] for pair in rollout.get("pairs", []))
    ]
    rng.shuffle(near_horizon_groups)
    mixed_outcome_groups = [group for group in grouped_rollouts if len(group_outcomes(group)) == 2]
    rng.shuffle(mixed_outcome_groups)

    # A single mixed prompt group is strictly more informative than a one-class
    # group: it retains prompt-level isolation while making both calibration and
    # ranking diagnostics available. With a larger budget, first retain the rare
    # near-horizon failures, then explicitly cover any missing outcome class.
    heldout_groups: list[tuple[tuple[int, ...], list[dict[str, Any]]]] = []
    if holdout_group_count == 1 and mixed_outcome_groups:
        heldout_groups.append(mixed_outcome_groups[0])
    else:
        reserved_count = min(len(near_horizon_groups), max(1, math.ceil(holdout_group_count / 2)))
        heldout_groups.extend(near_horizon_groups[:reserved_count])

    selected_group_ids = {id(group) for group in heldout_groups}
    observed_outcomes = set().union(*(group_outcomes(group) for group in heldout_groups))
    for missing_outcome in (True, False):
        if missing_outcome in observed_outcomes:
            continue
        if len(heldout_groups) >= holdout_group_count:
            break
        candidates = [
            group
            for group in grouped_rollouts
            if id(group) not in selected_group_ids and missing_outcome in group_outcomes(group)
        ]
        rng.shuffle(candidates)
        # Prefer a mixed group so the selected state panel contains both outcomes
        # even when the remaining prompt-group budget is only one.
        candidates.sort(key=lambda group: len(group_outcomes(group)), reverse=True)
        if candidates:
            heldout_groups.append(candidates[0])
            selected_group_ids.add(id(candidates[0]))

    remaining_groups = [group for group in grouped_rollouts if id(group) not in selected_group_ids]
    rng.shuffle(remaining_groups)
    heldout_groups.extend(remaining_groups[: holdout_group_count - len(heldout_groups)])
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
                "target_source": "sibling_empirical_return",
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
                "target_source": "single_sample_return",
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


def gen_value_validation_has_both_sampled_outcomes(examples: Sequence[dict[str, Any]]) -> bool:
    """Whether a fixed critic panel can measure correct-vs-incorrect ranking.

    Initial-state targets can be fractional sibling averages, so they do not
    establish that the panel contains both realized outcome classes.  Require
    both classes among the single-sample trajectory states before freezing an
    online validation panel.
    """
    sampled_outcomes = {
        float(example["target"]) > 0.5
        for example in examples
        if example.get("target_source") == "single_sample_return"
    }
    return sampled_outcomes == {False, True}


def gen_value_checkpoint_has_optimizer_state(checkpoint_path: str, checkpoint_tag: str) -> bool:
    """Whether a DeepSpeed critic checkpoint contains a separate optimizer shard."""
    checkpoint_dir = pathlib.Path(checkpoint_path) / checkpoint_tag
    return any(checkpoint_dir.glob("*optim_states.pt"))


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

    def add_binary_ranking_auc(
        prefix: str, correct: list[tuple[dict[str, Any], float]], incorrect: list[tuple[dict[str, Any], float]]
    ) -> None:
        """Add exact pairwise AUC, assigning half credit to prediction ties."""
        if not correct or not incorrect:
            return
        pairwise_credit = sum(
            float(correct_prediction > incorrect_prediction) + 0.5 * float(correct_prediction == incorrect_prediction)
            for _, correct_prediction in correct
            for _, incorrect_prediction in incorrect
        )
        metrics[f"gen_value/validation_{prefix}_auc"] = pairwise_credit / (len(correct) * len(incorrect))

    def add_binary_macro_mse(
        prefix: str, correct: list[tuple[dict[str, Any], float]], incorrect: list[tuple[dict[str, Any], float]]
    ) -> None:
        """Add equal-class-weight MSE without changing the natural value target.

        Math batches contain substantially more failed than successful trajectories. A
        prevalence-weighted validation MSE can therefore improve while calibration on
        successful prefixes regresses. Macro averaging is only a diagnostic: it gives
        each observed outcome class equal reporting weight while leaving both the heldout
        examples and the critic optimizer untouched.
        """
        if not correct or not incorrect:
            return
        correct_mse = sum((float(example["target"]) - prediction) ** 2 for example, prediction in correct) / len(
            correct
        )
        incorrect_mse = sum((float(example["target"]) - prediction) ** 2 for example, prediction in incorrect) / len(
            incorrect
        )
        metrics[f"gen_value/validation_{prefix}_macro_mse"] = (correct_mse + incorrect_mse) / 2.0

    initial = [(example, prediction) for example, prediction in parsed if example["kind"] == "initial"]
    empirical_return = [
        (example, prediction)
        for example, prediction in parsed
        if example.get("target_source") == "sibling_empirical_return"
    ]
    single_sample_return = [
        (example, prediction)
        for example, prediction in parsed
        if example.get("target_source") == "single_sample_return"
    ]
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
    # Use half-open bins except for the final interval so an exact boundary
    # cannot be reported in two position bands.
    prefix_position_bands = {"early": (0.0, 0.375, False), "middle": (0.375, 0.625, False), "late": (0.625, 1.0, True)}
    prefix_position_groups: dict[str, list[tuple[dict[str, Any], float]]] = {}
    for band, (lower, upper, include_upper) in prefix_position_bands.items():
        band_rows = [
            (example, prediction)
            for example, prediction in prefixes
            if example.get("trajectory_fraction") is not None
            and lower <= float(example["trajectory_fraction"])
            and (
                float(example["trajectory_fraction"]) <= upper
                if include_upper
                else float(example["trajectory_fraction"]) < upper
            )
        ]
        prefix_position_groups[f"prefix_{band}_correct"] = [
            (example, prediction) for example, prediction in band_rows if float(example["target"]) > 0.5
        ]
        prefix_position_groups[f"prefix_{band}_incorrect"] = [
            (example, prediction) for example, prediction in band_rows if float(example["target"]) <= 0.5
        ]
    # Trajectory-relative position and absolute context length answer different
    # questions.  A short failed rollout can already be "late" at 600 tokens,
    # while the live math actor may routinely reason for several thousand.  Use
    # non-overlapping absolute bands so validation exposes that distribution
    # shift without double-counting examples across the reported groups.
    prefix_token_bands = {
        "lt_1024": (0, 1024),
        "1024_2048": (1024, 2048),
        "2048_4096": (2048, 4096),
        "ge_4096": (4096, None),
    }
    prefix_token_groups: dict[str, list[tuple[dict[str, Any], float]]] = {}
    for band, (lower, upper) in prefix_token_bands.items():
        band_rows = [
            (example, prediction)
            for example, prediction in prefixes
            if example.get("response_tokens_used") is not None
            and lower <= int(example["response_tokens_used"])
            and (upper is None or int(example["response_tokens_used"]) < upper)
        ]
        prefix_token_groups[f"prefix_tokens_{band}_correct"] = [
            (example, prediction) for example, prediction in band_rows if float(example["target"]) > 0.5
        ]
        prefix_token_groups[f"prefix_tokens_{band}_incorrect"] = [
            (example, prediction) for example, prediction in band_rows if float(example["target"]) <= 0.5
        ]
    near_horizon_incorrect = [
        (example, prediction)
        for example, prediction in parsed
        if example["kind"] != "initial" and is_gen_value_near_horizon_incorrect(example)
    ]

    add_group("initial", initial)
    add_group("empirical_return", empirical_return)
    add_group("single_sample_return", single_sample_return)
    add_group("final_correct", final_correct)
    add_group("final_incorrect", final_incorrect)
    add_group("final_action_correct", final_action_correct)
    add_group("final_action_incorrect", final_action_incorrect)
    add_group("prefix_correct", prefix_correct)
    add_group("prefix_incorrect", prefix_incorrect)
    for name, rows in prefix_position_groups.items():
        add_group(name, rows)
    for name, rows in prefix_token_groups.items():
        add_group(name, rows)
    add_group("near_horizon_incorrect", near_horizon_incorrect)
    add_binary_macro_mse("final", final_correct, final_incorrect)
    add_binary_macro_mse("final_action", final_action_correct, final_action_incorrect)
    add_binary_macro_mse("prefix", prefix_correct, prefix_incorrect)
    add_binary_ranking_auc("final", final_correct, final_incorrect)
    add_binary_ranking_auc("final_action", final_action_correct, final_action_incorrect)
    add_binary_ranking_auc("prefix", prefix_correct, prefix_incorrect)
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
        add_binary_macro_mse(f"prefix_{band}", correct, incorrect)
        add_binary_ranking_auc(f"prefix_{band}", correct, incorrect)
        if correct and incorrect:
            metrics[f"gen_value/validation_prefix_{band}_value_gap"] = (
                metrics[f"gen_value/validation_prefix_{band}_correct_v_hat_mean"]
                - metrics[f"gen_value/validation_prefix_{band}_incorrect_v_hat_mean"]
            )
    for band in prefix_token_bands:
        correct = prefix_token_groups[f"prefix_tokens_{band}_correct"]
        incorrect = prefix_token_groups[f"prefix_tokens_{band}_incorrect"]
        add_binary_macro_mse(f"prefix_tokens_{band}", correct, incorrect)
        add_binary_ranking_auc(f"prefix_tokens_{band}", correct, incorrect)
        if correct and incorrect:
            metrics[f"gen_value/validation_prefix_tokens_{band}_value_gap"] = (
                metrics[f"gen_value/validation_prefix_tokens_{band}_correct_v_hat_mean"]
                - metrics[f"gen_value/validation_prefix_tokens_{band}_incorrect_v_hat_mean"]
            )
    for outcome in ("correct", "incorrect"):
        early = metrics.get(f"gen_value/validation_prefix_early_{outcome}_v_hat_mean")
        late = metrics.get(f"gen_value/validation_prefix_late_{outcome}_v_hat_mean")
        if early is not None and late is not None:
            metrics[f"gen_value/validation_prefix_{outcome}_early_to_late_delta"] = late - early
    early_gap = metrics.get("gen_value/validation_prefix_early_value_gap")
    late_gap = metrics.get("gen_value/validation_prefix_late_value_gap")
    if early_gap is not None and late_gap is not None:
        metrics["gen_value/validation_prefix_value_gap_early_to_late_delta"] = late_gap - early_gap
    return metrics


def gen_value_validation_prediction_change_metrics(
    previous_predictions: Sequence[float | None], current_predictions: Sequence[float | None]
) -> dict[str, float]:
    """Measure serving-side prediction changes on the same fixed critic panel.

    A completed weight-transfer RPC proves only that vLLM accepted the update
    request.  Comparing successive predictions on the frozen validation panel
    provides an end-to-end behavioral check that the serving critic changed.
    Parse transitions are reported separately so they are not silently treated
    as numeric score changes.
    """
    if len(previous_predictions) != len(current_predictions):
        raise ValueError(
            "Previous and current validation predictions differ in length "
            f"({len(previous_predictions)} != {len(current_predictions)})."
        )
    if not current_predictions:
        return {}

    paired_predictions = [
        (float(previous), float(current))
        for previous, current in zip(previous_predictions, current_predictions, strict=True)
        if previous is not None and current is not None
    ]
    parse_status_changes = sum(
        (previous is None) != (current is None)
        for previous, current in zip(previous_predictions, current_predictions, strict=True)
    )
    metrics = {
        "gen_value/validation_prediction_change_examples": float(len(current_predictions)),
        "gen_value/validation_prediction_change_paired_examples": float(len(paired_predictions)),
        "gen_value/validation_prediction_parse_status_changed_fraction": parse_status_changes
        / len(current_predictions),
    }
    if paired_predictions:
        changes = [current - previous for previous, current in paired_predictions]
        absolute_changes = [abs(change) for change in changes]
        metrics.update(
            {
                "gen_value/validation_prediction_mean_change": sum(changes) / len(changes),
                "gen_value/validation_prediction_mean_abs_change": sum(absolute_changes) / len(absolute_changes),
                "gen_value/validation_prediction_max_abs_change": max(absolute_changes),
                "gen_value/validation_prediction_changed_fraction": sum(change != 0.0 for change in changes)
                / len(changes),
            }
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


def write_gen_value_validation_panel(checkpoint_state_dir: str, examples: Sequence[dict[str, Any]]) -> pathlib.Path:
    """Persist the exact fixed critic panel alongside resumable trainer state."""
    panel_dir = pathlib.Path(checkpoint_state_dir) / "gen_value_validation"
    panel_dir.mkdir(parents=True, exist_ok=True)
    panel_path = panel_dir / "panel.jsonl"
    temporary_path = panel_path.with_suffix(".jsonl.tmp")
    with temporary_path.open("w", encoding="utf-8") as panel_file:
        for index, example in enumerate(examples):
            prompt_token_ids = example.get("prompt_token_ids")
            if (
                not isinstance(prompt_token_ids, list)
                or not prompt_token_ids
                or not all(
                    isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in prompt_token_ids
                )
            ):
                raise ValueError(f"Validation panel example {index} is missing integer prompt_token_ids.")
            if not isinstance(example.get("target"), int | float):
                raise ValueError(f"Validation panel example {index} is missing a numeric target.")
            panel_file.write(json.dumps(example, ensure_ascii=False) + "\n")
    temporary_path.replace(panel_path)
    return panel_path


def read_gen_value_validation_panel(path: pathlib.Path) -> list[dict[str, Any]]:
    """Read the exact fixed critic panel saved in a resumable checkpoint."""
    examples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as panel_file:
        for line_number, line in enumerate(panel_file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}, got {type(row).__name__}.")
            prompt_token_ids = row.get("prompt_token_ids")
            if (
                not isinstance(prompt_token_ids, list)
                or not prompt_token_ids
                or not all(
                    isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in prompt_token_ids
                )
            ):
                raise ValueError(f"Missing integer prompt_token_ids at {path}:{line_number}.")
            if not isinstance(row.get("target"), int | float):
                raise ValueError(f"Missing numeric target at {path}:{line_number}.")
            examples.append(row)
    if not examples:
        raise ValueError(f"Validation panel is empty: {path}.")
    return examples


def read_gen_value_validation_snapshot(path: pathlib.Path) -> list[dict[str, Any]]:
    """Read and validate a persisted fixed generative-critic holdout."""
    examples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as snapshot_file:
        for line_number, line in enumerate(snapshot_file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}, got {type(row).__name__}.")
            if not isinstance(row.get("prompt"), str) or not row["prompt"]:
                raise ValueError(f"Missing non-empty prompt at {path}:{line_number}.")
            if not isinstance(row.get("target"), int | float):
                raise ValueError(f"Missing numeric target at {path}:{line_number}.")
            examples.append(row)
    if not examples:
        raise ValueError(f"Validation snapshot is empty: {path}.")
    return examples


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
    """Select parsed, training-target-accurate critic traces for prompt-preserving SFT.

    Each retained prompt is used at most once so SFT cannot see contradictory
    completions for the same state. By default, sampling after the accuracy gate
    balances both outcomes and trajectory positions. This prevents final-action
    replay from crowding intermediate states out of the SFT set and explicitly
    retains the early-to-late coverage needed to learn value propagation.

    New reservoirs record error against both the sampled return and the pooled
    shared-state training target. Prefer the latter so a calibrated prediction
    for an ambiguous shared prefix is not rejected merely because one sibling
    rollout happened to succeed or fail. Older reservoirs fall back to their
    sampled-return ``squared_error`` field.
    """
    if max_squared_error < 0:
        raise ValueError(f"max_squared_error must be nonnegative, got {max_squared_error}.")
    if min_critic_version < 0:
        raise ValueError(f"min_critic_version must be nonnegative, got {min_critic_version}.")
    if max_examples_per_outcome is not None and max_examples_per_outcome <= 0:
        raise ValueError(f"max_examples_per_outcome must be positive when set, got {max_examples_per_outcome}.")

    best_by_prompt: dict[str, dict[str, Any]] = {}
    best_error_by_prompt: dict[str, float] = {}
    for example in examples:
        prompt = example.get("prompt")
        generation = example.get("generation")
        prediction = example.get("prediction")
        squared_error = example.get("training_target_squared_error")
        if not isinstance(squared_error, (float, int)) or not math.isfinite(float(squared_error)):
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
        previous_error = best_error_by_prompt.get(prompt)
        if previous_error is None or float(squared_error) < previous_error:
            best_by_prompt[prompt] = dict(example)
            best_error_by_prompt[prompt] = float(squared_error)

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
    """Pack critic examples in order up to the configured training token target.

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


def expected_gen_value_score_from_logprobs(
    class_scores: Sequence[float], class_sequence_logprobs: Sequence[float]
) -> tuple[float, list[float]]:
    """Return a normalized-class expected score and its probabilities.

    The generative critic emits a discrete score such as ``<answer>7</answer>``.
    Greedy parsing therefore only changes in 0.1 increments after rescaling and
    can hide useful sub-threshold weight movement. Exact sequence log
    probabilities for every valid score provide a diagnostic-only continuous
    prediction without changing generation or the training objective.
    """
    if len(class_scores) != len(class_sequence_logprobs):
        raise ValueError(
            "class_scores and class_sequence_logprobs must have the same length "
            f"({len(class_scores)} != {len(class_sequence_logprobs)})."
        )
    if not class_scores:
        raise ValueError("At least one generative-value class is required.")
    scores = [float(score) for score in class_scores]
    logprobs = [float(logprob) for logprob in class_sequence_logprobs]
    if not all(math.isfinite(score) for score in scores):
        raise ValueError(f"Generative-value class scores must be finite, got {scores}.")
    if not all(math.isfinite(logprob) for logprob in logprobs):
        raise ValueError(f"Generative-value class log probabilities must be finite, got {logprobs}.")

    max_logprob = max(logprobs)
    unnormalized = [math.exp(logprob - max_logprob) for logprob in logprobs]
    normalizer = sum(unnormalized)
    probabilities = [weight / normalizer for weight in unnormalized]
    expected_score = sum(score * probability for score, probability in zip(scores, probabilities, strict=True))
    return expected_score, probabilities


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


def decode_generative_value_problem(
    tokenizer: Any, prompt_token_ids: Sequence[int] | None, fallback_problem: str = ""
) -> str:
    """Recover the exact actor-prompt text used by online generative-value scoring.

    Value-estimation parquet rows retain both the unformatted problem (for stable
    problem identity) and the tokenized actor prompt (for exact critic inputs).
    Online GenAC scores the latter after decoding it with special tokens removed,
    so offline critic SFT and calibration must do the same rather than silently
    substituting the plain problem string.
    """
    if prompt_token_ids is None:
        return fallback_problem
    token_ids = [int(token_id) for token_id in prompt_token_ids]
    if not token_ids:
        return fallback_problem
    return tokenizer.decode(token_ids, skip_special_tokens=True)


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
    """Extract the score from an ``<answer>X</answer>`` element."""
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
