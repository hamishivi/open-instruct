# Copyright 2026 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Sibling training script for the GENERATIVE value model.

This script is a thin wrapper around ``open_instruct.grpo_fast`` that adds a *second* vLLM pool
hosting a generative value model, runs it at SAE (or fixed-chunk) segment boundaries, and trains
the generative value model in-place via REINFORCE using the rollout outcome as reward.

The generative value model has its own weights and vLLM pool.  During each policy training step
``grpo_fast.PolicyTrainerRayProcess.step()`` scores the actual rollout tokens at fixed-chunk
boundaries via the gen-value vLLM pool and uses the returned piecewise-constant scores as the
value function for GAE (replacing the scalar value head when ``use_generative_value_model=True``).

A background critic loop reads complete rollouts from a bounded queue and forms fixed-size
batches. Policy actors pipe through the exact prompt and sampled token IDs, generated score
text, rollout outcome, and source critic version without waiting for critic training. A
``GenValueTrainerActor`` holds a DeepSpeed-wrapped copy of the gen-value model and computes
REINFORCE gradients using the paper's
MSE-shaped critic reward ``R_v = 1 - (outcome - v_hat)**2``, following §5.2 of "Bringing Value
Models Back" (arXiv:2604.10701). Parse failures receive reward zero; the numeric value supplied
to GAE remains zero because GAE cannot consume a missing prediction.

Weight sync from the trainer actor back to the gen-value vLLM pool happens in-place over NCCL,
mirroring how the policy syncs to its own vLLM pool in
``grpo_fast.PolicyTrainerRayProcess.broadcast_to_vllm``.  A NCCL group is established once at
startup via ``GenValueTrainerActor.setup_model_update_group``. Critic training stays asynchronous;
the latest eligible version is published between policy steps, after distributed critic scoring
has completed. Set ``gen_value_sync_freq=0`` to keep the serving critic frozen while REINFORCE
gradients are still computed.

Usage::

    python open_instruct/grpo_fast_genvalue.py \\
        --model_name_or_path ... \\
        --use_generative_value_model \\
        --gen_value_model_name_or_path ... \\
        --gen_value_vllm_num_engines 1 \\
        --gen_value_segmentation sae \\
        --sae_threshold 0.2 \\
        --gen_value_score_min 0 --gen_value_score_max 10 \\
        --gen_value_conditioning gt \\
        --gen_value_reinforce_coef 0.1 \\
        ...
"""

from __future__ import annotations

import math
import os
import pathlib
import queue as queue_lib
import random
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent import futures
from dataclasses import dataclass
from queue import Queue
from typing import Any

import deepspeed
import ray
import torch
import torch.distributed as dist
from ray.util import queue as ray_queue
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm.distributed.weight_transfer.base import WeightTransferInitRequest
from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

from open_instruct import data_loader as data_loader_lib
from open_instruct import grpo_fast_resource_plan, grpo_utils, logger_utils, utils, value_model_utils, vllm_utils

# grpo_fast is heavy (pulls in vLLM) so it is imported lazily inside main().
from open_instruct.dataset_transformation import INPUT_IDS_PROMPT_KEY, TokenizerConfig
from open_instruct.environments.tools.utils import EnvsConfig
from open_instruct.ground_truth_utils import RewardConfig, build_all_verifiers
from open_instruct.model_utils import ModelConfig, disable_dropout_in_model, olmo_core_attn_to_hf
from open_instruct.utils import ArgumentParserPlus

logger = logger_utils.setup_logger(__name__)

_GEN_VALUE_SAMPLE_SIZE = 4  # prompts sampled per step for background scoring
_GEN_VALUE_OPERATION_TIMEOUT_S = 7200.0
_GEN_VALUE_HEALTH_TIMEOUT_S = 60.0


def _check_gen_value_engines(vllm_engines: list) -> None:
    """Raise promptly if a critic vLLM background task or loop has failed."""
    if not vllm_engines:
        return
    utils.ray_get_with_progress(
        [engine.check_background_threads.remote() for engine in vllm_engines],
        desc="Checking generative critic vLLM engine health",
        enable=False,
        timeout=_GEN_VALUE_HEALTH_TIMEOUT_S,
    )


@ray.remote(num_gpus=1, num_cpus=1)
class GenValueTrainerActor:
    """Ray actor that holds the gen-value model (DeepSpeed) and performs REINFORCE updates.

    The actor receives token-preserving training pairs via ``reinforce_step`` and optimises the
    paper's accuracy-shaped REINFORCE reward::

        R_v = 1 - (outcome - v_hat)**2    if v_hat was parsed from the generation
            = 0                           otherwise

    where ``outcome`` is clipped to [0, 1] and ``v_hat`` is the parsed/rescaled score
    from the critic's own generation. This matches §5.2 of
    "Bringing Value Models Back" (GenAC, arXiv:2604.10701).

    Weight sync to the gen-value vLLM pool is done in-place over NCCL via
    ``setup_model_update_group`` + ``broadcast_to_vllm``, mirroring how the policy
    pushes weights to its vLLM pool in ``grpo_fast.PolicyTrainerRayProcess``.
    """

    def __init__(
        self,
        model_path: str,
        model_revision: str | None,
        tokenizer_path: str,
        tokenizer_revision: str | None,
        learning_rate: float,
        score_min: float,
        score_max: float,
        max_sequence_tokens: int,
        pack_length: int,
        attn_implementation: str,
        gradient_checkpointing: bool,
        temperature: float = 1.0,
        truncated_importance_sampling_ratio_cap: float = 2.0,
        tis_mask_lower: float = 0.0,
        tis_mask_upper: float = 0.0,
        tensor_parallel_size: int = 1,
        reinforce_coef: float = 0.1,
        reinforce_baseline: str = "none",
        weight_decay: float = 0.0,
        set_weight_decay_on_bias_and_norm: bool = True,
        fused_optimizer: bool = True,
        max_grad_norm: float = 1.0,
        checkpoint_path: str | None = None,
        checkpoint_tag: str | None = None,
        trace_reservoir_size: int = 0,
        trace_seed: int = 0,
    ) -> None:
        self._score_min = score_min
        self._score_max = score_max
        self._tp_size = tensor_parallel_size
        self._max_sequence_tokens = max_sequence_tokens
        self._pack_length = pack_length
        self._temperature = temperature
        self._tis_ratio_cap = truncated_importance_sampling_ratio_cap
        self._tis_mask_lower = tis_mask_lower
        self._tis_mask_upper = tis_mask_upper
        self._reinforce_coef = reinforce_coef
        self._reinforce_baseline = reinforce_baseline
        self._step_count = 0
        self._trace_reservoir_size = trace_reservoir_size
        self._trace_rng = random.Random(trace_seed)
        self._trace_reservoirs: dict[str, list[dict[str, Any]]] = {"correct": [], "incorrect": []}
        self._trace_seen_by_outcome = {"correct": 0, "incorrect": 0}
        if self._reinforce_baseline not in {"none", "leave_one_out"}:
            raise ValueError(f"Unknown generative-value REINFORCE baseline: {self._reinforce_baseline!r}.")
        if self._trace_reservoir_size < 0:
            raise ValueError(f"trace_reservoir_size must be nonnegative, got {self._trace_reservoir_size}.")
        if self._pack_length <= 0:
            raise ValueError(f"Generative critic pack length must be > 0, got {self._pack_length}.")
        if self._pack_length > self._max_sequence_tokens:
            raise ValueError(
                "The policy pack length cannot exceed the generative critic context limit "
                f"({self._pack_length} > {self._max_sequence_tokens})."
            )
        torch.cuda.set_device(0)
        if dist.is_initialized():
            if dist.get_world_size() != 1:
                raise RuntimeError(
                    "The independent generative critic requires a private world-size-one process group, "
                    f"but found world size {dist.get_world_size()}."
                )
        else:
            master_address = "127.0.0.1"
            master_port = utils.find_free_port()
            os.environ.update(
                {
                    "LOCAL_RANK": "0",
                    "RANK": "0",
                    "WORLD_SIZE": "1",
                    "MASTER_ADDR": master_address,
                    "MASTER_PORT": str(master_port),
                }
            )
            deepspeed.init_distributed(
                dist_backend="nccl",
                auto_mpi_discovery=False,
                init_method=f"tcp://{master_address}:{master_port}",
                rank=0,
                world_size=1,
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_path, revision=model_revision, dtype=torch.bfloat16, attn_implementation=attn_implementation
        ).cuda()
        disable_dropout_in_model(model)
        model.config.use_cache = False
        if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()

        optimizer_parameters = (
            utils.get_optimizer_grouped_parameters(model, weight_decay)
            if set_weight_decay_on_bias_and_norm
            else model.parameters()
        )
        optimizer = torch.optim.AdamW(
            optimizer_parameters, lr=learning_rate, weight_decay=weight_decay, fused=fused_optimizer
        )
        ds_config = utils.get_train_ds_config(
            offload=False,
            adam_offload=False,
            stage=0,
            bf16=True,
            max_norm=max_grad_norm,
            zpg=1,
            grad_accum_dtype="fp32",
        )
        ds_config["train_micro_batch_size_per_gpu"] = 1
        ds_config["gradient_accumulation_steps"] = 1
        self._model, self._optimizer, _, _ = deepspeed.initialize(
            model=model, optimizer=optimizer, config=ds_config, dist_init_required=False
        )
        optimizer_dtypes = {parameter.dtype for group in self._optimizer.param_groups for parameter in group["params"]}
        if optimizer_dtypes != {torch.float32}:
            raise RuntimeError(
                "The generative critic requires DeepSpeed FP32 master parameters, "
                f"but its optimizer owns {sorted(str(dtype) for dtype in optimizer_dtypes)}."
            )

        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, revision=tokenizer_revision)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path, checkpoint_tag)

        self._vllm_engines: list | None = None
        self._model_update_group: Any | None = None

    def _score_from_text(self, text: str) -> float | None:
        raw = value_model_utils.parse_generative_value_score(text, self._score_min, self._score_max)
        if raw is None:
            return None
        return value_model_utils.rescale_gen_value_score(raw, self._score_min, self._score_max)

    def _load_checkpoint(self, checkpoint_path: str, checkpoint_tag: str | None) -> None:
        if checkpoint_tag is None:
            raise ValueError("Generative-value DeepSpeed checkpoints require an explicit checkpoint tag.")
        loaded_path, client_state = self._model.load_checkpoint(
            checkpoint_path,
            tag=checkpoint_tag,
            load_module_strict=True,
            load_optimizer_states=True,
            load_lr_scheduler_states=False,
            load_module_only=False,
        )
        if loaded_path is None:
            raise ValueError(f"Failed to load generative-value DeepSpeed checkpoint from {checkpoint_path}.")
        self._step_count = int(client_state.get("gen_value_version", client_state.get("reinforce_steps", 0)))
        logger.info(
            "[GenValue] Restored trainer checkpoint %s/%s at critic version %d.",
            checkpoint_path,
            checkpoint_tag,
            self._step_count,
        )

    def save_checkpoint(self, checkpoint_state_dir: str, policy_step: int) -> dict[str, Any]:
        checkpoint_dir = pathlib.Path(checkpoint_state_dir) / "gen_value_model" / "deepspeed"
        checkpoint_tag = f"global_step{policy_step}"
        client_state = {"gen_value_version": self._step_count, "reinforce_steps": self._step_count}
        saved = self._model.save_checkpoint(
            str(checkpoint_dir), tag=checkpoint_tag, client_state=client_state, save_latest=False
        )
        if saved is False:
            raise RuntimeError(f"Failed to save generative-value DeepSpeed checkpoint to {checkpoint_dir}.")
        trace_path = self._write_trace_reservoir(checkpoint_state_dir)
        checkpoint_metadata = {
            "gen_value_trainer_saved": True,
            "gen_value_trainer_checkpoint": str(checkpoint_dir.relative_to(checkpoint_state_dir)),
            "gen_value_trainer_checkpoint_tag": checkpoint_tag,
            "gen_value_version": self._step_count,
        }
        if trace_path is not None:
            checkpoint_metadata["gen_value_training_trace_reservoir"] = str(trace_path)
        return checkpoint_metadata

    def save_model(self, output_dir: str) -> None:
        gen_value_output_dir = pathlib.Path(output_dir) / "gen_value_model"
        gen_value_output_dir.mkdir(parents=True, exist_ok=True)
        self._model.module.save_pretrained(gen_value_output_dir, safe_serialization=True)
        self._tokenizer.save_pretrained(gen_value_output_dir)
        self._write_trace_reservoir(output_dir)

    def _trace_bucket_capacity(self, bucket: str) -> int:
        correct_capacity = self._trace_reservoir_size // 2
        return correct_capacity if bucket == "correct" else self._trace_reservoir_size - correct_capacity

    def _maybe_store_training_trace(
        self,
        pair: dict[str, Any],
        prompt_ids: list[int],
        generation: str,
        outcome: float,
        prediction: float | None,
        squared_error: float | None,
        reward: float,
    ) -> None:
        if self._trace_reservoir_size == 0:
            return
        bucket = "correct" if outcome > 0.5 else "incorrect"
        capacity = self._trace_bucket_capacity(bucket)
        self._trace_seen_by_outcome[bucket] += 1
        seen = self._trace_seen_by_outcome[bucket]
        reservoir = self._trace_reservoirs[bucket]
        if len(reservoir) < capacity:
            replacement_index: int | None = len(reservoir)
        else:
            sampled_index = self._trace_rng.randrange(seen)
            replacement_index = sampled_index if sampled_index < capacity else None
        if replacement_index is None:
            return
        example = {
            "source_critic_version": int(pair.get("critic_version", 0)),
            "state_kind": pair.get("state_kind"),
            "response_tokens_used": pair.get("response_tokens_used"),
            "response_token_limit": pair.get("response_token_limit"),
            "outcome": outcome,
            "prediction": prediction,
            "squared_error": squared_error,
            "reinforce_reward": reward,
            "prompt": self._tokenizer.decode(prompt_ids, skip_special_tokens=False),
            "generation": generation,
        }
        if replacement_index == len(reservoir):
            reservoir.append(example)
        else:
            reservoir[replacement_index] = example

    def _write_trace_reservoir(self, output_dir: str) -> pathlib.Path | None:
        examples = self._trace_reservoirs["correct"] + self._trace_reservoirs["incorrect"]
        if not examples:
            return None
        return value_model_utils.write_gen_value_training_trace_reservoir(
            output_dir, self._step_count, examples, self._trace_seen_by_outcome
        )

    def reinforce_step(self, training_pairs: list[dict]) -> dict:
        """Apply one REINFORCE gradient step with the MSE-shaped critic reward.

        For each pair we compute ``R_v = 1 - (r - v_hat)^2``, with both ``r`` and
        ``v_hat`` clipped to [0, 1]. Parse failures receive reward zero.
        """
        if not training_pairs:
            return {"gen_value/version": self._step_count, "gen_value/reinforce_steps": self._step_count}

        validated_examples: list[dict[str, Any]] = []
        skipped_empty_generation = 0
        mses: list[float] = []  # Parsed generations only; parse failures have no numeric prediction.
        parsed_v_hats: list[float] = []  # Only for pairs where parsing succeeded.
        near_horizon_incorrect_v_hats: list[float] = []
        near_horizon_incorrect_mses: list[float] = []
        for pair in training_pairs:
            if pair["outcome"] is None:
                continue
            request_output = pair["request_output"]
            if len(request_output.outputs) != 1 or request_output.outputs[0].text is None:
                raise ValueError("Gen-value REINFORCE requires exactly one completion with generated text.")
            completion = request_output.outputs[0]
            prompt_ids = request_output.prompt_token_ids
            generated_ids = completion.token_ids
            if len(completion.logprobs) != len(generated_ids):
                raise ValueError(
                    "Gen-value completion token IDs and rollout log-probabilities must have the same length "
                    f"({len(generated_ids)} != {len(completion.logprobs)})."
                )
            sequence_ids = prompt_ids + generated_ids
            if len(sequence_ids) > self._max_sequence_tokens:
                raise ValueError(
                    "Gen-value REINFORCE sequence exceeds the vLLM critic context "
                    f"({len(sequence_ids)} > {self._max_sequence_tokens} tokens)."
                )
            if not generated_ids:
                skipped_empty_generation += 1
                continue
            outcome = max(0.0, min(1.0, float(pair["outcome"])))
            v_hat = self._score_from_text(completion.text)
            if v_hat is not None:
                parsed_v_hats.append(v_hat)
            reward, squared_error = value_model_utils.generative_value_reinforce_reward(outcome, v_hat)
            self._maybe_store_training_trace(pair, prompt_ids, completion.text, outcome, v_hat, squared_error, reward)
            if squared_error is not None:
                mses.append(squared_error)
                response_tokens_used = pair.get("response_tokens_used")
                response_token_limit = pair.get("response_token_limit")
                if response_tokens_used is not None and response_token_limit is not None:
                    response_token_limit = int(response_token_limit)
                    remaining_tokens = response_token_limit - int(response_tokens_used)
                    near_horizon_threshold = max(512, math.ceil(0.1 * response_token_limit))
                    if outcome <= 0.5 and remaining_tokens <= near_horizon_threshold:
                        near_horizon_incorrect_v_hats.append(v_hat)
                        near_horizon_incorrect_mses.append(squared_error)
            validated_examples.append(
                {
                    "sequence_ids": sequence_ids,
                    "generated_ids": generated_ids,
                    "rollout_logprobs": completion.logprobs,
                    "outcome": outcome,
                    "reward": reward,
                }
            )

        if not validated_examples:
            return {"gen_value/version": self._step_count, "gen_value/reinforce_steps": self._step_count}

        raw_rewards = [float(example["reward"]) for example in validated_examples]
        reinforce_weights = value_model_utils.generative_value_reinforce_weights(raw_rewards, self._reinforce_baseline)
        for example, reinforce_weight in zip(validated_examples, reinforce_weights):
            example["reward"] = reinforce_weight

        packs = value_model_utils.pack_gen_value_examples(validated_examples, self._pack_length)
        # DeepSpeed's BF16 optimizer owns the FP32 accumulation buffers, so clear
        # it directly rather than only clearing the BF16 module gradients.
        self._optimizer.zero_grad()
        reinforce_token_denominator = sum(len(example["generated_ids"]) for example in validated_examples)
        gradient_scale = self._reinforce_coef / max(reinforce_token_denominator, 1)
        total_loss = 0.0
        rewards = raw_rewards
        outcomes = [example["outcome"] for example in validated_examples]
        parsed_count = len(parsed_v_hats)
        has_effective_training_signal = False
        reinforce_token_count = 0
        tis_ratio_sum = 0.0
        tis_token_count = 0
        tis_clipped_tokens = 0
        tis_mask_kept_tokens = 0
        tis_mask_total_tokens = 0

        pack_token_counts: list[int] = []
        for pack_idx, pack in enumerate(packs):
            flattened = value_model_utils.flatten_gen_value_pack(pack)
            (
                input_ids_list,
                position_ids_list,
                logit_positions_list,
                target_ids_list,
                rollout_logprobs_list,
                rewards_list,
            ) = flattened
            pack_token_counts.append(len(input_ids_list))
            input_ids = torch.tensor([input_ids_list], dtype=torch.long, device="cuda")
            position_ids = torch.tensor([position_ids_list], dtype=torch.long, device="cuda")
            logit_positions = torch.tensor(logit_positions_list, dtype=torch.long, device="cuda")
            target_ids = torch.tensor(target_ids_list, dtype=torch.long, device="cuda")

            # Reset position IDs at each example boundary, exactly as policy packing does.
            # Transformers uses these resets to select isolated variable-length FlashAttention,
            # so examples in one pack cannot attend to each other. Selecting the prediction
            # positions directly also avoids materializing prompt-token logits.
            self._model.set_gradient_accumulation_boundary(pack_idx == len(packs) - 1)
            outputs = self._model(
                input_ids=input_ids, attention_mask=None, position_ids=position_ids, logits_to_keep=logit_positions
            )
            logits = outputs.logits.float() / self._temperature
            token_logprobs = -torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1), reduction="none"
            )
            rollout_logprobs = torch.tensor(rollout_logprobs_list, dtype=torch.float32, device="cuda")
            response_mask = torch.ones_like(token_logprobs, dtype=torch.bool)
            rollout_logprobs = grpo_utils.mask_logprobs(rollout_logprobs, response_mask)
            tis_clamped, tis_unclamped = grpo_utils.compute_tis_weights(
                token_logprobs.detach(), rollout_logprobs, response_mask, self._tis_ratio_cap
            )
            tis_mask = grpo_utils.compute_tis_mask(
                token_logprobs, rollout_logprobs, response_mask, self._tis_mask_lower, self._tis_mask_upper
            )
            tis_weights = grpo_utils.combine_tis_terms(tis_clamped, tis_mask)
            if tis_clamped is not None and tis_unclamped is not None:
                tis_ratio_sum += float(tis_clamped.sum())
                tis_token_count += tis_clamped.numel()
                tis_clipped_tokens += int((tis_clamped < tis_unclamped).sum())
            if tis_mask is not None:
                tis_mask_kept_tokens += int(tis_mask.sum())
                tis_mask_total_tokens += tis_mask.numel()

            token_rewards = torch.tensor(rewards_list, dtype=torch.float32, device="cuda")
            per_token_loss = -token_logprobs * token_rewards
            if tis_weights is not None:
                per_token_loss = per_token_loss * tis_weights
            loss = per_token_loss.sum() * gradient_scale
            self._model.backward(loss)
            total_loss += float(loss.detach())
            reinforce_token_count += per_token_loss.numel()
            effective_weights = token_rewards != 0.0
            if tis_weights is not None:
                effective_weights &= tis_weights > 0.0
            if bool(effective_weights.any()):
                has_effective_training_signal = True

        grad_norm: float | None = None
        if self._reinforce_coef > 0.0 and has_effective_training_signal:
            self._model.step()
            global_grad_norm = self._model.get_global_grad_norm()
            if global_grad_norm is not None and math.isfinite(float(global_grad_norm)):
                grad_norm = float(global_grad_norm)
            self._step_count += 1

        metrics = {
            "gen_value/reinforce_loss": total_loss,
            "gen_value/reward_mean": sum(rewards) / len(rewards),
            "gen_value/reinforce_weight_mean": sum(reinforce_weights) / len(reinforce_weights),
            "gen_value/reinforce_weight_abs_mean": sum(abs(weight) for weight in reinforce_weights)
            / len(reinforce_weights),
            "gen_value/reinforce_weight_positive_frac": sum(weight > 0.0 for weight in reinforce_weights)
            / len(reinforce_weights),
            "gen_value/reinforce_weight_negative_frac": sum(weight < 0.0 for weight in reinforce_weights)
            / len(reinforce_weights),
            "gen_value/outcome_mean": sum(outcomes) / len(outcomes),
            "gen_value/mse": sum(mses) / len(mses) if mses else float("nan"),
            "gen_value/parse_rate": parsed_count / len(rewards),
            "gen_value/version": self._step_count,
            "gen_value/reinforce_steps": self._step_count,
            "gen_value/train_tokens": reinforce_token_count,
            "gen_value/train_examples": len(rewards),
            "gen_value/parsed_examples": parsed_count,
            "gen_value/train_packs": len(packs),
            "gen_value/train_pack_tokens": sum(pack_token_counts),
            "gen_value/train_examples_per_pack": len(rewards) / len(packs),
            "gen_value/train_mean_pack_tokens": sum(pack_token_counts) / len(packs),
            "gen_value/train_max_pack_tokens": max(pack_token_counts),
            "gen_value/lr": self._optimizer.param_groups[0]["lr"],
        }
        if self._trace_reservoir_size > 0:
            metrics["gen_value/trace_examples_seen"] = sum(self._trace_seen_by_outcome.values())
            metrics["gen_value/trace_examples_retained"] = sum(len(rows) for rows in self._trace_reservoirs.values())
        if grad_norm is not None:
            metrics["gen_value/grad_norm"] = grad_norm
        if skipped_empty_generation:
            metrics["gen_value/skipped_empty_generation"] = skipped_empty_generation
        if self._reinforce_coef <= 0.0 or not has_effective_training_signal:
            metrics["gen_value/update_skipped"] = 1.0
        if tis_token_count:
            metrics["gen_value/tis_tokens"] = tis_token_count
            metrics["gen_value/tis_ratio"] = tis_ratio_sum / tis_token_count
            metrics["gen_value/tis_clipfrac"] = tis_clipped_tokens / tis_token_count
        if tis_mask_total_tokens:
            metrics["gen_value/tis_mask_tokens"] = tis_mask_total_tokens
            metrics["gen_value/tis_mask_frac_kept"] = tis_mask_kept_tokens / tis_mask_total_tokens
        if parsed_v_hats:
            # Mean of parsed predictions -- tells us whether the critic is biased high/low
            # vs. ``outcome_mean`` and whether it's moving over training. Undefined when
            # no pair parsed this step, so we only emit the key when we have signal.
            metrics["gen_value/v_hat_mean"] = sum(parsed_v_hats) / len(parsed_v_hats)
        if near_horizon_incorrect_v_hats:
            metrics["gen_value/near_horizon_incorrect_v_hat_mean"] = sum(near_horizon_incorrect_v_hats) / len(
                near_horizon_incorrect_v_hats
            )
            metrics["gen_value/near_horizon_incorrect_mse"] = sum(near_horizon_incorrect_mses) / len(
                near_horizon_incorrect_mses
            )
            metrics["gen_value/near_horizon_incorrect_examples"] = len(near_horizon_incorrect_v_hats)
        return metrics

    def setup_model_update_group(self, vllm_engines: list) -> None:
        """One-time NCCL handshake between this trainer and the gen-value vLLM engines.

        World layout matches the policy pool: trainer is rank 0, then each vLLM
        engine owns ``tensor_parallel_size`` consecutive ranks starting from 1.
        """
        self._vllm_engines = vllm_engines
        if not vllm_engines:
            self._model_update_group = None
            return

        master_address = ray._private.services.get_node_ip_address().strip("[]")
        master_port = utils.find_free_port()
        world_size = len(vllm_engines) * self._tp_size + 1
        master_info = {"master_address": master_address, "master_port": master_port, "world_size": world_size}
        init_infos = [master_info | {"rank_offset": i * self._tp_size + 1} for i, _ in enumerate(vllm_engines)]

        # Submit the vLLM-side init RPCs first (async) so the NCCL handshake can
        # proceed on both sides in parallel, then wait.
        refs = [
            engine.init_weight_transfer_engine.remote(WeightTransferInitRequest(init_info=info))
            for engine, info in zip(vllm_engines, init_infos)
        ]
        torch.cuda.set_device(0)
        self._model_update_group = NCCLWeightTransferEngine.trainer_init(master_info)
        utils.ray_get_with_progress(refs, desc="Initializing gen-value vLLM weight transfer engines", timeout=600)

    def broadcast_to_vllm(self) -> dict[str, Any]:
        """Push current PyTorch weights to the gen-value vLLM pool over NCCL.

        Returns engine-side ``update_weights`` ObjectRefs together with the exact
        critic update watermarks represented by the broadcast.
        """
        if not self._vllm_engines or self._model_update_group is None:
            return {"engine_refs": [], "version": self._step_count}
        torch.cuda.empty_cache()
        torch.cuda.set_device(0)
        engine_refs = vllm_utils.broadcast_weights_to_vllm(
            model=self._model.module,
            vllm_engines=self._vllm_engines,
            model_update_group=self._model_update_group,
            model_step=self._step_count,
            gather_whole_model=True,
        )
        return {"engine_refs": engine_refs, "version": self._step_count}

    def get_version(self) -> int:
        return self._step_count

    def ready(self) -> bool:
        return True


@dataclass
class GenValueExperimentConfig(grpo_utils.GRPOExperimentConfig):
    """Extended experiment config for the generative-value training script."""

    # Whether to enable the generative value model path (required for this script).
    use_generative_value_model: bool = False
    # Generative value model: its own weights + its own vLLM pool.
    gen_value_model_name_or_path: str | None = None
    gen_value_model_revision: str | None = None
    # Defaults to the generative critic model's tokenizer and revision. This matters
    # when the actor and critic are different variants with distinct EOS/chat tokens.
    gen_value_tokenizer_name_or_path: str | None = None
    gen_value_tokenizer_revision: str | None = None
    gen_value_vllm_num_engines: int = 1
    gen_value_vllm_tensor_parallel_size: int = 1
    # Segmentation: 'sae' uses the policy-logprob-based SAE boundaries (requires --use_sae);
    # 'fixed' queries the gen value model every `gen_value_chunk_size` response tokens.
    gen_value_segmentation: str = "sae"
    gen_value_chunk_size: int = 512
    # Cap on SAE/fixed boundaries before mandatory tool-observation boundaries are added.
    # Tool transitions may exceed this limit because merging states across an observation
    # would make the following value non-causal.
    gen_value_max_segments: int = 16
    # Generation params for the gen value model's vLLM engine.
    # Default matches GenAC's "Maximum Critic Response Length" (Table 5): the critic
    # needs enough budget to actually do CoT reasoning before emitting the score.
    gen_value_max_new_tokens: int = 1024
    gen_value_temperature: float = 1.0
    # vLLM/trainer context length for the generative critic. When unset, use the
    # critic model's declared maximum. Requests must fit the full prompt and full
    # gen_value_max_new_tokens budget; neither side is silently shortened.
    gen_value_max_model_len: int | None = None
    # Score schema.
    gen_value_score_min: float = 0.0
    gen_value_score_max: float = 10.0
    # Training coefficients.
    gen_value_learning_rate: float | None = None
    gen_value_reinforce_coef: float = 0.1
    # Optional variance-reducing baseline for the raw GenAC reward. Leave-one-out
    # centering also makes malformed/inaccurate generations receive negative weight.
    gen_value_reinforce_baseline: str = "none"
    # Independent critic update cadence. The default critic batch contains the same
    # number of complete rollouts as one global policy batch, irrespective of policy world size.
    gen_value_batch_size: int | None = None
    # Conditioning for the gen-value prompt: one of none, gt, correct_demo, rollout_context.
    gen_value_conditioning: str = "none"
    # Paper-style In-Context Conditioning: identify the active actor and provide an EMA of
    # its observed success rate so the critic can calibrate values to the current policy.
    gen_value_use_icc: bool = True
    gen_value_icc_momentum: float = 0.9
    # How often (in critic optimizer updates) to publish gen-value weights to vLLM.
    # Set to 0 to keep the serving critic frozen while its trainer continues updating.
    gen_value_sync_freq: int = 1
    # Fixed held-out states captured from the first on-policy batch and excluded from
    # its REINFORCE update. Rescore them at critic version 0 and each frequency multiple.
    gen_value_validation_freq: int = 0
    gen_value_validation_max_examples: int = 0
    gen_value_validation_prompt_holdout_fraction: float = 0.125
    # Balanced, bounded reservoir of raw on-policy critic traces for inspection and an
    # optional later SFT stage. Zero disables collection.
    gen_value_trace_reservoir_size: int = 0

    def __post_init__(self):
        super().__post_init__()
        if not self.use_generative_value_model:
            raise ValueError(
                "grpo_fast_genvalue.py requires --use_generative_value_model. "
                "Use grpo_fast.py for runs without a generative value model."
            )
        if not self.use_value_model:
            raise ValueError("--use_generative_value_model requires --use_value_model.")
        if self.gen_value_vllm_num_engines <= 0:
            raise ValueError("--gen_value_vllm_num_engines must be > 0 for generative-value training.")
        if self.gen_value_vllm_tensor_parallel_size <= 0:
            raise ValueError(
                f"--gen_value_vllm_tensor_parallel_size must be > 0, got {self.gen_value_vllm_tensor_parallel_size}."
            )
        if self.gen_value_segmentation not in {"sae", "fixed"}:
            raise ValueError(
                f"--gen_value_segmentation must be 'sae' or 'fixed', got {self.gen_value_segmentation!r}."
            )
        if self.gen_value_segmentation == "sae" and not self.use_sae:
            raise ValueError(
                "--gen_value_segmentation=sae requires --use_sae (SAE boundaries come from the policy's vLLM logprobs)."
            )
        if self.gen_value_reinforce_baseline not in {"none", "leave_one_out"}:
            raise ValueError(
                "--gen_value_reinforce_baseline must be one of 'none' or 'leave_one_out', "
                f"got {self.gen_value_reinforce_baseline!r}."
            )
        if self.gen_value_chunk_size <= 0:
            raise ValueError(f"--gen_value_chunk_size must be > 0, got {self.gen_value_chunk_size}.")
        if self.gen_value_max_segments <= 0:
            raise ValueError(f"--gen_value_max_segments must be > 0, got {self.gen_value_max_segments}.")
        if self.gen_value_max_new_tokens <= 0:
            raise ValueError(f"--gen_value_max_new_tokens must be > 0, got {self.gen_value_max_new_tokens}.")
        if self.gen_value_temperature <= 0.0:
            raise ValueError(f"--gen_value_temperature must be > 0, got {self.gen_value_temperature}.")
        if self.gen_value_max_model_len is not None and self.gen_value_max_model_len <= self.gen_value_max_new_tokens:
            raise ValueError(
                "--gen_value_max_model_len must be greater than --gen_value_max_new_tokens "
                f"({self.gen_value_max_model_len} <= {self.gen_value_max_new_tokens})."
            )
        if self.gen_value_score_max <= self.gen_value_score_min:
            raise ValueError("--gen_value_score_max must be greater than --gen_value_score_min.")
        if self.gen_value_reinforce_coef < 0:
            raise ValueError(f"--gen_value_reinforce_coef must be >= 0, got {self.gen_value_reinforce_coef}.")
        if self.gen_value_learning_rate is not None and self.gen_value_learning_rate <= 0.0:
            raise ValueError(f"--gen_value_learning_rate must be > 0, got {self.gen_value_learning_rate}.")
        if self.gen_value_batch_size is not None and self.gen_value_batch_size <= 0:
            raise ValueError(f"--gen_value_batch_size must be > 0, got {self.gen_value_batch_size}.")
        if self.gen_value_sync_freq < 0:
            raise ValueError(f"--gen_value_sync_freq must be >= 0, got {self.gen_value_sync_freq}.")
        if self.gen_value_validation_freq < 0:
            raise ValueError(f"--gen_value_validation_freq must be >= 0, got {self.gen_value_validation_freq}.")
        if self.gen_value_validation_max_examples < 0:
            raise ValueError(
                f"--gen_value_validation_max_examples must be >= 0, got {self.gen_value_validation_max_examples}."
            )
        if not 0.0 < self.gen_value_validation_prompt_holdout_fraction <= 1.0:
            raise ValueError(
                "--gen_value_validation_prompt_holdout_fraction must be in (0, 1], got "
                f"{self.gen_value_validation_prompt_holdout_fraction}."
            )
        if self.gen_value_trace_reservoir_size < 0:
            raise ValueError(
                f"--gen_value_trace_reservoir_size must be >= 0, got {self.gen_value_trace_reservoir_size}."
            )
        if (self.gen_value_validation_freq == 0) != (self.gen_value_validation_max_examples == 0):
            raise ValueError(
                "--gen_value_validation_freq and --gen_value_validation_max_examples must either both be zero "
                "or both be positive."
            )
        if self.gen_value_conditioning not in value_model_utils.GEN_VALUE_CONDITIONING_TYPES:
            raise ValueError(
                f"--gen_value_conditioning must be one of "
                f"{sorted(value_model_utils.GEN_VALUE_CONDITIONING_TYPES)}, "
                f"got {self.gen_value_conditioning!r}."
            )
        if not 0.0 <= self.gen_value_icc_momentum < 1.0:
            raise ValueError(f"--gen_value_icc_momentum must be in [0, 1), got {self.gen_value_icc_momentum}.")


def _resolve_gen_value_tokenizer(args: GenValueExperimentConfig, tc: TokenizerConfig) -> tuple[str, str | None]:
    """Resolve a tokenizer that matches the critic model unless explicitly overridden."""
    if args.gen_value_tokenizer_name_or_path is not None:
        tokenizer_path = args.gen_value_tokenizer_name_or_path
        # A different repository must not inherit a revision belonging to the
        # policy tokenizer. None intentionally means that repository's default.
        tokenizer_revision = args.gen_value_tokenizer_revision
    elif args.gen_value_model_name_or_path is not None:
        tokenizer_path = args.gen_value_model_name_or_path
        tokenizer_revision = args.gen_value_tokenizer_revision or args.gen_value_model_revision
    else:
        tokenizer_path = tc.tokenizer_name_or_path
        tokenizer_revision = args.gen_value_tokenizer_revision or tc.tokenizer_revision
    if tokenizer_path is None:
        raise ValueError("The policy or generative critic tokenizer path must be configured.")
    return tokenizer_path, tokenizer_revision


def _resolve_gen_value_model(args: GenValueExperimentConfig, model_config: ModelConfig) -> tuple[str, str | None]:
    """Default to the policy model and revision, or use an independent critic override."""
    model_path = args.gen_value_model_name_or_path or model_config.model_name_or_path
    if args.gen_value_model_revision is not None:
        model_revision = args.gen_value_model_revision
    elif args.gen_value_model_name_or_path is None:
        model_revision = model_config.model_revision
    else:
        # A different repository must not inherit a revision belonging to the policy model.
        model_revision = None
    return model_path, model_revision


def score_partial_rollout_batch(
    vllm_engines, prompts: list[str], *, max_new_tokens: int, temperature: float, score_min: float, score_max: float
) -> tuple[list[float | None], list[str]]:
    """Send a batch of partial-rollout scoring prompts to the gen-value vLLM pool.

    Returns (parsed_scores_in_0_1, raw_generations). Parse failures are reported as ``None`` in
    the returned list so callers can track ``parse_rate`` as a metric. The in-graph value scorer
    substitutes ``0.0`` because GAE needs a numeric value, while REINFORCE assigns malformed
    generations reward zero.
    """
    n_eng = len(vllm_engines)
    buckets: list[list[tuple[int, str]]] = [[] for _ in range(n_eng)]
    for k, prompt in enumerate(prompts):
        buckets[k % n_eng].append((k, prompt))
    non_empty = [(e, b) for e, b in enumerate(buckets) if b]
    refs = [
        vllm_engines[e].generate_request_outputs.remote(
            [p for _, p in bucket],
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=1.0,
            stop=["</answer>"],
            include_stop_str_in_output=True,
            allow_prompt_truncation=False,
        )
        for e, bucket in non_empty
    ]
    engine_results, _ = utils.ray_get_with_progress(
        refs,
        desc="Scoring generative critic diagnostics",
        enable=False,
        timeout=_GEN_VALUE_OPERATION_TIMEOUT_S,
        health_check_fn=lambda: _check_gen_value_engines(vllm_engines),
    )
    raw: list[str] = [""] * len(prompts)
    for (_, bucket), bucket_outputs in zip(non_empty, engine_results):
        for (k, _), request_output in zip(bucket, bucket_outputs):
            if len(request_output.outputs) != 1 or request_output.outputs[0].text is None:
                raise RuntimeError("Generative-value scoring requires exactly one completion with generated text.")
            raw[k] = request_output.outputs[0].text

    scores: list[float | None] = []
    for text in raw:
        parsed = value_model_utils.parse_generative_value_score(text, score_min=score_min, score_max=score_max)
        if parsed is None:
            scores.append(None)
        else:
            scores.append(value_model_utils.rescale_gen_value_score(parsed, score_min, score_max))
    return scores, raw


def _build_sample_scoring_prompts(
    args: GenValueExperimentConfig, tokenizer: Any, train_dataset: Any, n: int, ground_truths_key: str = "ground_truth"
) -> list[str]:
    """Sample n prompts from the dataset and build gen-value scoring prompts from them.

    Used only by the diagnostic scoring thread; we probe the critic at the
    start-of-rollout state (i.e. ``partial_response=""``) so the logged score
    represents the prior value estimate for each sampled problem.
    """
    indices = random.sample(range(len(train_dataset)), min(n, len(train_dataset)))
    prompts = []
    for idx in indices:
        prompt_ids = train_dataset[idx][INPUT_IDS_PROMPT_KEY]
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        ground_truth = train_dataset[idx].get(ground_truths_key, "")
        scoring_prompt = value_model_utils.build_generative_value_prompt(
            partial_response="",
            conditioning=args.gen_value_conditioning,
            ground_truth=ground_truth,
            score_min=args.gen_value_score_min,
            score_max=args.gen_value_score_max,
            problem=prompt_text,
        )
        prompts.append(scoring_prompt)
    return prompts


def _put_gen_value_metrics(metrics_Q: Queue, metrics: dict[str, Any], source: str) -> None:
    """Send background-thread metrics through the main training step for aligned W&B logging."""
    metrics_Q.put_nowait({"_gen_value_metric_source": source, **metrics})


def _drain_gen_value_metrics(metrics_Q: Queue) -> dict[str, Any]:
    """Drain and aggregate every critic update emitted since the previous policy log.

    State metrics keep their latest value. Batch counters are summed, ranges span all
    updates, and means use their true example/token denominators.
    """
    emitted: list[dict[str, Any]] = []
    while True:
        try:
            emitted.append(metrics_Q.get_nowait())
        except queue_lib.Empty:
            break
    if not emitted:
        return {}

    merged: dict[str, Any] = {}
    reinforce_updates: list[dict[str, Any]] = []
    for item in emitted:
        item = dict(item)
        source = item.pop("_gen_value_metric_source", None)
        merged.update(item)
        if source == "REINFORCE":
            reinforce_updates.append(item)

    if not reinforce_updates:
        return merged

    def weighted_mean(metric: str, weight: str) -> float | None:
        numerator = 0.0
        denominator = 0.0
        for update in reinforce_updates:
            value = update.get(metric)
            count = float(update.get(weight, 0.0))
            if isinstance(value, int | float) and math.isfinite(float(value)) and count > 0.0:
                numerator += float(value) * count
                denominator += count
        return numerator / denominator if denominator > 0.0 else None

    weighted_metrics = {
        "gen_value/reinforce_loss": "gen_value/train_tokens",
        "gen_value/grad_norm": "gen_value/train_tokens",
        "gen_value/reward_mean": "gen_value/train_examples",
        "gen_value/reinforce_weight_mean": "gen_value/train_examples",
        "gen_value/reinforce_weight_abs_mean": "gen_value/train_examples",
        "gen_value/reinforce_weight_positive_frac": "gen_value/train_examples",
        "gen_value/reinforce_weight_negative_frac": "gen_value/train_examples",
        "gen_value/outcome_mean": "gen_value/train_examples",
        "gen_value/parse_rate": "gen_value/train_examples",
        "gen_value/mse": "gen_value/parsed_examples",
        "gen_value/v_hat_mean": "gen_value/parsed_examples",
        "gen_value/near_horizon_incorrect_v_hat_mean": "gen_value/near_horizon_incorrect_examples",
        "gen_value/near_horizon_incorrect_mse": "gen_value/near_horizon_incorrect_examples",
        "gen_value/tis_ratio": "gen_value/tis_tokens",
        "gen_value/tis_clipfrac": "gen_value/tis_tokens",
        "gen_value/tis_mask_frac_kept": "gen_value/tis_mask_tokens",
        "gen_value/train_examples_per_pack": "gen_value/train_packs",
        "gen_value/train_mean_pack_tokens": "gen_value/train_packs",
    }
    for metric, weight in weighted_metrics.items():
        mean = weighted_mean(metric, weight)
        if mean is not None:
            merged[metric] = mean

    summed_metrics = {
        "gen_value/batch_rollouts",
        "gen_value/batch_pairs",
        "gen_value/batch_tokens",
        "gen_value/batch_sequence_tokens",
        "gen_value/train_tokens",
        "gen_value/train_examples",
        "gen_value/parsed_examples",
        "gen_value/near_horizon_incorrect_examples",
        "gen_value/train_packs",
        "gen_value/train_pack_tokens",
        "gen_value/tis_tokens",
        "gen_value/tis_mask_tokens",
        "gen_value/skipped_empty_generation",
        "gen_value/update_skipped",
    }
    for metric in summed_metrics:
        values = [float(update[metric]) for update in reinforce_updates if metric in update]
        if values:
            merged[metric] = sum(values)

    for prefix in ("gen_value/source_policy_training_step", "gen_value/source_value_version"):
        minima = [float(update[f"{prefix}_min"]) for update in reinforce_updates if f"{prefix}_min" in update]
        maxima = [float(update[f"{prefix}_max"]) for update in reinforce_updates if f"{prefix}_max" in update]
        if minima and maxima:
            merged[f"{prefix}_min"] = min(minima)
            merged[f"{prefix}_max"] = max(maxima)
            merged[f"{prefix}_spread"] = max(maxima) - min(minima)

    for suffix, reducer in (("min", min), ("max", max)):
        metric = f"gen_value/source_value_lag_{suffix}"
        values = [float(update[metric]) for update in reinforce_updates if metric in update]
        if values:
            merged[metric] = reducer(values)
    max_pack_tokens = [
        float(update["gen_value/train_max_pack_tokens"])
        for update in reinforce_updates
        if "gen_value/train_max_pack_tokens" in update
    ]
    if max_pack_tokens:
        merged["gen_value/train_max_pack_tokens"] = max(max_pack_tokens)
    return merged


def _gen_value_scoring_loop(
    args: GenValueExperimentConfig,
    tokenizer: Any,
    train_dataset: Any,
    gen_value_vllm_engines: list,
    step_trigger: threading.Event,
    stop_event: threading.Event,
    engines_lock: threading.Lock,
    metrics_Q: Queue,
    progress_state: dict[str, Any],
    progress_lock: threading.Lock,
    ground_truths_key: str = "ground_truth",
) -> None:
    """Background thread: after each policy training step, score sample prompts with the gen-value
    pool and log ``gen_value/score_mean`` and ``gen_value/parse_rate`` to W&B.

    The local engine lock serializes this optional diagnostic with weight publication.
    Policy scoring itself is complete before publication is considered.
    """
    logger.info("[GenValue] Scoring thread started.")
    while not stop_event.is_set():
        triggered = step_trigger.wait(timeout=1.0)
        if not triggered:
            continue
        step_trigger.clear()
        if stop_event.is_set():
            break
        try:
            prompts = _build_sample_scoring_prompts(
                args, tokenizer, train_dataset, _GEN_VALUE_SAMPLE_SIZE, ground_truths_key
            )
            with engines_lock:
                with progress_lock:
                    critic_version = progress_state["synced_version"]
                scores, _ = score_partial_rollout_batch(
                    gen_value_vllm_engines,
                    prompts,
                    max_new_tokens=args.gen_value_max_new_tokens,
                    temperature=args.gen_value_temperature,
                    score_min=args.gen_value_score_min,
                    score_max=args.gen_value_score_max,
                )
            valid = [s for s in scores if s is not None]
            score_metrics = {
                "gen_value/score_mean": sum(valid) / len(valid) if valid else float("nan"),
                "gen_value/score_parse_rate": len(valid) / len(scores) if scores else 0.0,
                "gen_value/score_version": critic_version,
            }
            _put_gen_value_metrics(metrics_Q, score_metrics, "scoring")
            logger.debug(
                "[GenValue] scored %d prompts: mean=%.3f parse_rate=%.2f",
                len(scores),
                score_metrics["gen_value/score_mean"],
                score_metrics["gen_value/score_parse_rate"],
            )
        except Exception:
            logger.exception("[GenValue] scoring failed")
            raise
    logger.info("[GenValue] Scoring thread stopped.")


def _gen_value_reinforce_loop(
    trainer_actor: Any,
    training_queue: ray_queue.Queue,
    batch_size: int,
    stop_event: threading.Event,
    metrics_Q: Queue,
    progress_state: dict[str, Any],
    progress_lock: threading.Lock,
    validation_max_examples: int,
    validation_seed: int,
    validation_prompt_holdout_fraction: float,
    validation_state: dict[str, Any],
    validation_lock: threading.Lock,
) -> None:
    """Pipe queued rollouts into fixed-size critic batches and update asynchronously."""
    logger.info("[GenValue] asynchronous critic trainer started.")
    pending_rollouts: deque[dict[str, Any]] = deque()
    admitted_rollouts = 0
    trained_rollouts = 0

    while True:
        try:
            published_rollouts = training_queue.get(timeout=0.1)
        except queue_lib.Empty:
            if stop_event.is_set():
                break
            with progress_lock:
                progress_state["training_queue_size"] = training_queue.qsize()
                progress_state["pending_rollouts"] = len(pending_rollouts)
            continue
        pending_rollouts.extend(published_rollouts)
        admitted_rollouts += len(published_rollouts)

        while len(pending_rollouts) >= batch_size:
            rollouts = [pending_rollouts.popleft() for _ in range(batch_size)]
            trained_rollouts += len(rollouts)

            capture_metrics: dict[str, float] = {}
            with validation_lock:
                should_capture_validation = validation_max_examples > 0 and not validation_state["captured"]
                if should_capture_validation:
                    validation_examples, pairs = value_model_utils.build_gen_value_validation_holdout(
                        rollouts, validation_max_examples, validation_seed, validation_prompt_holdout_fraction
                    )
                    validation_state["examples"] = validation_examples
                    validation_state["captured"] = True
                    capture_metrics = {
                        "gen_value/validation_examples": float(len(validation_examples)),
                        "gen_value/validation_heldout_prompt_groups": float(
                            sum(example["kind"] == "initial" for example in validation_examples)
                        ),
                        "gen_value/validation_heldout_pairs": float(
                            sum(len(rollout["pairs"]) for rollout in rollouts) - len(pairs)
                        ),
                        "gen_value/validation_training_pairs": float(len(pairs)),
                    }
                else:
                    pairs = [pair for rollout in rollouts for pair in rollout["pairs"]]
            policy_training_steps = [int(rollout["policy_training_step"]) for rollout in rollouts]
            source_versions = [int(rollout["critic_version"]) for rollout in rollouts]
            batch_sequence_tokens = sum(
                len(pair["request_output"].prompt_token_ids)
                + sum(len(completion.token_ids) for completion in pair["request_output"].outputs)
                for pair in pairs
            )
            batch_response_tokens = sum(
                sum(len(completion.token_ids) for completion in pair["request_output"].outputs) for pair in pairs
            )
            metrics, _ = utils.ray_get_with_progress(
                [trainer_actor.reinforce_step.remote(pairs)],
                desc="Training generative critic",
                enable=False,
                timeout=_GEN_VALUE_OPERATION_TIMEOUT_S,
            )
            metrics = metrics[0]
            critic_version = int(metrics["gen_value/version"])
            with progress_lock:
                synced_version = progress_state["synced_version"]
                progress_state["version"] = critic_version
                latest_source_step = progress_state["latest_source_policy_training_step"]
                progress_state["latest_source_policy_training_step"] = (
                    max(policy_training_steps)
                    if latest_source_step is None
                    else max(latest_source_step, max(policy_training_steps))
                )
                progress_state["training_queue_size"] = training_queue.qsize()
                progress_state["pending_rollouts"] = len(pending_rollouts)
                progress_state["admitted_rollouts"] = admitted_rollouts
                progress_state["trained_rollouts"] = trained_rollouts
            metrics.update(
                {
                    "gen_value/source_policy_training_step_min": min(policy_training_steps),
                    "gen_value/source_policy_training_step_max": max(policy_training_steps),
                    "gen_value/source_policy_training_step_spread": max(policy_training_steps)
                    - min(policy_training_steps),
                    "gen_value/source_value_version_min": min(source_versions),
                    "gen_value/source_value_version_max": max(source_versions),
                    "gen_value/source_value_version_spread": max(source_versions) - min(source_versions),
                    "gen_value/source_value_lag_min": max(critic_version - max(source_versions), 0),
                    "gen_value/source_value_lag_max": max(critic_version - min(source_versions), 0),
                    "gen_value/batch_rollouts": len(rollouts),
                    "gen_value/batch_pairs": len(pairs),
                    "gen_value/batch_tokens": batch_response_tokens,
                    "gen_value/batch_sequence_tokens": batch_sequence_tokens,
                    "gen_value/training_queue_size": training_queue.qsize(),
                    "gen_value/pending_rollouts": len(pending_rollouts),
                    "gen_value/admitted_rollouts": admitted_rollouts,
                    "gen_value/trained_rollouts": trained_rollouts,
                    "gen_value/synced_version": synced_version,
                    "gen_value/serving_version_lag": max(critic_version - synced_version, 0),
                }
            )
            metrics.update(capture_metrics)
            _put_gen_value_metrics(metrics_Q, metrics, "REINFORCE")
            logger.debug("[GenValue] REINFORCE step: %s", metrics)

    with progress_lock:
        progress_state["training_queue_size"] = training_queue.qsize()
        progress_state["pending_rollouts"] = len(pending_rollouts)
    logger.info("[GenValue] asynchronous critic trainer stopped.")


def _sync_gen_value_weights(
    gen_value_trainer: Any,
    gen_value_vllm_engines: list,
    engines_lock: threading.Lock,
    health_check_fn: Callable[[], None] | None = None,
) -> dict[str, float]:
    """Push updated gen-value weights to the gen-value vLLM pool over NCCL.

    Mirrors the policy-side weight sync in ``grpo_fast.weight_sync_thread``:
    the engines are put to sleep inside ``broadcast_weights_to_vllm``, the
    trainer streams parameters over the NCCL group established at startup,
    and the engines are woken back up here.

    Callers invoke this only between distributed policy steps. The local lock
    additionally keeps diagnostic scoring from overlapping the publication.
    """
    if not gen_value_vllm_engines:
        return {}
    started_at = time.perf_counter()

    def check_health() -> None:
        _check_gen_value_engines(gen_value_vllm_engines)
        if health_check_fn is not None:
            health_check_fn()

    lock_deadline = started_at + _GEN_VALUE_OPERATION_TIMEOUT_S
    while not engines_lock.acquire(timeout=1.0):
        check_health()
        if time.perf_counter() >= lock_deadline:
            raise TimeoutError("Timed out waiting for generative critic scoring before weight sync.")

    try:
        transfer_error: BaseException | None = None
        try:
            sync_results, _ = utils.ray_get_with_progress(
                [gen_value_trainer.broadcast_to_vllm.remote()],
                desc="Broadcasting generative critic weights",
                enable=False,
                timeout=_GEN_VALUE_OPERATION_TIMEOUT_S,
                health_check_fn=check_health,
            )
            sync_result = sync_results[0]
            engine_refs = sync_result["engine_refs"]
            if engine_refs:
                utils.ray_get_with_progress(
                    engine_refs,
                    desc="Loading generative critic weights",
                    enable=False,
                    timeout=_GEN_VALUE_OPERATION_TIMEOUT_S,
                    health_check_fn=check_health,
                )
            synced_version = int(sync_result["version"])
        except BaseException as error:
            transfer_error = error
            raise
        finally:
            try:
                utils.ray_get_with_progress(
                    [engine.wake_up.remote() for engine in gen_value_vllm_engines],
                    desc="Waking generative critic vLLM engines",
                    enable=False,
                    timeout=_GEN_VALUE_HEALTH_TIMEOUT_S,
                )
            except Exception:
                if transfer_error is None:
                    raise
                logger.exception("[GenValue] Failed to wake critic vLLM engines after a failed weight transfer.")
    finally:
        engines_lock.release()

    logger.debug(
        "[GenValue] Weight sync complete (%d engine(s), critic version=%d).",
        len(gen_value_vllm_engines),
        synced_version,
    )
    return {
        "gen_value/synced_version": synced_version,
        "gen_value/evaluator_version": synced_version,
        "gen_value/weight_sync_seconds": time.perf_counter() - started_at,
    }


def main():
    """Entry point: parse GenValueExperimentConfig, bring up the second vLLM pool, then train.

    Mirrors grpo_fast.main() setup step-by-step.  After policy model init, creates the gen-value
    vLLM pool and starts a background scoring thread that fires after each policy training step.
    """
    import grpo_fast as _grpo_fast  # noqa: PLC0415

    utils.check_oe_eval_internal()

    parser = ArgumentParserPlus(
        (
            GenValueExperimentConfig,
            TokenizerConfig,
            ModelConfig,
            data_loader_lib.StreamingDataLoaderConfig,
            data_loader_lib.VLLMConfig,
            EnvsConfig,
        )
    )
    parser.set_defaults(exp_name="grpo_genvalue", warmup_ratio=0.0, max_grad_norm=1.0, per_device_train_batch_size=1)
    args, tc, model_config, streaming_config, vllm_config, tools_config = parser.parse_args_into_dataclasses()
    assert isinstance(args, GenValueExperimentConfig)

    # Log combined resource requirements (policy pool + gen-value pool).
    base_reqs = grpo_fast_resource_plan.build_grpo_fast_startup_requirements(
        num_learners_per_node=args.num_learners_per_node,
        single_gpu_mode=args.single_gpu_mode,
        vllm_num_engines=vllm_config.vllm_num_engines,
        vllm_tensor_parallel_size=vllm_config.vllm_tensor_parallel_size,
    )
    gen_value_pool_gpus = args.gen_value_vllm_num_engines * args.gen_value_vllm_tensor_parallel_size
    gen_value_trainer_gpus = 1
    gen_value_extra_gpus = gen_value_pool_gpus + gen_value_trainer_gpus
    combined_reqs = dict(base_reqs)
    combined_reqs["additional_topology_gpus"] = gen_value_extra_gpus
    combined_reqs["additional_topology_cpus"] = gen_value_pool_gpus + 1
    combined_reqs["min_total_cluster_gpus"] = base_reqs["min_total_cluster_gpus"] + gen_value_extra_gpus
    combined_reqs["min_total_cluster_cpus"] = base_reqs["min_total_cluster_cpus"] + gen_value_pool_gpus + 1
    logger.info(
        "Gen-value adds %d GPU(s): %d for the second vLLM pool (%d engine(s) × TP%d) "
        "and %d for the trainer. "
        "Policy needs ≥%d GPU(s); combined total ≥%d GPU(s).",
        gen_value_extra_gpus,
        gen_value_pool_gpus,
        args.gen_value_vllm_num_engines,
        args.gen_value_vllm_tensor_parallel_size,
        gen_value_trainer_gpus,
        base_reqs["min_total_cluster_gpus"],
        combined_reqs["min_total_cluster_gpus"],
    )

    # ── Step 1: mirror grpo_fast.main() pre-ray setup ─────────────────────────
    tokenizer = _grpo_fast.make_tokenizer(tc, model_config)
    gen_value_model_path, gen_value_model_revision = _resolve_gen_value_model(args, model_config)
    gen_value_tokenizer_path, gen_value_tokenizer_revision = _resolve_gen_value_tokenizer(args, tc)
    args = _grpo_fast.setup_runtime_variables(args, streaming_config, tools_config)
    _grpo_fast.validate_configs(
        streaming_config, vllm_config, tuple(args.num_learners_per_node), args.sequence_parallel_size
    )
    default_gen_value_batch_size = (
        streaming_config.num_unique_prompts_rollout * streaming_config.num_samples_per_prompt_rollout
    )
    gen_value_batch_size = args.gen_value_batch_size or default_gen_value_batch_size
    args.gen_value_batch_size = gen_value_batch_size

    if args.verbose:
        root_logger = logger_utils.setup_logger()
        root_logger.setLevel("DEBUG")
        for handler in root_logger.handlers:
            handler.setLevel("DEBUG")

    beaker_config, wandb_url = _grpo_fast.setup_experiment_tracking(
        args, tc, model_config, streaming_config, vllm_config
    )

    # ── Step 2: ray.init ──────────────────────────────────────────────────────
    ray.init(
        runtime_env={
            "excludes": [".git/"],
            "env_vars": {k: v for k, v in os.environ.items() if k not in _grpo_fast.EXCLUDED_ENV_VARS},
        }
    )
    _grpo_fast.wait_for_grpo_fast_minimum_cluster_resources(args, combined_reqs)

    pool_size = tools_config.pool_size
    if pool_size is None:
        pool_size = streaming_config.num_unique_prompts_rollout * streaming_config.num_samples_per_prompt_rollout

    pools, tool_definitions, tool_stop_sequences = _grpo_fast.initialize_tools_and_envs(
        tools_config,
        tokenizer,
        pool_size=pool_size,
        dataset_mixer_list=streaming_config.dataset_mixer_list,
        dataset_mixer_list_splits=streaming_config.dataset_mixer_list_splits,
    )
    if tool_stop_sequences:
        streaming_config.stop_strings.extend(tool_stop_sequences)

    train_dataset, eval_dataset = _grpo_fast.setup_datasets(
        args,
        tc,
        tokenizer,
        streaming_config,
        tool_definitions,
        pass_tools_to_chat_template=tools_config.pass_tools_to_chat_template,
        configured_tool_call_names=tools_config.tool_call_names if tools_config.enabled else None,
    )

    if len(train_dataset) < (
        needed := max(streaming_config.async_steps, 1) * streaming_config.num_unique_prompts_rollout
    ):
        raise ValueError(
            f"Train dataset is too small ({len(train_dataset)} prompts); need {needed}. "
            "Reduce async_steps / num_unique_prompts_rollout or increase the dataset."
        )

    if args.cache_dataset_only:
        return

    utils.ensure_hf_repo_cached(model_config.model_name_or_path, revision=model_config.model_revision)
    if tc.tokenizer_name_or_path and tc.tokenizer_name_or_path != model_config.model_name_or_path:
        utils.ensure_hf_repo_cached(tc.tokenizer_name_or_path, revision=tc.tokenizer_revision)
    if (
        gen_value_model_path != model_config.model_name_or_path
        or gen_value_model_revision != model_config.model_revision
    ):
        utils.ensure_hf_repo_cached(gen_value_model_path, revision=gen_value_model_revision)
    if gen_value_tokenizer_path != tc.tokenizer_name_or_path or gen_value_tokenizer_revision != tc.tokenizer_revision:
        utils.ensure_hf_repo_cached(gen_value_tokenizer_path, revision=gen_value_tokenizer_revision)
    gen_value_tokenizer = AutoTokenizer.from_pretrained(
        gen_value_tokenizer_path, revision=gen_value_tokenizer_revision
    )

    # ── Step 3: create policy model, optimizer, and policy vLLM pool ──────────
    num_eval_prompts = len(eval_dataset) if eval_dataset is not None else 0
    queue_size = (streaming_config.async_steps + 1) * streaming_config.num_unique_prompts_rollout + num_eval_prompts
    inference_results_Q = ray_queue.Queue(maxsize=queue_size)
    prompt_Q = ray_queue.Queue(maxsize=queue_size)
    evaluation_inference_results_Q = ray_queue.Queue()

    reward_config = RewardConfig(
        apply_r1_style_format_reward=streaming_config.apply_r1_style_format_reward,
        r1_style_format_reward=streaming_config.r1_style_format_reward,
        apply_verifiable_reward=streaming_config.apply_verifiable_reward,
        verification_reward=streaming_config.verification_reward,
        non_stop_penalty=streaming_config.non_stop_penalty,
        non_stop_penalty_value=streaming_config.non_stop_penalty_value,
        only_reward_good_outputs=tools_config.only_reward_good_outputs,
        additive_format_reward=streaming_config.additive_format_reward,
        verifier_functions=build_all_verifiers(args, streaming_config),
        reward_aggregator=streaming_config.reward_aggregator,
    )

    generation_configs = _grpo_fast.create_generation_configs(args, streaming_config, vllm_config)
    base_env_config = _grpo_fast.build_base_env_config(tools_config, pools)

    (
        policy_group,
        vllm_engines,
        resume_training_step,
        episode,
        actor_manager,
        model_dims,
        _data_prep_actor,
        checkpoint_state,
    ) = _grpo_fast.create_model_and_optimizer(
        args,
        tc,
        model_config,
        beaker_config,
        wandb_url,
        tokenizer,
        inference_results_Q,
        prompt_Q,
        evaluation_inference_results_Q,
        streaming_config,
        vllm_config,
        train_dataset,
        eval_dataset,
        reward_config,
        generation_configs["train"],
        base_env_config,
        tool_definitions,
        tools_config,
        pools,
        tool_stop_sequences,
    )

    if checkpoint_state:
        episode = checkpoint_state["episode"]
        logger.info("Restored episode count: %d", episode)

    # Several functions in grpo_fast.py reference module-level globals that are set by its
    # __main__ block in normal execution. Since we import grpo_fast as a module rather than
    # running it via __main__, we inject all required globals into its namespace here.
    _grpo_fast.vllm_config = vllm_config
    _grpo_fast.streaming_config = streaming_config
    _grpo_fast.args = args

    # ── Step 4: create gen-value vLLM pool ────────────────────────────────────
    gen_value_vllm_engines: list = []

    if args.gen_value_vllm_num_engines > 0:
        # The gen-value engines are queried directly via score_partial_rollout_batch(); they do
        # not participate in the queue-driven rollout loop.  We pass sentinel Ray queues so the
        # LLMRayActor's internal prefetch thread doesn't crash (it blocks on an empty queue).
        gen_value_prompt_Q: ray_queue.Queue = ray_queue.Queue()
        gen_value_results_Q: ray_queue.Queue = ray_queue.Queue()
        gen_value_eval_Q: ray_queue.Queue = ray_queue.Queue()

        gen_value_vllm_engines = vllm_utils.create_vllm_engines(
            args.gen_value_vllm_num_engines,
            args.gen_value_vllm_tensor_parallel_size,
            vllm_config.vllm_enforce_eager,
            gen_value_tokenizer_path,
            gen_value_model_path,
            gen_value_model_revision,
            args.seed,
            False,  # no prefix caching for value scoring
            args.gen_value_max_model_len,
            vllm_config.vllm_gpu_memory_utilization,
            False,  # gen-value pool never shares GPU with learners
            pg=None,
            tool_parser_type="legacy",
            tool_definitions=None,
            tool_stop_sequences=[],
            max_steps=1,
            per_turn_max_tokens=None,
            mask_tool_use=False,
            pools={},
            prompt_queue=gen_value_prompt_Q,
            results_queue=gen_value_results_Q,
            eval_results_queue=gen_value_eval_Q,
            actor_manager=None,
            inflight_updates=False,
            reward_config=reward_config,
            train_dataset=None,
            eval_dataset=None,
            vllm_attention_backend=vllm_config.vllm_attention_backend,
            vllm_gdn_prefill_backend=vllm_config.vllm_gdn_prefill_backend,
            tokenizer_revision=gen_value_tokenizer_revision,
        )
        utils.ray_get_with_progress(
            [engine.ready.remote() for engine in gen_value_vllm_engines],
            desc="Waiting for generative critic vLLM engines",
            timeout=300,
        )
        context_limits, _ = utils.ray_get_with_progress(
            [engine.get_max_model_len.remote() for engine in gen_value_vllm_engines],
            desc="Reading generative critic context limits",
            enable=False,
            timeout=_GEN_VALUE_HEALTH_TIMEOUT_S,
        )
        context_limits = [int(limit) for limit in context_limits]
        if len(set(context_limits)) != 1:
            raise RuntimeError(f"Generative critic vLLM engines disagree on context limits: {context_limits}.")
        gen_value_max_model_len = context_limits[0]
        if gen_value_max_model_len <= args.gen_value_max_new_tokens:
            raise ValueError(
                "The effective generative critic context must be greater than its completion budget "
                f"({gen_value_max_model_len} <= {args.gen_value_max_new_tokens})."
            )
        _check_gen_value_engines(gen_value_vllm_engines)
        logger.info(
            "======== ✅ Gen-value vLLM pool ready (%d engine(s), model=%s, max_model_len=%d) =========",
            len(gen_value_vllm_engines),
            gen_value_model_path,
            gen_value_max_model_len,
        )
    else:
        logger.warning(
            "gen_value_vllm_num_engines=0: gen-value pool skipped. "
            "Set --gen_value_vllm_num_engines 1 (requires one additional GPU) to enable scoring."
        )

    # ── Step 4a: gen-value trainer actor + injection wiring ───────────────────
    # When gen-value vLLM engines are available we also spin up a GenValueTrainerActor that
    # holds a DeepSpeed-wrapped copy of the gen-value model for REINFORCE gradient updates. Complete
    # rollouts flow through one bounded queue; the critic consumes fixed batches independently
    # of policy steps and policy world size.
    gen_value_trainer: Any = None
    gen_value_training_queue: ray_queue.Queue | None = None
    gen_value_checkpoint_path: str | None = None
    gen_value_checkpoint_tag: str | None = None
    initial_gen_value_version = 0
    initial_synced_version = 0
    gen_value_engines_lock = threading.Lock()

    if gen_value_vllm_engines:
        gv_lr = args.gen_value_learning_rate or 1e-6
        if checkpoint_state:
            if not checkpoint_state.get("gen_value_trainer_saved", False):
                raise ValueError(
                    "Cannot resume generative-value training because the policy checkpoint does not contain a "
                    "generative-value trainer checkpoint. Start a new run from the policy weights instead of "
                    "resuming partial optimizer state."
                )
            gen_value_checkpoint_path = checkpoint_state.get("gen_value_trainer_checkpoint")
            gen_value_checkpoint_tag = checkpoint_state.get("gen_value_trainer_checkpoint_tag")
            if not gen_value_checkpoint_path or not gen_value_checkpoint_tag:
                raise ValueError(
                    "Checkpoint says the generative-value trainer was saved, but its DeepSpeed path or tag "
                    "was not recorded."
                )
            if not os.path.isabs(gen_value_checkpoint_path):
                gen_value_checkpoint_path = os.path.join(args.checkpoint_state_dir, gen_value_checkpoint_path)
            if not os.path.isdir(os.path.join(gen_value_checkpoint_path, gen_value_checkpoint_tag)):
                raise ValueError(
                    "Generative-value trainer checkpoint is missing: "
                    f"{gen_value_checkpoint_path}/{gen_value_checkpoint_tag}"
                )
        gen_value_trainer = GenValueTrainerActor.remote(
            gen_value_model_path,
            gen_value_model_revision,
            gen_value_tokenizer_path,
            gen_value_tokenizer_revision,
            gv_lr,
            args.gen_value_score_min,
            args.gen_value_score_max,
            tensor_parallel_size=args.gen_value_vllm_tensor_parallel_size,
            max_sequence_tokens=gen_value_max_model_len,
            pack_length=streaming_config.pack_length,
            attn_implementation=olmo_core_attn_to_hf(model_config.attn_implementation),
            gradient_checkpointing=model_config.gradient_checkpointing,
            temperature=args.gen_value_temperature,
            truncated_importance_sampling_ratio_cap=args.truncated_importance_sampling_ratio_cap,
            tis_mask_lower=args.tis_mask_lower,
            tis_mask_upper=args.tis_mask_upper,
            reinforce_coef=args.gen_value_reinforce_coef,
            reinforce_baseline=args.gen_value_reinforce_baseline,
            weight_decay=args.weight_decay,
            set_weight_decay_on_bias_and_norm=args.set_weight_decay_on_bias_and_norm,
            fused_optimizer=args.fused_optimizer,
            max_grad_norm=args.max_grad_norm if args.max_grad_norm is not None else 0.0,
            checkpoint_path=gen_value_checkpoint_path,
            checkpoint_tag=gen_value_checkpoint_tag,
            trace_reservoir_size=args.gen_value_trace_reservoir_size,
            trace_seed=args.seed,
        )
        ready_results, _ = utils.ray_get_with_progress(
            [gen_value_trainer.ready.remote(), gen_value_trainer.get_version.remote()],
            desc="Starting generative critic trainer",
            timeout=_GEN_VALUE_OPERATION_TIMEOUT_S,
        )
        initial_gen_value_version = int(ready_results[1])
        logger.info(
            "======== ✅ Gen-value trainer actor ready "
            "(lr=%.2e, pack_length=%d, attention=%s, gradient_checkpointing=%s) =========",
            gv_lr,
            streaming_config.pack_length,
            olmo_core_attn_to_hf(model_config.attn_implementation),
            model_config.gradient_checkpointing,
        )

        # Establish the NCCL weight-transfer group between the trainer actor and the
        # gen-value vLLM engines. Mirrors `setup_model_update_group` on the policy
        # side (see grpo_fast.PolicyTrainerRayProcess) so we can push weights
        # in-place instead of killing and recreating engines.
        if args.gen_value_sync_freq > 0:
            utils.ray_get_with_progress(
                [gen_value_trainer.setup_model_update_group.remote(gen_value_vllm_engines)],
                desc="Setting up generative critic weight transfer",
                timeout=_GEN_VALUE_OPERATION_TIMEOUT_S,
                health_check_fn=lambda: _check_gen_value_engines(gen_value_vllm_engines),
            )
            logger.info("======== ✅ Gen-value NCCL weight-transfer group initialised =========")
            if gen_value_checkpoint_path is not None:
                sync_metrics = _sync_gen_value_weights(
                    gen_value_trainer, gen_value_vllm_engines, gen_value_engines_lock
                )
                initial_synced_version = int(sync_metrics["gen_value/synced_version"])
                logger.info(
                    "======== ✅ Restored gen-value weights published at critic version %d =========",
                    initial_synced_version,
                )

        queue_capacity = max(args.world_size * max(streaming_config.async_steps, 1), args.world_size)
        gen_value_training_queue = ray_queue.Queue(maxsize=queue_capacity)

        utils.ray_get_with_progress(
            [a.set_gen_value_engines.remote(gen_value_vllm_engines) for a in policy_group.models],
            desc="Wiring generative critic engines to policy trainers",
            timeout=_GEN_VALUE_OPERATION_TIMEOUT_S,
            health_check_fn=lambda: _check_gen_value_engines(gen_value_vllm_engines),
        )
        utils.ray_get_with_progress(
            [a.set_gen_value_training_queue.remote(gen_value_training_queue) for a in policy_group.models]
            + [a.set_gen_value_version.remote(initial_synced_version) for a in policy_group.models],
            desc="Wiring generative critic training state",
            timeout=_GEN_VALUE_OPERATION_TIMEOUT_S,
            health_check_fn=lambda: _check_gen_value_engines(gen_value_vllm_engines),
        )
        logger.info(
            "Gen-value injection wired: %d engine(s), critic batch=%d, queue capacity=%d → %d policy actor(s).",
            len(gen_value_vllm_engines),
            gen_value_batch_size,
            queue_capacity,
            len(policy_group.models),
        )

    # ── Step 5: background threads (scoring + REINFORCE) ──────────────────────
    gen_value_step_trigger = threading.Event()
    gen_value_stop_event = threading.Event()
    # Cross-thread metrics shuttle: background threads put() per-step metric
    # dicts here; _one_training_step_with_genvalue drains them and merges into
    # data_thread_metrics so they land in the main pretty-print + wandb log.
    gen_value_metrics_Q: Queue = Queue()
    gen_value_progress_lock = threading.Lock()
    gen_value_progress_state: dict[str, Any] = {
        "version": initial_gen_value_version,
        "synced_version": initial_synced_version,
        "latest_source_policy_training_step": None,
        "training_queue_size": 0,
        "pending_rollouts": 0,
        "admitted_rollouts": 0,
        "trained_rollouts": 0,
    }
    gen_value_validation_lock = threading.Lock()
    gen_value_validation_state: dict[str, Any] = {"captured": False, "examples": []}
    gen_value_scoring_future: futures.Future | None = None
    gen_value_reinforce_future: futures.Future | None = None

    # Shared executor for training support threads; both critic futures remain
    # observable from the main loop so any background failure aborts the run.
    weight_sync_metrics_Q: Queue = Queue(maxsize=streaming_config.async_steps)
    stop_event = threading.Event()
    executor = futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="grpo_genvalue")

    if gen_value_vllm_engines:
        gen_value_scoring_future = executor.submit(
            _gen_value_scoring_loop,
            args,
            tokenizer,
            train_dataset,
            gen_value_vllm_engines,
            gen_value_step_trigger,
            gen_value_stop_event,
            gen_value_engines_lock,
            gen_value_metrics_Q,
            gen_value_progress_state,
            gen_value_progress_lock,
            tc.ground_truths_key,
        )

        assert gen_value_trainer is not None
        gen_value_reinforce_future = executor.submit(
            _gen_value_reinforce_loop,
            gen_value_trainer,
            gen_value_training_queue,
            gen_value_batch_size,
            gen_value_stop_event,
            gen_value_metrics_Q,
            gen_value_progress_state,
            gen_value_progress_lock,
            args.gen_value_validation_max_examples,
            args.seed,
            args.gen_value_validation_prompt_holdout_fraction,
            gen_value_validation_state,
            gen_value_validation_lock,
        )

    # Wrap one_training_step to expose asynchronous critic progress and trigger diagnostics.
    _original_one_training_step = _grpo_fast.one_training_step

    def _raise_if_gen_value_background_failed() -> None:
        for background_future in (gen_value_scoring_future, gen_value_reinforce_future):
            if background_future is not None and background_future.done():
                background_future.result()

    def _wait_for_gen_value_background() -> None:
        deadline = time.monotonic() + _GEN_VALUE_OPERATION_TIMEOUT_S
        while any(
            background_future is not None and not background_future.done()
            for background_future in (gen_value_scoring_future, gen_value_reinforce_future)
        ):
            _raise_if_gen_value_background_failed()
            _check_gen_value_engines(gen_value_vllm_engines)
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for generative critic background work to stop.")
            time.sleep(1.0)
        _raise_if_gen_value_background_failed()

    last_gen_value_engine_health_check = 0.0
    last_gen_value_validation_version: int | None = None

    def _one_training_step_with_genvalue(*step_args, **step_kwargs):
        _raise_if_gen_value_background_failed()

        existing_health_check = step_kwargs.get("background_health_check")

        def _combined_health_check() -> None:
            nonlocal last_gen_value_engine_health_check
            if existing_health_check is not None:
                existing_health_check()
            _raise_if_gen_value_background_failed()
            now = time.monotonic()
            if now - last_gen_value_engine_health_check >= 5.0:
                _check_gen_value_engines(gen_value_vllm_engines)
                last_gen_value_engine_health_check = now

        step_kwargs["background_health_check"] = _combined_health_check
        policy_step = int(step_args[6] if len(step_args) > 6 else step_kwargs["training_step"])

        existing_post_training_metrics_callback = step_kwargs.get("post_training_metrics_callback")

        def _critic_post_training_metrics_callback() -> dict[str, Any]:
            nonlocal last_gen_value_validation_version
            # Drain critic updates exactly once per policy step. A second earlier drain
            # can split two updates across dictionaries and let the later one overwrite
            # the first when one_training_step merges its callback results.
            progress_metrics = (
                existing_post_training_metrics_callback()
                if existing_post_training_metrics_callback is not None
                else {}
            )
            if gen_value_trainer is not None:
                if policy_step >= args.num_training_steps:
                    # Finish every complete final critic batch before taking the
                    # final progress snapshot and writing the policy-step metrics.
                    gen_value_stop_event.set()
                    gen_value_step_trigger.set()
                    _wait_for_gen_value_background()
                with gen_value_progress_lock:
                    trainer_progress = dict(gen_value_progress_state)
                critic_version = int(trainer_progress["version"])
                synced_version = int(trainer_progress["synced_version"])

                # Publish only between policy steps. All learners have completed
                # critic scoring, so one simple boundary replaces a distributed gate.
                sync_freq = args.gen_value_sync_freq
                next_sync_version = (synced_version // sync_freq + 1) * sync_freq if sync_freq > 0 else None
                if next_sync_version is not None and critic_version >= next_sync_version:
                    sync_metrics = _sync_gen_value_weights(
                        gen_value_trainer,
                        gen_value_vllm_engines,
                        gen_value_engines_lock,
                        health_check_fn=_raise_if_gen_value_background_failed,
                    )
                    progress_metrics.update(sync_metrics)
                    synced_version = int(sync_metrics["gen_value/synced_version"])
                    utils.ray_get_with_progress(
                        [a.set_gen_value_version.remote(synced_version) for a in policy_group.models],
                        desc="Publishing generative critic version",
                        enable=False,
                        timeout=_GEN_VALUE_OPERATION_TIMEOUT_S,
                        health_check_fn=_combined_health_check,
                    )
                    with gen_value_progress_lock:
                        gen_value_progress_state["version"] = max(gen_value_progress_state["version"], synced_version)
                        gen_value_progress_state["synced_version"] = synced_version
                with gen_value_progress_lock:
                    trainer_progress = dict(gen_value_progress_state)
                synced_version = int(trainer_progress["synced_version"])

                progress_metrics.update(
                    {
                        "gen_value/version": trainer_progress["version"],
                        "gen_value/reinforce_steps": trainer_progress["version"],
                        "gen_value/synced_version": synced_version,
                        "gen_value/evaluator_version": synced_version,
                        "gen_value/serving_version_lag": max(trainer_progress["version"] - synced_version, 0),
                        "gen_value/training_queue_size": gen_value_training_queue.qsize(),
                        "gen_value/pending_rollouts": trainer_progress["pending_rollouts"],
                        "gen_value/admitted_rollouts": trainer_progress["admitted_rollouts"],
                        "gen_value/trained_rollouts": trainer_progress["trained_rollouts"],
                    }
                )
                latest_sampled_policy_training_step = trainer_progress["latest_source_policy_training_step"]
                if latest_sampled_policy_training_step is not None:
                    latest_sampled_policy_training_step = int(latest_sampled_policy_training_step)
                    progress_metrics.update(
                        {
                            "gen_value/latest_trained_policy_training_step": latest_sampled_policy_training_step,
                            "gen_value/policy_training_to_sample_lag": max(
                                policy_step - latest_sampled_policy_training_step, 0
                            ),
                        }
                    )
                if args.gen_value_validation_freq > 0:
                    with gen_value_validation_lock:
                        validation_examples = list(gen_value_validation_state["examples"])
                    if last_gen_value_validation_version is None:
                        # The held-out buffer is captured by the first critic batch while
                        # serving remains on version zero until the first publication.
                        should_validate = bool(validation_examples)
                    else:
                        next_validation_version = (
                            last_gen_value_validation_version // args.gen_value_validation_freq + 1
                        ) * args.gen_value_validation_freq
                        should_validate = bool(validation_examples) and synced_version >= next_validation_version
                    if should_validate:
                        prompts = [
                            gen_value_tokenizer.decode(example["prompt_token_ids"], skip_special_tokens=False)
                            for example in validation_examples
                        ]
                        with gen_value_engines_lock:
                            predictions, generations = score_partial_rollout_batch(
                                gen_value_vllm_engines,
                                prompts,
                                max_new_tokens=args.gen_value_max_new_tokens,
                                temperature=args.gen_value_temperature,
                                score_min=args.gen_value_score_min,
                                score_max=args.gen_value_score_max,
                            )
                        snapshot_path = value_model_utils.write_gen_value_validation_snapshot(
                            args.output_dir, synced_version, validation_examples, predictions, prompts, generations
                        )
                        logger.info(
                            "Saved held-out generative-critic predictions for version %d to %s",
                            synced_version,
                            snapshot_path,
                        )
                        progress_metrics.update(
                            value_model_utils.gen_value_validation_metrics(validation_examples, predictions)
                        )
                        progress_metrics["gen_value/validation_version"] = float(synced_version)
                        last_gen_value_validation_version = synced_version
            critic_metrics = _drain_gen_value_metrics(gen_value_metrics_Q)
            critic_metrics.update(progress_metrics)
            return critic_metrics

        step_kwargs["post_training_metrics_callback"] = _critic_post_training_metrics_callback
        result = _original_one_training_step(*step_args, **step_kwargs)
        if gen_value_vllm_engines and not gen_value_stop_event.is_set():
            gen_value_step_trigger.set()
        _raise_if_gen_value_background_failed()
        return result

    _grpo_fast.one_training_step = _one_training_step_with_genvalue

    def _save_gen_value_checkpoint(checkpoint_state_dir: str, training_step: int) -> dict[str, Any]:
        if gen_value_trainer is None:
            return {}
        results, _ = utils.ray_get_with_progress(
            [gen_value_trainer.save_checkpoint.remote(checkpoint_state_dir, training_step)],
            desc="Saving generative critic checkpoint",
            timeout=_GEN_VALUE_OPERATION_TIMEOUT_S,
        )
        return results[0]

    def _commit_gen_value_checkpoint(checkpoint_state_dir: str, training_step: int) -> None:
        del training_step
        if gen_value_trainer is None or args.keep_last_n_checkpoints < 0:
            return
        utils.clean_last_n_checkpoints_deepspeed(
            os.path.join(checkpoint_state_dir, "gen_value_model", "deepspeed"), args.keep_last_n_checkpoints
        )

    def _save_gen_value_model(output_dir: str, training_step: int) -> None:
        if gen_value_trainer is None:
            return
        _raise_if_gen_value_background_failed()
        if training_step >= args.num_training_steps:
            # Drain every complete queued batch before writing the final critic.
            # A final partial batch remains untrained rather than changing batch size.
            gen_value_stop_event.set()
            gen_value_step_trigger.set()
            _wait_for_gen_value_background()
        utils.ray_get_with_progress(
            [gen_value_trainer.save_model.remote(output_dir)],
            desc="Saving generative critic model",
            timeout=_GEN_VALUE_OPERATION_TIMEOUT_S,
        )
        _raise_if_gen_value_background_failed()
        logger.info("Saved generative-value model at policy step %d to %s", training_step, output_dir)

    # ── Step 6: run policy training loop ─────────────────────────────────────
    primary_exception: BaseException | None = None
    try:
        _grpo_fast.run_training(
            args,
            streaming_config,
            tokenizer,
            train_dataset,
            eval_dataset,
            policy_group,
            vllm_engines,
            generation_configs,
            resume_training_step,
            episode,
            wandb_url,
            tc,
            stop_event,
            executor,
            inference_results_Q,
            prompt_Q,
            evaluation_inference_results_Q,
            weight_sync_metrics_Q,
            actor_manager,
            model_dims,
            data_prep_actor=_data_prep_actor,
            checkpoint_state=checkpoint_state,
            base_env_config=base_env_config,
            checkpoint_callback=_save_gen_value_checkpoint,
            checkpoint_commit_callback=_commit_gen_value_checkpoint,
            model_save_callback=_save_gen_value_model,
        )

        if args.push_to_hub and (not dist.is_initialized() or dist.get_rank() == 0):
            _grpo_fast.push_folder_to_hub(args.output_dir, args.hf_repo_id, args.hf_repo_revision)
    except Exception as e:
        primary_exception = e
        if args.send_slack_alerts:
            utils.send_slack_message(f"<!here> A gen-value RL job has died. Error message: {e}.")
        raise
    finally:
        _grpo_fast.one_training_step = _original_one_training_step
        gen_value_stop_event.set()
        gen_value_step_trigger.set()
        _grpo_fast.cleanup_training_resources(
            stop_event, executor, [inference_results_Q, prompt_Q, evaluation_inference_results_Q], actor_manager
        )
        if primary_exception is None:
            _raise_if_gen_value_background_failed()

    logger.info("finished gen-value training")
    utils.check_runtime_leaks()


if __name__ == "__main__":
    main()
