#!/usr/bin/env bash
# Scale the established asynchronous GenAC math recipe to the Qwen3-8B base
# actor and a paper-sized rollout batch: 128 prompts x 8 responses = 1,024
# trajectories. This intentionally preserves the full local math horizons and
# the validated async critic objective; it is not a synchronous paper replica.
set -euo pipefail

export POLICY_MODEL_PATH="${POLICY_MODEL_PATH:-Qwen/Qwen3-8B-Base}"
export NUM_UNIQUE_PROMPTS_ROLLOUT="${NUM_UNIQUE_PROMPTS_ROLLOUT:-128}"
export NUM_SAMPLES_PER_PROMPT_ROLLOUT="${NUM_SAMPLES_PER_PROMPT_ROLLOUT:-8}"

# Use the durable four-H200 allocation without relying on the
# not-yet-production-validated data-parallel critic: one actor learner, one
# actor vLLM engine, one critic vLLM engine, and one critic trainer.
export NUM_POLICY_LEARNERS="${NUM_POLICY_LEARNERS:-1}"
export NUM_POLICY_VLLM_ENGINES="${NUM_POLICY_VLLM_ENGINES:-1}"
export GEN_VALUE_VLLM_NUM_ENGINES="${GEN_VALUE_VLLM_NUM_ENGINES:-1}"
export GEN_VALUE_TRAINER_NUM_GPUS="${GEN_VALUE_TRAINER_NUM_GPUS:-1}"

# Match the paper's actor optimizer and symmetric PPO clipping. Keep the
# critic's conservative learning rate from the completed 4B run. The critic
# consumes all fresh rollouts but uses an unbiased bounded optimizer subset so
# its long-context backward pass can remain concurrent with policy training.
export POLICY_LEARNING_RATE="${POLICY_LEARNING_RATE:-1e-6}"
export POLICY_WEIGHT_DECAY="${POLICY_WEIGHT_DECAY:-0.01}"
export POLICY_BETA="${POLICY_BETA:-0.0}"
export POLICY_CLIP_LOWER="${POLICY_CLIP_LOWER:-0.2}"
export POLICY_CLIP_HIGHER="${POLICY_CLIP_HIGHER:-0.2}"
export NUM_EPOCHS="${NUM_EPOCHS:-1}"
export GEN_VALUE_LEARNING_RATE="${GEN_VALUE_LEARNING_RATE:-2e-7}"
export GEN_VALUE_TRAIN_TARGET_EXAMPLES_PER_UPDATE="${GEN_VALUE_TRAIN_TARGET_EXAMPLES_PER_UPDATE:-1536}"
export GEN_VALUE_OPTIMIZER_SAMPLING_STRATEGY="${GEN_VALUE_OPTIMIZER_SAMPLING_STRATEGY:-length_outcome_stratified}"
export GEN_VALUE_MAX_ASYNC_STEPS="${GEN_VALUE_MAX_ASYNC_STEPS:-4}"
export GEN_VALUE_SYNC_FREQ="${GEN_VALUE_SYNC_FREQ:-1}"

# Preserve asynchronous actor execution, but keep its source window no wider
# than the critic freshness window.
export INFLIGHT_UPDATES="${INFLIGHT_UPDATES:-true}"
export ASYNC_STEPS="${ASYNC_STEPS:-4}"
export TRUNCATED_IMPORTANCE_SAMPLING_RATIO_CAP="${TRUNCATED_IMPORTANCE_SAMPLING_RATIO_CAP:-2.0}"
export USE_VLLM_LOGPROBS="${USE_VLLM_LOGPROBS:-false}"
export ACTIVE_SAMPLING="${ACTIVE_SAMPLING:-false}"
export FILTER_ZERO_STD_SAMPLES="${FILTER_ZERO_STD_SAMPLES:-false}"

# An 8B critic is initialized separately. Frozen-policy warmup adapts that
# critic to the 8B actor before the requested joint-training phase.
export VALUE_WARMUP_STEPS="${VALUE_WARMUP_STEPS:-100}"
export JOINT_TRAINING_STEPS="${JOINT_TRAINING_STEPS:-200}"
export DEEPSPEED_OFFLOAD_OPTIMIZER="${DEEPSPEED_OFFLOAD_OPTIMIZER:-true}"

exec bash scripts/train/debug/genac_math_joint_h200.sh \
    --gen_value_batch_size 256 \
    "$@"
