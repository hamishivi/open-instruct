#!/usr/bin/env bash
# Matched value-free control for genac_math_joint_8b_paper_batch_h200.sh.
# Both runs use Qwen3-8B-Base, 128 prompts x 8 responses, the same actor
# optimizer and policy objective, full 8,192-token responses, and the same
# evaluation cadence. Only the advantage estimator and resulting GPU topology
# differ.
set -euo pipefail

export POLICY_MODEL_PATH="${POLICY_MODEL_PATH:-Qwen/Qwen3-8B-Base}"
export NUM_UNIQUE_PROMPTS_ROLLOUT="${NUM_UNIQUE_PROMPTS_ROLLOUT:-128}"
export NUM_SAMPLES_PER_PROMPT_ROLLOUT="${NUM_SAMPLES_PER_PROMPT_ROLLOUT:-8}"

# Spend the same eight-H200 budget entirely on the actor: four DeepSpeed
# learners and four independent policy vLLM engines.
export NUM_POLICY_LEARNERS="${NUM_POLICY_LEARNERS:-4}"
export NUM_POLICY_VLLM_ENGINES="${NUM_POLICY_VLLM_ENGINES:-4}"

export POLICY_LEARNING_RATE="${POLICY_LEARNING_RATE:-1e-6}"
export POLICY_WEIGHT_DECAY="${POLICY_WEIGHT_DECAY:-0.01}"
export POLICY_BETA="${POLICY_BETA:-0.0}"
export POLICY_CLIP_LOWER="${POLICY_CLIP_LOWER:-0.2}"
export POLICY_CLIP_HIGHER="${POLICY_CLIP_HIGHER:-0.2}"
export NUM_EPOCHS="${NUM_EPOCHS:-1}"
export CONTROL_STEPS="${CONTROL_STEPS:-200}"

# Match the bounded asynchronous actor window used by GenAC without active
# group filtering, so every method consumes exactly 128 prompt groups/step.
export INFLIGHT_UPDATES="${INFLIGHT_UPDATES:-true}"
export ASYNC_STEPS="${ASYNC_STEPS:-4}"
export TRUNCATED_IMPORTANCE_SAMPLING_RATIO_CAP="${TRUNCATED_IMPORTANCE_SAMPLING_RATIO_CAP:-2.0}"
export USE_VLLM_LOGPROBS="${USE_VLLM_LOGPROBS:-false}"
export ACTIVE_SAMPLING="${ACTIVE_SAMPLING:-false}"
export FILTER_ZERO_STD_SAMPLES="${FILTER_ZERO_STD_SAMPLES:-false}"
export DEEPSPEED_OFFLOAD_OPTIMIZER="${DEEPSPEED_OFFLOAD_OPTIMIZER:-true}"

exec bash scripts/train/debug/grpo_math_h200.sh "$@"
