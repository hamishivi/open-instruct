#!/usr/bin/env bash
# Matched value-free control for the realistic Qwen3-4B GenAC math run.
#
# This deliberately keeps the policy/data/evaluation configuration identical to
# genac_math_joint_h200.sh while removing the value model and generative critic.
# Use the same one-policy-learner/one-policy-vLLM topology as the GenAC run.
# The prior two-learner control deadlocked after step 1 in the ZeRO-3/NCCL
# weight-transfer collective, so changing only this parallel topology gives a
# usable value-free comparison without changing its data or optimization recipe.
set -euo pipefail

if [[ "${PRESERVE_LD_LIBRARY_PATH:-0}" != "1" ]]; then
    unset LD_LIBRARY_PATH
fi
export NCCL_CUMEM_ENABLE=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/tmp/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

EXP_NAME="${EXP_NAME:-grpo-math-control-h200}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${PWD}/outputs/${EXP_NAME}}"
CHECKPOINT_STATE_DIR="${CHECKPOINT_STATE_DIR:-${RUN_OUTPUT_DIR}/checkpoint_states}"
CONTROL_STEPS="${CONTROL_STEPS:-300}"
NUM_UNIQUE_PROMPTS_ROLLOUT=32
NUM_SAMPLES_PER_PROMPT_ROLLOUT=8
NUM_POLICY_LEARNERS="${NUM_POLICY_LEARNERS:-1}"
NUM_POLICY_VLLM_ENGINES="${NUM_POLICY_VLLM_ENGINES:-1}"

if [[ ! "${CONTROL_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: CONTROL_STEPS must be at least 1" >&2
    exit 1
fi
if [[ ! "${NUM_POLICY_LEARNERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: NUM_POLICY_LEARNERS must be at least 1" >&2
    exit 1
fi
if [[ ! "${NUM_POLICY_VLLM_ENGINES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: NUM_POLICY_VLLM_ENGINES must be at least 1" >&2
    exit 1
fi
TOTAL_EPISODES=$((CONTROL_STEPS * NUM_UNIQUE_PROMPTS_ROLLOUT * NUM_SAMPLES_PER_PROMPT_ROLLOUT))

RAY_PORT="${RAY_PORT:-$((8000 + ${SLURM_JOB_ID:-0} % 1000))}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray-${USER}-${SLURM_JOB_ID:-local}}"
ray stop --force 2>/dev/null || true
ray start --head --port="${RAY_PORT}" --temp-dir="${RAY_TEMP_DIR}" --dashboard-host=0.0.0.0
trap 'ray stop --force' EXIT

mkdir -p "${HOME}/.triton/autotune"

python open_instruct/grpo_fast.py \
    --exp_name "${EXP_NAME}" \
    --output_dir "${RUN_OUTPUT_DIR}" \
    --checkpoint_state_dir "${CHECKPOINT_STATE_DIR}" \
    --dataset_mixer_list hamishivi/DAPO-Math-17k-Processed_filtered 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list mnoukhov/aime_2025_openinstruct 1.0 \
    --dataset_mixer_eval_list_splits train \
    --remap_verifier math=final_boxed_math,math_aime_2025=final_boxed_math \
    --max_prompt_token_length 2048 \
    --response_length 8192 \
    --pack_length 10240 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout "${NUM_UNIQUE_PROMPTS_ROLLOUT}" \
    --num_samples_per_prompt_rollout "${NUM_SAMPLES_PER_PROMPT_ROLLOUT}" \
    --active_sampling false \
    --filter_zero_std_samples false \
    --model_name_or_path Qwen/Qwen3-4B-Base \
    --chat_template_name qwen_instruct_user_boxed_math \
    --temperature 1.0 \
    --apply_verifiable_reward true \
    --verification_reward 1.0 \
    --non_stop_penalty false \
    --beta 0.01 \
    --loss_fn dapo \
    --clip_higher 0.272 \
    --use_vllm_logprobs \
    --truncated_importance_sampling_ratio_cap 0.0 \
    --advantage_normalization_type centered \
    --learning_rate 5e-7 \
    --lr_scheduler_type constant \
    --total_episodes "${TOTAL_EPISODES}" \
    --num_epochs 2 \
    --num_mini_batches 1 \
    --deepspeed_stage 3 \
    --deepspeed_offload_optimizer \
    --num_learners_per_node "${NUM_POLICY_LEARNERS}" \
    --sequence_parallel_size 1 \
    --vllm_num_engines "${NUM_POLICY_VLLM_ENGINES}" \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.85 \
    --vllm_top_p 1.0 \
    --vllm_enable_prefix_caching \
    --inflight_updates false \
    --async_steps 1 \
    --seed 1 \
    --local_eval_every 25 \
    --eval_on_step_0 true \
    --eval_pass_at_k 8 \
    --save_freq 100 \
    --checkpoint_state_freq 50 \
    --keep_last_n_checkpoints 2 \
    --gradient_checkpointing \
    --with_tracking \
    --push_to_hub false \
    "$@"
