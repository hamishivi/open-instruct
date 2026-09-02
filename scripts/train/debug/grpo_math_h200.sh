#!/usr/bin/env bash
# Value-free control for the realistic Qwen3-4B GenAC math run.
#
# This deliberately keeps the policy/data/evaluation configuration identical to
# genac_math_joint_h200.sh while removing the value model and generative critic.
# Use the same one-policy-learner/one-policy-vLLM topology as the GenAC run.
# Defaults retain the conservative matched control. Environment overrides allow
# the established strong GRPO recipe (four learners, four vLLM engines,
# inflight sampling, active sampling, and 1e-6 LR) without changing the shared
# data, response horizon, group size, or evaluation schedule.
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
NUM_UNIQUE_PROMPTS_ROLLOUT="${NUM_UNIQUE_PROMPTS_ROLLOUT:-32}"
NUM_SAMPLES_PER_PROMPT_ROLLOUT="${NUM_SAMPLES_PER_PROMPT_ROLLOUT:-8}"
NUM_POLICY_LEARNERS="${NUM_POLICY_LEARNERS:-1}"
NUM_POLICY_VLLM_ENGINES="${NUM_POLICY_VLLM_ENGINES:-1}"
POLICY_LEARNING_RATE="${POLICY_LEARNING_RATE:-5e-7}"
POLICY_BETA="${POLICY_BETA:-0.01}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
ACTIVE_SAMPLING="${ACTIVE_SAMPLING:-false}"
FILTER_ZERO_STD_SAMPLES="${FILTER_ZERO_STD_SAMPLES:-false}"
INFLIGHT_UPDATES="${INFLIGHT_UPDATES:-false}"
ASYNC_STEPS="${ASYNC_STEPS:-1}"
NO_RESAMPLING_PASS_RATE="${NO_RESAMPLING_PASS_RATE:-none}"
TRUNCATED_IMPORTANCE_SAMPLING_RATIO_CAP="${TRUNCATED_IMPORTANCE_SAMPLING_RATIO_CAP:-0.0}"
USE_VLLM_LOGPROBS="${USE_VLLM_LOGPROBS:-true}"
DEEPSPEED_OFFLOAD_OPTIMIZER="${DEEPSPEED_OFFLOAD_OPTIMIZER:-true}"

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
if [[ ! "${NUM_UNIQUE_PROMPTS_ROLLOUT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: NUM_UNIQUE_PROMPTS_ROLLOUT must be at least 1" >&2
    exit 1
fi
if [[ ! "${NUM_SAMPLES_PER_PROMPT_ROLLOUT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: NUM_SAMPLES_PER_PROMPT_ROLLOUT must be at least 1" >&2
    exit 1
fi
if [[ ! "${NUM_EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: NUM_EPOCHS must be at least 1" >&2
    exit 1
fi
if [[ ! "${ASYNC_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ASYNC_STEPS must be at least 1" >&2
    exit 1
fi
for bool_var in ACTIVE_SAMPLING FILTER_ZERO_STD_SAMPLES INFLIGHT_UPDATES USE_VLLM_LOGPROBS DEEPSPEED_OFFLOAD_OPTIMIZER; do
    case "${!bool_var}" in
        true | false) ;;
        *)
            echo "ERROR: ${bool_var} must be true or false" >&2
            exit 1
            ;;
    esac
done
if [[ "${ACTIVE_SAMPLING}" == "true" && "${FILTER_ZERO_STD_SAMPLES}" != "true" ]]; then
    echo "ERROR: ACTIVE_SAMPLING=true requires FILTER_ZERO_STD_SAMPLES=true" >&2
    exit 1
fi
NO_RESAMPLING_ARGS=()
if [[ "${NO_RESAMPLING_PASS_RATE}" != "none" ]]; then
    NO_RESAMPLING_ARGS+=(--no_resampling_pass_rate "${NO_RESAMPLING_PASS_RATE}")
fi
DEEPSPEED_OFFLOAD_ARGS=()
if [[ "${DEEPSPEED_OFFLOAD_OPTIMIZER}" == "true" ]]; then
    DEEPSPEED_OFFLOAD_ARGS+=(--deepspeed_offload_optimizer)
fi
VLLM_LOGPROB_ARGS=()
if [[ "${USE_VLLM_LOGPROBS}" == "true" ]]; then
    VLLM_LOGPROB_ARGS+=(--use_vllm_logprobs)
fi
TOTAL_EPISODES=$((CONTROL_STEPS * NUM_UNIQUE_PROMPTS_ROLLOUT * NUM_SAMPLES_PER_PROMPT_ROLLOUT))

# Ray defaults the dashboard to 8265. Keep the GCS and dashboard in disjoint,
# job-specific ranges so a job ID ending in 265 cannot collide with itself and
# colocated jobs do not all claim the default dashboard port.
RAY_JOB_OFFSET=$((${SLURM_JOB_ID:-0} % 10000))
RAY_PORT="${RAY_PORT:-$((9000 + RAY_JOB_OFFSET % 1000))}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-$((20000 + RAY_JOB_OFFSET))}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray-${USER}-${SLURM_JOB_ID:-local}}"
# `ray stop` is user-global on a host. Slurm can colocate two jobs owned by the
# same user, so invoking it here would kill the other job's Ray cluster. Slurm
# tears down this job's processes with its cgroup; retain explicit cleanup only
# for local runs where there is no scheduler-owned process boundary.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    ray stop --force 2>/dev/null || true
    trap 'ray stop --force' EXIT
fi
ray start \
    --head \
    --port="${RAY_PORT}" \
    --temp-dir="${RAY_TEMP_DIR}" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${RAY_DASHBOARD_PORT}"

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
    --active_sampling "${ACTIVE_SAMPLING}" \
    --filter_zero_std_samples "${FILTER_ZERO_STD_SAMPLES}" \
    "${NO_RESAMPLING_ARGS[@]}" \
    --model_name_or_path Qwen/Qwen3-4B-Base \
    --chat_template_name qwen_instruct_user_boxed_math \
    --temperature 1.0 \
    --apply_verifiable_reward true \
    --verification_reward 1.0 \
    --non_stop_penalty false \
    --beta "${POLICY_BETA}" \
    --loss_fn dapo \
    --clip_higher 0.272 \
    "${VLLM_LOGPROB_ARGS[@]}" \
    --truncated_importance_sampling_ratio_cap "${TRUNCATED_IMPORTANCE_SAMPLING_RATIO_CAP}" \
    --advantage_normalization_type centered \
    --learning_rate "${POLICY_LEARNING_RATE}" \
    --lr_scheduler_type constant \
    --total_episodes "${TOTAL_EPISODES}" \
    --num_epochs "${NUM_EPOCHS}" \
    --num_mini_batches 1 \
    --deepspeed_stage 3 \
    "${DEEPSPEED_OFFLOAD_ARGS[@]}" \
    --num_learners_per_node "${NUM_POLICY_LEARNERS}" \
    --sequence_parallel_size 1 \
    --vllm_num_engines "${NUM_POLICY_VLLM_ENGINES}" \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.85 \
    --vllm_top_p 1.0 \
    --vllm_enable_prefix_caching \
    --inflight_updates "${INFLIGHT_UPDATES}" \
    --async_steps "${ASYNC_STEPS}" \
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
