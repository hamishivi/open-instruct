#!/usr/bin/env bash
# Frozen-policy generative-critic pretraining for the realistic Qwen3-4B math setup.
#
# This deliberately ends before the actor is unfrozen. The resulting critic is only
# admitted to joint training after its held-distribution calibration, correct/incorrect
# ranking, and near-horizon incorrect values have been inspected.
#
# GPU layout (4 GPUs on one H200 node):
#   - 1 frozen policy learner
#   - 1 policy vLLM engine
#   - 1 generative-critic vLLM engine
#   - 1 generative-critic DeepSpeed trainer
#
# By default total_episodes = 100 critic-only steps * 32 prompts * 8 samples = 25,600.
# VALUE_PRETRAIN_STEPS and GEN_VALUE_MODEL_PATH can be overridden to continue
# critic-only training from a previously exported Hugging Face model.  The default
# seed intentionally differs from the seed-1 rollout stream used to build the SFT
# trace reservoir, so the post-SFT calibration stage exercises unseen prompts.
# Center critic rewards against other independently sampled completions with the
# same binary outcome. This preserves the policy-gradient expectation while
# turning inaccurate scores and parse failures into an explicit negative signal.
# Final-action replay copies are collapsed before computing the baseline.
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

EXP_NAME="${EXP_NAME:-genac-math-value-pretrain-h200}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${PWD}/outputs/${EXP_NAME}}"
CHECKPOINT_STATE_DIR="${CHECKPOINT_STATE_DIR:-${RUN_OUTPUT_DIR}/checkpoint_states}"
VALUE_PRETRAIN_STEPS="${VALUE_PRETRAIN_STEPS:-100}"
# The GenAC cold-start recipe SFTs and RL-pretrains the base model. Keep an
# explicit override for ablations and for continuing from an exported SFT/RL
# critic checkpoint.
GEN_VALUE_MODEL_PATH="${GEN_VALUE_MODEL_PATH:-Qwen/Qwen3-4B-Base}"
GEN_VALUE_CONDITIONING="${GEN_VALUE_CONDITIONING:-none}"
GEN_VALUE_REINFORCE_BASELINE="${GEN_VALUE_REINFORCE_BASELINE:-leave_one_out_by_outcome}"
SEED="${SEED:-17}"
NUM_UNIQUE_PROMPTS_ROLLOUT=32
NUM_SAMPLES_PER_PROMPT_ROLLOUT=8

if [[ ! "${VALUE_PRETRAIN_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: VALUE_PRETRAIN_STEPS must be at least 1" >&2
    exit 1
fi
if [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SEED must be a nonnegative integer" >&2
    exit 1
fi
TOTAL_EPISODES=$((VALUE_PRETRAIN_STEPS * NUM_UNIQUE_PROMPTS_ROLLOUT * NUM_SAMPLES_PER_PROMPT_ROLLOUT))

RAY_PORT="${RAY_PORT:-$((8000 + ${SLURM_JOB_ID:-0} % 1000))}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray-${USER}-${SLURM_JOB_ID:-local}}"
# `ray stop` is user-global on a host. Slurm can colocate two jobs owned by the
# same user, so invoking it here would kill the other job's Ray cluster. Slurm
# tears down this job's processes with its cgroup; retain explicit cleanup only
# for local runs where there is no scheduler-owned process boundary.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    ray stop --force 2>/dev/null || true
    trap 'ray stop --force' EXIT
fi
ray start --head --port="${RAY_PORT}" --temp-dir="${RAY_TEMP_DIR}" --dashboard-host=0.0.0.0

mkdir -p "${HOME}/.triton/autotune"

python open_instruct/grpo_fast_genvalue.py \
    --exp_name "${EXP_NAME}" \
    --output_dir "${RUN_OUTPUT_DIR}" \
    --checkpoint_state_dir "${CHECKPOINT_STATE_DIR}" \
    --dataset_mixer_list hamishivi/DAPO-Math-17k-Processed_filtered 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 8192 \
    --pack_length 10240 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout "${NUM_UNIQUE_PROMPTS_ROLLOUT}" \
    --num_samples_per_prompt_rollout "${NUM_SAMPLES_PER_PROMPT_ROLLOUT}" \
    --filter_zero_std_samples false \
    --model_name_or_path Qwen/Qwen3-4B-Base \
    --chat_template_name qwen_instruct_user_boxed_math \
    --temperature 1.0 \
    --apply_verifiable_reward true \
    --verification_reward 1.0 \
    --remap_verifier math=final_boxed_math \
    --non_stop_penalty false \
    --beta 0.0 \
    --loss_fn dapo \
    --clip_higher 0.272 \
    --truncated_importance_sampling_ratio_cap 2.0 \
    --advantage_normalization_type centered \
    --learning_rate 1e-6 \
    --weight_decay 0.01 \
    --lr_scheduler_type constant \
    --total_episodes "${TOTAL_EPISODES}" \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --deepspeed_stage 3 \
    --deepspeed_offload_optimizer \
    --num_learners_per_node 1 \
    --sequence_parallel_size 1 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.85 \
    --vllm_top_p 1.0 \
    --vllm_enable_prefix_caching \
    --inflight_updates true \
    --async_steps 8 \
    --seed "${SEED}" \
    --local_eval_every -1 \
    --save_freq 100 \
    --checkpoint_state_freq 25 \
    --keep_last_n_checkpoints 2 \
    --gradient_checkpointing \
    --with_tracking \
    --push_to_hub false \
    --use_value_model \
    --value_warmup_steps "${VALUE_PRETRAIN_STEPS}" \
    --gae_lambda 1.0 \
    --gamma 1.0 \
    --use_sae \
    --sae_threshold 0.2 \
    --use_generative_value_model \
    --gen_value_model_name_or_path "${GEN_VALUE_MODEL_PATH}" \
    --gen_value_vllm_num_engines 1 \
    --gen_value_vllm_tensor_parallel_size 1 \
    --gen_value_segmentation sae \
    --gen_value_max_segments 16 \
    --gen_value_score_min 0 \
    --gen_value_score_max 10 \
    --gen_value_max_new_tokens 1024 \
    --gen_value_max_model_len 32768 \
    --gen_value_temperature 1.0 \
    --gen_value_conditioning "${GEN_VALUE_CONDITIONING}" \
    --gen_value_use_icc true \
    --gen_value_icc_momentum 0.9 \
    --gen_value_learning_rate 1e-6 \
    --gen_value_reinforce_coef 1.0 \
    --gen_value_reinforce_baseline "${GEN_VALUE_REINFORCE_BASELINE}" \
    --gen_value_final_action_replay_weight 4 \
    --gen_value_sync_freq 5 \
    --gen_value_diagnostic_scoring_freq 0 \
    --gen_value_validation_freq 25 \
    --gen_value_validation_max_examples 128 \
    --gen_value_validation_prompt_holdout_fraction 0.125 \
    --gen_value_model_snapshot_freq 25 \
    --gen_value_trace_reservoir_size 4096 \
    "$@"
