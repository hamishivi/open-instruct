#!/usr/bin/env bash
# Joint 300-step actor/critic run using a separately validated generative critic.
# The critic path must be the final `gen_value_model` directory produced by
# genac_math_value_pretrain_h200.sh.
#
# GPU layout: one learner, one policy vLLM, one critic vLLM, one critic trainer.
# total_episodes = 300 joint steps * 32 prompts * 8 samples = 76,800
# Two half-rate actor epochs make DAPO clipping active on the second pass while
# keeping the nominal per-rollout update scale close to one 1e-6 pass.
# Continue the raw-reward REINFORCE objective used during critic pretraining and
# in Algorithm 1 of GenAC.
set -euo pipefail

if [[ -z "${GEN_VALUE_MODEL_PATH:-}" ]]; then
    echo "ERROR: set GEN_VALUE_MODEL_PATH to the validated pretrained gen_value_model directory" >&2
    exit 1
fi
if [[ ! -d "${GEN_VALUE_MODEL_PATH}" ]]; then
    echo "ERROR: GEN_VALUE_MODEL_PATH does not exist: ${GEN_VALUE_MODEL_PATH}" >&2
    exit 1
fi

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

RAY_PORT="${RAY_PORT:-$((8000 + ${SLURM_JOB_ID:-0} % 1000))}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray-${USER}-${SLURM_JOB_ID:-local}}"
ray stop --force 2>/dev/null || true
ray start --head --port="${RAY_PORT}" --temp-dir="${RAY_TEMP_DIR}" --dashboard-host=0.0.0.0
trap 'ray stop --force' EXIT

mkdir -p "${HOME}/.triton/autotune"

EXP_NAME="${EXP_NAME:-genac-math-joint-h200}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${PWD}/outputs/${EXP_NAME}}"
CHECKPOINT_STATE_DIR="${CHECKPOINT_STATE_DIR:-${RUN_OUTPUT_DIR}/checkpoint_states}"
GEN_VALUE_CONDITIONING="${GEN_VALUE_CONDITIONING:-none}"

python open_instruct/grpo_fast_genvalue.py \
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
    --num_unique_prompts_rollout 32 \
    --num_samples_per_prompt_rollout 8 \
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
    --weight_decay 0.01 \
    --lr_scheduler_type constant \
    --total_episodes 76800 \
    --num_epochs 2 \
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
    --use_value_model \
    --value_warmup_steps 0 \
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
    --gen_value_reinforce_baseline none \
    --gen_value_final_action_replay_weight 4 \
    --gen_value_sync_freq 5 \
    --gen_value_diagnostic_scoring_freq 0 \
    --gen_value_validation_freq 25 \
    --gen_value_validation_max_examples 128 \
    --gen_value_validation_prompt_holdout_fraction 0.125 \
    --gen_value_trace_reservoir_size 4096 \
    "$@"
