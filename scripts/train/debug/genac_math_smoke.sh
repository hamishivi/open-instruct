#!/usr/bin/env bash
# Short, realistic GenAC validation using the established Qwen3-4B DAPO-math
# setup. It preserves the production rollout geometry and context lengths while
# reducing the schedule to 2 critic-warmup steps plus 10 joint updates.
#
# GPU layout (8 GPUs on one node):
#   - 2 policy/scalar-critic learners (DeepSpeed stage 3)
#   - 2 policy vLLM engines
#   - 3 generative-critic vLLM engines
#   - 1 generative-critic DeepSpeed trainer
#
# total_episodes = (2 warmup + 10 joint) * 32 prompts * 8 samples = 3072
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

ray stop --force 2>/dev/null || true
ray start --head --port=8888 --dashboard-host=0.0.0.0
trap 'ray stop --force' EXIT

mkdir -p "${HOME}/.triton/autotune"

EXP_NAME="${EXP_NAME:-genac_math_smoke}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-/tmp/genac_math_smoke_output}"

python open_instruct/grpo_fast_genvalue.py \
    --exp_name "${EXP_NAME}" \
    --output_dir "${RUN_OUTPUT_DIR}" \
    --dataset_mixer_list hamishivi/DAPO-Math-17k-Processed_filtered 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 8192 \
    --pack_length 10240 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 32 \
    --num_samples_per_prompt_rollout 8 \
    --active_sampling \
    --no_resampling_pass_rate 0.875 \
    --model_name_or_path Qwen/Qwen3-4B-Base \
    --chat_template_name qwen_instruct_user_boxed_math \
    --temperature 1.0 \
    --apply_verifiable_reward true \
    --verification_reward 1.0 \
    --non_stop_penalty false \
    --beta 0.0 \
    --loss_fn dapo \
    --clip_higher 0.272 \
    --truncated_importance_sampling_ratio_cap 2.0 \
    --advantage_normalization_type centered \
    --learning_rate 1e-6 \
    --lr_scheduler_type constant \
    --total_episodes 3072 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --deepspeed_stage 3 \
    --num_learners_per_node 2 \
    --sequence_parallel_size 1 \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.85 \
    --vllm_top_p 1.0 \
    --vllm_enable_prefix_caching \
    --inflight_updates true \
    --async_steps 8 \
    --seed 1 \
    --local_eval_every -1 \
    --save_freq 100 \
    --gradient_checkpointing \
    --with_tracking \
    --push_to_hub false \
    --use_value_model \
    --value_learning_rate 2e-6 \
    --value_warmup_steps 2 \
    --reset_optimizer_after_value_warmup \
    --gae_lambda 0.95 \
    --gamma 1.0 \
    --value_loss_coef 0.5 \
    --vf_clip_range 0.2 \
    --use_sae \
    --sae_threshold 0.2 \
    --value_model_ground_truth_conditioning \
    --gt_conditioning_template answer_prefix \
    --use_generative_value_model \
    --gen_value_model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --gen_value_vllm_num_engines 3 \
    --gen_value_vllm_tensor_parallel_size 1 \
    --gen_value_segmentation sae \
    --gen_value_max_segments 16 \
    --gen_value_score_min 0 \
    --gen_value_score_max 10 \
    --gen_value_max_new_tokens 1024 \
    --gen_value_max_model_len 32768 \
    --gen_value_temperature 1.0 \
    --gen_value_conditioning gt \
    --gen_value_learning_rate 1e-6 \
    --gen_value_sync_freq 1 \
    "$@"
