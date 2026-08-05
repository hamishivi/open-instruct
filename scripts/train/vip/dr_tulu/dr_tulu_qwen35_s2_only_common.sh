#!/bin/bash
# Shared Qwen3.5-4B DR-Tulu launcher for the S2-only GRPO/SAE ablation.
# Invoke through one of the two thin variant launchers in this directory.

set -euo pipefail

: "${VARIANT:?Set VARIANT to grpo or sae_critic_whiten}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3.5-4B}"
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

DATASETS="${DATASETS:-rl-research/dr-tulu-rl-data 1.0}"
DATASET_SPLITS="${DATASET_SPLITS:-train}"
PRIORITY="${PRIORITY:-high}"
RL_STEPS="${RL_STEPS:-4000}"
VALUE_WARMUP_STEPS="${VALUE_WARMUP_STEPS:-100}"
ROLLOUT_BATCH_SIZE=$((8 * 32))

case "${VARIANT}" in
    grpo)
        EXP_NAME="${EXP_NAME:-dr_tulu_q35_4b_s2_grpo}"
        TOTAL_EPISODES=$((RL_STEPS * ROLLOUT_BATCH_SIZE))
        VALUE_ARGS=()
        ;;
    sae_critic_whiten)
        EXP_NAME="${EXP_NAME:-dr_tulu_q35_4b_s2_sae_whiten}"
        TOTAL_EPISODES=$(((RL_STEPS + VALUE_WARMUP_STEPS) * ROLLOUT_BATCH_SIZE))
        VALUE_ARGS=(
            --use_value_model
            --value_learning_rate 5e-7
            --value_num_epochs 1
            --value_loss mse
            --gae_lambda 0.95
            --decoupled_gae
            --skip_tool_outputs True
            --gamma 1.0
            --value_loss_coef 0.5
            --vf_clip_range 0.2
            --use_sae
            --sae_threshold 0.2
            --whiten_advantages
            --value_warmup_steps "${VALUE_WARMUP_STEPS}"
            --reset_optimizer_after_value_warmup
        )
        ;;
    *)
        echo "Unknown VARIANT=${VARIANT}; expected grpo or sae_critic_whiten" >&2
        exit 2
        ;;
esac

RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"

uv run mason.py \
    --task_name "${EXP_NAME}" \
    --description "${RUN_NAME}" \
    --cluster ai2/jupiter \
    --workspace ai2/dr-tulu-ablations \
    --priority "${PRIORITY}" \
    --pure_docker_mode \
    --image "${BEAKER_IMAGE}" \
    --preemptible \
    --num_nodes 2 \
    --gpus 8 \
    --no_auto_dataset_cache \
    --env RUBRIC_JUDGE_MODEL=gpt-4.1 \
    --env RUBRIC_GENERATION_MODEL=gpt-4.1 \
    --secret S2_API_KEY=hamishivi_S2_API_KEY \
    --secret OPENAI_API_KEY=hamishivi_OPENAI_API_KEY \
    -- \
source configs/beaker_configs/ray_node_setup.sh \
\&\& uv run open_instruct/grpo_fast.py \
    --run_name "${RUN_NAME}" \
    --exp_name "${EXP_NAME}" \
    --beta 0.001 \
    --load_ref_policy True \
    --async_steps 4 \
    --active_sampling \
    --inflight_updates \
    --advantage_normalization_type centered \
    --num_samples_per_prompt_rollout 32 \
    --num_unique_prompts_rollout 8 \
    --num_mini_batches 1 \
    --learning_rate 5e-7 \
    --per_device_train_batch_size 1 \
    --dataset_mixer_list ${DATASETS} \
    --dataset_mixer_list_splits ${DATASET_SPLITS} \
    --dataset_mixer_eval_list rl-research/dr-tulu-rl-data 8 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 16384 \
    --pack_length 18500 \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --non_stop_penalty False \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --sft_messages_key messages \
    --total_episodes "${TOTAL_EPISODES}" \
    --deepspeed_stage 3 \
    --num_learners_per_node 8 \
    --sequence_parallel_size 1 \
    --vllm_num_engines 8 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --apply_evolving_rubric_reward true \
    --max_active_rubrics 5 \
    --remap_verifier general_rubric=rubric \
    --tool_parser_type vllm_qwen3_xml \
    --tools s2_search \
    --tool_call_names snippet_search \
    --tool_configs '{"num_results": 10}' \
    --pool_size 256 \
    --system_prompt_override_file scripts/train/vip/dr_tulu/dr_tulu_s2_only.txt \
    --max_steps 10 \
    --backend_timeout 1800 \
    --save_traces \
    --seed 1 \
    --local_eval_every 100 \
    --save_freq 100 \
    --checkpoint_state_freq 100 \
    --gradient_checkpointing \
    --with_tracking \
    --vllm_enable_prefix_caching \
    --vllm_gdn_prefill_backend triton \
    --keep_last_n_checkpoints -1 \
    --kl_estimator 3 \
    --push_to_hub False \
    ${VALUE_ARGS[@]+"${VALUE_ARGS[@]}"}
