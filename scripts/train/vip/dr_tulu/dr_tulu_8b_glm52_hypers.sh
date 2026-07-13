#!/bin/bash
# DR-TULU: answer-prefix value conditioning with decoupled length-adaptive GAE
# (no SAE), group size 1, skip-tool-outputs GAE, and TIS ratio masking.
#
# Compared to `dr_tulu_8b_sae_ap_from_value_model.sh` (glm52-style hypers):
#   * trains the value model for 100 warmup steps instead of loading a checkpoint
#   * group size 1 with prompts scaled 8 -> 256 (same overall bsz = 256)
#   * length-adaptive / decoupled GAE instead of SAE
#   * --skip_tool_outputs
#   * --use_vllm_logprobs + --tis_mask_lower 0.8 --tis_mask_upper 3.0
#     (TIS cap disabled; incompatible with use_vllm_logprobs)
#   * --value_num_epochs 2
#
# Step budget:
#   total_episodes = (100 value warmup + 2000 RL) * 256 prompts * 1 sample
#                  = 537600
#
# Launch via Beaker:
#   ./scripts/train/build_image_and_launch.sh scripts/train/vip/dr_tulu/dr_tulu_8b_glm52_hypers.sh

EXP_NAME="${EXP_NAME:-dr_tulu_8b_glm52_hypers}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"

MODEL_NAME_OR_PATH="Qwen/Qwen3.5-4B"
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

DATASETS="rl-research/dr-tulu-rl-data 1.0"
DATASET_SPLITS="train"

PRIORITY="${PRIORITY:-high}"

uv run mason.py \
    --task_name ${EXP_NAME} \
    --description "${RUN_NAME}" \
    --cluster "ai2/jupiter" \
    --workspace ai2/olmo-instruct \
    --priority ${PRIORITY} \
    --pure_docker_mode \
    --image ${BEAKER_IMAGE} \
    --preemptible \
    --num_nodes 2 \
    --gpus 8 \
    --no_auto_dataset_cache \
    --env RUBRIC_JUDGE_MODEL=gpt-4.1 \
    --env RUBRIC_GENERATION_MODEL=gpt-4.1 \
    --secret SERPER_API_KEY=hamishivi_SERPER_API_KEY \
    --secret S2_API_KEY=hamishivi_S2_API_KEY \
    --secret JINA_API_KEY=hamishivi_JINA_API_KEY \
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
    --use_vllm_logprobs True \
    --truncated_importance_sampling_ratio_cap 0 \
    --tis_mask_lower 0.8 \
    --tis_mask_upper 3.0 \
    --num_samples_per_prompt_rollout 1 \
    --num_unique_prompts_rollout 256 \
    --num_mini_batches 1 \
    --learning_rate 5e-7 \
    --per_device_train_batch_size 1 \
    --dataset_mixer_list $DATASETS \
    --dataset_mixer_list_splits $DATASET_SPLITS \
    --max_prompt_token_length 2048 \
    --response_length 16384 \
    --pack_length 18500 \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --non_stop_penalty False \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --sft_messages_key messages \
    --total_episodes 537600 \
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
    --tools serper_search jina_browse s2_search \
    --tool_call_names google_search browse_webpage snippet_search \
    --tool_configs '{}' '{}' '{}' \
    --pool_size 1024 \
    --system_prompt_override_file scripts/train/dr-tulu/dr_tulu_adjusted.txt \
    --max_steps 100 \
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
    --use_value_model \
    --value_learning_rate 5e-7 \
    --value_num_epochs 2 \
    --gae_lambda 0.95 \
    --decoupled_gae \
    --length_adaptive_gae \
    --skip_tool_outputs \
    --gamma 1.0 \
    --value_loss_coef 0.5 \
    --vf_clip_range 0.2 \
    --value_model_ground_truth_conditioning \
    --gt_conditioning_template answer_prefix \
    --value_warmup_steps 100 \
    --reset_optimizer_after_value_warmup \
    --push_to_hub False
