#!/usr/bin/env bash
#SBATCH --job-name=vip-drtulu-glm52
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --time=72:00:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
#
# Slurm port of scripts/train/vip/dr_tulu/dr_tulu_8b_glm52_hypers.sh
#
# Prerequisites: export tool/judge API keys in the environment (or your
# ~/.bashrc / module setup), e.g.:
#   SERPER_API_KEY, S2_API_KEY, JINA_API_KEY, OPENAI_API_KEY
#   RUBRIC_JUDGE_MODEL=gpt-4.1 RUBRIC_GENERATION_MODEL=gpt-4.1
#
# Usage:
#   mkdir -p logs
#   sbatch scripts/slurm/vip/dr_tulu_8b_glm52_hypers.sh

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs

EXP_NAME="${EXP_NAME:-dr_tulu_8b_glm52_hypers}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${EXP_NAME}_$(date +%s)}"
export RUBRIC_JUDGE_MODEL="${RUBRIC_JUDGE_MODEL:-gpt-4.1}"
export RUBRIC_GENERATION_MODEL="${RUBRIC_GENERATION_MODEL:-gpt-4.1}"

for key in SERPER_API_KEY S2_API_KEY JINA_API_KEY OPENAI_API_KEY; do
  if [[ -z "${!key:-}" ]]; then
    echo "WARNING: ${key} is unset; tool / rubric calls may fail."
  fi
done

echo "=== VIP Dr. Tulu glm52 hypers (Slurm) ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Repo:   ${REPO_ROOT}"
echo "Exp:    ${EXP_NAME}"
echo "Run:    ${RUN_NAME}"
echo "Nodes:  ${SLURM_JOB_NUM_NODES}  GPUs/node: ${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-8}}"
echo "========================================"

# Start Ray on every allocated node (head on rank 0 runs training; workers block).
srun --cpu-bind=none bash -c '
  source scripts/slurm/vip/ray_node_setup.sh
  if [[ "${SLURM_PROCID:-0}" -eq 0 ]]; then
    uv run python open_instruct/grpo_fast.py \
      --run_name "'"${RUN_NAME}"'" \
      --exp_name "'"${EXP_NAME}"'" \
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
      --dataset_mixer_list rl-research/dr-tulu-rl-data 1.0 \
      --dataset_mixer_list_splits train \
      --max_prompt_token_length 2048 \
      --response_length 16384 \
      --pack_length 18500 \
      --model_name_or_path Qwen/Qwen3.5-4B \
      --non_stop_penalty False \
      --temperature 1.0 \
      --ground_truths_key ground_truth \
      --sft_messages_key messages \
      --total_episodes 537600 \
      --deepspeed_stage 3 \
      --num_learners_per_node 8 \
      --num_nodes 2 \
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
      --tool_configs "{}" "{}" "{}" \
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
  fi
'

echo "=== Training complete (or head exited) ==="
