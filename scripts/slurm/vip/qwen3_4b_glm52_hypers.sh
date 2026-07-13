#!/usr/bin/env bash
#SBATCH --job-name=vip-math-glm52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
#
# Slurm port of scripts/train/vip/math_train_rl/qwen3_4b_glm52_hypers.sh
#
# Usage:
#   mkdir -p logs
#   sbatch scripts/slurm/vip/qwen3_4b_glm52_hypers.sh
#
# Optional overrides (export before sbatch, or edit #SBATCH above):
#   PARTITION / ACCOUNT via: sbatch --partition=... --account=... this_script.sh
#   EXP_NAME, RUN_NAME, WANDB_PROJECT, NUM_LEARNERS_PER_NODE,
#   VLLM_NUM_ENGINES, UV_SYNC, APPTAINER_IMAGE, APPTAINER_BIND,
#   APPTAINER_LOCAL_VENV, etc.

set -euo pipefail

# Move to the repo root (directory from which sbatch was invoked, or script parent).
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs

DDMM=$(date +"%d%m")
EXP_NAME="${EXP_NAME:-vip_glm52_hypers_${DDMM}_qwen3_4b_math}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${EXP_NAME}_$(date +%s)}"
NUM_LEARNERS_PER_NODE="${NUM_LEARNERS_PER_NODE:-4}"
VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-4}"

# Ensure Ray and the training dependencies are available inside the allocation.
# Set UV_SYNC=0 when reusing an already-synced environment. On clusters whose
# host glibc is too old for vLLM wheels, prepare .venv-container inside a modern
# Apptainer image and set APPTAINER_IMAGE to run every Slurm task in that image.
SRUN_PREFIX=()
if [[ -n "${APPTAINER_IMAGE:-}" ]]; then
  if [[ ! -f "${APPTAINER_IMAGE}" ]]; then
    echo "ERROR: APPTAINER_IMAGE does not exist: ${APPTAINER_IMAGE}" >&2
    exit 1
  fi
  CONTAINER_VENV="${REPO_ROOT}/.venv-container"
  APPTAINER_LOCAL_BIND=""
  if [[ "${APPTAINER_LOCAL_VENV:-0}" == "1" ]]; then
    CONTAINER_VENV="${APPTAINER_LOCAL_VENV_DIR:-/scr/${USER}/open-instruct-${SLURM_JOB_ID}/.venv}"
    APPTAINER_LOCAL_BIND="$(dirname "${CONTAINER_VENV}")"
    mkdir -p "${APPTAINER_LOCAL_BIND}"
    APPTAINER_SYNC_ARGS=(exec --nv)
    if [[ -n "${APPTAINER_BIND:-}" ]]; then
      APPTAINER_SYNC_ARGS+=(--bind "${APPTAINER_BIND}")
    fi
    APPTAINER_SYNC_ARGS+=(--bind "${APPTAINER_LOCAL_BIND}:${APPTAINER_LOCAL_BIND}")
    echo "Syncing the locked uv environment to node-local storage: ${CONTAINER_VENV}"
    apptainer "${APPTAINER_SYNC_ARGS[@]}" "${APPTAINER_IMAGE}" bash -c '
      set -euo pipefail
      cd "$1"
      export UV_PROJECT_ENVIRONMENT="$2"
      uv sync --frozen --no-dev
    ' _ "${REPO_ROOT}" "${CONTAINER_VENV}"
  fi
  if [[ ! -x "${CONTAINER_VENV}/bin/python" ]]; then
    echo "ERROR: ${CONTAINER_VENV} is not prepared" >&2
    exit 1
  fi
  export PATH="${CONTAINER_VENV}/bin:${PATH}"
  SRUN_PREFIX=(apptainer exec --nv --env "PREPEND_PATH=${CONTAINER_VENV}/bin")
  if [[ -n "${APPTAINER_BIND:-}" ]]; then
    SRUN_PREFIX+=(--bind "${APPTAINER_BIND}")
  fi
  if [[ -n "${APPTAINER_LOCAL_BIND}" ]]; then
    SRUN_PREFIX+=(--bind "${APPTAINER_LOCAL_BIND}:${APPTAINER_LOCAL_BIND}")
  fi
  SRUN_PREFIX+=("${APPTAINER_IMAGE}")
else
  if [[ "${UV_SYNC:-1}" == "1" ]]; then
    uv sync --frozen --no-dev
  fi
  export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
fi

echo "=== VIP math glm52 hypers (Slurm) ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Repo:   ${REPO_ROOT}"
echo "Exp:    ${EXP_NAME}"
echo "Run:    ${RUN_NAME}"
echo "Nodes:  ${SLURM_JOB_NUM_NODES}  GPUs/node: ${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-8}}"
echo "Actors: learners=${NUM_LEARNERS_PER_NODE}  vLLM engines=${VLLM_NUM_ENGINES}"
echo "===================================="

# Start Ray on every allocated node (head on rank 0, workers block).
srun --cpu-bind=none "${SRUN_PREFIX[@]}" bash -c '
  source scripts/slurm/vip/ray_node_setup.sh
  if [[ "${SLURM_PROCID:-0}" -eq 0 ]]; then
    python open_instruct/grpo_fast.py \
      --exp_name "'"${EXP_NAME}"'" \
      --run_name "'"${RUN_NAME}"'" \
      --beta 0.0 \
      --async_steps 8 \
      --inflight_updates \
      --no_resampling_pass_rate 0.875 \
      --use_vllm_logprobs True \
      --truncated_importance_sampling_ratio_cap 0 \
      --tis_mask_lower 0.8 \
      --tis_mask_upper 3.0 \
      --advantage_normalization_type centered \
      --num_samples_per_prompt_rollout 1 \
      --filter_zero_std_samples False \
      --num_unique_prompts_rollout 256 \
      --num_mini_batches 1 \
      --learning_rate 1e-6 \
      --per_device_train_batch_size 1 \
      --dataset_mixer_list hamishivi/DAPO-Math-17k-Processed_filtered 1.0 \
      --dataset_mixer_list_splits train \
      --max_prompt_token_length 2048 \
      --response_length 8192 \
      --pack_length 10240 \
      --model_name_or_path Qwen/Qwen3-4B-Base \
      --chat_template_name qwen_instruct_user_boxed_math \
      --non_stop_penalty False \
      --temperature 1.0 \
      --total_episodes 281600 \
      --deepspeed_stage 3 \
      --num_learners_per_node "'"${NUM_LEARNERS_PER_NODE}"'" \
      --num_nodes 1 \
      --sequence_parallel_size 1 \
      --vllm_num_engines "'"${VLLM_NUM_ENGINES}"'" \
      --vllm_tensor_parallel_size 1 \
      --vllm_top_p 1.0 \
      --lr_scheduler_type constant \
      --apply_verifiable_reward true \
      --verification_reward 1.0 \
      --seed 1 \
      --local_eval_every 100 \
      --save_freq 100 \
      --gradient_checkpointing \
      --with_tracking \
      --push_to_hub False \
      --use_value_model \
      --value_learning_rate 2e-6 \
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
      --reset_optimizer_after_value_warmup
  fi
'

echo "=== Training complete (or head exited) ==="
