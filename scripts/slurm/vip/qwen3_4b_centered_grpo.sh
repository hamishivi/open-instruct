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
# Pure centered-GRPO Slurm launcher derived from the VIP GLM52 recipe.
#
# Usage:
#   mkdir -p logs
#   sbatch scripts/slurm/vip/qwen3_4b_centered_grpo.sh
#
# Optional overrides (export before sbatch, or edit #SBATCH above):
#   PARTITION / ACCOUNT via: sbatch --partition=... --account=... this_script.sh
#   EXP_NAME, RUN_NAME, WANDB_PROJECT, NUM_LEARNERS_PER_NODE,
#   VLLM_NUM_ENGINES, ASYNC_STEPS, POLICY_LEARNING_RATE,
#   ACTIVE_SAMPLING, FILTER_ZERO_STD_SAMPLES,
#   NUM_SAMPLES_PER_PROMPT_ROLLOUT,
#   NUM_UNIQUE_PROMPTS_ROLLOUT, NO_RESAMPLING_PASS_RATE,
#   USE_VLLM_LOGPROBS, TRUNCATED_IMPORTANCE_SAMPLING_RATIO_CAP,
#   TIS_MASK_LOWER, TIS_MASK_UPPER,
#   DATASET_NAME, DATASET_WEIGHT, SFT_MESSAGES_KEY, GROUND_TRUTHS_KEY,
#   VERIFIER_SOURCE_KEY, HINTS_KEY, CHAT_TEMPLATE_NAME,
#   TOTAL_EPISODES, UV_SYNC, APPTAINER_IMAGE,
#   CHECKPOINT_STATE_FREQ, CHECKPOINT_STATE_DIR,
#   APPTAINER_BIND,
#   APPTAINER_LOCAL_VENV, VLLM_ALLOW_INSECURE_SERIALIZATION, etc.

set -euo pipefail

# Move to the repo root (directory from which sbatch was invoked, or script parent).
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs

DDMM=$(date +"%d%m")
EXP_NAME="${EXP_NAME:-vip_grpo_centered_g8_p32_as_zvf_tis2_${DDMM}_q3_4b}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${EXP_NAME}_$(date +%s)}"
# vLLM 0.19's fast serializer cannot encode torch.dtype objects emitted by the
# weight-transfer engine. The fallback is safe here because serialization stays
# within this trusted, single-user Slurm allocation.
export VLLM_ALLOW_INSECURE_SERIALIZATION="${VLLM_ALLOW_INSECURE_SERIALIZATION:-1}"
# Hyak's default proxy bypass list includes localhost but not its numeric
# loopback address. vLLM's internal OpenAI client uses 127.0.0.1 explicitly.
export no_proxy="127.0.0.1,localhost${no_proxy:+,${no_proxy}}"
export NO_PROXY="${no_proxy}"
NUM_LEARNERS_PER_NODE="${NUM_LEARNERS_PER_NODE:-4}"
VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-4}"
ASYNC_STEPS="${ASYNC_STEPS:-2}"
POLICY_LEARNING_RATE="${POLICY_LEARNING_RATE:-1e-6}"
ACTIVE_SAMPLING="${ACTIVE_SAMPLING:-false}"
FILTER_ZERO_STD_SAMPLES="${FILTER_ZERO_STD_SAMPLES:-false}"
NUM_SAMPLES_PER_PROMPT_ROLLOUT="${NUM_SAMPLES_PER_PROMPT_ROLLOUT:-1}"
NUM_UNIQUE_PROMPTS_ROLLOUT="${NUM_UNIQUE_PROMPTS_ROLLOUT:-128}"
NO_RESAMPLING_PASS_RATE="${NO_RESAMPLING_PASS_RATE:-none}"
USE_VLLM_LOGPROBS="${USE_VLLM_LOGPROBS:-true}"
TRUNCATED_IMPORTANCE_SAMPLING_RATIO_CAP="${TRUNCATED_IMPORTANCE_SAMPLING_RATIO_CAP:-0}"
TIS_MASK_LOWER="${TIS_MASK_LOWER:-0.8}"
TIS_MASK_UPPER="${TIS_MASK_UPPER:-3.0}"
DATASET_NAME="${DATASET_NAME:-hamishivi/DAPO-Math-17k-Processed_filtered}"
DATASET_WEIGHT="${DATASET_WEIGHT:-1.0}"
SFT_MESSAGES_KEY="${SFT_MESSAGES_KEY:-messages}"
GROUND_TRUTHS_KEY="${GROUND_TRUTHS_KEY:-ground_truth}"
VERIFIER_SOURCE_KEY="${VERIFIER_SOURCE_KEY:-dataset}"
HINTS_KEY="${HINTS_KEY:-hint}"
CHAT_TEMPLATE_NAME="${CHAT_TEMPLATE_NAME-qwen_instruct_user_boxed_math}"
TOTAL_EPISODES="${TOTAL_EPISODES:-281600}"
WANDB_ENTITY_NAME="${WANDB_ENTITY:-hamishivi}"
WANDB_PROJECT_NAME="${WANDB_PROJECT:-VIP}"
CHECKPOINT_STATE_FREQ="${CHECKPOINT_STATE_FREQ:-50}"
CHECKPOINT_STATE_DIR="${CHECKPOINT_STATE_DIR:-/gscratch/h2lab/${USER}/tmp/checkpoint_states/${RUN_NAME}}"

for bool_var in ACTIVE_SAMPLING FILTER_ZERO_STD_SAMPLES USE_VLLM_LOGPROBS; do
  case "${!bool_var}" in
    true|false) ;;
    *)
      echo "ERROR: ${bool_var} must be true or false, got: ${!bool_var}" >&2
      exit 2
      ;;
  esac
done
if [[ "${ACTIVE_SAMPLING}" == "true" && "${FILTER_ZERO_STD_SAMPLES}" != "true" ]]; then
  echo "ERROR: ACTIVE_SAMPLING=true requires FILTER_ZERO_STD_SAMPLES=true" >&2
  exit 2
fi
if ! [[ "${CHECKPOINT_STATE_FREQ}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: CHECKPOINT_STATE_FREQ must be a positive integer, got: ${CHECKPOINT_STATE_FREQ}" >&2
  exit 2
fi
mkdir -p "$(dirname "${CHECKPOINT_STATE_DIR}")"

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
    APPTAINER_UV_CACHE_DIR="${APPTAINER_UV_CACHE_DIR:-/gscratch/h2lab/${USER}/uv-cache}"
    mkdir -p "${APPTAINER_LOCAL_BIND}"
    APPTAINER_SYNC_ARGS=(exec --nv)
    if [[ -n "${APPTAINER_BIND:-}" ]]; then
      APPTAINER_SYNC_ARGS+=(--bind "${APPTAINER_BIND}")
    fi
    # uv needs the Git executable whenever a locked Git dependency is not
    # already available in its artifact cache. The training image intentionally
    # omits Git, so expose the host binary and its helper programs during sync.
    APPTAINER_SYNC_ARGS+=(
      --bind /usr/bin/git:/usr/bin/git
      --bind /usr/libexec/git-core:/usr/libexec/git-core
      --bind /lib64/libcrypto.so.1.1:/lib64/libcrypto.so.1.1
      --env LD_LIBRARY_PATH=/lib64:/.singularity.d/libs:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
      --env "UV_CACHE_DIR=${APPTAINER_UV_CACHE_DIR}"
      --env UV_LINK_MODE=copy
    )
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
  # Keep model, dataset, and Xet downloads off the small home filesystem. The
  # gscratch mount is shared across jobs, so subsequent launches also reuse the
  # downloaded artifacts instead of filling node-local or home storage.
  APPTAINER_HF_HOME="${APPTAINER_HF_HOME:-/gscratch/h2lab/${USER}/huggingface}"
  APPTAINER_HF_HUB_CACHE="${APPTAINER_HF_HUB_CACHE:-${APPTAINER_HF_HOME}/hub}"
  APPTAINER_HF_DATASETS_CACHE="${APPTAINER_HF_DATASETS_CACHE:-${APPTAINER_HF_HOME}/datasets}"
  APPTAINER_HF_XET_CACHE="${APPTAINER_HF_XET_CACHE:-${APPTAINER_HF_HOME}/xet}"
  mkdir -p \
    "${APPTAINER_HF_HUB_CACHE}" \
    "${APPTAINER_HF_DATASETS_CACHE}" \
    "${APPTAINER_HF_XET_CACHE}"
  SRUN_PREFIX+=(
    --env "HF_HOME=${APPTAINER_HF_HOME}"
    --env "HF_HUB_CACHE=${APPTAINER_HF_HUB_CACHE}"
    --env "HF_DATASETS_CACHE=${APPTAINER_HF_DATASETS_CACHE}"
    --env "HF_XET_CACHE=${APPTAINER_HF_XET_CACHE}"
  )
  # Compilation artifacts are large, highly environment-specific, and cheap to
  # recreate. Keep them on node-local storage instead of the quota-limited home
  # directory or shared gscratch. When the venv is node-local, reuse its bound
  # parent so every Ray worker sees the same cache paths inside the container.
  if [[ -n "${APPTAINER_LOCAL_BIND}" ]]; then
    APPTAINER_RUNTIME_CACHE_DIR="${APPTAINER_RUNTIME_CACHE_DIR:-${APPTAINER_LOCAL_BIND}/cache}"
  else
    APPTAINER_RUNTIME_CACHE_DIR="${APPTAINER_RUNTIME_CACHE_DIR:-/tmp/open-instruct-${USER}-${SLURM_JOB_ID:-local}/cache}"
  fi
  APPTAINER_VLLM_CACHE_ROOT="${APPTAINER_VLLM_CACHE_ROOT:-${APPTAINER_RUNTIME_CACHE_DIR}/vllm}"
  APPTAINER_TORCHINDUCTOR_CACHE_DIR="${APPTAINER_TORCHINDUCTOR_CACHE_DIR:-${APPTAINER_RUNTIME_CACHE_DIR}/torchinductor}"
  APPTAINER_TRITON_CACHE_DIR="${APPTAINER_TRITON_CACHE_DIR:-${APPTAINER_RUNTIME_CACHE_DIR}/triton}"
  APPTAINER_CUDA_CACHE_PATH="${APPTAINER_CUDA_CACHE_PATH:-${APPTAINER_RUNTIME_CACHE_DIR}/cuda}"
  APPTAINER_XDG_CACHE_HOME="${APPTAINER_XDG_CACHE_HOME:-${APPTAINER_RUNTIME_CACHE_DIR}/xdg}"
  # Ray nests long session/socket names beneath TMPDIR, and AF_UNIX paths are
  # limited to 107 bytes. Use a deliberately short node-local path.
  APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/tmp/oi-${SLURM_JOB_ID:-local}}"
  mkdir -p \
    "${APPTAINER_VLLM_CACHE_ROOT}" \
    "${APPTAINER_TORCHINDUCTOR_CACHE_DIR}" \
    "${APPTAINER_TRITON_CACHE_DIR}" \
    "${APPTAINER_CUDA_CACHE_PATH}" \
    "${APPTAINER_XDG_CACHE_HOME}" \
    "${APPTAINER_TMPDIR}"
  SRUN_PREFIX+=(
    --env "VLLM_CACHE_ROOT=${APPTAINER_VLLM_CACHE_ROOT}"
    --env "TORCHINDUCTOR_CACHE_DIR=${APPTAINER_TORCHINDUCTOR_CACHE_DIR}"
    --env "TRITON_CACHE_DIR=${APPTAINER_TRITON_CACHE_DIR}"
    --env "CUDA_CACHE_PATH=${APPTAINER_CUDA_CACHE_PATH}"
    --env "XDG_CACHE_HOME=${APPTAINER_XDG_CACHE_HOME}"
    --env "TMPDIR=${APPTAINER_TMPDIR}"
  )
  # The locked PyTorch stack ships CUDA 12.8 libraries. Do not let a newer
  # container toolkit (for example CUDA 12.9) override those wheel libraries;
  # retain only Apptainer's host-driver bind on the global lookup path.
  APPTAINER_LD_LIBRARY_PATH="${APPTAINER_LD_LIBRARY_PATH:-/.singularity.d/libs}"
  SRUN_PREFIX+=(--env "LD_LIBRARY_PATH=${APPTAINER_LD_LIBRARY_PATH}")
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
echo "Async:  steps=${ASYNC_STEPS}"
echo "Rollout: samples_per_prompt=${NUM_SAMPLES_PER_PROMPT_ROLLOUT}  unique_prompts=${NUM_UNIQUE_PROMPTS_ROLLOUT}  active_sampling=${ACTIVE_SAMPLING}  filter_zero_std=${FILTER_ZERO_STD_SAMPLES}  no_resampling_pass_rate=${NO_RESAMPLING_PASS_RATE}"
echo "Off-policy correction: use_vllm_logprobs=${USE_VLLM_LOGPROBS}  tis_cap=${TRUNCATED_IMPORTANCE_SAMPLING_RATIO_CAP}  hard_ratio_mask=[${TIS_MASK_LOWER},${TIS_MASK_UPPER}]"
echo "Schedule: total_episodes=${TOTAL_EPISODES}"
echo "Dataset: ${DATASET_NAME} weight=${DATASET_WEIGHT} messages=${SFT_MESSAGES_KEY} ground_truth=${GROUND_TRUTHS_KEY} verifier=${VERIFIER_SOURCE_KEY} hints=${HINTS_KEY}"
echo "Chat template: ${CHAT_TEMPLATE_NAME:-model default}"
echo "Algorithm: pure centered GRPO (no critic/value model, GAE, SAE, or value warmup)"
echo "Tuning: policy_lr=${POLICY_LEARNING_RATE}"
echo "Resume state: every ${CHECKPOINT_STATE_FREQ} steps -> ${CHECKPOINT_STATE_DIR}"
echo "===================================="

# Start Ray on every allocated node (head on rank 0, workers block).
export \
  EXP_NAME RUN_NAME ASYNC_STEPS USE_VLLM_LOGPROBS \
  TRUNCATED_IMPORTANCE_SAMPLING_RATIO_CAP TIS_MASK_LOWER TIS_MASK_UPPER \
  ACTIVE_SAMPLING NUM_SAMPLES_PER_PROMPT_ROLLOUT NO_RESAMPLING_PASS_RATE \
  FILTER_ZERO_STD_SAMPLES NUM_UNIQUE_PROMPTS_ROLLOUT POLICY_LEARNING_RATE \
  DATASET_NAME DATASET_WEIGHT SFT_MESSAGES_KEY GROUND_TRUTHS_KEY \
  VERIFIER_SOURCE_KEY HINTS_KEY CHAT_TEMPLATE_NAME \
  TOTAL_EPISODES NUM_LEARNERS_PER_NODE VLLM_NUM_ENGINES \
  WANDB_ENTITY_NAME WANDB_PROJECT_NAME \
  CHECKPOINT_STATE_FREQ CHECKPOINT_STATE_DIR
srun --cpu-bind=none "${SRUN_PREFIX[@]}" bash -c '
  source scripts/slurm/vip/ray_node_setup.sh
  if [[ "${SLURM_PROCID:-0}" -eq 0 ]]; then
    train_args=(
      --exp_name "${EXP_NAME}"
      --run_name "${RUN_NAME}"
      --beta 0.0 \
      --async_steps "${ASYNC_STEPS}"
      --inflight_updates \
      --use_vllm_logprobs "${USE_VLLM_LOGPROBS}"
      --truncated_importance_sampling_ratio_cap "${TRUNCATED_IMPORTANCE_SAMPLING_RATIO_CAP}"
      --tis_mask_lower "${TIS_MASK_LOWER}"
      --tis_mask_upper "${TIS_MASK_UPPER}"
      --clip_lower 0.2
      --clip_higher 0.272
      --advantage_normalization_type centered \
      --active_sampling "${ACTIVE_SAMPLING}"
      --num_samples_per_prompt_rollout "${NUM_SAMPLES_PER_PROMPT_ROLLOUT}"
      --filter_zero_std_samples "${FILTER_ZERO_STD_SAMPLES}"
      --num_unique_prompts_rollout "${NUM_UNIQUE_PROMPTS_ROLLOUT}"
      --num_mini_batches 1 \
      --learning_rate "${POLICY_LEARNING_RATE}"
      --per_device_train_batch_size 1 \
      --dataset_mixer_list "${DATASET_NAME}" "${DATASET_WEIGHT}"
      --dataset_mixer_list_splits train \
      --sft_messages_key "${SFT_MESSAGES_KEY}"
      --ground_truths_key "${GROUND_TRUTHS_KEY}"
      --dataset_source_key "${VERIFIER_SOURCE_KEY}"
      --hints_key "${HINTS_KEY}"
      --max_prompt_token_length 2048 \
      --response_length 8192 \
      --pack_length 10240 \
      --model_name_or_path Qwen/Qwen3-4B-Base \
      --non_stop_penalty False \
      --temperature 1.0 \
      --total_episodes "${TOTAL_EPISODES}"
      --checkpoint_state_freq "${CHECKPOINT_STATE_FREQ}"
      --checkpoint_state_dir "${CHECKPOINT_STATE_DIR}"
      --deepspeed_stage 3 \
      --num_learners_per_node "${NUM_LEARNERS_PER_NODE}"
      --num_nodes 1 \
      --sequence_parallel_size 1 \
      --vllm_num_engines "${VLLM_NUM_ENGINES}"
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
      --wandb_entity "${WANDB_ENTITY_NAME}"
      --wandb_project "${WANDB_PROJECT_NAME}"
      --push_to_hub False
    )
    if [[ "${NO_RESAMPLING_PASS_RATE}" != "none" ]]; then
      train_args+=(--no_resampling_pass_rate "${NO_RESAMPLING_PASS_RATE}")
    fi
    if [[ -n "${CHAT_TEMPLATE_NAME}" ]]; then
      train_args+=(--chat_template_name "${CHAT_TEMPLATE_NAME}")
    fi
    python open_instruct/grpo_fast.py "${train_args[@]}"
  fi
'

echo "=== Training complete (or head exited) ==="
