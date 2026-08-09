#!/usr/bin/env bash
# Shared Slurm body for the S2-only Qwen3-4B-Instruct DR-Tulu ablation.

set -euo pipefail

: "${VARIANT:?Set VARIANT to grpo, sae_critic_whiten, or sae_critic_whiten_rubrics}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
DATASET_NAME="${DATASET_NAME:-rl-research/dr-tulu-rl-data}"
DATASET_WEIGHT="${DATASET_WEIGHT:-1.0}"
RL_STEPS="${RL_STEPS:-4000}"
VALUE_WARMUP_STEPS="${VALUE_WARMUP_STEPS:-100}"
NUM_UNIQUE_PROMPTS_ROLLOUT="${NUM_UNIQUE_PROMPTS_ROLLOUT:-8}"
NUM_SAMPLES_PER_PROMPT_ROLLOUT="${NUM_SAMPLES_PER_PROMPT_ROLLOUT:-32}"
ROLLOUT_BATCH_SIZE=$((NUM_UNIQUE_PROMPTS_ROLLOUT * NUM_SAMPLES_PER_PROMPT_ROLLOUT))
NUM_NODES="${SLURM_JOB_NUM_NODES:-1}"
NUM_LEARNERS_PER_NODE="${NUM_LEARNERS_PER_NODE:-2}"
VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-2}"
CHECKPOINT_STATE_FREQ="${CHECKPOINT_STATE_FREQ:-50}"
WANDB_ENTITY_NAME="${WANDB_ENTITY:-hamishivi}"
WANDB_PROJECT_NAME="${WANDB_PROJECT:-dr-tulu-ablations}"
TOOL_CONFIG='{"num_results": 10}'

case "${VARIANT}" in
  grpo)
    EXP_NAME="${EXP_NAME:-dr_tulu_q3_4b_inst_s2_grpo}"
    TOTAL_EPISODES=$((RL_STEPS * ROLLOUT_BATCH_SIZE))
    ;;
  sae_critic_whiten)
    EXP_NAME="${EXP_NAME:-dr_tulu_q3_4b_inst_s2_sae_whiten}"
    TOTAL_EPISODES=$(((RL_STEPS + VALUE_WARMUP_STEPS) * ROLLOUT_BATCH_SIZE))
    ;;
  sae_critic_whiten_rubrics)
    EXP_NAME="${EXP_NAME:-dr_tulu_q3_4b_inst_s2_sae_whiten_rubrics}"
    TOTAL_EPISODES=$(((RL_STEPS + VALUE_WARMUP_STEPS) * ROLLOUT_BATCH_SIZE))
    ;;
  *)
    echo "ERROR: unknown VARIANT=${VARIANT}; expected grpo, sae_critic_whiten, or sae_critic_whiten_rubrics" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT_STATE_DIR="${CHECKPOINT_STATE_DIR:-/gscratch/h2lab/${USER}/tmp/checkpoint_states/${RUN_NAME}}"
ROLLOUTS_SAVE_PATH="${ROLLOUTS_SAVE_PATH:-/gscratch/h2lab/${USER}/rollouts/${RUN_NAME}}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${RUN_NAME}}"
export RUBRIC_JUDGE_MODEL="${RUBRIC_JUDGE_MODEL:-gpt-4.1}"
export RUBRIC_GENERATION_MODEL="${RUBRIC_GENERATION_MODEL:-gpt-4.1}"
export VLLM_ALLOW_INSECURE_SERIALIZATION="${VLLM_ALLOW_INSECURE_SERIALIZATION:-1}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,${no_proxy}}"
export NO_PROXY="${no_proxy}"

for key in S2_API_KEY OPENAI_API_KEY; do
  if [[ -z "${!key:-}" ]]; then
    echo "ERROR: ${key} must be exported before submission" >&2
    exit 2
  fi
done

if [[ "${NUM_UNIQUE_PROMPTS_ROLLOUT}" != 8 || "${NUM_SAMPLES_PER_PROMPT_ROLLOUT}" != 32 ]]; then
  echo "ERROR: this matched ablation requires 8 prompts and group size 32" >&2
  exit 2
fi
if ! [[ "${CHECKPOINT_STATE_FREQ}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: CHECKPOINT_STATE_FREQ must be a positive integer, got: ${CHECKPOINT_STATE_FREQ}" >&2
  exit 2
fi

mkdir -p "$(dirname "${CHECKPOINT_STATE_DIR}")" "${ROLLOUTS_SAVE_PATH}"

SRUN_PREFIX=()
if [[ -n "${APPTAINER_IMAGE:-}" ]]; then
  if [[ ! -f "${APPTAINER_IMAGE}" ]]; then
    echo "ERROR: APPTAINER_IMAGE does not exist: ${APPTAINER_IMAGE}" >&2
    exit 2
  fi
  CONTAINER_VENV="${CONTAINER_VENV:-${REPO_ROOT}/.venv-container}"
  APPTAINER_LOCAL_BIND=""
  if [[ "${APPTAINER_LOCAL_VENV:-0}" == 1 ]]; then
    CONTAINER_VENV="${APPTAINER_LOCAL_VENV_DIR:-/scr/${USER}/open-instruct-${SLURM_JOB_ID}/.venv}"
    APPTAINER_LOCAL_BIND="$(dirname "${CONTAINER_VENV}")"
    APPTAINER_UV_CACHE_DIR="${APPTAINER_UV_CACHE_DIR:-/gscratch/h2lab/${USER}/uv-cache}"
    mkdir -p "${APPTAINER_LOCAL_BIND}" "${APPTAINER_UV_CACHE_DIR}"
    APPTAINER_SYNC_ARGS=(exec --nv)
    if [[ -n "${APPTAINER_BIND:-}" ]]; then
      APPTAINER_SYNC_ARGS+=(--bind "${APPTAINER_BIND}")
    fi
    APPTAINER_SYNC_ARGS+=(
      --bind /usr/bin/git:/usr/bin/git
      --bind /usr/libexec/git-core:/usr/libexec/git-core
      --bind /lib64/libcrypto.so.1.1:/lib64/libcrypto.so.1.1
      --bind "${APPTAINER_LOCAL_BIND}:${APPTAINER_LOCAL_BIND}"
      --env LD_LIBRARY_PATH=/lib64:/.singularity.d/libs:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
      --env GIT_EXEC_PATH=/usr/libexec/git-core
      --env "UV_CACHE_DIR=${APPTAINER_UV_CACHE_DIR}"
      --env UV_LINK_MODE=copy
    )
    echo "Syncing the locked environment to node-local storage: ${CONTAINER_VENV}"
    apptainer "${APPTAINER_SYNC_ARGS[@]}" "${APPTAINER_IMAGE}" bash -c '
      set -euo pipefail
      cd "$1"
      export UV_PROJECT_ENVIRONMENT="$2"
      uv sync --frozen --no-dev
    ' _ "${REPO_ROOT}" "${CONTAINER_VENV}"
  fi
  if [[ ! -x "${CONTAINER_VENV}/bin/python" ]]; then
    echo "ERROR: prepare ${CONTAINER_VENV} with scripts/slurm/vip/setup_apptainer_env.sh first" >&2
    exit 2
  fi
  export PATH="${CONTAINER_VENV}/bin:${PATH}"
  SRUN_PREFIX=(apptainer exec --nv --env "PREPEND_PATH=${CONTAINER_VENV}/bin")
  APPTAINER_HF_HOME="${APPTAINER_HF_HOME:-${HF_HOME:-/gscratch/h2lab/${USER}/huggingface}}"
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
  # The locked PyTorch stack ships CUDA 12.8 libraries. Do not let the
  # container's CUDA 12.9 toolkit override those wheel libraries; retain only
  # Apptainer's host-driver bind on the global lookup path.
  APPTAINER_LD_LIBRARY_PATH="${APPTAINER_LD_LIBRARY_PATH:-/.singularity.d/libs}"
  SRUN_PREFIX+=(--env "LD_LIBRARY_PATH=${APPTAINER_LD_LIBRARY_PATH}")
  if [[ -n "${APPTAINER_BIND:-}" ]]; then
    SRUN_PREFIX+=(--bind "${APPTAINER_BIND}")
  fi
  if [[ -n "${APPTAINER_LOCAL_BIND}" ]]; then
    SRUN_PREFIX+=(--bind "${APPTAINER_LOCAL_BIND}:${APPTAINER_LOCAL_BIND}")
  fi
  SRUN_PREFIX+=("${APPTAINER_IMAGE}")
elif [[ "${UV_SYNC:-1}" == 1 ]]; then
  uv sync --frozen --no-dev
  export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
fi

export VARIANT EXP_NAME RUN_NAME MODEL_NAME_OR_PATH DATASET_NAME DATASET_WEIGHT
export TOTAL_EPISODES NUM_NODES NUM_LEARNERS_PER_NODE VLLM_NUM_ENGINES
export NUM_UNIQUE_PROMPTS_ROLLOUT NUM_SAMPLES_PER_PROMPT_ROLLOUT
export VALUE_WARMUP_STEPS CHECKPOINT_STATE_FREQ CHECKPOINT_STATE_DIR
export ROLLOUTS_SAVE_PATH WANDB_ENTITY_NAME WANDB_PROJECT_NAME TOOL_CONFIG

echo "=== Qwen3-4B-Instruct-2507 DR-Tulu S2-only ${VARIANT} ==="
echo "Job:      ${SLURM_JOB_ID:-local}"
echo "Run:      ${RUN_NAME}"
echo "Nodes:    ${NUM_NODES}; GPUs/node: ${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-4}}"
echo "Actors:   ${NUM_LEARNERS_PER_NODE} learners/node; ${VLLM_NUM_ENGINES} vLLM engines total"
echo "Rollout:  ${NUM_UNIQUE_PROMPTS_ROLLOUT} prompts x ${NUM_SAMPLES_PER_PROMPT_ROLLOUT} samples = ${ROLLOUT_BATCH_SIZE}"
echo "Episodes: ${TOTAL_EPISODES}"
echo "State:    ${CHECKPOINT_STATE_DIR} every ${CHECKPOINT_STATE_FREQ} steps"
echo "Traces:   ${ROLLOUTS_SAVE_PATH}"
echo "Tools:    snippet_search (Semantic Scholar) only"
echo "================================================="

srun --cpu-bind=none "${SRUN_PREFIX[@]}" bash -c '
  source scripts/slurm/vip/ray_node_setup.sh
  if [[ "${SLURM_PROCID:-0}" -eq 0 ]]; then
    value_args=()
    if [[ "${VARIANT}" == sae_critic_whiten || "${VARIANT}" == sae_critic_whiten_rubrics ]]; then
      value_args=(
        --use_value_model
        --value_learning_rate 5e-7
        --value_num_epochs 1
        --value_loss mse
        --gae_lambda 0.95
        --decoupled_gae
        --skip_tool_outputs
        --gamma 1.0
        --value_loss_coef 0.5
        --vf_clip_range 0.2
        --use_sae
        --sae_threshold 0.2
        --whiten_advantages
        --value_warmup_steps "${VALUE_WARMUP_STEPS}"
        --reset_optimizer_after_value_warmup
      )
      if [[ "${VARIANT}" == sae_critic_whiten_rubrics ]]; then
        value_args+=(
          --value_model_ground_truth_conditioning
          --gt_conditioning_template rubrics
        )
      fi
    fi

    python open_instruct/grpo_fast.py \
      --run_name "${RUN_NAME}" \
      --exp_name "${EXP_NAME}" \
      --beta 0.001 \
      --load_ref_policy True \
      --async_steps 4 \
      --active_sampling \
      --inflight_updates \
      --advantage_normalization_type centered \
      --num_samples_per_prompt_rollout "${NUM_SAMPLES_PER_PROMPT_ROLLOUT}" \
      --num_unique_prompts_rollout "${NUM_UNIQUE_PROMPTS_ROLLOUT}" \
      --num_mini_batches 1 \
      --learning_rate 5e-7 \
      --per_device_train_batch_size 1 \
      --dataset_mixer_list "${DATASET_NAME}" "${DATASET_WEIGHT}" \
      --dataset_mixer_list_splits train \
      --dataset_mixer_eval_list "${DATASET_NAME}" 8 \
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
      --checkpoint_state_freq "${CHECKPOINT_STATE_FREQ}" \
      --checkpoint_state_dir "${CHECKPOINT_STATE_DIR}" \
      --deepspeed_stage 3 \
      --num_learners_per_node "${NUM_LEARNERS_PER_NODE}" \
      --num_nodes "${NUM_NODES}" \
      --sequence_parallel_size 1 \
      --vllm_num_engines "${VLLM_NUM_ENGINES}" \
      --lr_scheduler_type constant \
      --apply_verifiable_reward true \
      --apply_evolving_rubric_reward true \
      --max_active_rubrics 5 \
      --remap_verifier general_rubric=rubric \
      --tool_parser_type vllm_hermes \
      --tools s2_search \
      --tool_call_names snippet_search \
      --tool_configs "${TOOL_CONFIG}" \
      --pool_size 256 \
      --system_prompt_override_file scripts/train/vip/dr_tulu/dr_tulu_s2_only.txt \
      --max_steps 10 \
      --backend_timeout 1800 \
      --save_traces \
      --rollouts_save_path "${ROLLOUTS_SAVE_PATH}" \
      --seed 1 \
      --local_eval_every 100 \
      --save_freq 100 \
      --gradient_checkpointing \
      --with_tracking \
      --wandb_entity "${WANDB_ENTITY_NAME}" \
      --wandb_project "${WANDB_PROJECT_NAME}" \
      --vllm_enable_prefix_caching \
      --keep_last_n_checkpoints -1 \
      --kl_estimator 3 \
      --push_to_hub False \
      ${value_args[@]+"${value_args[@]}"}
  fi
'

echo "=== Training complete (or Ray head exited) ==="
