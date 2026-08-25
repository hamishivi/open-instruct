#!/usr/bin/env bash
#SBATCH --account=gpu-h200-h2lab
#SBATCH --partition=gpu-h200
#SBATCH --job-name=drt-asta-sqa
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%A_%a.drt-asta-sqa.out
#SBATCH --error=logs/%A_%a.drt-asta-sqa.err

set -euo pipefail

: "${S2_API_KEY:?Export S2_API_KEY before submission}"
: "${HF_TOKEN:?Export HF_TOKEN with access to allenai/asta-bench}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
TRAINING_ROOT="${TRAINING_ROOT:-/mmfs1/gscratch/h2lab/hamishiv/open-instruct-dr-tulu-s2}"
MODEL_RUNS=(base grpo sae)
MODEL_RUN="${MODEL_RUN:-${MODEL_RUNS[${SLURM_ARRAY_TASK_ID:-0}]}}"
REQUIRE_CHECKPOINT_MARKER=1

case "${MODEL_RUN}" in
  base)
    MODEL_LABEL="${MODEL_LABEL:-qwen3_4b_instruct_2507_step_0}"
    MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
    REQUIRE_CHECKPOINT_MARKER=0
    ;;
  grpo)
    MODEL_LABEL="${MODEL_LABEL:-dr_tulu_q3_4b_inst_s2_grpo_step_1000}"
    MODEL_PATH="${MODEL_PATH:-${TRAINING_ROOT}/output/dr_tulu_q3_4b_inst_s2_grpo__1__1786035513_checkpoints/step_1000}"
    ;;
  sae)
    MODEL_LABEL="${MODEL_LABEL:-dr_tulu_q3_4b_inst_s2_sae_whiten_rubrics_step_1100}"
    MODEL_PATH="${MODEL_PATH:-${TRAINING_ROOT}/output/dr_tulu_q3_4b_inst_s2_sae_whiten_rubrics__1__1786890301}"
    ;;
  custom)
    : "${MODEL_LABEL:?Set MODEL_LABEL when MODEL_RUN=custom}"
    : "${MODEL_PATH:?Set MODEL_PATH when MODEL_RUN=custom}"
    ;;
  *)
    echo "ERROR: MODEL_RUN must be base, grpo, sae, or custom; got ${MODEL_RUN}" >&2
    exit 2
    ;;
esac

ASTA_ROOT="${ASTA_ROOT:-/gscratch/h2lab/${USER}/asta-sqa}"
SCORER_ENV="${SCORER_ENV:-${ASTA_ROOT}/scorer-venv}"
ASTA_DATA_FILE="${ASTA_DATA_FILE:-${ASTA_ROOT}/data/tasks/sqa/rubrics_v2_recomputed.json}"
CONTAINER="${CONTAINER:-/gscratch/h2lab/${USER}/containers/vllm-openai-v0.19.1.sif}"
RESULTS_ROOT="${RESULTS_ROOT:-/mmfs1/gscratch/h2lab/${USER}/asta-sqa-results}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/${MODEL_LABEL}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_LABEL}}"
PORT="${PORT:-30001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
WORKERS="${WORKERS:-16}"
MAX_CONNECTIONS="${MAX_CONNECTIONS:-24}"
RUN_SCORING="${RUN_SCORING:-1}"

for required_file in \
  "${ASTA_DATA_FILE}" \
  "${CONTAINER}" \
  "${SCORER_ENV}/bin/python"; do
  if [[ ! -e "${required_file}" ]]; then
    echo "ERROR: required path is missing: ${required_file}" >&2
    exit 2
  fi
done
if [[ "${REQUIRE_CHECKPOINT_MARKER}" == 1 ]]; then
  for required_file in "${MODEL_PATH}/config.json" "${MODEL_PATH}/.checkpoint_complete"; do
    if [[ ! -e "${required_file}" ]]; then
      echo "ERROR: required checkpoint file is missing: ${required_file}" >&2
      exit 2
    fi
  done
fi
if [[ -z "${GOOGLE_API_KEY:-}" && "${RUN_SCORING}" == 1 ]]; then
  echo "ERROR: GOOGLE_API_KEY is required when RUN_SCORING=1" >&2
  exit 2
fi
if [[ "${RUN_SCORING}" != 0 && "${RUN_SCORING}" != 1 ]]; then
  echo "ERROR: RUN_SCORING must be 0 or 1" >&2
  exit 2
fi

JOB_SCRATCH="${JOB_SCRATCH:-/scr/${USER}/asta-sqa-${SLURM_JOB_ID:-local}}"
PROJECT_ENV="${PROJECT_ENV:-${JOB_SCRATCH}/open-instruct-venv}"
export HF_HOME="${HF_HOME:-/gscratch/h2lab/${USER}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export NLTK_DATA="${NLTK_DATA:-${ASTA_ROOT}/nltk_data}"
export NLTK_ALLOW_PROXIED_URLOPEN=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-/gscratch/h2lab/${USER}/uv-cache}"
export TMPDIR="${TMPDIR:-${JOB_SCRATCH}/tmp}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${JOB_SCRATCH}/vllm}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${JOB_SCRATCH}/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${JOB_SCRATCH}/triton}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,${no_proxy}}"
export NO_PROXY="${no_proxy}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy-not-used}"
export APPTAINERENV_S2_API_KEY="${S2_API_KEY}"
export APPTAINERENV_GOOGLE_API_KEY="${GOOGLE_API_KEY:-}"
export APPTAINERENV_HF_TOKEN="${HF_TOKEN}"
export APPTAINERENV_OPENAI_API_KEY="${OPENAI_API_KEY}"

mkdir -p \
  "${OUTPUT_DIR}" \
  "${TMPDIR}" \
  "${VLLM_CACHE_ROOT}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${UV_CACHE_DIR}"

APPTAINER_ARGS=(
  exec
  --nv
  --bind /mmfs1/gscratch:/mmfs1/gscratch
  --bind /mmfs1/gscratch:/gscratch
  --bind /usr/bin/git:/usr/bin/git
  --bind /usr/libexec/git-core:/usr/libexec/git-core
  --bind /lib64/libcrypto.so.1.1:/lib64/libcrypto.so.1.1
  --bind "${JOB_SCRATCH}:${JOB_SCRATCH}"
  --env "HF_HOME=${HF_HOME}"
  --env "HF_HUB_CACHE=${HF_HUB_CACHE}"
  --env "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
  --env "NLTK_DATA=${NLTK_DATA}"
  --env "NLTK_ALLOW_PROXIED_URLOPEN=${NLTK_ALLOW_PROXIED_URLOPEN}"
  --env "UV_CACHE_DIR=${UV_CACHE_DIR}"
  --env "TMPDIR=${TMPDIR}"
  --env "VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}"
  --env "TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR}"
  --env "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
  --env "GIT_EXEC_PATH=/usr/libexec/git-core"
  --env "LD_LIBRARY_PATH=/.singularity.d/libs"
  "${CONTAINER}"
)

echo "Syncing vip-work environment to ${PROJECT_ENV}"
apptainer "${APPTAINER_ARGS[@]}" bash -c '
  set -euo pipefail
  cd "$1"
  export UV_PROJECT_ENVIRONMENT="$2"
  export UV_LINK_MODE=copy
  export LD_LIBRARY_PATH=/lib64:/.singularity.d/libs:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
  uv sync --frozen --no-dev
' _ "${REPO_ROOT}" "${PROJECT_ENV}"

SERVER_LOG="${OUTPUT_DIR}/vllm-server.log"
apptainer "${APPTAINER_ARGS[@]}" \
  "${PROJECT_ENV}/bin/vllm" serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --dtype bfloat16 \
    --generation-config vllm \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.90 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --port "${PORT}" \
    >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  if curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null; then
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -n 200 "${SERVER_LOG}" >&2
    exit 1
  fi
  sleep 5
done
curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null

RAW_OUTPUT="${OUTPUT_DIR}/responses.jsonl"
ASTA_OUTPUT="${OUTPUT_DIR}/responses_asta_format.json"
apptainer "${APPTAINER_ARGS[@]}" \
  "${PROJECT_ENV}/bin/python" "${REPO_ROOT}/scripts/eval/asta_sqa_generate.py" \
    --input-file "${ASTA_DATA_FILE}" \
    --raw-output-file "${RAW_OUTPUT}" \
    --asta-output-file "${ASTA_OUTPUT}" \
    --api-base "http://127.0.0.1:${PORT}/v1" \
    --model "${SERVED_MODEL_NAME}" \
    --system-prompt-file "${REPO_ROOT}/scripts/eval/asta_sqa_system_prompt.txt" \
    --max-samples "${MAX_SAMPLES}" \
    --workers "${WORKERS}" \
    --max-tool-calls 10 \
    --num-docs 10 \
    --temperature 0.6 \
    --top-p 0.95 \
    --max-tokens "${MAX_TOKENS}" \
    --resume

if [[ "${RUN_SCORING}" == 1 ]]; then
  INSPECT_ARGS=(
    -m inspect_ai._cli.main
    eval astabench/sqa
    --display plain
    --solver "${REPO_ROOT}/scripts/eval/asta_sqa_cached_solver.py@cache_solver"
    -S "path=${ASTA_OUTPUT}"
    -T split=test
    -T with_search_tools=false
    -T simplified_eval=true
    -T assess_jointly=true
    -T sentence_wise_cit_eval=false
    -T all_at_once=true
    -T scorer_model=google/gemini-2.5-flash
    --max-connections "${MAX_CONNECTIONS}"
    --log-dir "${OUTPUT_DIR}/inspect"
  )
  if [[ "${MAX_SAMPLES}" != -1 ]]; then
    INSPECT_ARGS+=(--limit "${MAX_SAMPLES}")
  fi
  export INSPECT_EVAL_LOG_FILE_PATTERN="${MODEL_LABEL}"
  apptainer "${APPTAINER_ARGS[@]}" "${SCORER_ENV}/bin/python" "${INSPECT_ARGS[@]}"
fi

echo "ASTA ScholarQA evaluation complete: ${OUTPUT_DIR}"
