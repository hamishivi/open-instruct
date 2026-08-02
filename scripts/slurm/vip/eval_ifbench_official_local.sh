#!/usr/bin/env bash
#SBATCH --account=gpu-h200-h2lab
#SBATCH --partition=gpu-h200
#SBATCH --job-name=vip-ifbench-official
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.vip-ifbench-official.out
#SBATCH --error=logs/%j.vip-ifbench-official.err

set -euo pipefail

: "${MODEL_ID:?Set MODEL_ID to the Hugging Face repository to evaluate}"
: "${MODEL_REVISION:?Set MODEL_REVISION to the Hugging Face revision}"

OPEN_INSTRUCT_DIR="${OPEN_INSTRUCT_DIR:-/mmfs1/gscratch/h2lab/hamishiv/open-instruct-vip-ifbench-official}"
IFBENCH_DIR="${IFBENCH_DIR:-/mmfs1/gscratch/h2lab/hamishiv/IFBench-official}"
CONTAINER="${CONTAINER:-/gscratch/h2lab/hamishiv/containers/vllm-openai-v0.19.1.sif}"
ENV_DIR="${ENV_DIR:-/gscratch/h2lab/hamishiv/uv-envs/ifbench-official}"
RESULTS_ROOT="${RESULTS_ROOT:-/mmfs1/gscratch/h2lab/hamishiv/oe-eval-results/vip_ifbench_official}"
MODEL_SLUG="${MODEL_ID//\//__}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/${MODEL_SLUG}/${MODEL_REVISION}}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-ifbench-model}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
SEED="${SEED:-42}"
WORKERS="${WORKERS:-16}"

export HF_HOME="${HF_HOME:-/gscratch/h2lab/hamishiv/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/gscratch/h2lab/hamishiv/uv-cache}"
export UV_PROJECT_ENVIRONMENT="${ENV_DIR}"
export TMPDIR="${TMPDIR:-/tmp/oi-ifbench-official-${SLURM_JOB_ID}}"

mkdir -p "${OUTPUT_DIR}" "${TMPDIR}" "${HF_HUB_CACHE}" "${UV_CACHE_DIR}"
test -f "${IFBENCH_DIR}/data/IFBench_test.jsonl"
test -f "${IFBENCH_DIR}/uv.lock"
uv sync --project "${IFBENCH_DIR}" --frozen --no-dev

SERVER_LOG="${OUTPUT_DIR}/vllm-server.log"
apptainer exec --nv \
  --bind /mmfs1/gscratch:/mmfs1/gscratch \
  --bind /mmfs1/gscratch:/gscratch \
  --env "HF_HOME=${HF_HOME}" \
  --env "HF_HUB_CACHE=${HF_HUB_CACHE}" \
  "${CONTAINER}" \
  vllm serve "${MODEL_ID}" \
    --revision "${MODEL_REVISION}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --port "${PORT}" \
    >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  if curl --fail --silent "http://localhost:${PORT}/health" >/dev/null; then
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -n 200 "${SERVER_LOG}" >&2
    exit 1
  fi
  sleep 5
done
curl --fail --silent "http://localhost:${PORT}/health" >/dev/null

echo "Evaluating ${MODEL_ID}@${MODEL_REVISION} with official IFBench"
echo "temperature=${TEMPERATURE} top_p=${TOP_P} max_tokens=${MAX_TOKENS} max_model_len=${MAX_MODEL_LEN}"
uv run --project "${IFBENCH_DIR}" python "${OPEN_INSTRUCT_DIR}/scripts/eval/vip_ifbench_official.py" \
  --api-base "http://localhost:${PORT}/v1" \
  --model "${SERVED_MODEL_NAME}" \
  --input-file "${IFBENCH_DIR}/data/IFBench_test.jsonl" \
  --output-file "${OUTPUT_DIR}/responses.jsonl" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --max-tokens "${MAX_TOKENS}" \
  --seed "${SEED}" \
  --workers "${WORKERS}" \
  --resume

cd "${IFBENCH_DIR}"
uv run python -m run_eval \
  --input_data="${IFBENCH_DIR}/data/IFBench_test.jsonl" \
  --input_response_data="${OUTPUT_DIR}/responses.jsonl" \
  --output_dir="${OUTPUT_DIR}"
