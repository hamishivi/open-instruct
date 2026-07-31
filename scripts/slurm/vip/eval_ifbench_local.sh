#!/usr/bin/env bash
#SBATCH --account=gpu-h200-h2lab
#SBATCH --partition=gpu-h200
#SBATCH --job-name=vip-ifbench-local
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.vip-ifbench-local.out
#SBATCH --error=logs/%j.vip-ifbench-local.err

set -euo pipefail

: "${MODEL_ID:?Set MODEL_ID to the Hugging Face repository to evaluate}"
: "${MODEL_REVISION:?Set MODEL_REVISION to the Hugging Face revision}"

OPEN_INSTRUCT_DIR="${OPEN_INSTRUCT_DIR:-/mmfs1/gscratch/h2lab/hamishiv/open-instruct-vip-if-runs}"
OE_EVAL_INTERNAL_DIR="${OE_EVAL_INTERNAL_DIR:-/mmfs1/gscratch/h2lab/hamishiv/oe-eval-internal-vip-ifbench}"
CONTAINER="${CONTAINER:-/gscratch/h2lab/hamishiv/containers/vllm-openai-v0.19.1.sif}"
ENV_DIR="${ENV_DIR:-/gscratch/h2lab/hamishiv/uv-envs/oe-eval-ifbench}"
RESULTS_ROOT="${RESULTS_ROOT:-/mmfs1/gscratch/h2lab/hamishiv/oe-eval-results/vip_ifbench_local}"
MODEL_SLUG="${MODEL_ID//\//__}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/${MODEL_SLUG}/${MODEL_REVISION}}"

export HF_HOME="${HF_HOME:-/gscratch/h2lab/hamishiv/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/gscratch/h2lab/hamishiv/uv-cache}"
export UV_PROJECT_ENVIRONMENT="${ENV_DIR}"
export TMPDIR="${TMPDIR:-/tmp/oi-ifbench-${SLURM_JOB_ID}}"

mkdir -p "${OUTPUT_DIR}" "${TMPDIR}" "${UV_CACHE_DIR}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}"

echo "Syncing oe-eval environment: ${ENV_DIR}"
uv sync --project "${OE_EVAL_INTERNAL_DIR}" --frozen --no-dev

apptainer exec --nv \
  --bind /mmfs1/gscratch:/mmfs1/gscratch \
  --bind /mmfs1/gscratch:/gscratch \
  --env "PATH=${ENV_DIR}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --env "HF_HOME=${HF_HOME}" \
  --env "HF_HUB_CACHE=${HF_HUB_CACHE}" \
  --env "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}" \
  --env "UV_CACHE_DIR=${UV_CACHE_DIR}" \
  --env "UV_PROJECT_ENVIRONMENT=${ENV_DIR}" \
  --env "TMPDIR=${TMPDIR}" \
  "${CONTAINER}" \
  bash -lc "cd '${OPEN_INSTRUCT_DIR}' && \
    MODEL_ID='${MODEL_ID}' \
    MODEL_REVISION='${MODEL_REVISION}' \
    OE_EVAL_INTERNAL_DIR='${OE_EVAL_INTERNAL_DIR}' \
    RUN_LOCAL=1 \
    DRY_RUN=0 \
    GPUS=1 \
    OUTPUT_DIR='${OUTPUT_DIR}' \
    bash scripts/eval/vip_ifbench.sh"
