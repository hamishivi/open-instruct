#!/usr/bin/env bash
#SBATCH --job-name=frozen-aime-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err

set -euo pipefail

: "${MODEL_A_PATH:?set MODEL_A_PATH to a complete frozen policy checkpoint}"
: "${MODEL_A_NAME:?set MODEL_A_NAME to the first result name}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to the result directory}"
: "${APPTAINER_IMAGE:?set APPTAINER_IMAGE to the training container}"
: "${CONTAINER_VENV:?set CONTAINER_VENV to the prepared training environment}"

REPO_ROOT="${FROZEN_AIME_REPO_ROOT:-${SLURM_SUBMIT_DIR:-${PWD}}}"
cd "${REPO_ROOT}"
if [[ ! -f pyproject.toml ]]; then
    echo "ERROR: repository checkout not found at ${REPO_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${APPTAINER_IMAGE}" ]]; then
    echo "ERROR: Apptainer image not found at ${APPTAINER_IMAGE}" >&2
    exit 1
fi
if [[ ! -x "${CONTAINER_VENV}/bin/python" ]]; then
    echo "ERROR: prepared container Python not found at ${CONTAINER_VENV}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
RUNTIME_CACHE="${RUNTIME_CACHE:-/tmp/frozen-aime-${SLURM_JOB_ID}}"
mkdir -p "${RUNTIME_CACHE}"/{vllm,torchinductor,triton,cuda,xdg,tmp}

run_evaluation() {
    local model_path=$1
    local run_name=$2
    if [[ ! -f "${model_path}/.checkpoint_complete" ]]; then
        echo "ERROR: frozen checkpoint is incomplete: ${model_path}" >&2
        exit 1
    fi
    srun --cpu-bind=none apptainer exec --nv \
        --env "PREPEND_PATH=${CONTAINER_VENV}/bin" \
        --env PRESERVE_LD_LIBRARY_PATH=1 \
        --env LD_LIBRARY_PATH=/.singularity.d/libs \
        --env "PYTHONPATH=${REPO_ROOT}" \
        --env "HF_HOME=${HF_HOME:-/gpfs/scrubbed/${USER}/huggingface}" \
        --env "HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/gpfs/scrubbed/${USER}/huggingface/datasets}" \
        --env "VLLM_CACHE_ROOT=${RUNTIME_CACHE}/vllm" \
        --env "TORCHINDUCTOR_CACHE_DIR=${RUNTIME_CACHE}/torchinductor" \
        --env "TRITON_CACHE_DIR=${RUNTIME_CACHE}/triton" \
        --env "CUDA_CACHE_PATH=${RUNTIME_CACHE}/cuda" \
        --env "XDG_CACHE_HOME=${RUNTIME_CACHE}/xdg" \
        --env "TMPDIR=${RUNTIME_CACHE}/tmp" \
        --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
        --env TOKENIZERS_PARALLELISM=false \
        "${APPTAINER_IMAGE}" \
        "${CONTAINER_VENV}/bin/python" scripts/eval/frozen_aime_vllm.py \
        --model_name_or_path "${model_path}" \
        --output_dir "${OUTPUT_DIR}" \
        --run_name "${run_name}"
}

run_evaluation "${MODEL_A_PATH}" "${MODEL_A_NAME}"
if [[ -n "${MODEL_B_PATH:-}" || -n "${MODEL_B_NAME:-}" ]]; then
    : "${MODEL_B_PATH:?set both MODEL_B_PATH and MODEL_B_NAME}"
    : "${MODEL_B_NAME:?set both MODEL_B_PATH and MODEL_B_NAME}"
    run_evaluation "${MODEL_B_PATH}" "${MODEL_B_NAME}"
fi
