#!/usr/bin/env bash
#SBATCH --job-name=genac-pack-memory
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=180G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs

: "${APPTAINER_IMAGE:?set APPTAINER_IMAGE to the training container}"
: "${CONTAINER_VENV:?set CONTAINER_VENV to the prepared container environment}"
: "${GEN_VALUE_MODEL_PATH:?set GEN_VALUE_MODEL_PATH to the critic model}"

if [[ ! -f "${APPTAINER_IMAGE}" ]]; then
    echo "ERROR: APPTAINER_IMAGE does not exist: ${APPTAINER_IMAGE}" >&2
    exit 1
fi
if [[ ! -x "${CONTAINER_VENV}/bin/python" ]]; then
    echo "ERROR: CONTAINER_VENV does not contain Python: ${CONTAINER_VENV}" >&2
    exit 1
fi
if [[ ! -d "${GEN_VALUE_MODEL_PATH}" ]]; then
    echo "ERROR: GEN_VALUE_MODEL_PATH does not exist: ${GEN_VALUE_MODEL_PATH}" >&2
    exit 1
fi

export HF_HOME="${HF_HOME:-/gpfs/scrubbed/${USER}/huggingface}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
RUNTIME_CACHE="${RUNTIME_CACHE:-/tmp/genac-pack-memory-${SLURM_JOB_ID}}"
mkdir -p "${RUNTIME_CACHE}"/{torchinductor,triton,cuda,xdg,tmp}

srun --cpu-bind=none apptainer exec --nv \
    --env "PREPEND_PATH=${CONTAINER_VENV}/bin" \
    --env LD_LIBRARY_PATH=/.singularity.d/libs \
    --env "HF_HOME=${HF_HOME}" \
    --env "PYTHONPATH=${PYTHONPATH}" \
    --env "GEN_VALUE_MODEL_PATH=${GEN_VALUE_MODEL_PATH}" \
    --env "TORCHINDUCTOR_CACHE_DIR=${RUNTIME_CACHE}/torchinductor" \
    --env "TRITON_CACHE_DIR=${RUNTIME_CACHE}/triton" \
    --env "CUDA_CACHE_PATH=${RUNTIME_CACHE}/cuda" \
    --env "XDG_CACHE_HOME=${RUNTIME_CACHE}/xdg" \
    --env "TMPDIR=${RUNTIME_CACHE}/tmp" \
    "${APPTAINER_IMAGE}" bash -c '
        set -euo pipefail
        export PATH="${PREPEND_PATH}:${PATH}"
        python scripts/test/gen_value_pack_memory.py \
            --model "${GEN_VALUE_MODEL_PATH}" \
            --pack-length 10240
        python scripts/test/gen_value_pack_memory.py \
            --model "${GEN_VALUE_MODEL_PATH}" \
            --pack-length 32768
    '
