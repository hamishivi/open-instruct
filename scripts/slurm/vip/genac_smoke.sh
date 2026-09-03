#!/usr/bin/env bash
#SBATCH --job-name=genac-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err

set -euo pipefail

if [[ -n "${GENAC_REPO_ROOT:-}" ]]; then
    REPO_ROOT="${GENAC_REPO_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/pyproject.toml" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
elif [[ -f "${PWD}/pyproject.toml" ]]; then
    # `--chdir` controls the batch working directory but does not rewrite
    # SLURM_SUBMIT_DIR. Prefer the actual checkout when sbatch was invoked from
    # somewhere else (for example, a login-shell home directory).
    REPO_ROOT="${PWD}"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
cd "${REPO_ROOT}"
mkdir -p logs

# Prepared container environments may be editable installs created from a
# different worktree. Prefer the exact Slurm submission checkout so package
# imports cannot silently mix source files across experiment snapshots.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export NCCL_CUMEM_ENABLE=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
# A minimal outer `sbatch --export=NIL` is useful when a cluster cannot retrieve
# the login environment. Nested srun steps must still receive the explicit
# runtime paths and experiment configuration established by this script.
export SLURM_EXPORT_ENV=ALL
export HF_HOME="${HF_HOME:-/mmfs1/gscratch/h2lab/${USER}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,${no_proxy}}"
export NO_PROXY="${no_proxy}"

SRUN_PREFIX=()
if [[ -n "${APPTAINER_IMAGE:-}" ]]; then
    if [[ ! -f "${APPTAINER_IMAGE}" ]]; then
        echo "ERROR: APPTAINER_IMAGE does not exist: ${APPTAINER_IMAGE}" >&2
        exit 1
    fi
    APPTAINER_LOCAL_BIND=""
    if [[ "${APPTAINER_LOCAL_VENV:-0}" == "1" ]]; then
        CONTAINER_VENV="${APPTAINER_LOCAL_VENV_DIR:-/scr/${USER}/open-instruct-${SLURM_JOB_ID}/.venv}"
        APPTAINER_LOCAL_BIND="$(dirname "${CONTAINER_VENV}")"
        mkdir -p "${APPTAINER_LOCAL_BIND}"
        APPTAINER_UV_CACHE_DIR="${APPTAINER_UV_CACHE_DIR:-/mmfs1/gscratch/h2lab/${USER}/uv-cache}"
        apptainer exec --nv \
            --bind /usr/bin/git:/usr/bin/git \
            --bind /usr/libexec/git-core:/usr/libexec/git-core \
            --bind /lib64/libcrypto.so.1.1:/lib64/libcrypto.so.1.1 \
            --bind "${APPTAINER_LOCAL_BIND}:${APPTAINER_LOCAL_BIND}" \
            --env LD_LIBRARY_PATH=/lib64:/.singularity.d/libs:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
            --env "UV_CACHE_DIR=${APPTAINER_UV_CACHE_DIR}" \
            --env UV_LINK_MODE=copy \
            "${APPTAINER_IMAGE}" bash -c '
                set -euo pipefail
                cd "$1"
                export UV_PROJECT_ENVIRONMENT="$2"
                uv sync --frozen --no-dev
            ' _ "${REPO_ROOT}" "${CONTAINER_VENV}"
    else
        CONTAINER_VENV="${CONTAINER_VENV:-${REPO_ROOT}/.venv-container}"
    fi
    if [[ ! -x "${CONTAINER_VENV}/bin/python" ]]; then
        echo "ERROR: prepared container environment not found at ${CONTAINER_VENV}" >&2
        exit 1
    fi
    RUNTIME_CACHE="${RUNTIME_CACHE:-/tmp/oi-${SLURM_JOB_ID}}"
    mkdir -p "${RUNTIME_CACHE}"/{vllm,torchinductor,triton,cuda,xdg,tmp}
    SRUN_PREFIX=(
        apptainer exec --nv
        --env "PREPEND_PATH=${CONTAINER_VENV}/bin"
        --env PRESERVE_LD_LIBRARY_PATH=1
        --env LD_LIBRARY_PATH=/.singularity.d/libs
        --env "HF_HOME=${HF_HOME}"
        --env "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
        --env "VLLM_CACHE_ROOT=${RUNTIME_CACHE}/vllm"
        --env "TORCHINDUCTOR_CACHE_DIR=${RUNTIME_CACHE}/torchinductor"
        --env "TRITON_CACHE_DIR=${RUNTIME_CACHE}/triton"
        --env "CUDA_CACHE_PATH=${RUNTIME_CACHE}/cuda"
        --env "XDG_CACHE_HOME=${RUNTIME_CACHE}/xdg"
        --env "TMPDIR=${RUNTIME_CACHE}/tmp"
    )
    if [[ -n "${APPTAINER_LOCAL_BIND}" ]]; then
        SRUN_PREFIX+=(--bind "${APPTAINER_LOCAL_BIND}:${APPTAINER_LOCAL_BIND}")
    fi
    SRUN_PREFIX+=("${APPTAINER_IMAGE}")
else
    export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
    if [[ -d "${REPO_ROOT}/.vllm-overlay" ]]; then
        export PYTHONPATH="${REPO_ROOT}/.vllm-overlay${PYTHONPATH:+:${PYTHONPATH}}"
    fi
    if ! command -v nvcc >/dev/null 2>&1; then
        module load gcc/13.4.0
        module load cuda/12.9.1
    fi
fi

GENAC_LAUNCH_SCRIPT="${GENAC_LAUNCH_SCRIPT:-scripts/train/debug/genac_smoke.sh}"
if [[ ! -f "${GENAC_LAUNCH_SCRIPT}" ]]; then
    echo "ERROR: GENAC_LAUNCH_SCRIPT does not exist: ${GENAC_LAUNCH_SCRIPT}" >&2
    exit 1
fi
srun --cpu-bind=none "${SRUN_PREFIX[@]}" bash "${GENAC_LAUNCH_SCRIPT}" "$@"
