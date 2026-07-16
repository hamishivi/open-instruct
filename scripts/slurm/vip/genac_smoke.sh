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

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs

export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
if [[ -d "${REPO_ROOT}/.vllm-overlay" ]]; then
    export PYTHONPATH="${REPO_ROOT}/.vllm-overlay${PYTHONPATH:+:${PYTHONPATH}}"
fi
if ! command -v nvcc >/dev/null 2>&1; then
    module load gcc/13.4.0
    module load cuda/12.9.1
fi
export NCCL_CUMEM_ENABLE=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/gpfs/scrubbed/${USER}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,${no_proxy}}"
export NO_PROXY="${no_proxy}"

srun --cpu-bind=none bash scripts/train/debug/genac_smoke.sh "$@"
