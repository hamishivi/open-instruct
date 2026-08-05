#!/usr/bin/env bash
#SBATCH --job-name=drt-q35-s2-grpo
#SBATCH --account=gpu-h200-h2lab
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=1100G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err

set -euo pipefail
export VARIANT=grpo
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
exec bash "${REPO_ROOT}/scripts/slurm/vip/dr_tulu_qwen35_s2_only_common.sh"
