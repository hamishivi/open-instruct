#!/usr/bin/env bash
# Shared Ray bootstrap for Slurm multi-node VIP / GRPO jobs.
#
# Expects to be launched via `srun` with one task per node
# (`#SBATCH --ntasks-per-node=1`). Rank 0 starts the Ray head; other ranks
# join as workers and block until the head exits.
#
# Environment (optional overrides):
#   RAY_PORT           Ray GCS port (default: 6379)
#   RAY_DASHBOARD_PORT dashboard port (default: 8265)

set -euo pipefail

RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"

export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
mkdir -p "${HOME}/.triton/autotune"

if [[ "${SLURM_JOB_NUM_NODES:-1}" -eq 1 ]]; then
    head_node="$(hostname -s)"
else
    head_node="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
fi
# Prefer IPv4 from getent so workers do not need a nested srun.
head_node_ip="$(getent ahostsv4 "${head_node}" | awk '{print $1; exit}')"
if [[ -z "${head_node_ip}" ]]; then
  head_node_ip="$(getent hosts "${head_node}" | awk '{print $1; exit}')"
fi
if [[ -z "${head_node_ip}" && "${SLURM_PROCID:-0}" -eq 0 ]]; then
  head_node_ip="$(hostname --ip-address 2>/dev/null | awk '{print $1; exit}')"
fi
if [[ -z "${head_node_ip}" ]]; then
  echo "[slurm ray] ERROR: could not resolve head node IP for ${head_node}" >&2
  exit 1
fi

ip_head="${head_node_ip}:${RAY_PORT}"
export ip_head
export RAY_ADDRESS="${ip_head}"

echo "[slurm ray] head=${head_node} (${head_node_ip})"
echo "[slurm ray] procid=${SLURM_PROCID:-0} nodeid=${SLURM_NODEID:-0} nnodes=${SLURM_JOB_NUM_NODES}"

ray stop --force >/dev/null 2>&1 || true

gpus_per_node="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-8}}"
cpus_per_task="${SLURM_CPUS_PER_TASK:-$((gpus_per_node * 16))}"

if [[ "${SLURM_PROCID:-0}" -eq 0 ]]; then
    echo "[slurm ray] Starting Ray head"
    ray start \
        --head \
        --node-ip-address="${head_node_ip}" \
        --port="${RAY_PORT}" \
        --dashboard-host=0.0.0.0 \
        --dashboard-port="${RAY_DASHBOARD_PORT}" \
        --num-cpus="${cpus_per_task}" \
        --num-gpus="${gpus_per_node}"
else
    echo "[slurm ray] Starting Ray worker rank ${SLURM_PROCID} -> ${ip_head}"
    sleep 10
    ray start \
        --address="${ip_head}" \
        --dashboard-host=0.0.0.0 \
        --num-cpus="${cpus_per_task}" \
        --num-gpus="${gpus_per_node}"

    cleanup() {
        echo "[slurm ray] Worker cleanup: stopping Ray"
        ray stop --force >/dev/null 2>&1 || true
        trap - TERM INT HUP EXIT
        exit 0
    }
    trap cleanup TERM INT HUP EXIT

    while true; do
        if ! ray status --address="${ip_head}" >/dev/null 2>&1; then
            echo "[slurm ray] Head unreachable; exiting worker"
            cleanup
        fi
        sleep 5
    done
fi
