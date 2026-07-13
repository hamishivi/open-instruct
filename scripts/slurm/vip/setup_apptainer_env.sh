#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${APPTAINER_IMAGE:-}" ]]; then
  echo "ERROR: APPTAINER_IMAGE must point to a container image" >&2
  exit 1
fi

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CONTAINER_VENV="${REPO_ROOT}/.venv-container"
APPTAINER_ARGS=(exec --nv)
if [[ -n "${APPTAINER_BIND:-}" ]]; then
  APPTAINER_ARGS+=(--bind "${APPTAINER_BIND}")
fi
if [[ -n "${APPTAINER_EXTRA_BIND:-}" ]]; then
  APPTAINER_ARGS+=(--bind "${APPTAINER_EXTRA_BIND}")
fi
if [[ -n "${APPTAINER_GIT_CORE_BIND:-}" ]]; then
  APPTAINER_ARGS+=(--bind "${APPTAINER_GIT_CORE_BIND}")
fi
if [[ -n "${APPTAINER_LD_LIBRARY_PATH:-}" ]]; then
  APPTAINER_ARGS+=(--env "LD_LIBRARY_PATH=${APPTAINER_LD_LIBRARY_PATH}")
fi
if [[ -n "${APPTAINER_GIT_EXEC_PATH:-}" ]]; then
  APPTAINER_ARGS+=(--env "GIT_EXEC_PATH=${APPTAINER_GIT_EXEC_PATH}")
fi

cd "${REPO_ROOT}"

apptainer "${APPTAINER_ARGS[@]}" "${APPTAINER_IMAGE}" bash -c '
  set -euo pipefail
  cd "'"${REPO_ROOT}"'"
  export UV_PROJECT_ENVIRONMENT="'"${CONTAINER_VENV}"'"
  uv sync --frozen --no-dev
  "'"${CONTAINER_VENV}"'/bin/python" -c "import deepspeed, ray, torch, vllm, wandb; print(torch.__version__, vllm.__version__)"
'
