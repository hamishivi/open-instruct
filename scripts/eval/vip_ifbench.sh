#!/usr/bin/env bash

# Launch IFBench from olmo-eval-internal for one Hugging Face model revision.
#
# Required:
#   MODEL_ID=hamishivi/example-model
#   MODEL_REVISION=step-100
#
# Optional:
#   OLMO_EVAL_INTERNAL_DIR=/path/to/olmo-eval-internal
#   CLUSTER=h100
#   WORKSPACE=ai2/tulu-3-results
#   BUDGET=ai2/oe-adapt
#   GPUS=1
#   PRIORITY=normal
#   TIMEOUT=24h
#   EXPERIMENT_GROUP=vip-ifbench
#   EXPERIMENT_NAME=<derived>
#   DRY_RUN=1
#
# DRY_RUN defaults to 1. Set DRY_RUN=0 to submit.

set -euo pipefail

: "${MODEL_ID:?Set MODEL_ID to the Hugging Face repository to evaluate}"
: "${MODEL_REVISION:?Set MODEL_REVISION to the Hugging Face revision, for example step-100}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
OLMO_EVAL_INTERNAL_DIR="${OLMO_EVAL_INTERNAL_DIR:-$(cd "${REPO_ROOT}/.." && pwd)/olmo-eval-internal}"
CLUSTER="${CLUSTER:-h100}"
WORKSPACE="${WORKSPACE:-ai2/tulu-3-results}"
BUDGET="${BUDGET:-ai2/oe-adapt}"
GPUS="${GPUS:-1}"
PRIORITY="${PRIORITY:-normal}"
TIMEOUT="${TIMEOUT:-24h}"
EXPERIMENT_GROUP="${EXPERIMENT_GROUP:-vip-ifbench}"
MODEL_SLUG="${MODEL_ID##*/}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${MODEL_SLUG}-${MODEL_REVISION}-ifbench}"
DRY_RUN="${DRY_RUN:-1}"

if [[ ! -d "${OLMO_EVAL_INTERNAL_DIR}/.git" ]]; then
  echo "ERROR: olmo-eval-internal checkout not found: ${OLMO_EVAL_INTERNAL_DIR}" >&2
  exit 2
fi
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
  echo "ERROR: DRY_RUN must be 0 or 1, got: ${DRY_RUN}" >&2
  exit 2
fi

args=(
  beaker launch
  --name "${EXPERIMENT_NAME}"
  --model "${MODEL_ID}"
  --task ifbench
  --harness default
  --override "provider.revision=${MODEL_REVISION}"
  --cluster "${CLUSTER}"
  --workspace "${WORKSPACE}"
  --budget "${BUDGET}"
  --gpus "${GPUS}"
  --priority "${PRIORITY}"
  --timeout "${TIMEOUT}"
  --group "${EXPERIMENT_GROUP}"
  --force-download-model
  --no-follow
  --yes
)
if [[ "${DRY_RUN}" == "1" ]]; then
  args+=(--dry-run)
fi

echo "Model:    ${MODEL_ID}@${MODEL_REVISION}"
echo "Suite:    ifbench"
echo "Eval repo: ${OLMO_EVAL_INTERNAL_DIR}"
echo "Mode:     $([[ "${DRY_RUN}" == "1" ]] && echo dry-run || echo submit)"

cd "${OLMO_EVAL_INTERNAL_DIR}"
uv run olmo-eval "${args[@]}"
