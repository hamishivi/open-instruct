#!/usr/bin/env bash

# Launch IFBench from oe-eval-internal for one Hugging Face model revision.
#
# IFBench is registered in oe-eval-internal as the `ifbench::tulu` suite. The
# newer olmo-eval-internal launcher does not currently expose this suite.
#
# Required:
#   MODEL_ID=hamishivi/example-model
#   MODEL_REVISION=step-100
#
# Optional:
#   OE_EVAL_INTERNAL_DIR=/path/to/oe-eval-internal
#   CLUSTER=ai2/jupiter
#   WORKSPACE=ai2/tulu-3-results
#   GPUS=1
#   PRIORITY=normal
#   TIMEOUT=24h
#   MAX_LENGTH=8192
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
OE_EVAL_INTERNAL_DIR="${OE_EVAL_INTERNAL_DIR:-$(cd "${REPO_ROOT}/.." && pwd)/oe-eval-internal}"
CLUSTER="${CLUSTER:-ai2/jupiter}"
WORKSPACE="${WORKSPACE:-ai2/tulu-3-results}"
GPUS="${GPUS:-1}"
PRIORITY="${PRIORITY:-normal}"
TIMEOUT="${TIMEOUT:-24h}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
EXPERIMENT_GROUP="${EXPERIMENT_GROUP:-vip-ifbench}"
MODEL_SLUG="${MODEL_ID##*/}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${MODEL_SLUG}-${MODEL_REVISION}-ifbench}"
DRY_RUN="${DRY_RUN:-1}"

if [[ ! -d "${OE_EVAL_INTERNAL_DIR}/.git" ]]; then
  echo "ERROR: oe-eval-internal checkout not found: ${OE_EVAL_INTERNAL_DIR}" >&2
  exit 2
fi
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
  echo "ERROR: DRY_RUN must be 0 or 1, got: ${DRY_RUN}" >&2
  exit 2
fi

args=(
  run
  --project "${OE_EVAL_INTERNAL_DIR}"
  python
  "${OE_EVAL_INTERNAL_DIR}/oe_eval/launch.py"
  --model "${EXPERIMENT_NAME}"
  --model-type vllm
  --revision "${MODEL_REVISION}"
  --model-args "{\"model_path\":\"${MODEL_ID}\",\"max_length\":${MAX_LENGTH},\"trust_remote_code\":\"true\"}"
  --task "ifbench::tulu"
  --batch-size auto
  --gpus "${GPUS}"
  --cluster "${CLUSTER}"
  --beaker-workspace "${WORKSPACE}"
  --beaker-timeout "${TIMEOUT}"
  --beaker-priority "${PRIORITY}"
  --push-datalake
  --datalake-tags "experiment_group=${EXPERIMENT_GROUP},revision=${MODEL_REVISION}"
)
if [[ "${DRY_RUN}" == "1" ]]; then
  args+=(--dry-run)
fi

echo "Model:    ${MODEL_ID}@${MODEL_REVISION}"
echo "Suite:    ifbench::tulu"
echo "Eval repo: ${OE_EVAL_INTERNAL_DIR}"
echo "Mode:     $([[ "${DRY_RUN}" == "1" ]] && echo dry-run || echo submit)"

uv "${args[@]}"
