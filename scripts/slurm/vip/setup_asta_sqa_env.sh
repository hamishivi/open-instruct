#!/usr/bin/env bash

# Prepare the shared scorer environment and gated ScholarQA test data.
# Run this once from a Hyak login node before submitting eval_asta_sqa_local.sh.

set -euo pipefail

: "${HF_TOKEN:?Export HF_TOKEN with access to allenai/asta-bench}"

ASTA_ROOT="${ASTA_ROOT:-/gscratch/h2lab/${USER}/asta-sqa}"
SCORER_ENV="${SCORER_ENV:-${ASTA_ROOT}/scorer-venv}"
ASTA_DATA_DIR="${ASTA_DATA_DIR:-${ASTA_ROOT}/data}"
ASTA_DATA_FILE="${ASTA_DATA_FILE:-${ASTA_DATA_DIR}/tasks/sqa/rubrics_v2_recomputed.json}"
NLTK_DATA_DIR="${NLTK_DATA_DIR:-${ASTA_ROOT}/nltk_data}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/gscratch/h2lab/${USER}/uv-cache}"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required" >&2
  exit 2
fi

mkdir -p "${ASTA_ROOT}" "${ASTA_DATA_DIR}" "${NLTK_DATA_DIR}" "${UV_CACHE_DIR}"
export UV_CACHE_DIR ASTA_DATA_DIR NLTK_DATA_DIR

if [[ ! -x "${SCORER_ENV}/bin/python" ]]; then
  uv venv --python 3.11 "${SCORER_ENV}"
fi
uv pip install --python "${SCORER_ENV}/bin/python" \
  "astabench==0.3.1" \
  "datasets" \
  "inspect_ai" \
  "openai==1.78.0"

"${SCORER_ENV}/bin/python" -c '
import os
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="allenai/asta-bench",
    repo_type="dataset",
    filename="tasks/sqa/rubrics_v2_recomputed.json",
    local_dir=os.environ["ASTA_DATA_DIR"],
)
print(path)
'

NLTK_ALLOW_PROXIED_URLOPEN=1 NLTK_DATA="${NLTK_DATA_DIR}" \
  "${SCORER_ENV}/bin/python" -c '
import os

import nltk

for package in ("punkt", "punkt_tab"):
    if not nltk.download(package, download_dir=os.environ["NLTK_DATA_DIR"], quiet=True):
        raise RuntimeError(f"Failed to download NLTK package: {package}")
'

test -f "${ASTA_DATA_FILE}"
echo "ASTA scorer environment: ${SCORER_ENV}"
echo "ScholarQA test data:     ${ASTA_DATA_FILE}"
echo "NLTK data:               ${NLTK_DATA_DIR}"
