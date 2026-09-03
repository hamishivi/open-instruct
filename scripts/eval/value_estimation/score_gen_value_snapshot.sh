#!/bin/bash
set -euo pipefail

VALUE_MODEL_PATH="${1:-${VALUE_MODEL_PATH:-}}"
INPUT_SNAPSHOT="${2:-${INPUT_SNAPSHOT:-}}"
OUTPUT_JSONL="${3:-${OUTPUT_JSONL:-}}"

: "${VALUE_MODEL_PATH:?Pass a value model path or set VALUE_MODEL_PATH.}"
: "${INPUT_SNAPSHOT:?Pass a fixed validation snapshot JSONL or set INPUT_SNAPSHOT.}"
: "${OUTPUT_JSONL:?Pass an output JSONL path or set OUTPUT_JSONL.}"

PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-python}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
SEED=${SEED:-0}

exec "${PYTHON_EXECUTABLE}" scripts/eval/value_estimation/score_gen_value_snapshot.py \
    --input_snapshot "${INPUT_SNAPSHOT}" \
    --model_path "${VALUE_MODEL_PATH}" \
    --output_jsonl "${OUTPUT_JSONL}" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --seed "${SEED}"
