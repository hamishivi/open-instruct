#!/bin/bash
# Score with a generative value model served via vLLM.
set -euo pipefail

VALUE_MODEL_PATH="${1:-${VALUE_MODEL_PATH:-}}"
: "${VALUE_MODEL_PATH:?Pass a value model path or set VALUE_MODEL_PATH.}"
INPUT_DATASET_PATH="${2:-${INPUT_DATASET_PATH:-./value_estimation_data/dapo_math_100pairs.parquet}}"
OUTPUT_PATH="${3:-${OUTPUT_PATH:-./value_estimation_data/generative_value_scores.parquet}}"
CONDITIONING="${4:-${CONDITIONING:-none}}"  # none | gt | correct_demo | rollout_context
GEN_VALUE_MAX_NEW_TOKENS="${GEN_VALUE_MAX_NEW_TOKENS:-1024}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
RUN_NAME="${RUN_NAME:-generative_value_${CONDITIONING}}"
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-python}

"${PYTHON_EXECUTABLE}" -m open_instruct.value_estimation score_dataset \
    --input_dataset_path "${INPUT_DATASET_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --value_model_path "${VALUE_MODEL_PATH}" \
    --value_model_type generative \
    --gen_value_conditioning "${CONDITIONING}" \
    --gen_value_score_min 0 \
    --gen_value_score_max 10 \
    --gen_value_max_new_tokens "${GEN_VALUE_MAX_NEW_TOKENS}" \
    --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}" \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --run_name "${RUN_NAME}"
