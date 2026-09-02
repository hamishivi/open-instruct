#!/usr/bin/env bash
# Compare greedy critic scores with the expected discrete score conditioned on
# each model's own generated rationale, using one identical fixed MC panel.
set -euo pipefail

: "${BASELINE_MODEL_PATH:?Set BASELINE_MODEL_PATH to the deployed critic checkpoint.}"
: "${CANDIDATE_MODEL_PATH:?Set CANDIDATE_MODEL_PATH to the candidate critic checkpoint.}"
: "${INPUT_DATASET_PATH:?Set INPUT_DATASET_PATH to the fixed held-out MC parquet.}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new output directory.}"
: "${GEN_VALUE_CONDITIONING:?Set GEN_VALUE_CONDITIONING to match critic serving.}"

if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "Refusing to reuse OUTPUT_ROOT: ${OUTPUT_ROOT}" >&2
    exit 1
fi

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python}"
BASH_EXECUTABLE="${BASH_EXECUTABLE:-bash}"
NUM_GPUS="${NUM_GPUS:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
GEN_VALUE_MAX_NEW_TOKENS="${GEN_VALUE_MAX_NEW_TOKENS:-1024}"
ACTOR_TOKENIZER_NAME_OR_PATH="${ACTOR_TOKENIZER_NAME_OR_PATH:-Qwen/Qwen3-4B-Base}"

mkdir -p "${OUTPUT_ROOT}/scores" "${OUTPUT_ROOT}/comparisons"

score_model() {
    local run_name="$1"
    local model_path="$2"
    local output_path="${OUTPUT_ROOT}/scores/${run_name}.parquet"
    PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}" \
    VLLM_TENSOR_PARALLEL_SIZE="${NUM_GPUS}" \
    VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION}" \
    VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN}" \
    VLLM_DISABLE_CUSTOM_ALL_REDUCE=1 \
    GEN_VALUE_MAX_NEW_TOKENS="${GEN_VALUE_MAX_NEW_TOKENS}" \
    GEN_VALUE_SOFT_CLASS_PROBABILITIES=1 \
    ACTOR_TOKENIZER_NAME_OR_PATH="${ACTOR_TOKENIZER_NAME_OR_PATH}" \
    RUN_NAME="${run_name}" \
    "${BASH_EXECUTABLE}" scripts/eval/value_estimation/score_generative_value.sh \
        "${model_path}" "${INPUT_DATASET_PATH}" "${output_path}" "${GEN_VALUE_CONDITIONING}"
}

score_model baseline "${BASELINE_MODEL_PATH}"
score_model candidate "${CANDIDATE_MODEL_PATH}"

"${PYTHON_EXECUTABLE}" scripts/eval/value_estimation/compare_gen_value_scores.py \
    --baseline "${OUTPUT_ROOT}/scores/baseline.parquet" \
    --candidate "${OUTPUT_ROOT}/scores/candidate.parquet" \
    --output_json "${OUTPUT_ROOT}/comparisons/greedy.json"

"${PYTHON_EXECUTABLE}" scripts/eval/value_estimation/compare_gen_value_scores.py \
    --baseline "${OUTPUT_ROOT}/scores/baseline.parquet" \
    --candidate "${OUTPUT_ROOT}/scores/candidate.parquet" \
    --prediction_column soft_predicted_values \
    --output_json "${OUTPUT_ROOT}/comparisons/soft_expected.json"
