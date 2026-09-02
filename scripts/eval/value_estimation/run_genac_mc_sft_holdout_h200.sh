#!/usr/bin/env bash
# Split a current-policy exact-MC panel, score the starting critic, train only
# on the problem-disjoint training split, and score every exported SFT epoch on
# the identical held-out problems.
set -euo pipefail

: "${MC_VALUE_PARQUET:?Set MC_VALUE_PARQUET to the completed paired exact-MC parquet.}"
: "${MODEL_PATH:?Set MODEL_PATH to the starting generative-critic checkpoint.}"
: "${EXPERIMENT_ROOT:?Set EXPERIMENT_ROOT to a new output directory.}"
: "${GEN_VALUE_CONDITIONING:?Set GEN_VALUE_CONDITIONING to none or gt, matching the deployed critic.}"

if [[ ! -f "${MC_VALUE_PARQUET}" ]]; then
    echo "MC_VALUE_PARQUET is not a file: ${MC_VALUE_PARQUET}" >&2
    exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "MODEL_PATH is not a model directory: ${MODEL_PATH}" >&2
    exit 1
fi
if [[ -e "${EXPERIMENT_ROOT}" ]]; then
    echo "Refusing to reuse EXPERIMENT_ROOT: ${EXPERIMENT_ROOT}" >&2
    exit 1
fi
if [[ "${GEN_VALUE_CONDITIONING}" != "none" && "${GEN_VALUE_CONDITIONING}" != "gt" ]]; then
    echo "GEN_VALUE_CONDITIONING must be none or gt: ${GEN_VALUE_CONDITIONING}" >&2
    exit 1
fi

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python}"
BASH_EXECUTABLE="${BASH_EXECUTABLE:-bash}"
HELDOUT_PROBLEM_COUNT="${HELDOUT_PROBLEM_COUNT:-80}"
SPLIT_SEED="${SPLIT_SEED:-37}"
NUM_GPUS="${NUM_GPUS:-4}"
MIN_LONG_PREFIX_FRACTION="${MIN_LONG_PREFIX_FRACTION:-0.15}"
LONG_PREFIX_TOKEN_THRESHOLD="${LONG_PREFIX_TOKEN_THRESHOLD:-2048}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
GEN_VALUE_MAX_NEW_TOKENS="${GEN_VALUE_MAX_NEW_TOKENS:-1024}"
ACTOR_TOKENIZER_NAME_OR_PATH="${ACTOR_TOKENIZER_NAME_OR_PATH:-Qwen/Qwen3-4B-Base}"

mkdir -p "${EXPERIMENT_ROOT}/data" "${EXPERIMENT_ROOT}/scores" "${EXPERIMENT_ROOT}/comparisons"
TRAIN_PARQUET="${EXPERIMENT_ROOT}/data/train.parquet"
HELDOUT_PARQUET="${EXPERIMENT_ROOT}/data/heldout.parquet"
SFT_OUTPUT_DIR="${EXPERIMENT_ROOT}/sft"
BASELINE_SCORE="${EXPERIMENT_ROOT}/scores/baseline.parquet"

"${PYTHON_EXECUTABLE}" scripts/data/split_gen_value_mc_dataset.py \
    "${MC_VALUE_PARQUET}" \
    --train_output "${TRAIN_PARQUET}" \
    --heldout_output "${HELDOUT_PARQUET}" \
    --heldout_problem_count "${HELDOUT_PROBLEM_COUNT}" \
    --seed "${SPLIT_SEED}"

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}" \
VLLM_TENSOR_PARALLEL_SIZE="${NUM_GPUS}" \
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION}" \
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN}" \
GEN_VALUE_MAX_NEW_TOKENS="${GEN_VALUE_MAX_NEW_TOKENS}" \
ACTOR_TOKENIZER_NAME_OR_PATH="${ACTOR_TOKENIZER_NAME_OR_PATH}" \
RUN_NAME="baseline" \
"${BASH_EXECUTABLE}" scripts/eval/value_estimation/score_generative_value.sh \
    "${MODEL_PATH}" "${HELDOUT_PARQUET}" "${BASELINE_SCORE}" "${GEN_VALUE_CONDITIONING}"

MC_VALUE_PARQUET="${TRAIN_PARQUET}" \
HELDOUT_VALUE_PARQUET="${HELDOUT_PARQUET}" \
MODEL_PATH="${MODEL_PATH}" \
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}" \
NUM_GPUS="${NUM_GPUS}" \
LONG_PREFIX_TOKEN_THRESHOLD="${LONG_PREFIX_TOKEN_THRESHOLD}" \
MIN_LONG_PREFIX_FRACTION="${MIN_LONG_PREFIX_FRACTION}" \
GEN_VALUE_CONDITIONING="${GEN_VALUE_CONDITIONING}" \
ACTOR_TOKENIZER_NAME_OR_PATH="${ACTOR_TOKENIZER_NAME_OR_PATH}" \
MC_SFT_JSONL="${EXPERIMENT_ROOT}/data/train.jsonl" \
OUTPUT_DIR="${SFT_OUTPUT_DIR}" \
EXP_NAME="${EXP_NAME:-genac-math-current-policy-direct-mc-sft}" \
"${BASH_EXECUTABLE}" scripts/train/debug/genac_math_mc_value_sft_h200.sh

shopt -s nullglob
candidate_models=("${SFT_OUTPUT_DIR}"/epoch_*_model)
if [[ ${#candidate_models[@]} -eq 0 ]]; then
    echo "No epoch model exports found under ${SFT_OUTPUT_DIR}" >&2
    exit 1
fi

for candidate_model in "${candidate_models[@]}"; do
    candidate_name="$(basename "${candidate_model}")"
    candidate_score="${EXPERIMENT_ROOT}/scores/${candidate_name}.parquet"
    PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}" \
    VLLM_TENSOR_PARALLEL_SIZE="${NUM_GPUS}" \
    VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION}" \
    VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN}" \
    GEN_VALUE_MAX_NEW_TOKENS="${GEN_VALUE_MAX_NEW_TOKENS}" \
    ACTOR_TOKENIZER_NAME_OR_PATH="${ACTOR_TOKENIZER_NAME_OR_PATH}" \
    RUN_NAME="${candidate_name}" \
    "${BASH_EXECUTABLE}" scripts/eval/value_estimation/score_generative_value.sh \
        "${candidate_model}" "${HELDOUT_PARQUET}" "${candidate_score}" "${GEN_VALUE_CONDITIONING}"
    "${PYTHON_EXECUTABLE}" scripts/eval/value_estimation/compare_gen_value_scores.py \
        --baseline "${BASELINE_SCORE}" \
        --candidate "${candidate_score}" \
        --output_json "${EXPERIMENT_ROOT}/comparisons/${candidate_name}.json"
done
