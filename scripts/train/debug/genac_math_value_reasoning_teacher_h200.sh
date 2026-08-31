#!/usr/bin/env bash
# Build a GenAC-style value-reasoning SFT dataset from an on-policy trace reservoir.
#
# The source reservoir supplies critic prompts sampled from the active policy. A
# stronger local chat model writes value-focused reasoning traces, but never sees
# the sampled rollout outcome. The output remains an SFT cold start rather than a
# noisy oracle: later RL pretraining must ground its scores in empirical returns.
set -euo pipefail

: "${TRACE_RESERVOIR:?Set TRACE_RESERVOIR to gen_value_training_traces/reservoir.jsonl.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new directory for teacher artifacts.}"
: "${HOLDOUT_DATASET_PATH:?Set HOLDOUT_DATASET_PATH to the fixed-MC holdout parquet.}"

if [[ ! -s "${TRACE_RESERVOIR}" ]]; then
    echo "ERROR: trace reservoir is missing or empty: ${TRACE_RESERVOIR}" >&2
    exit 1
fi
if [[ ! -s "${HOLDOUT_DATASET_PATH}" ]]; then
    echo "ERROR: holdout dataset is missing or empty: ${HOLDOUT_DATASET_PATH}" >&2
    exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "ERROR: refusing to overwrite teacher output directory: ${OUTPUT_DIR}" >&2
    exit 1
fi

TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-32B}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-1024}"
MAX_EXAMPLES_PER_OUTCOME="${MAX_EXAMPLES_PER_OUTCOME:-512}"
MIN_TRACE_EXAMPLES="${MIN_TRACE_EXAMPLES:-512}"
SEED="${SEED:-29}"
ALLOW_GROUND_TRUTH_CONDITIONING="${ALLOW_GROUND_TRUTH_CONDITIONING:-0}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python}"

for variable_name in MAX_OUTPUT_TOKENS MAX_EXAMPLES_PER_OUTCOME MIN_TRACE_EXAMPLES SEED; do
    variable_value="${!variable_name}"
    if [[ ! "${variable_value}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: ${variable_name} must be a nonnegative integer: ${variable_value}" >&2
        exit 1
    fi
done
if ((MAX_OUTPUT_TOKENS == 0 || MAX_EXAMPLES_PER_OUTCOME == 0 || MIN_TRACE_EXAMPLES == 0)); then
    echo "ERROR: token, selection, and audit sizes must be positive" >&2
    exit 1
fi
if [[ "${ALLOW_GROUND_TRUTH_CONDITIONING}" != "0" && "${ALLOW_GROUND_TRUTH_CONDITIONING}" != "1" ]]; then
    echo "ERROR: ALLOW_GROUND_TRUTH_CONDITIONING must be 0 or 1" >&2
    exit 1
fi
if [[ "${ENABLE_THINKING}" != "0" && "${ENABLE_THINKING}" != "1" ]]; then
    echo "ERROR: ENABLE_THINKING must be 0 or 1" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
BATCH_INPUT="${OUTPUT_DIR}/teacher_batch_input.jsonl"
METADATA_OUTPUT="${OUTPUT_DIR}/teacher_metadata.jsonl"
BATCH_OUTPUT="${OUTPUT_DIR}/teacher_batch_output.jsonl"
TRACE_OUTPUT="${OUTPUT_DIR}/value_reasoning_sft.jsonl"

PREPARE_ARGS=(
    "${TRACE_RESERVOIR}"
    --batch_output "${BATCH_INPUT}"
    --metadata_output "${METADATA_OUTPUT}"
    --exclude_problem_dataset "${HOLDOUT_DATASET_PATH}"
    --model "${TEACHER_MODEL}"
    --request_format chat_completions
    --max_output_tokens "${MAX_OUTPUT_TOKENS}"
    --max_examples_per_outcome "${MAX_EXAMPLES_PER_OUTCOME}"
    --seed "${SEED}"
)
AUDIT_ARGS=("${TRACE_OUTPUT}" --min_examples "${MIN_TRACE_EXAMPLES}")
if [[ "${ALLOW_GROUND_TRUTH_CONDITIONING}" == "1" ]]; then
    PREPARE_ARGS+=(--allow_ground_truth_conditioning)
    AUDIT_ARGS+=(--allow_ground_truth_conditioning)
fi
if [[ "${ENABLE_THINKING}" == "1" ]]; then
    PREPARE_ARGS+=(--enable_thinking)
fi

"${PYTHON_EXECUTABLE}" scripts/data/synthesize_gen_value_sft.py prepare "${PREPARE_ARGS[@]}"

export BATCH_INPUT BATCH_OUTPUT TEACHER_MODEL
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
bash scripts/train/debug/genac_value_local_teacher_h200.sh

"${PYTHON_EXECUTABLE}" scripts/data/synthesize_gen_value_sft.py collect \
    --metadata "${METADATA_OUTPUT}" \
    --results "${BATCH_OUTPUT}" \
    --output "${TRACE_OUTPUT}" \
    --teacher_model "${TEACHER_MODEL}" \
    --skip_invalid_scores
"${PYTHON_EXECUTABLE}" scripts/data/synthesize_gen_value_sft.py audit "${AUDIT_ARGS[@]}"

echo "Prepared audited GenAC value-reasoning traces: ${TRACE_OUTPUT}"
