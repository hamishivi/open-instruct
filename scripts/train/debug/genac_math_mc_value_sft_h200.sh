#!/usr/bin/env bash
# Direct Monte Carlo value SFT for the Qwen3-4B generative critic.
#
# The input parquet must contain exact token-prefix continuation estimates from
# value_estimation.py. Held-out calibration problems are checked again before
# conversion. Final-action and late-trajectory examples are intentionally
# repeated, then audited as declared repeat groups before packed SFT.
set -euo pipefail

: "${MC_VALUE_PARQUET:?Set MC_VALUE_PARQUET to the disjoint MC training parquet.}"
: "${HELDOUT_VALUE_PARQUET:?Set HELDOUT_VALUE_PARQUET to the fixed MC evaluation parquet.}"
: "${MODEL_PATH:?Set MODEL_PATH to the critic checkpoint to calibrate.}"

for required_path in "${MC_VALUE_PARQUET}" "${HELDOUT_VALUE_PARQUET}"; do
    if [[ ! -f "${required_path}" ]]; then
        echo "Required parquet is not a file: ${required_path}" >&2
        exit 1
    fi
done
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "MODEL_PATH is not a critic model directory: ${MODEL_PATH}" >&2
    exit 1
fi

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python}"
MC_SFT_JSONL="${MC_SFT_JSONL:-${PWD}/inputs/mc_sft/direct_mc_value_sft.jsonl}"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-Qwen/Qwen3-4B-Base}"
MIN_CONTINUATIONS="${MIN_CONTINUATIONS:-16}"
MIN_MC_EXAMPLES="${MIN_MC_EXAMPLES:-512}"
FINAL_ACTION_REPEAT="${FINAL_ACTION_REPEAT:-4}"
LATE_STATE_REPEAT="${LATE_STATE_REPEAT:-2}"
LATE_STATE_FRACTION="${LATE_STATE_FRACTION:-0.75}"

"${PYTHON_EXECUTABLE}" scripts/data/prepare_gen_value_mc_sft.py \
    "${MC_VALUE_PARQUET}" \
    --output "${MC_SFT_JSONL}" \
    --tokenizer_name_or_path "${TOKENIZER_NAME_OR_PATH}" \
    --exclude_problem_dataset_path "${HELDOUT_VALUE_PARQUET}" \
    --min_continuations "${MIN_CONTINUATIONS}" \
    --min_examples "${MIN_MC_EXAMPLES}" \
    --final_action_repeat "${FINAL_ACTION_REPEAT}" \
    --late_state_repeat "${LATE_STATE_REPEAT}" \
    --late_state_fraction "${LATE_STATE_FRACTION}"

export TRACE_JSONL="${MC_SFT_JSONL}"
export MIN_TRACE_EXAMPLES="${MIN_MC_EXAMPLES}"
export NUM_GPUS="${NUM_GPUS:-4}"
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-32768}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
export LEARNING_RATE="${LEARNING_RATE:-5e-7}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-4}"
export SAVE_MODEL_EACH_EPOCH="${SAVE_MODEL_EACH_EPOCH:-1}"
export EXP_NAME="${EXP_NAME:-genac-math-value-direct-mc-sft}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PWD}/outputs/${EXP_NAME}}"

exec bash scripts/train/debug/genac_math_value_trace_sft_h200.sh
