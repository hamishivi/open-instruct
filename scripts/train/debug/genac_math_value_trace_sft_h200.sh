#!/bin/bash
set -euo pipefail

: "${TRACE_JSONL:?Set TRACE_JSONL to a filtered raw-prompt trace JSONL file.}"

if [[ ! -f "${TRACE_JSONL}" ]]; then
    echo "TRACE_JSONL is not a file: ${TRACE_JSONL}" >&2
    exit 1
fi

MIN_TRACE_EXAMPLES=${MIN_TRACE_EXAMPLES:-512}
ALLOW_GROUND_TRUTH_CONDITIONING=${ALLOW_GROUND_TRUTH_CONDITIONING:-0}
TRACE_DATASET_WEIGHT=${TRACE_DATASET_WEIGHT:-1.0}
if [[ ! "${MIN_TRACE_EXAMPLES}" =~ ^[0-9]+$ ]]; then
    echo "MIN_TRACE_EXAMPLES must be a nonnegative integer: ${MIN_TRACE_EXAMPLES}" >&2
    exit 1
fi
AUDIT_ARGS=("${TRACE_JSONL}" --min_examples "${MIN_TRACE_EXAMPLES}")
if [[ "${ALLOW_GROUND_TRUTH_CONDITIONING}" == "1" ]]; then
    AUDIT_ARGS+=(--allow_ground_truth_conditioning)
fi
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-python}
"${PYTHON_EXECUTABLE}" scripts/data/synthesize_gen_value_sft.py audit "${AUDIT_ARGS[@]}"

DATASET_MIXER_ARGS=("${TRACE_JSONL}" "${TRACE_DATASET_WEIGHT}")
if [[ -n "${AUX_TRACE_JSONL:-}" ]]; then
    if [[ ! -f "${AUX_TRACE_JSONL}" ]]; then
        echo "AUX_TRACE_JSONL is not a file: ${AUX_TRACE_JSONL}" >&2
        exit 1
    fi
    AUX_MIN_TRACE_EXAMPLES=${AUX_MIN_TRACE_EXAMPLES:-${MIN_TRACE_EXAMPLES}}
    AUX_TRACE_DATASET_WEIGHT=${AUX_TRACE_DATASET_WEIGHT:-1.0}
    if [[ ! "${AUX_MIN_TRACE_EXAMPLES}" =~ ^[0-9]+$ ]]; then
        echo "AUX_MIN_TRACE_EXAMPLES must be a nonnegative integer: ${AUX_MIN_TRACE_EXAMPLES}" >&2
        exit 1
    fi
    AUX_AUDIT_ARGS=("${AUX_TRACE_JSONL}" --min_examples "${AUX_MIN_TRACE_EXAMPLES}")
    if [[ "${ALLOW_GROUND_TRUTH_CONDITIONING}" == "1" ]]; then
        AUX_AUDIT_ARGS+=(--allow_ground_truth_conditioning)
    fi
    "${PYTHON_EXECUTABLE}" scripts/data/synthesize_gen_value_sft.py audit "${AUX_AUDIT_ARGS[@]}"
    DATASET_MIXER_ARGS+=("${AUX_TRACE_JSONL}" "${AUX_TRACE_DATASET_WEIGHT}")
fi

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Base}
NUM_GPUS=${NUM_GPUS:-4}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-32768}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-2}
EXP_NAME=${EXP_NAME:-genac-math-value-teacher-sft}
OUTPUT_DIR=${OUTPUT_DIR:-output/${EXP_NAME}}
SAVE_MODEL_EACH_EPOCH=${SAVE_MODEL_EACH_EPOCH:-0}
SCORE_TOKEN_ONLY_LOSS=${SCORE_TOKEN_ONLY_LOSS:-0}
if [[ ! "${SCORE_TOKEN_ONLY_LOSS}" =~ ^[01]$ ]]; then
    echo "SCORE_TOKEN_ONLY_LOSS must be 0 or 1: ${SCORE_TOKEN_ONLY_LOSS}" >&2
    exit 1
fi
GEN_VALUE_TOKENIZE_TRANSFORM=gen_value_sft_tokenize_and_truncate_v1
if [[ "${SCORE_TOKEN_ONLY_LOSS}" == "1" ]]; then
    GEN_VALUE_TOKENIZE_TRANSFORM=gen_value_sft_score_tokenize_and_truncate_v1
fi
MODEL_CHECKPOINT_ARGS=()
if [[ "${SAVE_MODEL_EACH_EPOCH}" == "1" ]]; then
    MODEL_CHECKPOINT_ARGS+=(--save_model_each_epoch)
fi
ACCELERATE_NETWORK_ARGS=()
if [[ -n "${MAIN_PROCESS_PORT:-}" ]]; then
    ACCELERATE_NETWORK_ARGS+=(--main_process_port "${MAIN_PROCESS_PORT}")
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
    # Multiple four-GPU jobs can share one eight-GPU node. Accelerate otherwise
    # defaults every single-node job to port 29500, so concurrent jobs collide
    # before distributed initialization even though their GPUs are disjoint.
    MAIN_PROCESS_PORT="$((10000 + SLURM_JOB_ID % 50000))"
    ACCELERATE_NETWORK_ARGS+=(--main_process_port "${MAIN_PROCESS_PORT}")
fi

accelerate launch \
    "${ACCELERATE_NETWORK_ARGS[@]}" \
    --mixed_precision bf16 \
    --num_processes "${NUM_GPUS}" \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --exp_name "${EXP_NAME}" \
    --model_name_or_path "${MODEL_PATH}" \
    --tokenizer_name "${MODEL_PATH}" \
    --max_seq_length "${MAX_SEQ_LENGTH}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --dataset_mixer_list "${DATASET_MIXER_ARGS[@]}" \
    --dataset_mixer_list_splits train \
    --dataset_transform_fn "${GEN_VALUE_TOKENIZE_TRANSFORM}" sft_tulu_filter_v1 \
    --dataset_target_columns input_ids attention_mask labels \
    --dataset_skip_cache \
    --gradient_checkpointing \
    --packing \
    "${MODEL_CHECKPOINT_ARGS[@]}" \
    --output_dir "${OUTPUT_DIR}" \
    --report_to wandb \
    --with_tracking \
    --logging_steps 1 \
    --seed 123 \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false
