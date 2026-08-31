#!/usr/bin/env bash
set -euo pipefail

# Exact-token Monte Carlo value calibration for the Qwen3-4B math actor. The
# defaults are a cheap pipeline/throughput pilot; set TARGET_NUM_PAIRS and
# CONTINUATIONS_PER_PROBE higher for the fixed production calibration set.
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-4B-Base}"
ACTOR_MODEL_NAME="${ACTOR_MODEL_NAME:-${MODEL_NAME_OR_PATH}}"
OUTPUT_PATH="${OUTPUT_PATH:-./value_estimation_data/dapo_math_mc_pilot.parquet}"
TARGET_NUM_PAIRS="${TARGET_NUM_PAIRS:-10}"
NUM_PROMPTS_TO_SAMPLE="${NUM_PROMPTS_TO_SAMPLE:-200}"
ROLLOUTS_PER_PROMPT="${ROLLOUTS_PER_PROMPT:-8}"
CONTINUATIONS_PER_PROBE="${CONTINUATIONS_PER_PROBE:-8}"
PROBE_INTERVAL="${PROBE_INTERVAL:-2000}"
MAX_PROBES="${MAX_PROBES:-6}"
PROBE_MODE="${PROBE_MODE:-fixed}"
SAE_THRESHOLD="${SAE_THRESHOLD:-0.2}"
SEED="${SEED:-1}"
EXCLUDE_PROBLEM_DATASET_PATH="${EXCLUDE_PROBLEM_DATASET_PATH:-}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

MAKE_DATASET_ARGS=(
    --model_name_or_path "${MODEL_NAME_OR_PATH}"
    --actor_model_name "${ACTOR_MODEL_NAME}"
    --output_path "${OUTPUT_PATH}"
    --dataset_name hamishivi/DAPO-Math-17k-Processed_filtered
    --num_prompts_to_sample "${NUM_PROMPTS_TO_SAMPLE}"
    --target_num_pairs "${TARGET_NUM_PAIRS}"
    --rollouts_per_prompt "${ROLLOUTS_PER_PROMPT}"
    --continuations_per_probe "${CONTINUATIONS_PER_PROBE}"
    --probe_interval "${PROBE_INTERVAL}"
    --probe_mode "${PROBE_MODE}"
    --sae_threshold "${SAE_THRESHOLD}"
    --max_probes "${MAX_PROBES}"
    --max_prompt_length 2048
    --max_response_length 8192
    --chat_template_name qwen_instruct_user_boxed_math
    --seed "${SEED}"
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
    --data_parallel_size "${DATA_PARALLEL_SIZE}"
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
)
if [[ -n "${EXCLUDE_PROBLEM_DATASET_PATH}" ]]; then
    MAKE_DATASET_ARGS+=(--exclude_problem_dataset_path "${EXCLUDE_PROBLEM_DATASET_PATH}")
fi

python -m open_instruct.value_estimation make_dataset "${MAKE_DATASET_ARGS[@]}"
