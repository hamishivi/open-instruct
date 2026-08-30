#!/usr/bin/env bash
# Generate cold-start GenAC value-reasoning traces with a local chat model.
#
# The input must use the OpenAI chat-completions batch format produced by:
#   scripts/data/synthesize_gen_value_sft.py prepare \
#     --request_format chat_completions ...
#
# This is an offline-teacher ablation for environments without an OpenAI API
# credential. Teacher outputs must still pass parsing and empirical-return
# checks before they are used to SFT the base critic.
set -euo pipefail

: "${BATCH_INPUT:?Set BATCH_INPUT to the audited chat-completions JSONL file.}"
: "${BATCH_OUTPUT:?Set BATCH_OUTPUT to a new output JSONL path.}"

if [[ ! -f "${BATCH_INPUT}" ]]; then
    echo "ERROR: BATCH_INPUT does not exist: ${BATCH_INPUT}" >&2
    exit 1
fi
if [[ -e "${BATCH_OUTPUT}" ]]; then
    echo "ERROR: refusing to overwrite BATCH_OUTPUT: ${BATCH_OUTPUT}" >&2
    exit 1
fi

if [[ "${PRESERVE_LD_LIBRARY_PATH:-0}" != "1" ]]; then
    unset LD_LIBRARY_PATH
fi
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/tmp/hf_home}"

TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-8B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

mkdir -p "$(dirname "${BATCH_OUTPUT}")"

vllm run-batch \
    --input-file "${BATCH_INPUT}" \
    --output-file "${BATCH_OUTPUT}" \
    --model "${TEACHER_MODEL}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --dtype bfloat16 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --enable-prefix-caching
