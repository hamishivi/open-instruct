#!/usr/bin/env bash
# Select a warmed generative critic only if its deterministic fixed-MC gate
# remains competitive with the direct-MC SFT checkpoint, then launch the full
# joint math run. This script is intended to run inside the experiment
# container after the scorer dependency has completed.
set -euo pipefail

: "${SFT_GEN_VALUE_MODEL_PATH:?Set SFT_GEN_VALUE_MODEL_PATH to the validated fallback critic.}"
: "${WARMUP_OUTPUT_ROOT:?Set WARMUP_OUTPUT_ROOT to the critic-only run output root.}"
: "${WARMUP_GATE_SUMMARY:?Set WARMUP_GATE_SUMMARY to the fixed-MC summary JSON.}"
JOINT_LAUNCH_SCRIPT="${JOINT_LAUNCH_SCRIPT:-scripts/train/debug/genac_math_joint_h200.sh}"

if [[ ! -s "${SFT_GEN_VALUE_MODEL_PATH}/model.safetensors" ]]; then
    echo "ERROR: fallback critic is incomplete: ${SFT_GEN_VALUE_MODEL_PATH}" >&2
    exit 1
fi
if [[ ! -s "${WARMUP_GATE_SUMMARY}" ]]; then
    echo "ERROR: warmup fixed-gate summary is missing: ${WARMUP_GATE_SUMMARY}" >&2
    exit 1
fi

mapfile -t warmup_candidates < <(
    find "${WARMUP_OUTPUT_ROOT}" \
        -type d \
        -path '*/gen_value_model_checkpoints/version_000025/gen_value_model' \
        -print
)
if [[ "${#warmup_candidates[@]}" -ne 1 ]]; then
    echo "ERROR: expected exactly one version_000025 warmup model, found ${#warmup_candidates[@]}" >&2
    printf '%s\n' "${warmup_candidates[@]}" >&2
    exit 1
fi
WARMUP_GEN_VALUE_MODEL_PATH="${warmup_candidates[0]}"
if [[ ! -s "${WARMUP_GEN_VALUE_MODEL_PATH}/model.safetensors" ]]; then
    echo "ERROR: warmed critic is incomplete: ${WARMUP_GEN_VALUE_MODEL_PATH}" >&2
    exit 1
fi

GATE_MIN_PARSE_RATE="${GATE_MIN_PARSE_RATE:-0.99}"
GATE_MAX_MSE="${GATE_MAX_MSE:-0.04}"
GATE_MIN_PEARSON="${GATE_MIN_PEARSON:-0.90}"
GATE_MIN_SPEARMAN="${GATE_MIN_SPEARMAN:-0.87}"
GATE_MIN_FINAL_CORRECT="${GATE_MIN_FINAL_CORRECT:-0.90}"
GATE_MAX_FINAL_INCORRECT="${GATE_MAX_FINAL_INCORRECT:-0.08}"
GATE_MIN_INTERMEDIATE_CORRECT="${GATE_MIN_INTERMEDIATE_CORRECT:-0.45}"
GATE_MAX_INTERMEDIATE_INCORRECT="${GATE_MAX_INTERMEDIATE_INCORRECT:-0.20}"

if python - \
    "${WARMUP_GATE_SUMMARY}" \
    "${GATE_MIN_PARSE_RATE}" \
    "${GATE_MAX_MSE}" \
    "${GATE_MIN_PEARSON}" \
    "${GATE_MIN_SPEARMAN}" \
    "${GATE_MIN_FINAL_CORRECT}" \
    "${GATE_MAX_FINAL_INCORRECT}" \
    "${GATE_MIN_INTERMEDIATE_CORRECT}" \
    "${GATE_MAX_INTERMEDIATE_INCORRECT}" <<'PY'
import json
import sys

summary_path = sys.argv[1]
threshold_values = [float(value) for value in sys.argv[2:]]
thresholds = dict(
    zip(
        (
            "min_parse_rate",
            "max_mse",
            "min_pearson",
            "min_spearman",
            "min_final_correct",
            "max_final_incorrect",
            "min_intermediate_correct",
            "max_intermediate_incorrect",
        ),
        threshold_values,
    )
)
with open(summary_path) as summary_file:
    metrics = json.load(summary_file)["metrics"]

observed = {
    "parse_rate": metrics["parse_rate"],
    "mse": metrics["penalized_mse"],
    "pearson": metrics["pearson"],
    "spearman": metrics["spearman"],
    "final_correct": metrics["final_action_correct_pred_mean"],
    "final_incorrect": metrics["final_action_incorrect_pred_mean"],
    "intermediate_correct": metrics["intermediate_correct_pred_mean"],
    "intermediate_incorrect": metrics["intermediate_incorrect_pred_mean"],
}
checks = {
    "parse_rate": observed["parse_rate"] >= thresholds["min_parse_rate"],
    "mse": observed["mse"] <= thresholds["max_mse"],
    "pearson": observed["pearson"] >= thresholds["min_pearson"],
    "spearman": observed["spearman"] >= thresholds["min_spearman"],
    "final_correct": observed["final_correct"] >= thresholds["min_final_correct"],
    "final_incorrect": observed["final_incorrect"] <= thresholds["max_final_incorrect"],
    "intermediate_correct": observed["intermediate_correct"] >= thresholds["min_intermediate_correct"],
    "intermediate_incorrect": observed["intermediate_incorrect"] <= thresholds["max_intermediate_incorrect"],
}
accepted = all(checks.values())
print(json.dumps({"accepted": accepted, "checks": checks, "observed": observed, "thresholds": thresholds}, sort_keys=True))
raise SystemExit(0 if accepted else 1)
PY
then
    GEN_VALUE_MODEL_PATH="${WARMUP_GEN_VALUE_MODEL_PATH}"
    echo "Selected warmed critic: ${GEN_VALUE_MODEL_PATH}"
else
    GEN_VALUE_MODEL_PATH="${SFT_GEN_VALUE_MODEL_PATH}"
    echo "Warmup missed the fixed gate; selected fallback critic: ${GEN_VALUE_MODEL_PATH}"
fi

export GEN_VALUE_MODEL_PATH
exec bash "${JOINT_LAUNCH_SCRIPT}" "$@"
