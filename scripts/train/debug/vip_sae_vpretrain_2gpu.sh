#!/bin/bash
# Local 2-GPU smoke test for VIP SAE + value pretraining.
#
# This reuses vip_local_2gpu.sh's Qwen3-0.6B setup and adds the SAE/value
# warmup flags used by the larger VIP Beaker launchers. The warmup window is
# intentionally short so the smoke test exercises both value-only and joint
# policy/value training.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

exec bash "${SCRIPT_DIR}/vip_local_2gpu.sh" \
    --exp_name vip_sae_vpretrain_2gpu_smoke \
    --output_dir /tmp/vip_sae_vpretrain_2gpu_output \
    --total_episodes 384 \
    --use_sae \
    --sae_threshold 0.2 \
    --value_model_ground_truth_conditioning \
    --gt_conditioning_template answer_prefix \
    --value_warmup_steps 3 \
    --reset_optimizer_after_value_warmup \
    "${@}"
