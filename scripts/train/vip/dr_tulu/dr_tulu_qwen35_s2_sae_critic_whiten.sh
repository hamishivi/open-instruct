#!/bin/bash
# Qwen3.5-4B DR-Tulu SAE + scalar critic + global whitening, S2 search only.
# The critic receives no rubric/answer-prefix/ground-truth conditioning.
# Launch with:
#   ./scripts/train/build_image_and_launch.sh scripts/train/vip/dr_tulu/dr_tulu_qwen35_s2_sae_critic_whiten.sh

set -euo pipefail
export VARIANT=sae_critic_whiten
exec "$(dirname "$0")/dr_tulu_qwen35_s2_only_common.sh" "$@"
