#!/bin/bash
# Qwen3.5-4B DR-Tulu GRPO baseline with only Semantic Scholar snippet search.
# Launch with:
#   ./scripts/train/build_image_and_launch.sh scripts/train/vip/dr_tulu/dr_tulu_qwen35_s2_grpo.sh

set -euo pipefail
export VARIANT=grpo
exec "$(dirname "$0")/dr_tulu_qwen35_s2_only_common.sh" "$@"
