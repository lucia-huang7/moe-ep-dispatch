#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/gpu_environment.sh"
timeout -k 15s 300 .venv/bin/torchrun --standalone --nproc_per_node=2 -m scripts.validate_deepep --output "${1:-results/deepep-world2.json}" "${@:2}"
