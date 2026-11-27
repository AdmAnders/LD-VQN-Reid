#!/usr/bin/env bash
# D-TCR cross-modal attention map for one image + caption (Fig. 9).
# Usage: bash scripts/extract_attention.sh <image> "<caption>" [weights] [config]
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${1:?image path required}"
CAPTION="${2:?caption required}"
WEIGHTS="${3:-weights/ldvqn_veri776.pth}"
CONFIG="${4:-configs/veri776_ldvqn.yml}"
OUT="outputs/attention/$(basename "${IMAGE%.*}")_attn.png"

python tools/visualize_attention.py --config "${CONFIG}" --weights "${WEIGHTS}" \
  --image "${IMAGE}" --caption "${CAPTION}" --output "${OUT}"
