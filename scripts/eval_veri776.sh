#!/usr/bin/env bash
# Evaluate LD-VQN on T2I-VeRi-776.
# Usage: bash scripts/eval_veri776.sh [weights] [extra test.py args, e.g. --no_rerank]
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/veri776_ldvqn.yml"
WEIGHTS="${1:-weights/ldvqn_veri776.pth}"
[ $# -gt 0 ] && shift
GPU_ID="${GPU_ID:-0}"

if [ ! -f "${WEIGHTS}" ]; then
  echo "[!] Weights not found: ${WEIGHTS} (see weights/README.md)"; exit 1
fi
mkdir -p logs results
LOG="logs/eval_veri776_$(date +%Y%m%d_%H%M%S).log"

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u test.py --config "${CONFIG}" --weights "${WEIGHTS}" \
  --output results/veri776.json "$@" 2>&1 | tee "${LOG}"
