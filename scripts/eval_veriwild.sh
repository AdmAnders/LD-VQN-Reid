#!/usr/bin/env bash
# Evaluate LD-VQN on T2I-VeRi-Wild (3000 / 5000 / 10000 splits).
# Usage: bash scripts/eval_veriwild.sh [weights] [extra test.py args, e.g. --no_rerank]
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/veriwild_ldvqn.yml"
WEIGHTS="${1:-weights/ldvqn_veriwild.pth}"
[ $# -gt 0 ] && shift
GPU_ID="${GPU_ID:-0}"

if [ ! -f "${WEIGHTS}" ]; then
  echo "[!] Weights not found: ${WEIGHTS} (see weights/README.md)"; exit 1
fi
mkdir -p logs results
LOG="logs/eval_veriwild_$(date +%Y%m%d_%H%M%S).log"

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u test.py --config "${CONFIG}" --weights "${WEIGHTS}" \
  --test_sizes 3000 5000 10000 --output results/veriwild.json "$@" 2>&1 | tee "${LOG}"
