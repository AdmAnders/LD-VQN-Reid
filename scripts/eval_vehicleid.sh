#!/usr/bin/env bash
# Evaluate LD-VQN on T2I-VehicleID (800 / 1600 / 2400 splits).
# Usage: bash scripts/eval_vehicleid.sh [weights] [extra test.py args, e.g. --no_rerank]
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/vehicleid_ldvqn.yml"
WEIGHTS="${1:-weights/ldvqn_vehicleid.pth}"
[ $# -gt 0 ] && shift
GPU_ID="${GPU_ID:-0}"

if [ ! -f "${WEIGHTS}" ]; then
  echo "[!] Weights not found: ${WEIGHTS} (see weights/README.md)"; exit 1
fi
mkdir -p logs results
LOG="logs/eval_vehicleid_$(date +%Y%m%d_%H%M%S).log"

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u test.py --config "${CONFIG}" --weights "${WEIGHTS}" \
  --test_sizes 800 1600 2400 --output results/vehicleid.json "$@" 2>&1 | tee "${LOG}"
