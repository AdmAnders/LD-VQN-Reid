#!/usr/bin/env bash
# Train LD-VQN on T2I-VeRi-776.
# Usage: bash scripts/train_veri776.sh [extra train.py args, e.g. --opts SOLVER.EPOCHS 10]
# GPU selection: GPU_ID=1 bash scripts/train_veri776.sh
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/veri776_ldvqn.yml"
GPU_ID="${GPU_ID:-0}"
mkdir -p logs
LOG="logs/train_veri776_$(date +%Y%m%d_%H%M%S).log"

echo "[*] Training on T2I-VeRi-776 | GPU ${GPU_ID} | log: ${LOG}"
CUDA_VISIBLE_DEVICES=${GPU_ID} python -u train.py --config "${CONFIG}" "$@" 2>&1 | tee "${LOG}"
