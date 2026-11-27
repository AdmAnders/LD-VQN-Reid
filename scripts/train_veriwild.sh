#!/usr/bin/env bash
# Train LD-VQN on T2I-VeRi-Wild.
# Usage: bash scripts/train_veriwild.sh [extra train.py args, e.g. --opts SOLVER.EPOCHS 10]
# GPU selection: GPU_ID=1 bash scripts/train_veriwild.sh
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/veriwild_ldvqn.yml"
GPU_ID="${GPU_ID:-0}"
mkdir -p logs
LOG="logs/train_veriwild_$(date +%Y%m%d_%H%M%S).log"

echo "[*] Training on T2I-VeRi-Wild | GPU ${GPU_ID} | log: ${LOG}"
CUDA_VISIBLE_DEVICES=${GPU_ID} python -u train.py --config "${CONFIG}" "$@" 2>&1 | tee "${LOG}"
