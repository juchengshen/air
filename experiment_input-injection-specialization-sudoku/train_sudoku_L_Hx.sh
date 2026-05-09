#!/bin/bash
# Runs used a single H200 GPU, 16GB CPU memory, and a 4-hour wall time.

set -euo pipefail

# Activate an environment with the project dependencies before running this script.
# For example: conda activate <your-env>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

wandb login "${WANDB_API_KEY}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting AIR training suite"

# Full sweeps use seeds 0-4 with matching run names run1-run5; one run per seed.
OMP_NUM_THREADS=8 python pretrain.py \
  data_path=data/sudoku-extreme-1k-aug-1000 \
  epochs=20000 \
  eval_interval=2000 \
  global_batch_size=768 \
  lr=1e-4 puzzle_emb_lr=1e-4 \
  weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  +seed=0 \
  +project_name=Sudoku \
  +run_name=air_L_Hx_lr1e4_bs768_20k_epochs_run1 \
  arch.name=air.air_1net_L_Hx@AIR

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training suite completed"
