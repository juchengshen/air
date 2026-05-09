#!/bin/bash
# Runs used 8 H200 GPUs, 16GB CPU memory, and a 3-hour wall time.

set -euo pipefail

# Activate an environment with the project dependencies before running this script.
# For example: conda activate <your-env>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

wandb login "${WANDB_API_KEY}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting AIR training suite"

# Full sweeps use seeds 0-4 with matching run names run1-run5; one run per seed.
OMP_NUM_THREADS=8 torchrun --nproc_per_node=8 pretrain.py \
  data_path=data/maze-30x30-hard-1k \
  epochs=20000 \
  eval_interval=2000 \
  global_batch_size=768 \
  lr=1e-4 puzzle_emb_lr=1e-4 \
  weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  +seed=0 \
  +project_name=Maze \
  +run_name=air_Lx_H_bs768_lr1e4_20k_epochs_run1 \
  arch.name=air.air_1net_Lx_H@AIR

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training suite completed"
