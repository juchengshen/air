#!/bin/bash
# Uses a single H200 GPU and 16GB CPU memory; allow about twice the wall time of the 20,000-epoch level-token runs.
# Trains the Lx_Hx prepend (no strip) level-token control to 40,000 epochs, twice the
# 20,000-epoch budget of the main level-token comparison. checkpoint_every_eval=True
# (the default in config/cfg_pretrain.yaml) keeps a checkpoint at every 2,000-epoch eval.

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
  epochs=40000 \
  eval_interval=2000 \
  checkpoint_every_eval=true \
  global_batch_size=768 \
  lr=1e-4 puzzle_emb_lr=1e-4 \
  weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  +seed=0 \
  +project_name=Sudoku_new \
  +run_name=air_Lx_Hx_input_token_prepend_no_strip_lr1e4_bs768_40k_epochs_run1 \
  arch.name=air.air_1net_Lx_Hx_input_token_prepend_no_strip@AIR

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training suite completed"
