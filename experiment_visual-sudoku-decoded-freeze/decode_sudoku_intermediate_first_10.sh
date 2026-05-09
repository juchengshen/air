#!/bin/bash
# Runs used a single H200 GPU, 16GB CPU memory, and a 1-hour wall time.

set -euo pipefail

# Activate an environment with the project dependencies before running this script.
# For example: conda activate <your-env>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export MPLBACKEND=Agg

mkdir -p "$SCRIPT_DIR"

SCRIPT="experiment_visual-sudoku-decoded-freeze/decode_sudoku_intermediate.py"
CKPT_PATH="checkpoints/Sudoku/air_Lx_H_lr1e4_bs768_20k_epochs_run1/step_26040"
OUTPUT_DIR="$SCRIPT_DIR/sudoku_asymmetric_decode_first_10"

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "Checkpoint file not found: $CKPT_PATH" >&2
  exit 1
fi

if [[ ! -f "$(dirname "$CKPT_PATH")/all_config.yaml" ]]; then
  echo "Config file not found next to checkpoint: $(dirname "$CKPT_PATH")/all_config.yaml" >&2
  exit 1
fi

export AIR_SUDOKU_CKPT_PATH="$CKPT_PATH"
export AIR_SUDOKU_OUT_DIR="$OUTPUT_DIR"
export AIR_VISUAL_NUM_QUESTIONS=10
mkdir -p "$AIR_SUDOKU_OUT_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting visual Sudoku intermediate decode (first ${AIR_VISUAL_NUM_QUESTIONS})"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Using checkpoint ${AIR_SUDOKU_CKPT_PATH}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Writing outputs to ${AIR_SUDOKU_OUT_DIR}"
OMP_NUM_THREADS=8 python "$SCRIPT"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed visual Sudoku intermediate decode"
