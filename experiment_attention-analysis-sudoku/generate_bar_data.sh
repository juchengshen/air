#!/bin/bash
# Runs used a single H200 GPU, 24GB CPU memory, and a 6-hour wall time.

set -euo pipefail

# Activate an environment with the project dependencies before running this script.
# For example: conda activate <your-env>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export MPLBACKEND=Agg

CKPT_PATH="checkpoints/Sudoku/air_Lx_H_lr1e4_bs768_20k_epochs_run1/step_26040"
OUTPUT_DIR="$SCRIPT_DIR/bar_data"

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "Checkpoint file not found: $CKPT_PATH" >&2
  exit 1
fi
if [[ ! -f "$(dirname "$CKPT_PATH")/all_config.yaml" ]]; then
  echo "Config file not found next to checkpoint: $(dirname "$CKPT_PATH")/all_config.yaml" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Generating Sudoku attention bar-chart data"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Using checkpoint $CKPT_PATH"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Writing per-layer JSONs to $OUTPUT_DIR"
OMP_NUM_THREADS=8 python "$SCRIPT_DIR/generate_bar_data.py" \
  --ckpt "$CKPT_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --num-puzzles 1000
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed Sudoku attention bar-chart data generation"
