#!/bin/bash
# Runs used a single H200 GPU, 16GB CPU memory, and a 24-hour wall time.
# Freeze experiments on symmetric Lx_Hx Sudoku checkpoints for seeds 0-2 (run names run1-run3), as in the paper.

set -euo pipefail

# Activate an environment with the project dependencies before running this script.
# For example: conda activate <your-env>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export MPLBACKEND=Agg
export AIR_SUDOKU_NUM_QUESTIONS=1000

mkdir -p "$SCRIPT_DIR"

RUN_ROOT="$SCRIPT_DIR/sudoku_freeze_symmetric_3runs"
mkdir -p "$RUN_ROOT"
MANIFEST_FILE="${RUN_ROOT}/run_manifest.txt"
: > "$MANIFEST_FILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting visual Sudoku freeze experiments for Lx_Hx across 3 runs (full split)"

for run_label in run1 run2 run3
do
  ckpt_step="step_26040"
  ckpt_dir="checkpoints/Sudoku/air_Lx_Hx_lr1e4_bs768_20k_epochs_${run_label}"

  export AIR_SUDOKU_CKPT_PATH="${ckpt_dir}/${ckpt_step}"
  export AIR_SUDOKU_TEST_OUT_DIR="${RUN_ROOT}/${run_label}"

  if [[ ! -f "$AIR_SUDOKU_CKPT_PATH" ]]; then
    echo "Checkpoint file not found: $AIR_SUDOKU_CKPT_PATH" >&2
    exit 1
  fi
  if [[ ! -f "$(dirname "$AIR_SUDOKU_CKPT_PATH")/all_config.yaml" ]]; then
    echo "Config file not found next to checkpoint: $(dirname "$AIR_SUDOKU_CKPT_PATH")/all_config.yaml" >&2
    exit 1
  fi

  mkdir -p "$AIR_SUDOKU_TEST_OUT_DIR"
  echo "${run_label}|${AIR_SUDOKU_TEST_OUT_DIR}" >> "$MANIFEST_FILE"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Run=${run_label}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Using checkpoint ${AIR_SUDOKU_CKPT_PATH}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Writing outputs to ${AIR_SUDOKU_TEST_OUT_DIR}"

  for script in \
    experiment_visual-sudoku-decoded-freeze/sudoku_freeze_zH.py \
    experiment_visual-sudoku-decoded-freeze/sudoku_freeze_zL.py
  do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running ${script}"
    OMP_NUM_THREADS=8 python "$script"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished ${script}"
  done
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Computing 3-run aggregate summaries"
python experiment_visual-sudoku-decoded-freeze/plot_sudoku_freeze.py --aggregate-manifest "$MANIFEST_FILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Plotting per-seed results"
python experiment_visual-sudoku-decoded-freeze/plot_sudoku_freeze_per_seed.py --manifest "$MANIFEST_FILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed visual Sudoku freeze experiments for Lx_Hx across 3 runs"
