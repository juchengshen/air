#!/bin/bash
# Runs used a single H200 GPU, 16GB CPU memory, and a 2-hour wall time.

set -euo pipefail

# Activate an environment with the project dependencies before running this script.
# For example: conda activate <your-env>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export MPLBACKEND=Agg
export AIR_MAZE_NUM_QUESTIONS=1000  # Note: Maze whole test split is 1000 questions.

mkdir -p "$SCRIPT_DIR"

RUN_ROOT="$SCRIPT_DIR/maze_freeze_asymmetric_5runs"
mkdir -p "$RUN_ROOT"
MANIFEST_FILE="${RUN_ROOT}/run_manifest.txt"
: > "$MANIFEST_FILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting visual Maze freeze experiments across 5 runs (1000 samples each)"

for run_label in run1 run2 run3 run4 run5
do
  ckpt_step="step_26040"
  ckpt_dir="checkpoints/Maze/air_Lx_H_bs768_lr1e4_20k_epochs_${run_label}"

  export AIR_MAZE_CKPT_PATH="${ckpt_dir}/${ckpt_step}"
  export AIR_MAZE_TEST_OUT_DIR="${RUN_ROOT}/${run_label}"

  if [[ ! -f "$AIR_MAZE_CKPT_PATH" ]]; then
    echo "Checkpoint file not found: $AIR_MAZE_CKPT_PATH" >&2
    exit 1
  fi
  if [[ ! -f "$(dirname "$AIR_MAZE_CKPT_PATH")/all_config.yaml" ]]; then
    echo "Config file not found next to checkpoint: $(dirname "$AIR_MAZE_CKPT_PATH")/all_config.yaml" >&2
    exit 1
  fi

  mkdir -p "$AIR_MAZE_TEST_OUT_DIR"
  echo "${run_label}|${AIR_MAZE_TEST_OUT_DIR}" >> "$MANIFEST_FILE"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Run=${run_label}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Using checkpoint ${AIR_MAZE_CKPT_PATH}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Writing outputs to ${AIR_MAZE_TEST_OUT_DIR}"


  for script in \
    experiment_visual-maze-decoded-freeze/maze_freeze_zH.py \
    experiment_visual-maze-decoded-freeze/maze_freeze_zL.py
  do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running ${script}"
    OMP_NUM_THREADS=8 python "$script"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished ${script}"
  done
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Computing 5-run aggregate summaries"
python experiment_visual-maze-decoded-freeze/plot_maze_freeze.py --aggregate-manifest "$MANIFEST_FILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed visual Maze freeze experiments across 5 runs"
