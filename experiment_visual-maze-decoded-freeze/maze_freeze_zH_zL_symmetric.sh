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

export AIR_MAZE_CKPT_PATH="checkpoints/Maze/air_L2x_Hx_bs768_lr1e4_20k_epochs_run1/step_26040"
export AIR_MAZE_TEST_OUT_DIR="$SCRIPT_DIR/maze_freeze_symmetric_1run"

if [[ ! -f "$AIR_MAZE_CKPT_PATH" ]]; then
  echo "Checkpoint file not found: $AIR_MAZE_CKPT_PATH" >&2
  exit 1
fi
if [[ ! -f "$(dirname "$AIR_MAZE_CKPT_PATH")/all_config.yaml" ]]; then
  echo "Config file not found next to checkpoint: $(dirname "$AIR_MAZE_CKPT_PATH")/all_config.yaml" >&2
  exit 1
fi

mkdir -p "$AIR_MAZE_TEST_OUT_DIR"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting visual Maze freeze experiments for L2x_Hx checkpoint (full split)"
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

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed visual Maze freeze experiments for L2x_Hx checkpoint"
