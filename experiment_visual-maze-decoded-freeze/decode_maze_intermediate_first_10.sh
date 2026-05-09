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

SCRIPT="experiment_visual-maze-decoded-freeze/decode_maze_intermediate.py"
CKPT_PATH="checkpoints/Maze/air_Lx_H_bs768_lr1e4_20k_epochs_run1/step_26040"
BASE_OUT_DIR="$SCRIPT_DIR/maze_Lx_H_decode_first_10"

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "Checkpoint file not found: $CKPT_PATH" >&2
  exit 1
fi

if [[ ! -f "$(dirname "$CKPT_PATH")/all_config.yaml" ]]; then
  echo "Config file not found next to checkpoint: $(dirname "$CKPT_PATH")/all_config.yaml" >&2
  exit 1
fi

export AIR_MAZE_CKPT_PATH="$CKPT_PATH"
mkdir -p "$BASE_OUT_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting visual Maze intermediate decode (first 10)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Using checkpoint ${AIR_MAZE_CKPT_PATH}"

for batch_index in $(seq 0 9)
do
  export AIR_MAZE_BATCH_INDEX="$batch_index"
  export AIR_MAZE_OUT_DIR="${BASE_OUT_DIR}/question_$(printf '%02d' "$batch_index")"
  mkdir -p "$AIR_MAZE_OUT_DIR"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Decoding batch index ${AIR_MAZE_BATCH_INDEX}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Writing outputs to ${AIR_MAZE_OUT_DIR}"
  OMP_NUM_THREADS=8 python "$SCRIPT"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed visual Maze intermediate decode (first 10)"
