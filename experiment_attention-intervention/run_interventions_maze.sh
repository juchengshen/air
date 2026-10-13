#!/bin/bash
# Runs used a single NVIDIA GH200 GPU, 48GB CPU memory, and a 2-hour wall time.

set -euo pipefail

# Activate an environment with the project dependencies before running this script.
# For example: conda activate <your-env>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR="$SCRIPT_DIR/results_maze"
mkdir -p "$OUTPUT_DIR"

# Each seed is evaluated at its best-evaluation checkpoint.
declare -A CKPT_STEPS=(["run1"]="step_26040" ["run2"]="step_18228" ["run3"]="step_20832" ["run4"]="step_18228" ["run5"]="step_13020")

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Maze attention interventions across 5 runs"

for run_label in run1 run2 run3 run4 run5
do
  ckpt_step="${CKPT_STEPS[$run_label]}"
  ckpt_dir="checkpoints/Maze/air_Lx_H_bs768_lr1e4_20k_epochs_${run_label}"
  ckpt_path="${ckpt_dir}/${ckpt_step}"

  if [[ ! -f "$ckpt_path" ]]; then
    echo "Checkpoint file not found: $ckpt_path" >&2
    exit 1
  fi
  if [[ ! -f "$(dirname "$ckpt_path")/all_config.yaml" ]]; then
    echo "Config file not found next to checkpoint: $(dirname "$ckpt_path")/all_config.yaml" >&2
    exit 1
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Run=${run_label}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Using checkpoint ${ckpt_path}"
  OMP_NUM_THREADS=8 python "$SCRIPT_DIR/run_interventions.py" \
    --task maze \
    --ckpt "$ckpt_path" \
    --num-puzzles 1000 \
    --output "$OUTPUT_DIR/${run_label}.json"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Aggregating 5-run results"
python "$SCRIPT_DIR/aggregate_interventions.py" \
  --inputs "$OUTPUT_DIR/run*.json" \
  --output-md "$OUTPUT_DIR/summary.md" \
  --output-json "$OUTPUT_DIR/summary.json"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed Maze attention interventions"
