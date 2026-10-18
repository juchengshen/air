#!/bin/bash
# Renders the longer-training figure for the Lx_Hx prepend (no strip) control from
# precomputed example_data/no_strip_training_data.json. No model run required.

set -euo pipefail

# Activate an environment with the project dependencies before running this script.
# For example: conda activate <your-env>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export MPLBACKEND=Agg

OUTPUT_DIR="${1:-$SCRIPT_DIR/example_data}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Rendering no-strip longer-training figure"
python "$SCRIPT_DIR/plot_no_strip_training.py" \
  --data "$SCRIPT_DIR/example_data/no_strip_training_data.json" \
  --output-dir "$OUTPUT_DIR"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done -> $OUTPUT_DIR"
