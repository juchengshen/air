#!/bin/bash
# Renders the Maze attention example figures (core comparison + temporal
# evolution) from precomputed example_data/*.npz. No model run required.

set -euo pipefail

# Activate an environment with the project dependencies before running this script.
# For example: conda activate <your-env>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export MPLBACKEND=Agg

OUTPUT_DIR="${1:-$SCRIPT_DIR/example_data}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Rendering Maze attention example figures"
python "$SCRIPT_DIR/render_example.py" \
  --example-dir "$SCRIPT_DIR/example_data" \
  --output-dir "$OUTPUT_DIR"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done -> $OUTPUT_DIR"
