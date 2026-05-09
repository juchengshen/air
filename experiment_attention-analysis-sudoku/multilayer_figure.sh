#!/bin/bash
# Renders the Sudoku 1x4 multilayer attention bar chart from the per-layer
# JSONs in bar_data/. CPU only.

set -euo pipefail

# Activate an environment with the project dependencies before running this script.
# For example: conda activate <your-env>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export MPLBACKEND=Agg

OUTPUT_PNG="$SCRIPT_DIR/quant_summary_multilayer.png"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Rendering Sudoku multilayer bar chart"
python "$SCRIPT_DIR/multilayer_figure.py" \
  --base-dir "$SCRIPT_DIR/bar_data" \
  --layers 0,1,2,3 \
  --output-png "$OUTPUT_PNG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Wrote $OUTPUT_PNG"
