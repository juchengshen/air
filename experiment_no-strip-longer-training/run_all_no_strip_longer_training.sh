#!/bin/bash
# Submit the 40k-epoch Lx_Hx prepend (no strip) level-token control once.

set -euo pipefail

# Activate an environment with the project dependencies before running this script.
# For example: conda activate <your-env>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Longer-training run for the persistent (no strip) level-token control

scripts=(
  train_sudoku_Lx_Hx_input_token_prepend_no_strip_40k.sh
)

for s in "${scripts[@]}"; do
  echo "Submitting $s ..."
  sbatch "$s"
done

echo "Submitted ${#scripts[@]} jobs."
