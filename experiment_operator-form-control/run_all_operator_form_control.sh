#!/bin/bash
# Submit all task-disambiguator variants once.

set -euo pipefail

# Activate an environment with the project dependencies before running this script.
# For example: conda activate <your-env>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

scripts=(
  train_sudoku_minus_x.sh
  train_sudoku_gx_linear.sh
  train_sudoku_gx_non_linear.sh
  train_sudoku_had_prod.sh
)

for s in "${scripts[@]}"; do
  echo "Submitting $s ..."
  sbatch "$s"
done

echo "Submitted ${#scripts[@]} jobs."
