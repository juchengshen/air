#!/bin/bash
# Submit all input-injection-specialization variants once.

set -euo pipefail

# Activate an environment with the project dependencies before running this script.
# For example: conda activate <your-env>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Input-injection-specialization variants

scripts=(
  train_sudoku_L_Hx.sh
  train_sudoku_Lx_H.sh  # default Lx_H
  train_sudoku_L_H2x.sh
  train_sudoku_L2x_H.sh
  train_sudoku_Lx_H2x.sh
  train_sudoku_L2x_Hx.sh
  train_sudoku_Lx_Hx.sh
  train_sudoku_L2x_H2x.sh
)

for s in "${scripts[@]}"; do
  echo "Submitting $s ..."
  sbatch "$s"
done

echo "Submitted ${#scripts[@]} jobs."
