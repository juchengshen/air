#!/bin/bash
# Submit all addition/prepend/strip/no-strip 1net variants once.

set -euo pipefail

# Activate an environment with the project dependencies before running this script.
# For example: conda activate <your-env>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Addition/prepend/strip/no-strip 1net variants on the symmetric Lx_Hx base (paper Table 2).
# The matching no-token Lx_Hx baseline is
# experiment_input-injection-specialization-sudoku/train_sudoku_Lx_Hx.sh.

scripts=(
  train_sudoku_Lx_Hx_input_token_addition.sh
  train_sudoku_Lx_Hx_input_token_prepend.sh
  train_sudoku_Lx_Hx_input_token_prepend_no_strip.sh
)

for s in "${scripts[@]}"; do
  echo "Submitting $s ..."
  sbatch "$s"
done

echo "Submitted ${#scripts[@]} jobs."
