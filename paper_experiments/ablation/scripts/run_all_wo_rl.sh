#!/bin/bash
# =============================================================================
# Run all Ablation 1 (w/o RL) experiments — Hartmann-6D + Alkox only
#
# Usage:
#   SURROGATE=gp bash paper_experiments/ablation/scripts/run_all_wo_rl.sh
#   SURROGATE=tabpfn_base bash paper_experiments/ablation/scripts/run_all_wo_rl.sh
# =============================================================================
set -euo pipefail

SURROGATE="${SURROGATE:-gp}"
SCRIPT_DIR="paper_experiments/ablation/scripts"

echo "=============================================="
echo "  Ablation 1: w/o RL — Hartmann-6D + Alkox"
echo "  Surrogate: ${SURROGATE}"
echo "=============================================="

for TASK in hartmann6d alkox; do
    echo ""
    echo ">>> Starting ${TASK} ..."
    SURROGATE="${SURROGATE}" bash "${SCRIPT_DIR}/eval_wo_rl_${TASK}.sh"
    echo ">>> Finished ${TASK}"
done

echo ""
echo "=============================================="
echo "  All Ablation 1 experiments complete!"
echo "=============================================="
