#!/bin/bash
# =============================================================================
# run_one_ablation.sh — Train + Eval(GP) + Eval(TabPFN) + Plot for one ablation
#
# Each step is skippable if already completed (resume-safe).
#
# Required env vars:
#   TASK          - "hartmann_6d_family" or "alkox_emulator"
#   ABLATION_NAME - e.g., "pool_size_small", "default"
#
# Optional env vars (passed through to train/eval):
#   AB_N_PERSISTENT_BASE  AB_N_TOTAL_CANDIDATES  AB_K_CENTERS
#   AB_K_VARIANTS         AB_MAX_STEPS           AB_TOTAL_EPISODES
#   AB_N_WORKERS          AB_DATA_SUFFIX         N_WORKERS
#
# Usage:
#   TASK=hartmann_6d_family ABLATION_NAME=pool_size_small \
#     AB_N_PERSISTENT_BASE=64 AB_N_TOTAL_CANDIDATES=128 \
#     bash paper_experiments/ablation/scripts/run_one_ablation.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}"
SCRIPTS="${REPO_ROOT}/paper_experiments/ablation/scripts"
PYTHON_BIN="${PYTHON_BIN:-python}"

: "${TASK:?TASK must be set}"
: "${ABLATION_NAME:?ABLATION_NAME must be set}"

export N_WORKERS="${N_WORKERS:-8}"
export AB_N_WORKERS="${AB_N_WORKERS:-8}"

TASK_SHORT="${TASK//_emulator/}"
TASK_SHORT="${TASK_SHORT//_family/}"
EXP_DIR="paper_experiments/ablation/sensitivity/${TASK_SHORT}/${ABLATION_NAME}"

start_time=$(date +%s)

echo ""
echo "################################################################"
echo "  ${TASK_SHORT} / ${ABLATION_NAME}"
echo "################################################################"
echo ""

# ===== Step 1: Train =====
echo "[Step 1/4] Training..."
bash "${SCRIPTS}/train_ablation.sh"

# ===== Step 2: Eval (GP surrogate) =====
echo ""
echo "[Step 2/4] Eval (GP surrogate)..."
SURROGATE=gp bash "${SCRIPTS}/eval_ablation.sh"

# ===== Step 3: Eval (TabPFN surrogate) =====
echo ""
echo "[Step 3/4] Eval (TabPFN surrogate)..."
SURROGATE=tabpfn_base bash "${SCRIPTS}/eval_ablation.sh"

# ===== Step 4: Plot =====
echo ""
echo "[Step 4/4] Generating plots..."
for SURR in gp tabpfn_base; do
    PKL="${REPO_ROOT}/${EXP_DIR}/results_${SURR}/scale_sweep_data.pkl"
    PLOT_DIR="${REPO_ROOT}/${EXP_DIR}/results_${SURR}/replot"
    if [[ -f "${PKL}" ]]; then
        mkdir -p "${PLOT_DIR}"
        "${PYTHON_BIN}" -u MYRL/scripts/plot_scale_sweep.py \
            --data "${PKL}" \
            --save_dir "${PLOT_DIR}" \
            --individual \
            --formats png pdf
        echo "  Plots: ${PLOT_DIR}/"
    else
        echo "  SKIP plot for ${SURR} — no data"
    fi
done

end_time=$(date +%s)
runtime=$((end_time - start_time))
echo ""
echo "================================================================"
echo "  DONE: ${TASK_SHORT} / ${ABLATION_NAME}"
echo "  Time: $((runtime / 3600))h $(((runtime % 3600) / 60))m $((runtime % 60))s"
echo "================================================================"
