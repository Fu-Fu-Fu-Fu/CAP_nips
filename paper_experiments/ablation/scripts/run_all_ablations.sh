#!/bin/bash
# =============================================================================
# run_all_ablations.sh — Master orchestrator for all ablation experiments
#
# ALL experiments use the NEW version (regret_balanced + z-norm).
# Each experiment: Train(n_workers=8) → Eval(GP) → Eval(TabPFN) → Plot
#
# 4 ablation dimensions × 2 tasks × 2 variants = 16 training + 2 default = 18 configs
#
# Usage:
#   bash paper_experiments/ablation/scripts/run_all_ablations.sh [phase]
#
# Phases:
#   data    - Generate K=5, K=20, and budget data
#   run     - Run all 18 experiments (train + eval + plot each)
#   all     - data + run (default)
#
# To run a single experiment:
#   TASK=hartmann_6d_family ABLATION_NAME=pool_size_small \
#     AB_N_PERSISTENT_BASE=64 AB_N_TOTAL_CANDIDATES=128 \
#     bash paper_experiments/ablation/scripts/run_one_ablation.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}"
cd "${REPO_ROOT}"

SCRIPTS="paper_experiments/ablation/scripts"
PHASE="${1:-all}"

export N_WORKERS="${N_WORKERS:-8}"
export AB_N_WORKERS="${AB_N_WORKERS:-8}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ===== Helper: run one full experiment =====
run_exp() {
    local task="$1" name="$2"
    shift 2
    log ">>> START: ${task} / ${name}"
    env TASK="${task}" ABLATION_NAME="${name}" "$@" \
        bash "${SCRIPTS}/run_one_ablation.sh" 2>&1 | \
        tee -a "paper_experiments/ablation/sensitivity/${name}_${task}.log"
    log ">>> DONE:  ${task} / ${name}"
    echo ""
}

# =============================================================================
# Phase: DATA GENERATION
# =============================================================================
if [[ "${PHASE}" == "data" ]] || [[ "${PHASE}" == "all" ]]; then
    log "========== DATA GENERATION =========="

    # K=5 truncation (from existing K=10 data)
    log "Generating K=5 data..."
    ${PYTHON_BIN:-python} "${SCRIPTS}/generate_k5_data.py"

    # K=20 + different budgets
    log "Generating K=20 and budget data..."
    bash "${SCRIPTS}/generate_data.sh"

    log "Data generation complete."
fi

# =============================================================================
# Phase: RUN (train + eval_gp + eval_tabpfn + plot for each)
# =============================================================================
if [[ "${PHASE}" == "run" ]] || [[ "${PHASE}" == "all" ]]; then
    log "========== RUNNING 18 EXPERIMENTS =========="

    # =========================================================================
    # 0. Default baselines (new version, standard hyperparams)
    # =========================================================================
    log "=== Default baselines ==="
    run_exp hartmann_6d_family default
    run_exp alkox_emulator default

    # =========================================================================
    # 1. Pool Size
    # =========================================================================
    log "=== Ablation 1: Pool Size ==="

    # Hartmann-6D
    run_exp hartmann_6d_family pool_size_small \
        AB_N_PERSISTENT_BASE=64 AB_N_TOTAL_CANDIDATES=128
    run_exp hartmann_6d_family pool_size_large \
        AB_N_PERSISTENT_BASE=256 AB_N_TOTAL_CANDIDATES=512

    # Alkox
    run_exp alkox_emulator pool_size_small \
        AB_N_PERSISTENT_BASE=64 AB_N_TOTAL_CANDIDATES=96
    run_exp alkox_emulator pool_size_large \
        AB_N_PERSISTENT_BASE=256 AB_N_TOTAL_CANDIDATES=384

    # =========================================================================
    # 2. Pool Construction (local candidate ratio)
    # =========================================================================
    log "=== Ablation 2: Pool Construction ==="

    # Hartmann-6D
    run_exp hartmann_6d_family pool_construct_no_local \
        AB_N_PERSISTENT_BASE=256 AB_N_TOTAL_CANDIDATES=256 AB_K_CENTERS=0
    run_exp hartmann_6d_family pool_construct_heavy_local \
        AB_N_PERSISTENT_BASE=64 AB_N_TOTAL_CANDIDATES=256 AB_K_CENTERS=6

    # Alkox
    run_exp alkox_emulator pool_construct_no_local \
        AB_N_PERSISTENT_BASE=192 AB_N_TOTAL_CANDIDATES=192 AB_K_CENTERS=0
    run_exp alkox_emulator pool_construct_heavy_local \
        AB_N_PERSISTENT_BASE=48 AB_N_TOTAL_CANDIDATES=192 AB_K_CENTERS=6

    # =========================================================================
    # 3. K Variants (training task diversity)
    # =========================================================================
    log "=== Ablation 3: K Variants ==="

    run_exp hartmann_6d_family k_variants_5 AB_K_VARIANTS=5
    run_exp hartmann_6d_family k_variants_20 AB_K_VARIANTS=20
    run_exp alkox_emulator k_variants_5 AB_K_VARIANTS=5
    run_exp alkox_emulator k_variants_20 AB_K_VARIANTS=20

    # =========================================================================
    # 4. Budget (Trajectory Length)
    # =========================================================================
    log "=== Ablation 4: Budget ==="

    # Hartmann-6D: short=20 evals, long=80 evals (default=50)
    run_exp hartmann_6d_family budget_short \
        AB_MAX_STEPS=18 AB_DATA_SUFFIX="_budget20"
    run_exp hartmann_6d_family budget_long \
        AB_MAX_STEPS=78 AB_DATA_SUFFIX="_budget80"

    # Alkox: short=15 evals, long=50 evals (default=30)
    run_exp alkox_emulator budget_short \
        AB_MAX_STEPS=13 AB_DATA_SUFFIX="_transform_budget15"
    run_exp alkox_emulator budget_long \
        AB_MAX_STEPS=48 AB_DATA_SUFFIX="_transform_budget50"

    log "========== ALL 18 EXPERIMENTS COMPLETE =========="
fi

log "========== ALL DONE =========="
echo ""
echo "Results: paper_experiments/ablation/sensitivity/"
echo ""
echo "Quick summary:"
echo "  find paper_experiments/ablation/sensitivity -name 'scale_sweep_data.pkl'"
echo "  find paper_experiments/ablation/sensitivity -name '*.png' -path '*/replot/*'"
