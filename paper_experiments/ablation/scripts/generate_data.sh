#!/bin/bash
# =============================================================================
# generate_data.sh — Generate ablation data: K=20 variants + different budget trajectories
#
# This generates:
#   1. K=20 variants + trajectories + TAF for Hartmann-6D and Alkox
#   2. Short/Long budget trajectories + TAF for Hartmann-6D and Alkox
#
# Usage:
#   bash paper_experiments/ablation/scripts/generate_data.sh
# =============================================================================

set -euo pipefail

export TF_USE_LEGACY_KERAS=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2
export TABPFN_DISABLE_TELEMETRY=1
export TABPFN_ENABLE_TELEMETRY_LOGS=0
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/MYRL:${REPO_ROOT}/olympus/src:${PYTHONPATH:-}"
export TABPFN_MODEL_CACHE_DIR="${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"

VARIANT_SEED=2026
BO_SEED=2026

# Helper: generate variants + trajectories + TAF
generate_step1_and_taf() {
    local TASK="$1"
    local K="$2"
    local TOTAL_EVALS="$3"
    local N_INIT="$4"
    local SUFFIX="$5"  # e.g., "_transform" for alkox

    local VARIANTS_CACHE="data/${TASK}_variants_k${K}_seed${VARIANT_SEED}${SUFFIX}.npz"
    local TRAJS_CACHE="data/${TASK}_bo_trajs_k${K}_boSeed${BO_SEED}${SUFFIX}.npz"
    local TAF_DATA="data/taf_source_data_${TASK}_k${K}${SUFFIX}.pkl"

    echo ""
    echo "--- ${TASK} K=${K} budget=${TOTAL_EVALS} ---"

    if [[ -f "${VARIANTS_CACHE}" ]] && [[ -f "${TRAJS_CACHE}" ]]; then
        echo "[Step 1] SKIP — data caches exist: ${VARIANTS_CACHE}"
    else
        echo "[Step 1] Generate variants + trajectories..."
        local MAX_CTX=$(( TOTAL_EVALS - 1 ))
        "${PYTHON_BIN}" -u MYRL/scripts/finetune.py \
            --task "${TASK}" \
            --stage generate \
            --k_variants "${K}" \
            --variant_seed "${VARIANT_SEED}" \
            --bo_seed "${BO_SEED}" \
            --variants_cache "${VARIANTS_CACHE}" \
            --trajectories_cache "${TRAJS_CACHE}" \
            --n_trials_per_variant 5 \
            --n_synth_trajectories_per_variant 0 \
            --total_evals "${TOTAL_EVALS}" \
            --max_context "${MAX_CTX}" \
            --n_init "${N_INIT}" \
            --xi 0.01 \
            --gp_n_restarts_optimizer 5
        echo "[Step 1] Done."
    fi

    if [[ -f "${TAF_DATA}" ]]; then
        echo "[Step 1.5] SKIP — TAF data exists: ${TAF_DATA}"
    else
        echo "[Step 1.5] Generate TAF source data..."
        "${PYTHON_BIN}" -u -c "
from myrl.rl.train_rl import prepare_taf_data
prepare_taf_data('${TRAJS_CACHE}', '${TAF_DATA}')
print('TAF saved: ${TAF_DATA}')
"
        echo "[Step 1.5] Done."
    fi
}

echo "================================================================"
echo "  Ablation Data Generation"
echo "================================================================"

# =====================================================================
# 1. K=20 variants (default budget)
# =====================================================================
echo ""
echo "=== K=20 variant data ==="

# Hartmann-6D: K=20, budget=50 (2+48)
generate_step1_and_taf "hartmann_6d_family" 20 50 2 ""

# Alkox: K=20, budget=30 (2+28)
generate_step1_and_taf "alkox_emulator" 20 30 2 "_transform"

# Alkox K=20 also needs BNN surrogates — train BNN for 20 variants
ALKOX_BNN_K20="data/bnn_surrogates_alkox_emulator_k20_kl0.001_transform.npz"
if [[ -f "${ALKOX_BNN_K20}" ]]; then
    echo "[Step 1.5b] SKIP — BNN surrogates K=20 exist: ${ALKOX_BNN_K20}"
else
    echo "[Step 1.5b] Train BNN surrogates for Alkox K=20..."
    "${PYTHON_BIN}" -u MYRL/scripts/train_bnn_surrogates.py \
        --trajs_path "data/alkox_emulator_bo_trajs_k20_boSeed${BO_SEED}_transform.npz" \
        --output_path "${ALKOX_BNN_K20}" \
        --kl_weight 0.001
    echo "[Step 1.5b] Done."
fi

# =====================================================================
# 2. Different budget trajectories (K=10, varied total_evals)
# =====================================================================
echo ""
echo "=== Different budget trajectory data ==="

# --- Hartmann-6D ---
# Short: 20 evals (2+18)
generate_step1_and_taf "hartmann_6d_family" 10 20 2 "_budget20"
# Long: 80 evals (2+78)
generate_step1_and_taf "hartmann_6d_family" 10 80 2 "_budget80"

# --- Alkox (budget variants share the same K=10 BNN surrogates) ---
# The BNN surrogates are variant-level, not budget-dependent.
# But variants/trajs npz have different budget suffixes, while BNN params
# are keyed to the base K=10 transform data. We symlink if needed.
# Short: 15 evals (2+13)
generate_step1_and_taf "alkox_emulator" 10 15 2 "_transform_budget15"
# Long: 50 evals (2+48)
generate_step1_and_taf "alkox_emulator" 10 50 2 "_transform_budget50"

# Budget variants reuse the same K=10 BNN surrogates (variant params don't change with budget)
for BSUF in "_transform_budget15" "_transform_budget50"; do
    BNN_LINK="data/bnn_surrogates_alkox_emulator_k10_kl0.001${BSUF}.npz"
    BNN_SRC="data/bnn_surrogates_alkox_emulator_k10_kl0.001_transform.npz"
    if [[ ! -f "${BNN_LINK}" ]] && [[ -f "${BNN_SRC}" ]]; then
        cp "${BNN_SRC}" "${BNN_LINK}"
        echo "  Copied BNN params: ${BNN_LINK}"
    fi
done

echo ""
echo "================================================================"
echo "  All data generation complete!"
echo "================================================================"
echo ""
echo "Generated files:"
echo "  K=20: hartmann_6d_family_*_k20_*, alkox_emulator_*_k20_*"
echo "  Budget: *_budget{15,20,50,80}*"
echo ""
echo "K=5 data can be generated separately with:"
echo "  python paper_experiments/ablation/scripts/generate_k5_data.py"
