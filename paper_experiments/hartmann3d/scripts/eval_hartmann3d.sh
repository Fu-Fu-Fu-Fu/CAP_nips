#!/bin/bash
# =============================================================================
# eval_hartmann3d.sh — Paper-level scale sweep evaluation for Hartmann-3D
#
# Usage:
#   bash paper_experiments/hartmann3d/eval_hartmann3d.sh
#
# Environment variables:
#   SURROGATE  - "gp" or "tabpfn_base" (default: tabpfn_base)
#   N_WORKERS  - parallel workers (default: 1)
#   PROGRESS_DETAIL - summary|variant|run|method (default: method)
#   SHOW_THIRD_PARTY_OUTPUT - 1 to show third-party warnings/logs (default: 0)
#   FRESH_RUN - 1 to ignore/delete existing resume checkpoints (default: 0)
#   PYTHON_BIN - python path (default: ${PYTHON_BIN:-python})
# =============================================================================

start_time=$(date +%s)
set -euo pipefail

export TF_USE_LEGACY_KERAS=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2
export TABPFN_DISABLE_TELEMETRY=1
export TABPFN_ENABLE_TELEMETRY_LOGS=0
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

cleanup () {
    end_time=$(date +%s)
    runtime=$((end_time - start_time))
    echo "Total time: $((runtime / 3600))h $(((runtime % 3600) / 60))m $((runtime % 60))s"
}
trap cleanup EXIT

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/MYRL:${REPO_ROOT}/olympus/src:${PYTHONPATH:-}"
export TABPFN_MODEL_CACHE_DIR="${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SURROGATE="${SURROGATE:-tabpfn_base}"
N_WORKERS="${N_WORKERS:-1}"
PROGRESS_DETAIL="${PROGRESS_DETAIL:-method}"
SHOW_THIRD_PARTY_OUTPUT="${SHOW_THIRD_PARTY_OUTPUT:-0}"
FRESH_RUN="${FRESH_RUN:-0}"

# --- Paths ---
MODEL_PATH="paper_experiments/hartmann3d/models/default/ppo_final.pt"
TAF_DATA="paper_experiments/hartmann3d/data/taf_source_data_hartmann_3d_family_k10.pkl"
SAVE_DIR="paper_experiments/hartmann3d/results/${SURROGATE}"

echo "=== Hartmann-3D Paper Evaluation ==="
echo "Surrogate: ${SURROGATE}"
echo "Model:     ${MODEL_PATH}"
echo "Save:      ${SAVE_DIR}"
echo "Workers:   ${N_WORKERS}"
echo "Progress:  ${PROGRESS_DETAIL}"
echo "3rd-party: ${SHOW_THIRD_PARTY_OUTPUT}"
echo "Resume:    $([[ "${FRESH_RUN}" == "1" ]] && echo fresh || echo enabled)"
echo ""

# Verify prerequisites
for path in "${MODEL_PATH}" "${TAF_DATA}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Missing: ${path}" >&2
        exit 1
    fi
done

EXTRA_ARGS=()
if [[ "${SHOW_THIRD_PARTY_OUTPUT}" == "1" ]]; then
    EXTRA_ARGS+=(--show_third_party_output)
fi
if [[ "${FRESH_RUN}" == "1" ]]; then
    EXTRA_ARGS+=(--fresh_run)
fi

"${PYTHON_BIN}" -u MYRL/scripts/eval_scale_sweep.py \
    --task hartmann_3d_family \
    --rl_model_path "${MODEL_PATH}" \
    --taf_data_path "${TAF_DATA}" \
    --surrogate "${SURROGATE}" \
    --scales 0.5 0.75 1.0 1.25 1.5 1.75 2.0 \
    --n_variants 20 \
    --n_runs 3 \
    --max_steps 18 \
    --n_init 2 \
    --n_persistent_base 128 \
    --n_total_candidates 192 \
    --k_centers 2 \
    --local_h 0.17 \
    --local_h_decay 0.95 \
    --n_candidates_baseline 2048 \
    --methods Random EI UCB PI FunBO PFNs4BO TuRBO TAF_me TAF_ranking CAP-PPO \
    --n_workers "${N_WORKERS}" \
    --progress_detail "${PROGRESS_DETAIL}" \
    --seed 2026 \
    --save_dir "${SAVE_DIR}" \
    "${EXTRA_ARGS[@]}"

# =============================================================================
# Step 2: Generate normalized regret plots (PNG + PDF)
# =============================================================================
PKL_PATH="${SAVE_DIR}/scale_sweep_data.pkl"
if [[ -f "${PKL_PATH}" ]]; then
    REPLOT_DIR="${SAVE_DIR}/replot"
    echo ""
    echo "[Step 2] Generating normalized regret plots (PNG + PDF)..."
    mkdir -p "${REPLOT_DIR}"
    "${PYTHON_BIN}" -u MYRL/scripts/plot_scale_sweep.py \
        --data "${PKL_PATH}" \
        --save_dir "${REPLOT_DIR}" \
        --individual \
        --formats png pdf
    echo "[Step 2] Plots saved to ${REPLOT_DIR}/"
fi

echo ""
echo "=== Done ==="
echo "Results: ${SAVE_DIR}/"
