#!/bin/bash
# =============================================================================
# eval_hartmann6d_znorm_rb.sh — Evaluate the NEW znorm+regret_balanced model
#
# Evaluates: runs/ppo_hartmann_6d_family_znorm_rb_ep5000/ppo_best.pt
# Saves to:  results_gp_znorm_rb/ (separate from old model results)
#
# Usage:
#   bash paper_experiments/hartmann6d/eval_hartmann6d_znorm_rb.sh
#
# Environment variables:
#   SURROGATE  - "gp" or "tabpfn_base" (default: gp)
#   N_WORKERS  - parallel workers (default: 8)
#   FRESH_RUN  - 1 to ignore existing resume checkpoints (default: 0)
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

SURROGATE="${SURROGATE:-gp}"
N_WORKERS="${N_WORKERS:-8}"
FRESH_RUN="${FRESH_RUN:-0}"

# --- Key difference: use the NEW znorm_rb model ---
EXP_DIR="paper_experiments/hartmann6d"
MODEL_PATH="${EXP_DIR}/runs/ppo_hartmann_6d_family_znorm_rb_ep5000/ppo_best.pt"
TAF_DATA="./data/taf_source_data_hartmann_6d_family_k10.pkl"
SAVE_DIR="${EXP_DIR}/results/znorm_rb_${SURROGATE}"

echo "=== Hartmann 6D — NEW znorm+regret_balanced Model Evaluation ==="
echo "Surrogate: ${SURROGATE}"
echo "Model:     ${MODEL_PATH} (NEW: znorm + regret_balanced + shuffled_cycle + ent_anneal)"
echo "Save:      ${SAVE_DIR}"
echo "Workers:   ${N_WORKERS}"
echo ""
echo "Compare with old model results in: ${EXP_DIR}/results_${SURROGATE}/"
echo ""

# Verify prerequisites
for path in "${MODEL_PATH}" "${TAF_DATA}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Missing: ${path}" >&2
        exit 1
    fi
done

EXTRA_ARGS=()
if [[ "${FRESH_RUN}" == "1" ]]; then
    EXTRA_ARGS+=(--fresh_run)
fi

# --- Run evaluation ---
echo "[Step 1] Running scale sweep evaluation..."

"${PYTHON_BIN}" -u MYRL/scripts/eval_scale_sweep.py \
    --task hartmann_6d_family \
    --rl_model_path "${MODEL_PATH}" \
    --taf_data_path "${TAF_DATA}" \
    --surrogate "${SURROGATE}" \
    --scales 0.5 0.75 1.0 1.25 1.5 1.75 2.0 \
    --n_variants 20 \
    --n_runs 3 \
    --max_steps 48 \
    --n_init 2 \
    --n_persistent_base 128 \
    --n_total_candidates 256 \
    --k_centers 3 \
    --local_h 0.15 \
    --local_h_decay 0.95 \
    --n_candidates_baseline 2048 \
    --methods Random EI UCB PI FunBO PFNs4BO TuRBO TAF_me TAF_ranking CAP-PPO \
    --n_workers "${N_WORKERS}" \
    --progress_detail method \
    --seed 2026 \
    --save_dir "${SAVE_DIR}" \
    "${EXTRA_ARGS[@]}"

echo "[Step 1] Evaluation complete."

# --- Generate plots ---
PKL_PATH="${SAVE_DIR}/scale_sweep_data.pkl"
if [[ -f "${PKL_PATH}" ]]; then
    REPLOT_DIR="${SAVE_DIR}/replot"
    echo ""
    echo "[Step 2] Generating plots..."
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
echo "New model results: ${SAVE_DIR}/"
echo "Old model results: ${EXP_DIR}/results_${SURROGATE}/"
echo ""
echo "To compare, load both pkl files:"
echo "  old: ${EXP_DIR}/results_${SURROGATE}/scale_sweep_data.pkl"
echo "  new: ${SAVE_DIR}/scale_sweep_data.pkl"
