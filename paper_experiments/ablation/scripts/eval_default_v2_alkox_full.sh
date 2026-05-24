#!/bin/bash
# =============================================================================
# Evaluate Alkox Default v2 with ALL 10 baselines (GP + TabPFN in parallel)
#
# Usage:
#   bash paper_experiments/ablation/scripts/eval_default_v2_alkox_full.sh
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

MODEL_PATH="paper_experiments/ablation/component/alkox/default_v2/ppo_best.pt"
TAF_DATA="./data/taf_source_data_alkox_emulator_k10_transform.pkl"

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "ERROR: Model not found at ${MODEL_PATH}"
    exit 1
fi

COMMON_ARGS=(
    --task alkox_emulator
    --rl_model_path "${MODEL_PATH}"
    --taf_data_path "${TAF_DATA}"
    --scales 0.5 0.75 1.0 1.25 1.5 1.75 2.0
    --n_variants 20 --n_runs 3 --max_steps 28 --n_init 2
    --n_persistent_base 128 --n_total_candidates 192
    --k_centers 2 --local_h 0.17 --local_h_decay 0.95
    --n_candidates_baseline 2048
    --methods Random EI UCB PI FunBO PFNs4BO TuRBO TAF_me TAF_ranking CAP-PPO
    --n_workers 4 --progress_detail method --seed 2026
    --fresh_run
)

run_eval() {
    local SURROGATE="$1"
    local SAVE_DIR="paper_experiments/ablation/component/alkox/default_v2/results_${SURROGATE}"

    echo "=== Eval Alkox Default v2 — ${SURROGATE} (all 10 methods) ==="

    "${PYTHON_BIN}" -u MYRL/scripts/eval_scale_sweep.py \
        "${COMMON_ARGS[@]}" \
        --surrogate "${SURROGATE}" \
        --save_dir "${SAVE_DIR}"

    # Replot
    PKL_PATH="${SAVE_DIR}/scale_sweep_data.pkl"
    if [[ -f "${PKL_PATH}" ]]; then
        REPLOT_DIR="${SAVE_DIR}/replot"
        mkdir -p "${REPLOT_DIR}"
        "${PYTHON_BIN}" -u MYRL/scripts/plot_scale_sweep.py \
            --pkl "${PKL_PATH}" --save_dir "${REPLOT_DIR}" --normalize 2>/dev/null \
            || echo "  [WARN] Replot failed for ${SURROGATE}."
    fi

    echo "=== Done: ${SAVE_DIR} ==="
}

# Run GP and TabPFN in parallel
run_eval "gp" &
PID_GP=$!

run_eval "tabpfn_base" &
PID_TABPFN=$!

echo "Running in parallel: gp (PID ${PID_GP}), tabpfn_base (PID ${PID_TABPFN})"
wait ${PID_GP}
EXIT_GP=$?
wait ${PID_TABPFN}
EXIT_TABPFN=$?

echo ""
echo "================================================================"
echo "  Eval complete"
echo "  gp:          exit=${EXIT_GP}"
echo "  tabpfn_base: exit=${EXIT_TABPFN}"
echo "================================================================"
