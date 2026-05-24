#!/bin/bash
# =============================================================================
# Ablation 1: w/o RL — Hartmann-6D (shared candidate pool)
# Uses the same trained CAP-PPO model, but all baselines see CAP's candidate pool.
#
# Usage:
#   SURROGATE=gp bash paper_experiments/ablation/scripts/eval_wo_rl_hartmann6d.sh
#   SURROGATE=tabpfn_base bash paper_experiments/ablation/scripts/eval_wo_rl_hartmann6d.sh
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

SURROGATE="${SURROGATE:-gp}"
N_WORKERS="${N_WORKERS:-4}"
SAVE_DIR="paper_experiments/ablation/wo_rl/hartmann6d/results_${SURROGATE}"

echo "=== Ablation 1: w/o RL — Hartmann-6D (${SURROGATE}) ==="

"${PYTHON_BIN}" -u paper_experiments/ablation/scripts/eval_shared_pool.py \
    --task hartmann_6d_family \
    --rl_model_path paper_experiments/hartmann6d/models/default/ppo_best.pt \
    --taf_data_path ./data/taf_source_data_hartmann_6d_family_k10.pkl \
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
    --methods Random EI UCB PI TAF_ranking CAP-PPO \
    --seed 2026 \
    --n_workers "${N_WORKERS}" \
    --save_dir "${SAVE_DIR}"

# Generate plots if pkl exists
PKL_PATH="${SAVE_DIR}/scale_sweep_data.pkl"
if [[ -f "${PKL_PATH}" ]]; then
    REPLOT_DIR="${SAVE_DIR}/replot"
    echo ""
    echo "[Replot] Generating normalized regret plots..."
    mkdir -p "${REPLOT_DIR}"
    "${PYTHON_BIN}" -u MYRL/scripts/plot_scale_sweep.py \
        --pkl "${PKL_PATH}" \
        --save_dir "${REPLOT_DIR}" \
        --normalize 2>/dev/null || echo "  [WARN] Replot failed, data still saved."
fi

echo "=== Done: ${SAVE_DIR} ==="
