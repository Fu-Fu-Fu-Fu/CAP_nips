#!/bin/bash
# =============================================================================
# Ablation 3: w/o Cross-Attention — Hartmann-6D
#
# Evaluates a model trained with cross-attention replaced by mean-pooling.
# Model must be trained first with the ablation architecture code.
#
# Usage:
#   SURROGATE=gp bash paper_experiments/ablation/scripts/eval_wo_crossattn_hartmann6d.sh
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

# Model trained with mean-pooling instead of cross-attention
MODEL_PATH="paper_experiments/ablation/component/hartmann6d/wo_crossattn/model/ppo_best.pt"
TAF_DATA="./data/taf_source_data_hartmann_6d_family_k10.pkl"
SAVE_DIR="paper_experiments/ablation/component/hartmann6d/wo_crossattn/results_${SURROGATE}"

echo "=== Ablation 3: w/o Cross-Attention — Hartmann-6D (${SURROGATE}) ==="

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "ERROR: Model not found at ${MODEL_PATH}"
    echo "Please train the ablation model first (w/o cross-attention)."
    exit 1
fi

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
    --methods Random EI CAP-PPO \
    --seed 2026 \
    --save_dir "${SAVE_DIR}"

PKL_PATH="${SAVE_DIR}/scale_sweep_data.pkl"
if [[ -f "${PKL_PATH}" ]]; then
    REPLOT_DIR="${SAVE_DIR}/replot"
    mkdir -p "${REPLOT_DIR}"
    "${PYTHON_BIN}" -u MYRL/scripts/plot_scale_sweep.py \
        --pkl "${PKL_PATH}" --save_dir "${REPLOT_DIR}" --normalize 2>/dev/null \
        || echo "  [WARN] Replot failed."
fi

echo "=== Done: ${SAVE_DIR} ==="
