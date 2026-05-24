#!/bin/bash
# =============================================================================
# Alkox Default v2 — Retrain with new recipe (regret_balanced + z-norm)
#
# Same hyperparams as the ablation models, but keeping ALL components:
#   - bnn_rff_alpha=5.0 (original default, NOT the ablation script's 1.5)
#   - cross-attention enabled (standard train_rl.py)
#
# This produces a fair baseline for comparing against wo_synthetic / wo_crossattn.
#
# Usage:
#   bash paper_experiments/ablation/scripts/train_default_v2_alkox.sh
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

RUN_DIR="paper_experiments/ablation/component/alkox/default_v2"
mkdir -p "${RUN_DIR}"

if [[ -f "${RUN_DIR}/ppo_final.pt" ]]; then
    echo "[TRAIN] SKIP — model already exists: ${RUN_DIR}/ppo_final.pt"
    exit 0
fi

echo "================================================================"
echo "  Alkox Default v2 (new training recipe, all components)"
echo "================================================================"
echo "  objective_source : bnn"
echo "  bnn_rff_alpha    : 1.5  (same as ablation models)"
echo "  cross-attention  : YES"
echo "  reward_mode      : regret_balanced"
echo "  normalize_bnn    : YES"
echo "  ent_coef         : 0.02 -> 0.002"
echo "  episodes         : 5000"
echo "  n_workers        : 8"
echo "================================================================"

"${PYTHON_BIN}" -u MYRL/scripts/train_rl.py \
    --task alkox_emulator \
    --objective_source bnn \
    --variants_path "data/alkox_emulator_variants_k10_seed2026_transform.npz" \
    --trajectories_path "data/alkox_emulator_bo_trajs_k10_boSeed2026_transform.npz" \
    --taf_data_path "data/taf_source_data_alkox_emulator_k10_transform.pkl" \
    --bnn_params_path "data/bnn_surrogates_alkox_emulator_k10_kl0.001_transform.npz" \
    --bnn_rff_alpha 1.5 \
    --bnn_rff_length_scale 0.3 \
    --normalize_bnn \
    --total_episodes 5000 \
    --max_steps 28 \
    --n_init_context 2 \
    --n_persistent_base 128 \
    --n_total_candidates 192 \
    --k_centers 2 \
    --local_h 0.17 \
    --local_h_decay 0.95 \
    --oracle_gp_min_grid_size 100 \
    --oracle_gp_min_n_lbfgs_starts 10 \
    --inference_precision float32 \
    --reward_mode regret_balanced \
    --reward_terminal_weight 1.5 \
    --reward_regret_auc_weight 0.2 \
    --reward_regret_delta_weight 1.0 \
    --reward_regret_early_power 0.5 \
    --reward_regret_terminal_power 3.0 \
    --reward_regret_scale_floor_ratio 0.02 \
    --variant_sampling shuffled_cycle \
    --ent_coef_start 0.02 \
    --ent_coef_end 0.002 \
    --update_every 40 \
    --save_every 500 \
    --seed 2026 \
    --n_workers 8 \
    --save_dir "${RUN_DIR}"

echo ""
echo "Training complete. Model: ${RUN_DIR}/ppo_best.pt"
