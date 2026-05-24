#!/bin/bash
# =============================================================================
# Trajectory Ablation: Random trajectories — Hartmann-6D
#
# Tests whether BO expert trajectories matter for Oracle GP fitting.
# Replaces BO trajectories with uniform random trajectories, keeping
# everything else identical to the default Hartmann-6D config.
#
# Pipeline: generate random trajs -> TAF -> train CAP-PPO
#
# Usage:
#   bash paper_experiments/ablation/scripts/train_random_traj_hartmann6d.sh
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

TASK="hartmann_6d_family"
K_VARIANTS=10

# --- Paths ---
EXP_DIR="paper_experiments/ablation/trajectory/hartmann6d/random_traj"
DATA_DIR="${EXP_DIR}/data"
mkdir -p "${DATA_DIR}" "${EXP_DIR}"

VARIANTS_PATH="data/${TASK}_variants_k${K_VARIANTS}_seed2026.npz"
RANDOM_TRAJS="${DATA_DIR}/${TASK}_random_trajs_k${K_VARIANTS}.npz"
TAF_DATA="${DATA_DIR}/taf_source_data_${TASK}_random_k${K_VARIANTS}.pkl"

# 1 random trajectory per variant (matching the 1 real BO trajectory in original)
N_TRAJS_PER_VARIANT=1
TOTAL_EVALS=50  # 2 init + 48 steps

echo "================================================================"
echo "  Trajectory Ablation: Random — Hartmann-6D"
echo "================================================================"

# =============================================================================
# Step 1: Generate random trajectories
# =============================================================================
if [[ -f "${RANDOM_TRAJS}" ]]; then
    echo "[Step 1] SKIP — random trajectories exist: ${RANDOM_TRAJS}"
else
    echo "[Step 1] Generate random trajectories..."
    "${PYTHON_BIN}" -u MYRL/scripts/generate_random_trajs.py \
        --task "${TASK}" \
        --variants_path "${VARIANTS_PATH}" \
        --output_path "${RANDOM_TRAJS}" \
        --n_trajs_per_variant ${N_TRAJS_PER_VARIANT} \
        --total_evals ${TOTAL_EVALS} \
        --seed 2026
    echo "[Step 1] Done."
fi

# =============================================================================
# Step 1.5: Generate TAF source data from random trajectories
# =============================================================================
if [[ -f "${TAF_DATA}" ]]; then
    echo "[Step 1.5] SKIP — TAF data exists: ${TAF_DATA}"
else
    echo "[Step 1.5] Generate TAF source data..."
    "${PYTHON_BIN}" -u -c "
from myrl.rl.train_rl import prepare_taf_data
prepare_taf_data('${RANDOM_TRAJS}', '${TAF_DATA}')
print('TAF source data saved to ${TAF_DATA}')
"
    echo "[Step 1.5] Done."
fi

# =============================================================================
# Step 2: Train CAP-PPO (same config as default, but with random trajectories)
# =============================================================================
if [[ -f "${EXP_DIR}/ppo_final.pt" ]]; then
    echo "[Step 2] SKIP — model already exists"
else
    echo "[Step 2] Train CAP-PPO..."
    "${PYTHON_BIN}" -u MYRL/scripts/train_rl.py \
        --task "${TASK}" \
        --objective_source oracle_gp \
        --normalize_oracle_gp \
        --variants_path "${VARIANTS_PATH}" \
        --trajectories_path "${RANDOM_TRAJS}" \
        --taf_data_path "${TAF_DATA}" \
        --total_episodes 5000 \
        --max_steps 48 \
        --n_init_context 2 \
        --n_persistent_base 128 \
        --n_total_candidates 256 \
        --k_centers 3 \
        --local_h 0.15 \
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
        --save_dir "${EXP_DIR}"
    echo "[Step 2] Done."
fi

echo ""
echo "=== Training complete ==="
echo "Model: ${EXP_DIR}/ppo_best.pt"
