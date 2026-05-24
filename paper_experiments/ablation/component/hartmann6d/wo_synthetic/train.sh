#!/bin/bash
# =============================================================================
# Ablation 2: w/o Synthetic Objective Generation — Hartmann-6D
#
# Trains CAP-PPO with objective_source=direct instead of oracle_gp.
# This removes RFF-based synthetic function diversity: the agent sees the
# same deterministic variant function every episode (no GP→RFF sampling).
#
# Everything else matches the original Hartmann-6D training config.
#
# Usage:
#   bash paper_experiments/ablation/wo_synthetic/hartmann6d/train.sh
# =============================================================================

start_time=$(date +%s)
set -euo pipefail

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

# ===== Task config (same as original) =====
TASK="hartmann_6d_family"
K_VARIANTS=10
VARIANT_SEED=2026
BO_TRAJ_SEED=2026

# ===== Paths (reuse existing data) =====
DATA_DIR="data"
VARIANTS_CACHE="${DATA_DIR}/${TASK}_variants_k${K_VARIANTS}_seed${VARIANT_SEED}.npz"
TRAJS_CACHE="${DATA_DIR}/${TASK}_bo_trajs_k${K_VARIANTS}_boSeed${BO_TRAJ_SEED}.npz"
TAF_DATA_PATH="${DATA_DIR}/taf_source_data_${TASK}_k${K_VARIANTS}.pkl"

# ===== Budget (same as original) =====
N_INIT=2
MAX_STEPS=48

# ===== Training params (same as original EXCEPT objective_source) =====
RL_TOTAL_EPISODES="${RL_TOTAL_EPISODES:-5000}"
RL_N_PERSISTENT_BASE=128
RL_N_TOTAL_CANDIDATES=256
RL_K_CENTERS=3
RL_LOCAL_H=0.15
RL_LOCAL_H_DECAY=0.95
RL_UPDATE_EVERY=40
RL_SAVE_EVERY=500
RL_SEED=2026

# ===== Reward (same as original) =====
RL_REWARD_MODE="regret_balanced"
RL_VARIANT_SAMPLING="shuffled_cycle"
RL_ENT_COEF_START=0.02
RL_ENT_COEF_END=0.002
RL_REWARD_TERMINAL_WEIGHT=1.5
RL_REWARD_REGRET_AUC_WEIGHT=0.2
RL_REWARD_REGRET_DELTA_WEIGHT=1.0
RL_REWARD_REGRET_EARLY_POWER=0.5
RL_REWARD_REGRET_TERMINAL_POWER=3.0
RL_REWARD_REGRET_SCALE_FLOOR_RATIO=0.02

# ===== Acceleration =====
RL_INFERENCE_PRECISION="float32"
RL_GRID_SIZE=100
RL_N_LBFGS_STARTS=10

# ===== Output =====
EXP_DIR="paper_experiments/ablation/wo_synthetic/hartmann6d"
RUN_DIR="${EXP_DIR}/runs/ppo_direct_ep${RL_TOTAL_EPISODES}"
mkdir -p "${RUN_DIR}"

echo "=== Ablation 2: w/o Synthetic Objective — Hartmann-6D ==="
echo "KEY CHANGE: objective_source=direct (no oracle_gp → RFF sampling)"
echo "Train: ${RL_TOTAL_EPISODES} episodes, $((N_INIT + MAX_STEPS)) evals/episode"
echo "Run dir: ${RUN_DIR}"
echo ""

# Verify data exists
for path in "${VARIANTS_CACHE}" "${TRAJS_CACHE}" "${TAF_DATA_PATH}"; do
    if [[ ! -f "${path}" ]]; then
        echo "ERROR: Missing ${path}" >&2
        echo "Run the original Hartmann-6D pipeline first to generate data."
        exit 1
    fi
done

# =============================================================================
# Train CAP-PPO with objective_source=direct (NO synthetic diversity)
# =============================================================================
echo "[Train] CAP-PPO with objective_source=direct..."
"${PYTHON_BIN}" -u MYRL/scripts/train_rl.py \
  --task "${TASK}" \
  --objective_source direct \
  --variants_path "${VARIANTS_CACHE}" \
  --trajectories_path "${TRAJS_CACHE}" \
  --taf_data_path "${TAF_DATA_PATH}" \
  --total_episodes "${RL_TOTAL_EPISODES}" \
  --max_steps "${MAX_STEPS}" \
  --n_init_context "${N_INIT}" \
  --n_persistent_base "${RL_N_PERSISTENT_BASE}" \
  --n_total_candidates "${RL_N_TOTAL_CANDIDATES}" \
  --k_centers "${RL_K_CENTERS}" \
  --local_h "${RL_LOCAL_H}" \
  --local_h_decay "${RL_LOCAL_H_DECAY}" \
  --oracle_gp_min_grid_size "${RL_GRID_SIZE}" \
  --oracle_gp_min_n_lbfgs_starts "${RL_N_LBFGS_STARTS}" \
  --inference_precision "${RL_INFERENCE_PRECISION}" \
  --reward_mode "${RL_REWARD_MODE}" \
  --reward_terminal_weight "${RL_REWARD_TERMINAL_WEIGHT}" \
  --reward_regret_auc_weight "${RL_REWARD_REGRET_AUC_WEIGHT}" \
  --reward_regret_delta_weight "${RL_REWARD_REGRET_DELTA_WEIGHT}" \
  --reward_regret_early_power "${RL_REWARD_REGRET_EARLY_POWER}" \
  --reward_regret_terminal_power "${RL_REWARD_REGRET_TERMINAL_POWER}" \
  --reward_regret_scale_floor_ratio "${RL_REWARD_REGRET_SCALE_FLOOR_RATIO}" \
  --variant_sampling "${RL_VARIANT_SAMPLING}" \
  --ent_coef_start "${RL_ENT_COEF_START}" \
  --ent_coef_end "${RL_ENT_COEF_END}" \
  --update_every "${RL_UPDATE_EVERY}" \
  --save_every "${RL_SAVE_EVERY}" \
  --seed "${RL_SEED}" \
  --save_dir "${RUN_DIR}"
echo "[Train] Done."

echo ""
echo "=== Training complete ==="
echo "Output: ${RUN_DIR}/"
echo ""
echo "Next steps:"
echo "  1. cp ${RUN_DIR}/ppo_best.pt ${EXP_DIR}/model/"
echo "  2. cp ${RUN_DIR}/config.json ${EXP_DIR}/model/"
echo "  3. SURROGATE=gp bash paper_experiments/ablation/scripts/eval_wo_synthetic_hartmann6d.sh"
