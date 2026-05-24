#!/bin/bash
# =============================================================================
# train_hartmann6d.sh — Hartmann-6D Family (6D) training pipeline
#
# Pipeline: Step 1 (data gen) -> Step 1.5 (TAF) -> Step 2 (train CAP-PPO)
#
# Hartmann-6D is a 6D synthetic function family.  Uses oracle_gp objective
# with z-norm + regret_balanced reward (aligned with HPLC proven config).
#
# Usage:
#   bash paper_experiments/hartmann6d/train_hartmann6d.sh
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

echo "=== Hartmann-6D Family (6D) Training Pipeline ==="
echo "Repo: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-none}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi

# ===== Task config =====
TASK="hartmann_6d_family"
K_VARIANTS=10
VARIANT_SEED=2026
BO_TRAJ_SEED=2026

# ===== Paths (use existing data from previous training) =====
EXP_DIR="paper_experiments/hartmann6d"
DATA_DIR="data"
VARIANTS_CACHE="${DATA_DIR}/${TASK}_variants_k${K_VARIANTS}_seed${VARIANT_SEED}.npz"
TRAJS_CACHE="${DATA_DIR}/${TASK}_bo_trajs_k${K_VARIANTS}_boSeed${BO_TRAJ_SEED}.npz"
TAF_DATA_PATH="${DATA_DIR}/taf_source_data_${TASK}_k${K_VARIANTS}.pkl"

# ===== Budget (dim=6, 50 evals) =====
N_INIT=2
MAX_STEPS=48
TOTAL_EVALS=$((N_INIT + MAX_STEPS))

# ===== Training params (dim=6) =====
RL_TOTAL_EPISODES=5000
RL_N_PERSISTENT_BASE=128
RL_N_TOTAL_CANDIDATES=256
RL_K_CENTERS=3
RL_LOCAL_H=0.15              # [0,1]^6 domain
RL_LOCAL_H_DECAY=0.95
RL_UPDATE_EVERY=40
RL_SAVE_EVERY=500
RL_SEED=2026

# ===== Reward (regret_balanced + z-norm — aligned with HPLC proven config) =====
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
RL_GRID_SIZE=100              # Sobol 10000 points for dim=6
RL_N_LBFGS_STARTS=10

# ===== Output =====
RUN_DIR="${EXP_DIR}/runs/ppo_${TASK}_znorm_rb_ep${RL_TOTAL_EPISODES}"
mkdir -p "${RUN_DIR}"

echo ""
echo "Train: ${RL_TOTAL_EPISODES} episodes, ${TOTAL_EVALS} evals/episode (${N_INIT} init + ${MAX_STEPS} steps)"
echo "Reward: ${RL_REWARD_MODE}"
echo "Objective: oracle_gp (z-norm)"
echo "Candidates: N_base=${RL_N_PERSISTENT_BASE}, N_total=${RL_N_TOTAL_CANDIDATES}, k_centers=${RL_K_CENTERS}"
echo "Run dir: ${RUN_DIR}"
echo ""

# =============================================================================
# Step 1: Generate training data (variants + BO trajectories)
# =============================================================================
if [[ -f "${VARIANTS_CACHE}" ]] && [[ -f "${TRAJS_CACHE}" ]]; then
  echo "[Step 1] SKIP — data caches already exist"
else
  echo "[Step 1] Generate variants + trajectories..."
  "${PYTHON_BIN}" -u MYRL/scripts/finetune.py \
    --task "${TASK}" \
    --stage generate \
    --k_variants "${K_VARIANTS}" \
    --variant_seed "${VARIANT_SEED}" \
    --bo_seed "${BO_TRAJ_SEED}" \
    --variants_cache "${VARIANTS_CACHE}" \
    --trajectories_cache "${TRAJS_CACHE}" \
    --n_trials_per_variant 5 \
    --n_synth_trajectories_per_variant 0 \
    --total_evals "${TOTAL_EVALS}" \
    --n_init "${N_INIT}" \
    --xi 0.01 \
    --gp_n_restarts_optimizer 5
  echo "[Step 1] Done."
fi

# =============================================================================
# Step 1.5: Generate TAF source data
# =============================================================================
if [[ -f "${TAF_DATA_PATH}" ]]; then
  echo "[Step 1.5] SKIP — TAF source data already exists: ${TAF_DATA_PATH}"
else
  echo "[Step 1.5] Generate TAF source data..."
  "${PYTHON_BIN}" -u -c "
from myrl.rl.train_rl import prepare_taf_data
prepare_taf_data('${TRAJS_CACHE}', '${TAF_DATA_PATH}')
print('TAF source data saved to ${TAF_DATA_PATH}')
"
  echo "[Step 1.5] Done."
fi

# =============================================================================
# Step 2: Train CAP-PPO (oracle_gp + z-norm + regret_balanced)
# =============================================================================
echo "[Step 2] Train CAP-PPO (${RL_TOTAL_EPISODES} ep, dim=6, oracle_gp, z-norm, regret_balanced)..."
"${PYTHON_BIN}" -u MYRL/scripts/train_rl.py \
  --task "${TASK}" \
  --objective_source oracle_gp \
  --normalize_oracle_gp \
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
echo "[Step 2] Done."

echo ""
echo "=== Training complete ==="
echo "Output: ${RUN_DIR}/"
echo ""
echo "Next steps:"
echo "  1. Copy best model: cp ${RUN_DIR}/ppo_best.pt ${EXP_DIR}/models/default/"
echo "  2. Copy config:     cp ${RUN_DIR}/config.json ${EXP_DIR}/models/default/"
echo "  3. Run eval:        bash ${EXP_DIR}/eval_hartmann6d.sh"
