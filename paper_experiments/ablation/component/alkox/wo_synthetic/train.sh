#!/bin/bash
# =============================================================================
# Ablation 2: w/o Synthetic Objective Generation — Alkox
#
# Trains CAP-PPO with bnn_rff_alpha=0 (BNN mean only, no RFF perturbation).
# This removes synthetic diversity: each variant always sees the same
# deterministic BNN posterior mean function every episode.
#
# Everything else matches the original Alkox training config.
#
# Usage:
#   bash paper_experiments/ablation/wo_synthetic/alkox/train.sh
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

# ===== Task config (same as original) =====
TASK="alkox_emulator"
K_VARIANTS=10
VARIANT_SEED=2026
BO_TRAJ_SEED=2026

# ===== Paths (reuse existing data) =====
DATA_DIR="data"
VARIANTS_CACHE="${DATA_DIR}/${TASK}_variants_k${K_VARIANTS}_seed${VARIANT_SEED}_transform.npz"
TRAJS_CACHE="${DATA_DIR}/${TASK}_bo_trajs_k${K_VARIANTS}_boSeed${BO_TRAJ_SEED}_transform.npz"
TAF_DATA_PATH="${DATA_DIR}/taf_source_data_${TASK}_k${K_VARIANTS}_transform.pkl"
BNN_PARAMS_PATH="${DATA_DIR}/bnn_surrogates_${TASK}_k${K_VARIANTS}_kl0.001_transform.npz"

# ===== Budget (same as original) =====
N_INIT=2
MAX_STEPS=28

# ===== Training params (same as original EXCEPT bnn_rff_alpha=0) =====
RL_TOTAL_EPISODES="${RL_TOTAL_EPISODES:-5000}"
RL_N_PERSISTENT_BASE=128
RL_N_TOTAL_CANDIDATES=192
RL_K_CENTERS=2
RL_LOCAL_H=0.17
RL_LOCAL_H_DECAY=0.95
RL_UPDATE_EVERY=20
RL_SAVE_EVERY=500
RL_SEED=2026

# KEY ABLATION CHANGE: alpha=0 → no RFF perturbation, BNN mean only
RL_BNN_RFF_ALPHA=0.0
RL_BNN_RFF_LENGTH_SCALE=0.3

# ===== Acceleration =====
RL_INFERENCE_PRECISION="float32"
RL_GRID_SIZE=100
RL_N_LBFGS_STARTS=10

# ===== Output =====
EXP_DIR="paper_experiments/ablation/wo_synthetic/alkox"
RUN_DIR="${EXP_DIR}/runs/ppo_bnn_alpha0_ep${RL_TOTAL_EPISODES}"
mkdir -p "${RUN_DIR}"

echo "=== Ablation 2: w/o Synthetic Objective — Alkox ==="
echo "KEY CHANGE: bnn_rff_alpha=0 (BNN mean only, no RFF diversity)"
echo "Train: ${RL_TOTAL_EPISODES} episodes, $((N_INIT + MAX_STEPS)) evals/episode"
echo "Run dir: ${RUN_DIR}"
echo ""

# Verify data exists
for path in "${VARIANTS_CACHE}" "${TRAJS_CACHE}" "${TAF_DATA_PATH}" "${BNN_PARAMS_PATH}"; do
    if [[ ! -f "${path}" ]]; then
        echo "ERROR: Missing ${path}" >&2
        echo "Run the original Alkox pipeline first to generate data."
        exit 1
    fi
done

# =============================================================================
# Train CAP-PPO with bnn_rff_alpha=0 (NO synthetic diversity)
# =============================================================================
echo "[Train] CAP-PPO with bnn_rff_alpha=0..."
"${PYTHON_BIN}" -u MYRL/scripts/train_rl.py \
  --task "${TASK}" \
  --objective_source bnn \
  --bnn_params_path "${BNN_PARAMS_PATH}" \
  --bnn_rff_alpha "${RL_BNN_RFF_ALPHA}" \
  --bnn_rff_length_scale "${RL_BNN_RFF_LENGTH_SCALE}" \
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
echo "  3. SURROGATE=gp bash paper_experiments/ablation/scripts/eval_wo_synthetic_alkox.sh"
