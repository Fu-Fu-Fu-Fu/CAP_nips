#!/bin/bash
# =============================================================================
# Branin calibrated multi-scale RFF + medium pool tuning
#
# Hyperparameter-only variant. Keeps the calibrated oracle-GP + normalized
# multi-scale RFF setup, but uses a less aggressive pool than wide_pool:
#   n_total_candidates=256, k_centers=3, local_h=2.25,
#   local_h_decay=0.95, explore_fraction=0.25
# =============================================================================
set -euo pipefail

start_time=$(date +%s)
cleanup () {
  end_time=$(date +%s)
  runtime=$((end_time - start_time))
  echo "Total time: $((runtime / 3600))h $(((runtime % 3600) / 60))m $((runtime % 60))s"
}
trap cleanup EXIT

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
export MPLCONFIGDIR="${REPO_ROOT}/.matplotlib"
PYTHON_BIN="${PYTHON_BIN:-python}"

TAG="calibrated_multiscale_rff_medium_pool"
RUN_DIR="paper_experiments/branin/runs/ppo_branin_family_${TAG}_ep${TOTAL_EPISODES:-5000}"
MODEL_DIR="paper_experiments/branin/models/${TAG}"
LOG_DIR="paper_experiments/branin/logs"
mkdir -p "${RUN_DIR}" "${MODEL_DIR}" "${LOG_DIR}" "${MPLCONFIGDIR}"

RFF_LENGTH_SCALES="${RFF_LENGTH_SCALES:-0.08,0.18,0.40}"
RFF_ALPHAS="${RFF_ALPHAS:-0.2302,0.4143,0.6445}"
RFF_TRIGGER_ALPHA="${RFF_TRIGGER_ALPHA:-1.0}"

N_PERSISTENT_BASE="${N_PERSISTENT_BASE:-128}"
N_TOTAL_CANDIDATES="${N_TOTAL_CANDIDATES:-256}"
K_CENTERS="${K_CENTERS:-3}"
LOCAL_H="${LOCAL_H:-2.25}"
LOCAL_H_DECAY="${LOCAL_H_DECAY:-0.95}"
EXPLORE_FRACTION="${EXPLORE_FRACTION:-0.25}"

N_WORKERS="${N_WORKERS:-8}"
TOTAL_EPISODES="${TOTAL_EPISODES:-5000}"

if [[ -f "${RUN_DIR}/ppo_final.pt" ]]; then
    echo "[TRAIN] SKIP — model already exists: ${RUN_DIR}/ppo_final.pt"
    echo "Model: ${MODEL_DIR}/ppo_best.pt"
    exit 0
fi

echo "================================================================"
echo "  Branin calibrated multi-scale RFF + medium pool"
echo "================================================================"
echo "  objective_source          : oracle_gp"
echo "  normalize_oracle_gp       : YES"
echo "  oracle_gp_rff_normalize_x : YES"
echo "  multiscale_ls             : ${RFF_LENGTH_SCALES}"
echo "  multiscale_alphas         : ${RFF_ALPHAS}"
echo "  n_total_candidates        : ${N_TOTAL_CANDIDATES}"
echo "  k_centers                 : ${K_CENTERS}"
echo "  local_h                   : ${LOCAL_H}"
echo "  local_h_decay             : ${LOCAL_H_DECAY}"
echo "  explore_fraction          : ${EXPLORE_FRACTION}"
echo "  episodes                  : ${TOTAL_EPISODES}"
echo "  n_workers                 : ${N_WORKERS}"
echo "================================================================"

"${PYTHON_BIN}" -u MYRL/scripts/train_rl.py \
    --task branin_family \
    --objective_source oracle_gp \
    --normalize_oracle_gp \
    --oracle_gp_rff_alpha "${RFF_TRIGGER_ALPHA}" \
    --oracle_gp_rff_multiscale "${RFF_LENGTH_SCALES}" \
    --oracle_gp_rff_multiscale_alphas "${RFF_ALPHAS}" \
    --oracle_gp_rff_normalize_x \
    --variants_path "data/branin_family_variants_k10_seed2026.npz" \
    --trajectories_path "data/branin_family_bo_trajs_k10_boSeed2026.npz" \
    --taf_data_path "data/taf_source_data_branin_family_k10.pkl" \
    --total_episodes "${TOTAL_EPISODES}" \
    --max_steps 18 \
    --n_init_context 2 \
    --n_persistent_base "${N_PERSISTENT_BASE}" \
    --n_total_candidates "${N_TOTAL_CANDIDATES}" \
    --k_centers "${K_CENTERS}" \
    --local_h "${LOCAL_H}" \
    --local_h_decay "${LOCAL_H_DECAY}" \
    --explore_fraction "${EXPLORE_FRACTION}" \
    --oracle_gp_min_grid_size 80 \
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
    --n_workers "${N_WORKERS}" \
    --save_dir "${RUN_DIR}"

cp "${RUN_DIR}/ppo_best.pt" "${MODEL_DIR}/"
cp "${RUN_DIR}/config.json" "${MODEL_DIR}/"

echo "=== Training complete ==="
echo "Model: ${MODEL_DIR}/ppo_best.pt"
