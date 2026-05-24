#!/bin/bash
# Ackley-5D family training: fixed oracle GP length-scale + conservative smooth multiscale RFF.
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

TASK="ackley_5d_family"
EXP_DIR="paper_experiments/ackley5d"
DATA_DIR="data"
K_VARIANTS="${K_VARIANTS:-10}"
VARIANT_SEED="${VARIANT_SEED:-2026}"
BO_TRAJ_SEED="${BO_TRAJ_SEED:-2026}"

N_INIT="${N_INIT:-2}"
MAX_STEPS="${MAX_STEPS:-48}"
TOTAL_EVALS=$((N_INIT + MAX_STEPS))

TOTAL_EPISODES="${TOTAL_EPISODES:-5000}"
N_WORKERS="${N_WORKERS:-8}"
TAG="fixedgp_ls07_smoothweak"
RUN_DIR="${RUN_DIR:-${EXP_DIR}/runs/ppo_${TASK}_${TAG}_ep${TOTAL_EPISODES}}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}/models/${TAG}}"

VARIANTS_CACHE="${DATA_DIR}/${TASK}_variants_k${K_VARIANTS}_seed${VARIANT_SEED}.npz"
TRAJS_CACHE="${DATA_DIR}/${TASK}_bo_trajs_k${K_VARIANTS}_boSeed${BO_TRAJ_SEED}.npz"
TAF_DATA_PATH="${DATA_DIR}/taf_source_data_${TASK}_k${K_VARIANTS}.pkl"

ORACLE_GP_FIXED_LS="${ORACLE_GP_FIXED_LS:-0.7}"
ORACLE_GP_ALPHA="${ORACLE_GP_ALPHA:-0.001}"
RFF_LENGTH_SCALES="${RFF_LENGTH_SCALES:-0.16,0.32,0.64}"
RFF_ALPHAS="${RFF_ALPHAS:-0.06,0.10,0.14}"

mkdir -p "${RUN_DIR}" "${MODEL_DIR}" "${EXP_DIR}/logs" "${MPLCONFIGDIR}"

echo "=== Ackley-5D training (${TAG}) ==="
echo "Task=${TASK}, evals=${TOTAL_EVALS}, episodes=${TOTAL_EPISODES}, workers=${N_WORKERS}"
echo "Fixed oracle GP length_scale=${ORACLE_GP_FIXED_LS}, alpha=${ORACLE_GP_ALPHA}"
echo "RFF ls=${RFF_LENGTH_SCALES}, alphas=${RFF_ALPHAS}"
echo "Run dir=${RUN_DIR}"

if [[ ! -f "${VARIANTS_CACHE}" || ! -f "${TRAJS_CACHE}" ]]; then
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
else
  echo "SKIP data generation: ${VARIANTS_CACHE}, ${TRAJS_CACHE}"
fi

if [[ ! -f "${TAF_DATA_PATH}" ]]; then
  "${PYTHON_BIN}" -u -c "from myrl.rl.train_rl import prepare_taf_data; prepare_taf_data('${TRAJS_CACHE}', '${TAF_DATA_PATH}'); print('TAF saved: ${TAF_DATA_PATH}')"
else
  echo "SKIP TAF generation: ${TAF_DATA_PATH}"
fi

"${PYTHON_BIN}" -u MYRL/scripts/train_rl.py \
  --task "${TASK}" \
  --objective_source oracle_gp \
  --normalize_oracle_gp \
  --oracle_gp_fixed_length_scale "${ORACLE_GP_FIXED_LS}" \
  --oracle_gp_alpha "${ORACLE_GP_ALPHA}" \
  --oracle_gp_rff_alpha 1.0 \
  --oracle_gp_rff_multiscale "${RFF_LENGTH_SCALES}" \
  --oracle_gp_rff_multiscale_alphas "${RFF_ALPHAS}" \
  --oracle_gp_rff_normalize_x \
  --variants_path "${VARIANTS_CACHE}" \
  --trajectories_path "${TRAJS_CACHE}" \
  --taf_data_path "${TAF_DATA_PATH}" \
  --total_episodes "${TOTAL_EPISODES}" \
  --max_steps "${MAX_STEPS}" \
  --n_init_context "${N_INIT}" \
  --n_persistent_base 128 \
  --n_total_candidates 256 \
  --k_centers 3 \
  --local_h 0.15 \
  --local_h_decay 0.95 \
  --explore_fraction 0.25 \
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
echo "Model: ${MODEL_DIR}/ppo_best.pt"
