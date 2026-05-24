#!/bin/bash
# =============================================================================
# train_ablation.sh — Unified ablation training script (NEW VERSION)
#
# Both Hartmann-6D and Alkox use regret_balanced + z-norm.
# Training parallelism: --n_workers (default 8).
#
# Required env vars:
#   TASK          - "hartmann_6d_family" or "alkox_emulator"
#   ABLATION_NAME - e.g., "pool_size_small", "k_variants_20", "default"
#
# Optional env vars (override defaults per task):
#   AB_N_PERSISTENT_BASE  AB_N_TOTAL_CANDIDATES  AB_K_CENTERS
#   AB_K_VARIANTS         AB_MAX_STEPS           AB_TOTAL_EPISODES
#   AB_N_WORKERS          AB_DATA_SUFFIX
#   AB_OBJECTIVE_SOURCE   - override objective source (e.g., "direct" for w/o synthetic)
#   AB_BNN_RFF_ALPHA      - override BNN RFF alpha (e.g., "0" for w/o synthetic)
#   AB_NO_CROSS_ATTN      - set to "1" to use mean-pooling instead of cross-attention
#
# Usage:
#   TASK=hartmann_6d_family ABLATION_NAME=pool_size_small \
#     AB_N_PERSISTENT_BASE=64 AB_N_TOTAL_CANDIDATES=128 \
#     bash paper_experiments/ablation/scripts/train_ablation.sh
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

# ===== Validate required vars =====
: "${TASK:?TASK must be set (hartmann_6d_family or alkox_emulator)}"
: "${ABLATION_NAME:?ABLATION_NAME must be set}"

# ===== Task-specific defaults (ALL use new version: regret_balanced + z-norm) =====
if [[ "${TASK}" == "hartmann_6d_family" ]]; then
    DEF_N_PERSISTENT_BASE=128
    DEF_N_TOTAL_CANDIDATES=256
    DEF_K_CENTERS=3
    DEF_LOCAL_H=0.15
    DEF_LOCAL_H_DECAY=0.95
    DEF_MAX_STEPS=48
    DEF_N_INIT=2
    DEF_K_VARIANTS=10
    DEF_OBJECTIVE_SOURCE="oracle_gp"
    DEF_DATA_SUFFIX=""
    DEF_UPDATE_EVERY=40
    DEF_GRID_SIZE=100
    DEF_N_LBFGS=10
elif [[ "${TASK}" == "alkox_emulator" ]]; then
    DEF_N_PERSISTENT_BASE=128
    DEF_N_TOTAL_CANDIDATES=192
    DEF_K_CENTERS=2
    DEF_LOCAL_H=0.17
    DEF_LOCAL_H_DECAY=0.95
    DEF_MAX_STEPS=28
    DEF_N_INIT=2
    DEF_K_VARIANTS=10
    DEF_OBJECTIVE_SOURCE="bnn"
    DEF_DATA_SUFFIX="_transform"
    DEF_UPDATE_EVERY=40
    DEF_GRID_SIZE=100
    DEF_N_LBFGS=10
else
    echo "ERROR: Unknown TASK=${TASK}" >&2
    exit 1
fi

# ===== Apply overrides =====
N_PERSISTENT_BASE="${AB_N_PERSISTENT_BASE:-${DEF_N_PERSISTENT_BASE}}"
N_TOTAL_CANDIDATES="${AB_N_TOTAL_CANDIDATES:-${DEF_N_TOTAL_CANDIDATES}}"
K_CENTERS="${AB_K_CENTERS:-${DEF_K_CENTERS}}"
LOCAL_H="${AB_LOCAL_H:-${DEF_LOCAL_H}}"
LOCAL_H_DECAY="${AB_LOCAL_H_DECAY:-${DEF_LOCAL_H_DECAY}}"
MAX_STEPS="${AB_MAX_STEPS:-${DEF_MAX_STEPS}}"
N_INIT="${AB_N_INIT:-${DEF_N_INIT}}"
K_VARIANTS="${AB_K_VARIANTS:-${DEF_K_VARIANTS}}"
TOTAL_EPISODES="${AB_TOTAL_EPISODES:-5000}"
SEED="${AB_SEED:-2026}"
N_WORKERS="${AB_N_WORKERS:-8}"

TOTAL_EVALS=$((N_INIT + MAX_STEPS))

# ===== Objective source override (for w/o synthetic ablation) =====
OBJECTIVE_SOURCE="${AB_OBJECTIVE_SOURCE:-${DEF_OBJECTIVE_SOURCE}}"
BNN_RFF_ALPHA="${AB_BNN_RFF_ALPHA:-1.5}"

# ===== Data paths =====
DATA_SUFFIX="${AB_DATA_SUFFIX:-${DEF_DATA_SUFFIX}}"
VARIANTS_CACHE="data/${TASK}_variants_k${K_VARIANTS}_seed2026${DATA_SUFFIX}.npz"
TRAJS_CACHE="data/${TASK}_bo_trajs_k${K_VARIANTS}_boSeed2026${DATA_SUFFIX}.npz"
TAF_DATA="data/taf_source_data_${TASK}_k${K_VARIANTS}${DATA_SUFFIX}.pkl"

BNN_PARAMS_PATH=""
if [[ "${DEF_OBJECTIVE_SOURCE}" == "bnn" ]]; then
    BNN_PARAMS_PATH="data/bnn_surrogates_${TASK}_k${K_VARIANTS}_kl0.001${DATA_SUFFIX}.npz"
fi

# ===== Output directory =====
TASK_SHORT="${TASK//_emulator/}"
TASK_SHORT="${TASK_SHORT//_family/}"
EXP_DIR="paper_experiments/ablation/sensitivity/${TASK_SHORT}/${ABLATION_NAME}"
RUN_DIR="${EXP_DIR}"
mkdir -p "${RUN_DIR}"

# ===== Skip if already trained =====
if [[ -f "${RUN_DIR}/ppo_final.pt" ]]; then
    echo "[TRAIN] SKIP — model already exists: ${RUN_DIR}/ppo_final.pt"
    exit 0
fi

# ===== Validate data exists =====
for path in "${VARIANTS_CACHE}" "${TRAJS_CACHE}" "${TAF_DATA}"; do
    if [[ ! -f "${path}" ]]; then
        echo "ERROR: Missing data file: ${path}" >&2
        exit 1
    fi
done
if [[ -n "${BNN_PARAMS_PATH}" ]] && [[ ! -f "${BNN_PARAMS_PATH}" ]]; then
    echo "ERROR: Missing BNN params: ${BNN_PARAMS_PATH}" >&2
    exit 1
fi

# ===== Print config =====
echo "================================================================"
echo "  Ablation Training: ${ABLATION_NAME}"
echo "================================================================"
echo "Task:       ${TASK}"
echo "Objective:  ${OBJECTIVE_SOURCE} (default: ${DEF_OBJECTIVE_SOURCE})"
if [[ "${OBJECTIVE_SOURCE}" == "bnn" ]]; then
echo "BNN alpha:  ${BNN_RFF_ALPHA}"
fi
echo "Reward:     regret_balanced (new version, z-norm)"
echo "Episodes:   ${TOTAL_EPISODES}"
echo "Budget:     ${TOTAL_EVALS} (${N_INIT} init + ${MAX_STEPS} steps)"
echo "Pool:       pers=${N_PERSISTENT_BASE}, total=${N_TOTAL_CANDIDATES}, k=${K_CENTERS}"
echo "Local:      h=${LOCAL_H}, decay=${LOCAL_H_DECAY}"
echo "K variants: ${K_VARIANTS}"
echo "Workers:    ${N_WORKERS}"
echo "Output:     ${RUN_DIR}"
echo "================================================================"

# ===== Build train command =====
TRAIN_ARGS=(
    --task "${TASK}"
    --objective_source "${OBJECTIVE_SOURCE}"
    --variants_path "${VARIANTS_CACHE}"
    --trajectories_path "${TRAJS_CACHE}"
    --taf_data_path "${TAF_DATA}"
    --total_episodes "${TOTAL_EPISODES}"
    --max_steps "${MAX_STEPS}"
    --n_init_context "${N_INIT}"
    --n_persistent_base "${N_PERSISTENT_BASE}"
    --n_total_candidates "${N_TOTAL_CANDIDATES}"
    --k_centers "${K_CENTERS}"
    --local_h "${LOCAL_H}"
    --local_h_decay "${LOCAL_H_DECAY}"
    --oracle_gp_min_grid_size "${DEF_GRID_SIZE}"
    --oracle_gp_min_n_lbfgs_starts "${DEF_N_LBFGS}"
    --inference_precision float32
    --reward_mode regret_balanced
    --reward_terminal_weight 1.5
    --reward_regret_auc_weight 0.2
    --reward_regret_delta_weight 1.0
    --reward_regret_early_power 0.5
    --reward_regret_terminal_power 3.0
    --reward_regret_scale_floor_ratio 0.02
    --variant_sampling shuffled_cycle
    --ent_coef_start 0.02
    --ent_coef_end 0.002
    --update_every "${DEF_UPDATE_EVERY}"
    --save_every 500
    --seed "${SEED}"
    --n_workers "${N_WORKERS}"
    --save_dir "${RUN_DIR}"
)

# Add objective-source-specific args
if [[ "${OBJECTIVE_SOURCE}" == "oracle_gp" ]]; then
    TRAIN_ARGS+=(--normalize_oracle_gp)
elif [[ "${OBJECTIVE_SOURCE}" == "bnn" ]]; then
    TRAIN_ARGS+=(
        --bnn_params_path "${BNN_PARAMS_PATH}"
        --bnn_rff_alpha "${BNN_RFF_ALPHA}"
        --bnn_rff_length_scale 0.3
        --normalize_bnn
    )
fi

# ===== Select training script =====
NO_CROSS_ATTN="${AB_NO_CROSS_ATTN:-0}"
if [[ "${NO_CROSS_ATTN}" == "1" ]]; then
    TRAIN_SCRIPT="MYRL/scripts/train_rl_no_crossattn.py"
    echo "Architecture: NO cross-attention (mean-pooling ablation)"
else
    TRAIN_SCRIPT="MYRL/scripts/train_rl.py"
fi

# ===== Run training =====
start_time=$(date +%s)

"${PYTHON_BIN}" -u "${TRAIN_SCRIPT}" "${TRAIN_ARGS[@]}"

end_time=$(date +%s)
runtime=$((end_time - start_time))
echo ""
echo "Training complete: ${ABLATION_NAME}"
echo "Time: $((runtime / 3600))h $(((runtime % 3600) / 60))m $((runtime % 60))s"
echo "Model: ${RUN_DIR}/ppo_final.pt"
