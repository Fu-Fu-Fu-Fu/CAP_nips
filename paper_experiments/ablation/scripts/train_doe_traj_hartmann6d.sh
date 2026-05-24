#!/bin/bash
# =============================================================================
# Trajectory Ablation: DOE prior trajectories — Hartmann-6D
#
# Trains one DOE prior-data variant per invocation.
# Training parallelism is handled inside train_rl.py via --n_workers.
#
# Usage:
#   bash paper_experiments/ablation/scripts/train_doe_traj_hartmann6d.sh doe_static
#   bash paper_experiments/ablation/scripts/train_doe_traj_hartmann6d.sh doe_seq_local
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
export MPLCONFIGDIR="${REPO_ROOT}/.matplotlib"
PYTHON_BIN="${PYTHON_BIN:-python}"
N_WORKERS_PER_JOB="${N_WORKERS_PER_JOB:-8}"

TASK="hartmann_6d_family"
K_VARIANTS=10
VARIANTS_PATH="data/${TASK}_variants_k${K_VARIANTS}_seed2026.npz"
BASE_DIR="paper_experiments/ablation/trajectory/hartmann6d"
METHOD="${1:-${DOE_METHOD:-}}"

if [[ "${METHOD}" != "doe_static" && "${METHOD}" != "doe_seq_local" ]]; then
    echo "Usage: bash $0 <doe_static|doe_seq_local>" >&2
    exit 2
fi

run_train() {
    local METHOD="$1"
    local EXP_DIR="${BASE_DIR}/${METHOD}"
    local DATA_DIR="${EXP_DIR}/data"
    local MODE
    local TRAJS
    local TAF_DATA
    local DOE_ARGS=()

    mkdir -p "${DATA_DIR}" "${EXP_DIR}"

    case "${METHOD}" in
        doe_static)
            MODE="static_sobol"
            TRAJS="${DATA_DIR}/${TASK}_doe_static_trajs_k${K_VARIANTS}.npz"
            TAF_DATA="${DATA_DIR}/taf_source_data_${TASK}_doe_static_k${K_VARIANTS}.pkl"
            ;;
        doe_seq_local)
            MODE="seq_local_sobol"
            TRAJS="${DATA_DIR}/${TASK}_doe_seq_local_trajs_k${K_VARIANTS}.npz"
            TAF_DATA="${DATA_DIR}/taf_source_data_${TASK}_doe_seq_local_k${K_VARIANTS}.pkl"
            DOE_ARGS=(--init_global 8 --batch_size 4 --radius0 0.20 --radius_min 0.05 --decay 0.90 --global_frac 0.25)
            ;;
        *)
            echo "Unknown METHOD=${METHOD}" >&2
            return 2
            ;;
    esac

    echo "================================================================"
    echo "  Train Hartmann-6D ${METHOD}"
    echo "  EXP_DIR=${EXP_DIR}"
    echo "  rollout workers=${N_WORKERS_PER_JOB}"
    echo "================================================================"

    if [[ -f "${TRAJS}" ]]; then
        echo "[${METHOD}] SKIP trajectories: ${TRAJS}"
    else
        "${PYTHON_BIN}" -u MYRL/scripts/generate_doe_trajs.py \
            --task "${TASK}" \
            --variants_path "${VARIANTS_PATH}" \
            --output_path "${TRAJS}" \
            --mode "${MODE}" \
            --n_trajs_per_variant 1 \
            --total_evals 50 \
            --seed 2026 \
            "${DOE_ARGS[@]}"
    fi

    if [[ -f "${TAF_DATA}" ]]; then
        echo "[${METHOD}] SKIP TAF: ${TAF_DATA}"
    else
        "${PYTHON_BIN}" -u -c "
from myrl.rl.train_rl import prepare_taf_data
prepare_taf_data('${TRAJS}', '${TAF_DATA}')
print('TAF source data saved to ${TAF_DATA}')
"
    fi

    if [[ -f "${EXP_DIR}/ppo_final.pt" ]]; then
        echo "[${METHOD}] SKIP model: ${EXP_DIR}/ppo_final.pt"
        return 0
    fi

    "${PYTHON_BIN}" -u MYRL/scripts/train_rl.py \
        --task "${TASK}" \
        --objective_source oracle_gp \
        --normalize_oracle_gp \
        --variants_path "${VARIANTS_PATH}" \
        --trajectories_path "${TRAJS}" \
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
        --n_workers "${N_WORKERS_PER_JOB}" \
        --save_dir "${EXP_DIR}"
}

run_train "${METHOD}"
