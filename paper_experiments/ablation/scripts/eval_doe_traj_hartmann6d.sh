#!/bin/bash
# =============================================================================
# Eval Trajectory Ablation: DOE — Hartmann-6D
#
# Evaluates one DOE prior-data variant per invocation.
# Surrogates gp and tabpfn_base are run in parallel, matching existing eval scripts.
#
# Usage:
#   bash paper_experiments/ablation/scripts/eval_doe_traj_hartmann6d.sh doe_static
#   bash paper_experiments/ablation/scripts/eval_doe_traj_hartmann6d.sh doe_seq_local
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

BASE_DIR="paper_experiments/ablation/trajectory/hartmann6d"
METHOD="${1:-${DOE_METHOD:-}}"

if [[ "${METHOD}" != "doe_static" && "${METHOD}" != "doe_seq_local" ]]; then
    echo "Usage: bash $0 <doe_static|doe_seq_local>" >&2
    exit 2
fi

taf_path() {
    case "$1" in
        doe_static) echo "${BASE_DIR}/doe_static/data/taf_source_data_hartmann_6d_family_doe_static_k10.pkl" ;;
        doe_seq_local) echo "${BASE_DIR}/doe_seq_local/data/taf_source_data_hartmann_6d_family_doe_seq_local_k10.pkl" ;;
        *) echo "Unknown method $1" >&2; return 2 ;;
    esac
}

run_eval() {
    local METHOD="$1"
    local SURROGATE="$2"
    local EXP_DIR="${BASE_DIR}/${METHOD}"
    local MODEL_PATH="${EXP_DIR}/ppo_best.pt"
    local TAF_DATA
    TAF_DATA="$(taf_path "${METHOD}")"
    local SAVE_DIR="${EXP_DIR}/results_${SURROGATE}"

    if [[ ! -f "${MODEL_PATH}" ]]; then
        echo "ERROR: Model not found: ${MODEL_PATH}" >&2
        return 1
    fi
    if [[ ! -f "${TAF_DATA}" ]]; then
        echo "ERROR: TAF data not found: ${TAF_DATA}" >&2
        return 1
    fi

    echo "=== Eval Hartmann-6D ${METHOD} (${SURROGATE}) ==="
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

    local PKL_PATH="${SAVE_DIR}/scale_sweep_data.pkl"
    if [[ -f "${PKL_PATH}" ]]; then
        mkdir -p "${SAVE_DIR}/replot"
        "${PYTHON_BIN}" -u MYRL/scripts/plot_scale_sweep.py \
            --pkl "${PKL_PATH}" --save_dir "${SAVE_DIR}/replot" --normalize 2>/dev/null \
            || echo "  [WARN] Replot failed for ${METHOD}/${SURROGATE}."
    fi
}

PIDS=()
LABELS=()
for SURROGATE in gp tabpfn_base; do
    run_eval "${METHOD}" "${SURROGATE}" &
    PIDS+=("$!")
    LABELS+=("${METHOD}/${SURROGATE}")
done

echo "Waiting for parallel Hartmann-6D ${METHOD} eval jobs..."
set +e
FAIL=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"
    STATUS=$?
    echo "  ${LABELS[$i]}: exit=${STATUS}"
    if [[ "${STATUS}" -ne 0 ]]; then
        FAIL=1
    fi
done
set -e

exit "${FAIL}"
