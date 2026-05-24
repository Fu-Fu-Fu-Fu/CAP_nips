#!/bin/bash
# =============================================================================
# Eval Branin calibrated multi-scale RFF + medium pool model.
# Runs GP and TabPFN eval concurrently; each eval also uses worker parallelism.
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
MODEL_PATH="${MODEL_PATH:-paper_experiments/branin/models/${TAG}/ppo_best.pt}"
TAF_DATA="${TAF_DATA:-data/taf_source_data_branin_family_k10.pkl}"
LOG_DIR="paper_experiments/branin/logs"
mkdir -p "${LOG_DIR}" "${MPLCONFIGDIR}"

N_WORKERS="${N_WORKERS:-4}"
PROGRESS_DETAIL="${PROGRESS_DETAIL:-summary}"
SHOW_THIRD_PARTY_OUTPUT="${SHOW_THIRD_PARTY_OUTPUT:-0}"
FRESH_RUN="${FRESH_RUN:-0}"

N_PERSISTENT_BASE="${N_PERSISTENT_BASE:-128}"
N_TOTAL_CANDIDATES="${N_TOTAL_CANDIDATES:-256}"
K_CENTERS="${K_CENTERS:-3}"
LOCAL_H="${LOCAL_H:-2.25}"
LOCAL_H_DECAY="${LOCAL_H_DECAY:-0.95}"
EXPLORE_FRACTION="${EXPLORE_FRACTION:-0.25}"

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "ERROR: Model not found at ${MODEL_PATH}"
    echo "Run train_branin_calibrated_multiscale_rff_medium_pool.sh first."
    exit 1
fi

EXTRA_ARGS=()
if [[ "${SHOW_THIRD_PARTY_OUTPUT}" == "1" ]]; then
    EXTRA_ARGS+=(--show_third_party_output)
fi
if [[ "${FRESH_RUN}" == "1" ]]; then
    EXTRA_ARGS+=(--fresh_run)
fi

if [[ -n "${METHODS:-}" ]]; then
    read -r -a METHOD_ARGS <<< "${METHODS}"
else
    METHOD_ARGS=(Random EI UCB PI FunBO PFNs4BO TuRBO TAF_me TAF_ranking CAP-PPO)
fi

run_eval() {
    local SURROGATE="$1"
    local SAVE_DIR="paper_experiments/branin/results/${TAG}_${SURROGATE}"

    echo "=== Branin medium-pool eval (${SURROGATE}) ==="
    echo "Save: ${SAVE_DIR}"

    "${PYTHON_BIN}" -u MYRL/scripts/eval_scale_sweep.py \
        --task branin_family \
        --rl_model_path "${MODEL_PATH}" \
        --taf_data_path "${TAF_DATA}" \
        --surrogate "${SURROGATE}" \
        --scales 0.5 0.75 1.0 1.25 1.5 1.75 2.0 \
        --n_variants 20 \
        --n_runs 3 \
        --max_steps 18 \
        --n_init 2 \
        --n_persistent_base "${N_PERSISTENT_BASE}" \
        --n_total_candidates "${N_TOTAL_CANDIDATES}" \
        --k_centers "${K_CENTERS}" \
        --local_h "${LOCAL_H}" \
        --local_h_decay "${LOCAL_H_DECAY}" \
        --explore_fraction "${EXPLORE_FRACTION}" \
        --n_candidates_baseline 2048 \
        --methods "${METHOD_ARGS[@]}" \
        --n_workers "${N_WORKERS}" \
        --progress_detail "${PROGRESS_DETAIL}" \
        --seed 2026 \
        --save_dir "${SAVE_DIR}" \
        "${EXTRA_ARGS[@]}"

    echo "=== Done: ${SAVE_DIR} ==="
}

run_eval "gp" &
PID_GP=$!

run_eval "tabpfn_base" &
PID_TABPFN=$!

echo "Waiting for parallel eval jobs (gp: ${PID_GP}, tabpfn: ${PID_TABPFN})..."
set +e
wait "${PID_GP}"
EXIT_GP=$?
wait "${PID_TABPFN}"
EXIT_TABPFN=$?
set -e

echo "Eval complete — gp: exit=${EXIT_GP}, tabpfn_base: exit=${EXIT_TABPFN}"
if [[ "${EXIT_GP}" != "0" || "${EXIT_TABPFN}" != "0" ]]; then
    exit 1
fi
