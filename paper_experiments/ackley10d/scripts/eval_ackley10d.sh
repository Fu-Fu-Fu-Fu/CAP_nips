#!/bin/bash
# Ackley-10D scale sweep eval; GP and TabPFN run concurrently, each with workers.
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

TASK="ackley_10d_family"
EXP_DIR="paper_experiments/ackley10d"
MODEL_PATH="${MODEL_PATH:-${EXP_DIR}/models/msrff_mid/ppo_best.pt}"
TAF_DATA="${TAF_DATA:-data/taf_source_data_${TASK}_k10.pkl}"

N_WORKERS="${N_WORKERS:-4}"
FRESH_RUN="${FRESH_RUN:-0}"
PROGRESS_DETAIL="${PROGRESS_DETAIL:-summary}"
METHODS="${METHODS:-Random EI UCB PI FunBO PFNs4BO TuRBO TAF_me TAF_ranking CAP-PPO}"

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Missing model: ${MODEL_PATH}" >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ "${FRESH_RUN}" == "1" ]]; then
  EXTRA_ARGS+=(--fresh_run)
fi
read -r -a METHOD_ARGS <<< "${METHODS}"

run_eval() {
  local SURROGATE="$1"
  local SAVE_DIR="${EXP_DIR}/results/msrff_mid_${SURROGATE}"
  echo "=== Ackley-10D eval: ${SURROGATE} ==="
  "${PYTHON_BIN}" -u MYRL/scripts/eval_scale_sweep.py \
    --task "${TASK}" \
    --rl_model_path "${MODEL_PATH}" \
    --taf_data_path "${TAF_DATA}" \
    --surrogate "${SURROGATE}" \
    --scales 0.5 0.75 1.0 1.25 1.5 1.75 2.0 \
    --n_variants 20 \
    --n_runs 3 \
    --max_steps 78 \
    --n_init 2 \
    --n_persistent_base 192 \
    --n_total_candidates 384 \
    --k_centers 4 \
    --local_h 0.12 \
    --local_h_decay 0.96 \
    --explore_fraction 0.30 \
    --n_candidates_baseline 4096 \
    --methods "${METHOD_ARGS[@]}" \
    --n_workers "${N_WORKERS}" \
    --progress_detail "${PROGRESS_DETAIL}" \
    --seed 2026 \
    --save_dir "${SAVE_DIR}" \
    "${EXTRA_ARGS[@]}"
}

run_eval gp &
PID_GP=$!
run_eval tabpfn_base &
PID_TABPFN=$!

wait "${PID_GP}"
EXIT_GP=$?
wait "${PID_TABPFN}"
EXIT_TABPFN=$?

echo "Eval exits: gp=${EXIT_GP}, tabpfn_base=${EXIT_TABPFN}"
if [[ "${EXIT_GP}" != "0" || "${EXIT_TABPFN}" != "0" ]]; then
  exit 1
fi
