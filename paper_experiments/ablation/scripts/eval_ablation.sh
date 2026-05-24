#!/bin/bash
# =============================================================================
# eval_ablation.sh — Unified ablation evaluation script
#
# Runs 7-scale sweep with resume support (saves per-variant results).
# 20 variants × 3 runs per scale. Supports both GP and TabPFN surrogates.
#
# Required env vars:
#   TASK          - "hartmann_6d_family" or "alkox_emulator"
#   ABLATION_NAME - e.g., "pool_size_small", "default"
#   SURROGATE     - "gp" or "tabpfn_base"
#
# Optional env vars (must match training config):
#   AB_N_PERSISTENT_BASE  AB_N_TOTAL_CANDIDATES  AB_K_CENTERS
#   AB_MAX_STEPS          AB_K_VARIANTS          AB_DATA_SUFFIX
#   AB_NO_CROSS_ATTN      - set to "1" to use no-crossattn eval script
#   N_WORKERS
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

: "${TASK:?TASK must be set}"
: "${ABLATION_NAME:?ABLATION_NAME must be set}"
: "${SURROGATE:?SURROGATE must be set (gp or tabpfn_base)}"

N_WORKERS="${N_WORKERS:-8}"

# ===== Task-specific defaults =====
if [[ "${TASK}" == "hartmann_6d_family" ]]; then
    DEF_N_PERSISTENT_BASE=128
    DEF_N_TOTAL_CANDIDATES=256
    DEF_K_CENTERS=3
    DEF_LOCAL_H=0.15
    DEF_LOCAL_H_DECAY=0.95
    DEF_MAX_STEPS=48
    DEF_N_INIT=2
    DEF_K_VARIANTS=10
    DEF_DATA_SUFFIX=""
elif [[ "${TASK}" == "alkox_emulator" ]]; then
    DEF_N_PERSISTENT_BASE=128
    DEF_N_TOTAL_CANDIDATES=192
    DEF_K_CENTERS=2
    DEF_LOCAL_H=0.17
    DEF_LOCAL_H_DECAY=0.95
    DEF_MAX_STEPS=28
    DEF_N_INIT=2
    DEF_K_VARIANTS=10
    DEF_DATA_SUFFIX="_transform"
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
DATA_SUFFIX="${AB_DATA_SUFFIX:-${DEF_DATA_SUFFIX}}"

# ===== Paths =====
TASK_SHORT="${TASK//_emulator/}"
TASK_SHORT="${TASK_SHORT//_family/}"
EXP_DIR="paper_experiments/ablation/sensitivity/${TASK_SHORT}/${ABLATION_NAME}"

# Use ppo_final.pt (last checkpoint), fall back to ppo_best.pt
if [[ -f "${EXP_DIR}/ppo_final.pt" ]]; then
    MODEL_PATH="${EXP_DIR}/ppo_final.pt"
elif [[ -f "${EXP_DIR}/ppo_best.pt" ]]; then
    MODEL_PATH="${EXP_DIR}/ppo_best.pt"
else
    echo "ERROR: No model found in ${EXP_DIR}/" >&2
    exit 1
fi

TAF_DATA="data/taf_source_data_${TASK}_k${K_VARIANTS}${DATA_SUFFIX}.pkl"
SAVE_DIR="${EXP_DIR}/results_${SURROGATE}"

# ===== Skip if already complete =====
if [[ -f "${SAVE_DIR}/scale_sweep_data.pkl" ]]; then
    echo "[EVAL ${SURROGATE}] SKIP — results exist: ${SAVE_DIR}/scale_sweep_data.pkl"
    exit 0
fi

# ===== Validate =====
if [[ ! -f "${TAF_DATA}" ]]; then
    echo "ERROR: TAF data not found: ${TAF_DATA}" >&2
    exit 1
fi

echo "================================================================"
echo "  Ablation Evaluation: ${ABLATION_NAME} (${SURROGATE})"
echo "================================================================"
echo "Task:       ${TASK}"
echo "Model:      ${MODEL_PATH}"
echo "Surrogate:  ${SURROGATE}"
echo "Budget:     $((N_INIT + MAX_STEPS)) (${N_INIT} init + ${MAX_STEPS} steps)"
echo "Pool:       pers=${N_PERSISTENT_BASE}, total=${N_TOTAL_CANDIDATES}, k=${K_CENTERS}"
echo "K variants: ${K_VARIANTS}"
echo "Workers:    ${N_WORKERS}"
echo "Save:       ${SAVE_DIR}"
echo "================================================================"

NO_CROSS_ATTN="${AB_NO_CROSS_ATTN:-0}"
if [[ "${NO_CROSS_ATTN}" == "1" ]]; then
    EVAL_SCRIPT="MYRL/scripts/eval_scale_sweep_no_crossattn.py"
    echo "Architecture: NO cross-attention (mean-pooling ablation)"
else
    EVAL_SCRIPT="MYRL/scripts/eval_scale_sweep.py"
fi

start_time=$(date +%s)

"${PYTHON_BIN}" -u "${EVAL_SCRIPT}" \
    --task "${TASK}" \
    --rl_model_path "${MODEL_PATH}" \
    --taf_data_path "${TAF_DATA}" \
    --surrogate "${SURROGATE}" \
    --scales 0.5 0.75 1.0 1.25 1.5 1.75 2.0 \
    --n_variants 20 \
    --n_runs 3 \
    --max_steps "${MAX_STEPS}" \
    --n_init "${N_INIT}" \
    --n_persistent_base "${N_PERSISTENT_BASE}" \
    --n_total_candidates "${N_TOTAL_CANDIDATES}" \
    --k_centers "${K_CENTERS}" \
    --local_h "${LOCAL_H}" \
    --local_h_decay "${LOCAL_H_DECAY}" \
    --n_candidates_baseline 2048 \
    --methods Random EI UCB PI FunBO PFNs4BO TuRBO TAF_me TAF_ranking CAP-PPO \
    --n_workers "${N_WORKERS}" \
    --progress_detail method \
    --seed 2026 \
    --save_dir "${SAVE_DIR}"

end_time=$(date +%s)
runtime=$((end_time - start_time))
echo ""
echo "Eval complete: ${ABLATION_NAME} (${SURROGATE})"
echo "Time: $((runtime / 3600))h $(((runtime % 3600) / 60))m $((runtime % 60))s"
echo "Results: ${SAVE_DIR}/"
