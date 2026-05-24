# CAP-PPO / Context-Aware Policy for Bayesian Optimization

This repository is a cleaned, reproducible snapshot of the CAP-PPO experiments. It was prepared for migration to a new server and for AI-assisted continuation of the project.

The project trains and evaluates a PPO policy for Bayesian optimization. The policy selects the next query point from a candidate pool using context from previous observations, candidate features, and task-family transfer information.

## Current Snapshot

This is not the original messy working directory. Failed trials, old raw runs, intermediate checkpoints, tensorboard folders, logs, `_resume/` evaluation caches, and obsolete archives were moved out of the repository.

External backup location on the old machine:

```text
/mnt/ai4sci_develop_fast/home/linfu/CAP_no_normalized_regret_cleanup_backup_20260520
```

Approximate retained sizes after cleanup:

- Repository excluding `.git`: about `1.5G`
- `paper_experiments/`: about `1.3G`
- External backup: about `6.5G`

Important cleanup guarantees already checked:

- No `ppo_ep*.pt` intermediate checkpoints remain.
- No `_resume/`, `tensorboard/`, `logs/`, `.log`, or `__pycache__/` remain.
- No non-git file larger than GitHub's 100MB single-file limit remains.
- `paper_experiments/**/scale_sweep_results.json` has `0` missing model references.
- Python source files parse successfully.
- Shell experiment scripts pass `bash -n`.
- Task registry imports work for Branin, Hartmann, Ackley, Alkox, and HPLC tasks.

## Repository Layout

```text
.
├── MYRL/                         # Main CAP-PPO code: tasks, policy, PPO training, evaluation
├── data/                         # Shared retained variants, BO trajectories, TAF data, BNN params
├── olympus/src/                  # Local Olympus emulator code for chemistry tasks
├── paper_experiments/            # Paper models, results, figures, and ablations
├── md/                           # Retained notes/reports
├── tabpfn-v2-regressor.ckpt      # Local TabPFN checkpoint used by evaluation
├── README.md                     # This file
└── .gitignore
```

Core code locations:

- `MYRL/myrl/tasks/`: task definitions and task registry.
- `MYRL/myrl/policies/`: CAP policy implementation.
- `MYRL/myrl/rl/`: PPO training logic and ablation variants.
- `MYRL/myrl/eval/`: evaluation utilities.
- `MYRL/scripts/train_rl.py`: main PPO training entrypoint.
- `MYRL/scripts/eval_scale_sweep.py`: main scale-sweep evaluation entrypoint.
- `MYRL/scripts/plot_scale_sweep.py`: plotting from evaluation results.

## Environment Setup

Use Python 3.10 or 3.11 if possible. The original environment used many scientific packages plus PyTorch, scikit-learn, BoTorch/GPyTorch, TabPFN, and TensorFlow/Olympus dependencies.

Recommended environment variables after cloning:

```bash
export REPO_ROOT="$(pwd)"
export PYTHONPATH="${REPO_ROOT}/MYRL:${REPO_ROOT}/olympus/src:${PYTHONPATH}"
export TF_USE_LEGACY_KERAS=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TABPFN_MODEL_CACHE_DIR="${REPO_ROOT}"
```

Most experiment shell scripts now infer `REPO_ROOT` from git and default to `python`. Override explicitly when needed:

```bash
export PYTHON_BIN=/path/to/python
```

If the server cannot access HuggingFace and the local TabPFN checkpoint is present:

```bash
export HF_HUB_OFFLINE=1
export TABPFN_MODEL_CACHE_DIR="${REPO_ROOT}"
```

The local Olympus requirements are listed in `olympus/requirements.txt`, but this file includes old TensorFlow-era dependencies. For migration, prefer first making the existing Python environment import the retained tasks, then run full evaluations.

## GitHub / Large File Notes

The retained `.pt`, `.ckpt`, and `.pkl` files are needed for reproducing paper figures and verifying final results. They are all below 100MB each after cleanup, but they are binary artifacts and should ideally be tracked with Git LFS:

```bash
git lfs track "*.pt"
git lfs track "*.ckpt"
git lfs track "*.pkl"
git add .gitattributes
```

`git-lfs` was not installed on the old machine during cleanup.

## Paper Experiment Snapshot

All retained paper artifacts live under `paper_experiments/`.

Evaluation protocol for main results:

- Methods: `Random`, `EI`, `UCB`, `PI`, `FunBO`, `PFNs4BO`, `TuRBO`, `TAF_me`, `TAF_ranking`, `CAP-PPO`
- Scales: `0.5`, `0.75`, `1.0`, `1.25`, `1.5`, `1.75`, `2.0`
- Per scale: `20` variants x `3` runs
- Surrogates: GP and TabPFN
- Main metric: final simple regret, lower is better

### Retained Main Models and Results

| Task | Retained model | Retained results |
|---|---|---|
| Branin 2D | `paper_experiments/branin/models/calibrated_multiscale_rff_pool/ppo_best.pt` | `results/calibrated_multiscale_rff_pool_gp/`, `results/calibrated_multiscale_rff_pool_tabpfn_base/` |
| Branin 2D | `paper_experiments/branin/models/calibrated_multiscale_rff_medium_pool/ppo_best.pt` | `results/calibrated_multiscale_rff_medium_pool_gp/`, `results/calibrated_multiscale_rff_medium_pool_tabpfn_base/` |
| Hartmann3D | `paper_experiments/hartmann3d/models/default/ppo_final.pt` | `results/gp/`, `results/tabpfn_base/` |
| Hartmann6D | `paper_experiments/hartmann6d/runs/ppo_hartmann_6d_family_znorm_rb_ep5000/ppo_best.pt` | `results/znorm_rb_gp/`, `results/znorm_rb_tabpfn_base/` |
| HPLC | `paper_experiments/hplc/models/default/ppo_best.pt` | `results/gp/`, `results/tabpfn/` |
| Alkox | `paper_experiments/alkox/models/default/ppo_best.pt` | `results/gp/`, `results/tabpfn/` |
| Ackley5D | `paper_experiments/ackley5d/models/fixedgp_ls07_smoothweak/ppo_best.pt` | `results/fixedgp_ls07_smoothweak_gp/`, `results/fixedgp_ls07_smoothweak_tabpfn_base/` |
| Ackley10D | `paper_experiments/ackley10d/models/msrff_mid/ppo_best.pt` | `results/msrff_mid_gp/`, `results/msrff_mid_tabpfn_base/` |

Each result directory generally contains:

```text
scale_sweep_results.json       # Summary statistics by method and scale
scale_sweep_data.pkl           # Full trajectory data for replotting and analysis
scale_sweep_*.png/pdf          # Retained final plots
```

### Ablation Snapshot

Retained ablation artifacts are under:

```text
paper_experiments/ablation/
paper_experiments/ablation_plot_data/
```

The retained ablation tree is the paper reproduction version. It keeps final or best checkpoints, configs, result JSON/PKL files, figures, and plot data. Intermediate checkpoints and `_resume/` caches were moved to backup.

Ablation categories:

- `ablation/component/`: component ablations such as w/o synthetic objective and w/o cross-attention.
- `ablation/sensitivity/`: sensitivity analysis for pool size, pool construction, number of variants, and budget.
- `ablation/trajectory/`: random/DOE trajectory ablations.
- `ablation/wo_rl/`: w/o RL/shared-pool evaluation.
- `ablation/figures/`: retained ablation figures.
- `ablation/sensitivity_plot_data/`: compact/enhanced JSON plot data for sensitivity figures.

Read:

- `paper_experiments/ablation/README.md`
- `paper_experiments/ablation/README_sensitivity.md`

## Shared Data

`data/` contains retained files required by current configs and scripts:

- Synthetic family variants and BO trajectories for Branin, Hartmann3D, Hartmann6D, Ackley5D, Ackley10D.
- Alkox and HPLC emulator variant files.
- BNN surrogate parameter files for Alkox and HPLC.
- TAF source data files for transfer acquisition baselines.
- Budget/K-variant variants used by ablations.

Some task-specific data is also kept inside paper experiment folders, for example:

- `paper_experiments/branin/data/`
- `paper_experiments/hartmann3d/data/`

## Common Commands

Always run commands from the repository root unless stated otherwise.

### 1. Quick Import Smoke Test

```bash
export REPO_ROOT="$(pwd)"
export PYTHONPATH="${REPO_ROOT}/MYRL:${REPO_ROOT}/olympus/src:${PYTHONPATH}"
export TF_USE_LEGACY_KERAS=1

PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
from myrl.tasks.registry import get_task
for name in [
    "branin_family",
    "hartmann_3d_family",
    "hartmann_6d_family",
    "ackley_5d_family",
    "ackley_10d_family",
    "alkox_emulator",
    "hplc_emulator",
]:
    task = get_task(name)
    print(name, task.__class__.__name__)
PY
```

### 2. Check Paper Result Model References

```bash
python - <<'PY'
import json, glob, os
missing = []
for p in sorted(glob.glob("paper_experiments/**/scale_sweep_results.json", recursive=True)):
    d = json.load(open(p))
    m = d.get("model")
    if m and not os.path.exists(m):
        missing.append((p, m))
print("scale_sweep_results", len(glob.glob("paper_experiments/**/scale_sweep_results.json", recursive=True)))
print("missing_model_refs", len(missing))
for row in missing[:20]:
    print(row)
PY
```

### 3. Re-evaluate a Retained Main Model

Use the task-specific wrapper scripts when possible:

```bash
bash paper_experiments/hartmann6d/scripts/eval_hartmann6d_znorm_rb.sh
bash paper_experiments/alkox/scripts/eval_alkox.sh
bash paper_experiments/hplc/scripts/eval_hplc.sh
bash paper_experiments/branin/scripts/eval_branin_calibrated_multiscale_rff_pool.sh
bash paper_experiments/ackley5d/scripts/eval_ackley5d_fixedgp_ls07_smoothweak.sh
bash paper_experiments/ackley10d/scripts/eval_ackley10d.sh
```

Most scripts support environment-variable overrides such as:

```bash
SURROGATE=gp N_WORKERS=4 PYTHON_BIN=python bash paper_experiments/alkox/scripts/eval_alkox.sh
```

### 4. Train a Retained Configuration

Use the retained training wrappers:

```bash
bash paper_experiments/hartmann6d/scripts/train_hartmann6d.sh
bash paper_experiments/hartmann3d/scripts/train_hartmann3d.sh
bash paper_experiments/branin/scripts/train_branin_calibrated_multiscale_rff_pool.sh
bash paper_experiments/branin/scripts/train_branin_calibrated_multiscale_rff_medium_pool.sh
bash paper_experiments/ackley5d/scripts/train_ackley5d_fixedgp_ls07_smoothweak.sh
bash paper_experiments/ackley10d/scripts/train_ackley10d.sh
```

Training chemistry tasks may require TensorFlow/Olympus emulator dependencies to be working.

### 5. Replot from Existing Results

Use `MYRL/scripts/plot_scale_sweep.py` or the existing plot-data extraction scripts. The safest approach is to reuse existing `scale_sweep_data.pkl` and `scale_sweep_results.json` files rather than rerunning full evaluations.

Useful retained plot directories:

```text
paper_experiments/figures/
paper_experiments/ablation/figures/
paper_experiments/ablation_plot_data/
paper_experiments/ablation/sensitivity_plot_data/
```

## Important Implementation Notes

### Objective Direction

The framework minimizes. Tasks that are naturally maximization tasks, such as chemistry emulator outputs, negate the emulator output internally.

### Task Families

Synthetic task families generate variants through affine transformations:

- translation `dx`
- rotation `rot`
- scaling `sx`

Chemistry emulator tasks use Olympus neural-network emulators plus BNN/RFF variants.

### Training Objective Sources

Common `objective_source` values:

- `direct`: use the true task/emulator directly.
- `oracle_gp`: fit GP to true function values and sample RFF posterior objectives.
- `bnn`: chemistry BNN mean plus RFF prior perturbation.

### Reward

Most retained final models use `reward_mode=regret_balanced`, with entropy annealing and shuffled-cycle variant sampling. Check each model's `config.json` for exact hyperparameters.

## For AI Agents Reading This Repo

Start here:

1. Read this `README.md`.
2. Read `paper_experiments/README.md`.
3. Read the relevant task subdirectory README if present, such as `paper_experiments/branin/README.md` or `paper_experiments/hartmann6d/README.md`.
4. Inspect `paper_experiments/**/models/*/config.json` for exact training settings.
5. Inspect `paper_experiments/**/results/*/scale_sweep_results.json` for final summarized performance.
6. Use `scale_sweep_data.pkl` only when full trajectories or replotting are needed.

Do not assume old run directories exist. The current tree intentionally keeps only final/best checkpoints and paper-relevant result artifacts.

Do not recreate `_resume/`, tensorboard, logs, or intermediate `ppo_ep*.pt` files in the repository unless debugging locally. They are ignored by `.gitignore`.

If a path in an old note points to the old machine or to a `*_20260413` / `ablation_20260422` folder, prefer the current paths under `paper_experiments/` and `data/`. The active result JSON/config references were already normalized during cleanup.

## Migration Checklist

After cloning on a new server:

1. Install project dependencies and make sure `python` points to the intended environment.
2. Set `PYTHONPATH`, `TF_USE_LEGACY_KERAS`, `TF_FORCE_GPU_ALLOW_GROWTH`, and `TABPFN_MODEL_CACHE_DIR`.
3. Confirm `tabpfn-v2-regressor.ckpt` is present in the repo root.
4. Run the quick import smoke test above.
5. Run the model-reference check above.
6. Run one small evaluation or plotting command before launching full sweeps.
7. If pushing to GitHub, install Git LFS and track `*.pt`, `*.ckpt`, and `*.pkl`.

## Cleanup History

Moved out of the repo:

- obsolete `MetaBO/`
- old root `runs/` and `results_policies/`
- `_archive/`
- old Olympus docs/dev/tests/case studies
- raw failed/superseded `paper_experiments` runs
- `_resume/` eval caches
- tensorboard directories
- training logs
- intermediate PPO checkpoints `ppo_ep*.pt`
- oversized duplicate Alkox JSON result
- Python bytecode caches

The backup was intentionally placed outside this repo so the current directory is suitable for GitHub and server migration.
