# Ablation Experiments (2026-04-22)

## Overview

4 ablation dimensions x 2 tasks (Hartmann-6D + Alkox) x 2 variants = 16 experiments + 2 default baselines = **18 total**.

ALL experiments use the **new version**: `regret_balanced` + z-norm (`normalize_oracle_gp` for Hartmann-6D, `normalize_bnn` for Alkox).

Alkox new version key params (aligned with HPLC):
- `bnn_rff_alpha=1.5` (old version was 5.0, z-norm changes effective scale)
- `normalize_bnn=True`
- `reward_mode=regret_balanced`
- `ent_coef: 0.02 -> 0.002`
- `update_every=40`

Each experiment: Train(5000ep, n_workers=8) -> Eval(GP, 7scales x 20variants x 3runs) -> Eval(TabPFN, same) -> Plot

## Prerequisites: Data Generation

```bash
bash paper_experiments/ablation/scripts/run_all_ablations.sh data
```

## Commands (18 experiments)

### 0. Default baselines

```bash
# Hartmann-6D default (model already trained, will skip training, only eval+plot)
TASK=hartmann_6d_family ABLATION_NAME=default \
  bash paper_experiments/ablation/scripts/run_one_ablation.sh \
  2>&1 | tee paper_experiments/ablation/hartmann6d_default.log

# Alkox default (new version, needs training)
TASK=alkox_emulator ABLATION_NAME=default \
  bash paper_experiments/ablation/scripts/run_one_ablation.sh \
  2>&1 | tee paper_experiments/ablation/alkox_default.log
```

### 1. Pool Size

```bash
# Hartmann-6D small pool (pers=64, total=128)
TASK=hartmann_6d_family ABLATION_NAME=pool_size_small \
  AB_N_PERSISTENT_BASE=64 AB_N_TOTAL_CANDIDATES=128 \
  bash paper_experiments/ablation/scripts/run_one_ablation.sh \
  2>&1 | tee paper_experiments/ablation/hartmann6d_pool_size_small.log

# Hartmann-6D large pool (pers=256, total=512)
TASK=hartmann_6d_family ABLATION_NAME=pool_size_large \
  AB_N_PERSISTENT_BASE=256 AB_N_TOTAL_CANDIDATES=512 \
  bash paper_experiments/ablation/scripts/run_one_ablation.sh \
  2>&1 | tee paper_experiments/ablation/hartmann6d_pool_size_large.log

# Alkox small pool (pers=64, total=96)
TASK=alkox_emulator ABLATION_NAME=pool_size_small \
  AB_N_PERSISTENT_BASE=64 AB_N_TOTAL_CANDIDATES=96 \
  bash paper_experiments/ablation/scripts/run_one_ablation.sh \
  2>&1 | tee paper_experiments/ablation/alkox_pool_size_small.log

# Alkox large pool (pers=256, total=384)
TASK=alkox_emulator ABLATION_NAME=pool_size_large \
  AB_N_PERSISTENT_BASE=256 AB_N_TOTAL_CANDIDATES=384 \
  bash paper_experiments/ablation/scripts/run_one_ablation.sh \
  2>&1 | tee paper_experiments/ablation/alkox_pool_size_large.log
```

### 2. Pool Construction

```bash
# Hartmann-6D no local candidates (pure Sobol, k=0)
TASK=hartmann_6d_family ABLATION_NAME=pool_construct_no_local \
  AB_N_PERSISTENT_BASE=256 AB_N_TOTAL_CANDIDATES=256 AB_K_CENTERS=0 \
  bash paper_experiments/ablation/scripts/run_one_ablation.sh \
  2>&1 | tee paper_experiments/ablation/hartmann6d_pool_construct_no_local.log

# Hartmann-6D heavy local (k=6)
TASK=hartmann_6d_family ABLATION_NAME=pool_construct_heavy_local \
  AB_N_PERSISTENT_BASE=64 AB_N_TOTAL_CANDIDATES=256 AB_K_CENTERS=6 \
  bash paper_experiments/ablation/scripts/run_one_ablation.sh \
  2>&1 | tee paper_experiments/ablation/hartmann6d_pool_construct_heavy_local.log

# Alkox no local candidates (pure Sobol, k=0)
TASK=alkox_emulator ABLATION_NAME=pool_construct_no_local \
  AB_N_PERSISTENT_BASE=192 AB_N_TOTAL_CANDIDATES=192 AB_K_CENTERS=0 \
  bash paper_experiments/ablation/scripts/run_one_ablation.sh \
  2>&1 | tee paper_experiments/ablation/alkox_pool_construct_no_local.log

# Alkox heavy local (k=6)
TASK=alkox_emulator ABLATION_NAME=pool_construct_heavy_local \
  AB_N_PERSISTENT_BASE=48 AB_N_TOTAL_CANDIDATES=192 AB_K_CENTERS=6 \
  bash paper_experiments/ablation/scripts/run_one_ablation.sh \
  2>&1 | tee paper_experiments/ablation/alkox_pool_construct_heavy_local.log
```

### 3. K Variants (training task diversity)

```bash
# Hartmann-6D K=5
TASK=hartmann_6d_family ABLATION_NAME=k_variants_5 AB_K_VARIANTS=5 \
  bash paper_experiments/ablation/scripts/run_one_ablation.sh \
  2>&1 | tee paper_experiments/ablation/hartmann6d_k_variants_5.log

# Hartmann-6D K=20
TASK=hartmann_6d_family ABLATION_NAME=k_variants_20 AB_K_VARIANTS=20 \
  bash paper_experiments/ablation/scripts/run_one_ablation.sh \
  2>&1 | tee paper_experiments/ablation/hartmann6d_k_variants_20.log

# Alkox K=5
TASK=alkox_emulator ABLATION_NAME=k_variants_5 AB_K_VARIANTS=5 \
  bash paper_experiments/ablation/scripts/run_one_ablation.sh \
  2>&1 | tee paper_experiments/ablation/alkox_k_variants_5.log

# Alkox K=20
TASK=alkox_emulator ABLATION_NAME=k_variants_20 AB_K_VARIANTS=20 \
  bash paper_experiments/ablation/scripts/run_one_ablation.sh \
  2>&1 | tee paper_experiments/ablation/alkox_k_variants_20.log
```

### 4. Budget (trajectory length)

```bash
# Hartmann-6D short budget (18 steps, 20 evals total)
TASK=hartmann_6d_family ABLATION_NAME=budget_short \
  AB_MAX_STEPS=18 AB_DATA_SUFFIX="_budget20" \
  bash paper_experiments/ablation/scripts/run_one_ablation.sh \
  2>&1 | tee paper_experiments/ablation/hartmann6d_budget_short.log

# Hartmann-6D long budget (78 steps, 80 evals total)
TASK=hartmann_6d_family ABLATION_NAME=budget_long \
  AB_MAX_STEPS=78 AB_DATA_SUFFIX="_budget80" \
  bash paper_experiments/ablation/scripts/run_one_ablation.sh \
  2>&1 | tee paper_experiments/ablation/hartmann6d_budget_long.log

# Alkox short budget (13 steps, 15 evals total)
TASK=alkox_emulator ABLATION_NAME=budget_short \
  AB_MAX_STEPS=13 AB_DATA_SUFFIX="_transform_budget15" \
  bash paper_experiments/ablation/scripts/run_one_ablation.sh \
  2>&1 | tee paper_experiments/ablation/alkox_budget_short.log

# Alkox long budget (48 steps, 50 evals total)
TASK=alkox_emulator ABLATION_NAME=budget_long \
  AB_MAX_STEPS=48 AB_DATA_SUFFIX="_transform_budget50" \
  bash paper_experiments/ablation/scripts/run_one_ablation.sh \
  2>&1 | tee paper_experiments/ablation/alkox_budget_long.log
```

## Run all at once

```bash
# Full pipeline (data + all 18 experiments)
bash paper_experiments/ablation/scripts/run_all_ablations.sh all

# Or just the experiments (if data already generated)
bash paper_experiments/ablation/scripts/run_all_ablations.sh run
```

## Ablation Summary Table

| Dimension | Variant | Hartmann-6D | Alkox |
|-----------|---------|-------------|-------|
| Default | baseline | pers=128, total=256, k=3 | pers=128, total=192, k=2 |
| Pool Size | small | pers=64, total=128 | pers=64, total=96 |
| Pool Size | large | pers=256, total=512 | pers=256, total=384 |
| Pool Construction | no_local (k=0) | pers=256, total=256 | pers=192, total=192 |
| Pool Construction | heavy_local (k=6) | pers=64, total=256 | pers=48, total=192 |
| K Variants | K=5 | 5 training variants | 5 training variants |
| K Variants | K=20 | 20 training variants | 20 training variants |
| Budget | short | 18 steps (budget 20) | 13 steps (budget 15) |
| Budget | long | 78 steps (budget 80) | 48 steps (budget 50) |

## Notes

- Hartmann-6D default model is reused from `paper_experiments/hartmann6d/runs/ppo_hartmann_6d_family_znorm_rb_ep5000/ppo_best.pt` (already trained)
- All other 17 experiments require training from scratch
- Each step (train/eval_gp/eval_tabpfn) skips if output already exists
- Historical `_resume/` eval checkpoints and run logs were moved to the external cleanup backup
