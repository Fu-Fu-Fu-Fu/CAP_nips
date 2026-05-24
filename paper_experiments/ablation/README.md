# Ablation Study

## Overview

Three categories of ablation experiments on Hartmann-6D and Alkox tasks.

### 1. Component Ablation (`component/`)

Removes one component at a time from CAP-PPO:

| Ablation | What's Removed | Implementation |
|----------|---------------|----------------|
| **w/o Synthetic Obj.** | Synthetic objective generation (GP->RFF / BNN+RFF) | Hartmann: `objective_source=direct`; Alkox: `bnn_rff_alpha=0` |
| **w/o Cross-Attention** | Cross-attention replaced with mean-pooling | `CAP_NO_CROSS_ATTN=1` env var (monkey-patches model) |

Each subdirectory contains one retained paper checkpoint (`ppo_final.pt` or `ppo_best.pt`), config, and evaluation results.

### 2. w/o RL (`wo_rl/`)

Evaluates a pretrained CAP-PPO model in a shared candidate pool setting without RL-learned strategy. See `scripts/eval_shared_pool.py`.

### 3. Sensitivity Analysis (`sensitivity/`)

4 hyperparameter dimensions, 2 settings each (+ default baseline):

| Dimension | Small/Short | Large/Long | Default |
|-----------|------------|------------|---------|
| Pool Size | pers=64 | pers=256 | pers=128 |
| Pool Construction | no_local (k=0) | heavy_local (k=6) | k=2/3 |
| K Variants | k=5 | k=20 | k=10 |
| Budget | short (18-20 steps) | long (48-80 steps) | 28-48 steps |

## Figures

All plots are in `figures/`:
- `ablation_component_comparison.{png,pdf}` — Full vs w/o Synthetic vs w/o Cross-Attention
- `ablation_lineplots_v*.{png,pdf}` — Sensitivity analysis line plots
- `ablation_normalized_regret_v*.{png,pdf}` — Normalized regret heatmaps

## Design Document

Older design logs were moved to the external cleanup backup. The retained tree is the paper reproduction snapshot.

## Scripts

- `scripts/train_ablation.sh` — Unified training script (env-var driven)
- `scripts/eval_ablation.sh` — Unified evaluation script
- `scripts/run_one_ablation.sh` — Train + Eval pipeline for one experiment
- `scripts/run_all_ablations.sh` — Master orchestrator for all 18 sensitivity experiments
- `scripts/eval_wo_rl_*.sh` — w/o RL evaluation scripts
- `scripts/eval_wo_synthetic_*.sh` — w/o Synthetic evaluation scripts
- `scripts/eval_wo_crossattn_*.sh` — w/o Cross-Attention evaluation scripts
