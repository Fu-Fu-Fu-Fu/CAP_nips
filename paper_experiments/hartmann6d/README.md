# Hartmann-6D Family Experiments

## Model Versions

| Version | Location | Reward Mode | Z-norm | Notes |
|---------|----------|-------------|--------|-------|
| **old** | `models/default/` | None (original) | No | First-generation model |
| **znorm_rb** | `runs/ppo_hartmann_6d_family_znorm_rb_ep5000/` | regret_balanced | Yes | Current best; used in paper |

The znorm_rb model uses: regret_balanced reward, z-normalized oracle_gp, shuffled_cycle variant sampling, entropy annealing 0.02 -> 0.002.

## Results

| Version | GP | TabPFN |
|---------|----|--------|
| old | `results/old_gp/` | `results/old_tabpfn/` |
| **znorm_rb** (paper) | `results/znorm_rb_gp/` | `results/znorm_rb_tabpfn/` |

## Scripts

- `scripts/train_hartmann6d.sh` — trains the znorm_rb model
- `scripts/eval_hartmann6d.sh` — evaluates old model
- `scripts/eval_hartmann6d_znorm_rb.sh` — evaluates znorm_rb model
- `scripts/eval_hartmann6d_full.sh` — eval + normalized-regret plots

## Usage

```bash
SURROGATE=gp N_WORKERS=4 bash paper_experiments/hartmann6d/scripts/eval_hartmann6d_znorm_rb.sh
```
