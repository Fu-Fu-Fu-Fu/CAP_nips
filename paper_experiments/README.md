# Paper Experiments — CAP-PPO

## Directory Structure

| Directory | Task | Dim | Objective | Status |
|-----------|------|-----|-----------|--------|
| `branin/` | Branin Family (2D) | 2 | oracle_gp | Multiple model versions |
| `hartmann3d/` | Hartmann-3D Family | 3 | oracle_gp | Trained + evaluated |
| `hartmann6d/` | Hartmann-6D Family | 6 | oracle_gp | Trained + evaluated; paper model is `znorm_rb` |
| `hplc/` | HPLC Emulator (6D) | 6 | bnn | Trained + evaluated |
| `alkox/` | Alkox Emulator (4D) | 4 | bnn/direct | Trained + evaluated |
| `ackley5d/` | Ackley-5D Family | 5 | oracle_gp | Retained best fixed-GP smoothweak result |
| `ackley10d/` | Ackley-10D Family | 10 | oracle_gp | Retained best MSRFF result |
| `ablation/` | Ablation & Sensitivity | — | — | Paper reproduction snapshot |
| `figures/` | Cross-task figures | — | — | Comparison tables |

## Evaluation Protocol

All tasks evaluated via `eval_scale_sweep.py`:
- 10 methods: Random, EI, UCB, PI, FunBO, PFNs4BO, TuRBO, TAF_me, TAF_ranking, CAP-PPO
- 7 variant scales: 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
- 20 variants x 3 runs per scale
- 2 surrogates: GP, TabPFN

## Subdirectory Layout (per task)

```
task/
├── scripts/        # Training and evaluation shell scripts
├── data/           # Generated variants, BO trajectories, TAF data
├── models/         # Trained model checkpoints (ppo_best.pt + config.json)
└── results/        # Evaluation results (scale_sweep_data.pkl, plots)
```

## Key Files

- Model: `models/{version}/ppo_best.pt` (or `ppo_final.pt`)
- Config: `models/{version}/config.json` — full training hyperparameters
- Results: `results/{name}/scale_sweep_results.json` — per-scale per-method regret
- Data: `results/{name}/scale_sweep_data.pkl` — full trajectory data for replotting

Intermediate checkpoints, `_resume/`, tensorboard outputs, logs, and failed/superseded raw runs were moved to the cleanup backup outside this repository.
