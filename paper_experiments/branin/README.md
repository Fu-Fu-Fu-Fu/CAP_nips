# Branin Family (2D) Experiments

This directory keeps the final Branin variants used after the failed early trials were removed.

## Retained Models

| Version | Directory | Notes |
|---------|-----------|-------|
| `calibrated_multiscale_rff_pool` | `models/calibrated_multiscale_rff_pool/` | Final calibrated multi-scale RFF model |
| `calibrated_multiscale_rff_medium_pool` | `models/calibrated_multiscale_rff_medium_pool/` | Medium-pool final variant |

Both retained model directories contain `ppo_best.pt` and `config.json`.

## Retained Results

| Version | GP Results | TabPFN Results |
|---------|------------|----------------|
| `calibrated_multiscale_rff_pool` | `results/calibrated_multiscale_rff_pool_gp/` | `results/calibrated_multiscale_rff_pool_tabpfn_base/` |
| `calibrated_multiscale_rff_medium_pool` | `results/calibrated_multiscale_rff_medium_pool_gp/` | `results/calibrated_multiscale_rff_medium_pool_tabpfn_base/` |

Each result directory keeps `scale_sweep_results.json`, `scale_sweep_data.pkl`, and final plot PDFs/PNGs.

## Usage

```bash
bash paper_experiments/branin/scripts/train_branin_calibrated_multiscale_rff_pool.sh
SURROGATE=gp bash paper_experiments/branin/scripts/eval_branin_calibrated_multiscale_rff_pool.sh
```
