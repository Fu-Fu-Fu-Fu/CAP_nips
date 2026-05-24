# CAP-PPO Experiment Report

> This document is a comprehensive reference for understanding the experimental design, execution pipeline, evaluation protocol, and ablation studies in this project. It is intended for LLM-assisted paper writing and experiment comprehension.

---

## 1. Project Overview

**CAP-PPO** (Context-Aware Policy for Bayesian Optimization) is a reinforcement-learning-trained policy that selects the next evaluation point in Bayesian Optimization (BO). Unlike classical acquisition functions (EI, UCB, PI) that follow fixed heuristics, CAP-PPO learns a selection strategy via PPO (Proximal Policy Optimization) by training on a family of related optimization tasks.

**Core Idea**: Train a policy on K task variants (from the same function family or emulator), so it learns a general-purpose BO strategy that transfers to unseen variants at test time. The policy uses cross-attention to integrate historical observations and candidate features.

**Framework Convention**: The framework always **minimizes**. Tasks that maximize (e.g., alkox conversion, hplc peak_area) negate their emulator output internally.

---

## 2. Experimental Tasks

Five benchmark tasks spanning synthetic functions and real-world chemistry emulators:

| Task | Dimension | Type | Objective Source | Domain | Difficulty |
|------|-----------|------|-----------------|--------|------------|
| **Branin (2D)** | 2 | Synthetic family | `oracle_gp` | [0,1]^2 | Low |
| **Hartmann-3D** | 3 | Synthetic family | `oracle_gp` | [0,1]^3 | Medium |
| **Hartmann-6D** | 6 | Synthetic family | `oracle_gp` | [0,1]^6 | Medium-High |
| **HPLC (6D)** | 6 | Real chemistry emulator | `bnn` | [0,1]^6 | High |
| **Alkox (4D)** | 4 | Real chemistry emulator | `bnn` / `direct` | [0,1]^4 | Very High |

### 2.1 Synthetic Function Families (Branin, Hartmann-3D, Hartmann-6D)

- Each family generates **K=10 variants** by applying affine transformations (translation `dx`, rotation `rot`, scaling `sx`) to the base function
- Variants are sampled with seed control (`variant_seed=2026` or `69`)
- During training, the objective is generated via **oracle_gp**: fit a GP to the true function values, then sample from the GP posterior using Random Fourier Features (RFF), which creates diverse but realistic training objectives
- Z-normalization (`normalize_oracle_gp`) is applied to standardize GP outputs across variants

### 2.2 Chemistry Emulator Tasks (HPLC, Alkox)

- Use Olympus NeuralNet emulators (require `TF_USE_LEGACY_KERAS=1`)
- Variants are created via affine transforms of the 4D/6D input space
- During training, the objective is generated via **BNN mode**: `f(x) = BNN_mean(x) + alpha * RFF_prior(x)`, where:
  - BNN_mean is the posterior mean of a Bayesian Neural Network trained on historical experiments (accurate surrogate)
  - RFF_prior provides smooth diversity from a GP prior (Matern 2.5 kernel, no data fitting)
  - Params: `bnn_rff_alpha=1.5`, `bnn_rff_length_scale=0.3`

---

## 3. Training Pipeline

Each task follows a 3-step pipeline:

### Step 1: Data Generation
```
python MYRL/scripts/finetune.py --task <TASK> --stage generate \
    --k_variants 10 --variant_seed 2026 --bo_seed 2026 \
    --n_trials_per_variant 5 --total_evals <BUDGET> --n_init 2
```
- Generates K variant parameter sets
- Runs 5 EI-based BO trajectories per variant as demonstration/warmstart data
- Outputs: `variants_k10_seed2026.npz`, `bo_trajs_k10_boSeed2026.npz`

### Step 1.5: TAF Source Data
```python
from myrl.rl.train_rl import prepare_taf_data
prepare_taf_data(trajectories_path, taf_data_path)
```
- Extracts Transfer Acquisition Function (TAF) source GPs from BO trajectories
- Needed for TAF_me / TAF_ranking baselines and as an optional feature for CAP-PPO

### Step 1.5b (BNN tasks only): BNN Surrogate Training
- Trains BNN surrogates for each variant (Alkox, HPLC)
- Outputs: `bnn_surrogates_<task>_k10_kl0.001.npz`

### Step 2: Train CAP-PPO
```
python MYRL/scripts/train_rl.py --task <TASK> \
    --objective_source <oracle_gp|bnn> --normalize_oracle_gp \
    --total_episodes 5000 --reward_mode regret_balanced \
    --variant_sampling shuffled_cycle ...
```

**Key Training Hyperparameters** (shared across tasks unless noted):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `total_episodes` | 5000 | Number of training episodes |
| `reward_mode` | `regret_balanced` | Multi-component reward function |
| `variant_sampling` | `shuffled_cycle` | Cycle through variants, shuffled each epoch |
| `ent_coef_start` → `ent_coef_end` | 0.02 → 0.002 | Entropy annealing for exploration→exploitation |
| `update_every` | 40 | PPO update frequency (episodes) |
| `n_init_context` | 2 | Initial random points per episode |

**Task-Specific Parameters**:

| Parameter | Branin (2D) | Hartmann-3D | Hartmann-6D | HPLC (6D) | Alkox (4D) |
|-----------|-------------|-------------|-------------|-----------|------------|
| `max_steps` | 18 | 18 | 48 | 48 | 28 |
| `n_persistent_base` | 128 | 128 | 128 | 128 | 128 |
| `n_total_candidates` | 192 | 192 | 256 | 256 | 192 |
| `k_centers` | 2 | 2 | 3 | 3 | 2 |
| `local_h` | 1.5 | 0.17 | 0.15 | 0.15 | 0.17 |
| `local_h_decay` | 0.9 | 0.95 | 0.95 | 0.95 | 0.95 |

**Reward Function** (`regret_balanced` mode):
- Terminal weight: 1.5
- Regret AUC weight: 0.2
- Regret delta weight: 1.0
- Early power: 0.5 (gentler penalty early)
- Terminal power: 3.0 (harsh penalty for late-stage failure)
- Scale floor ratio: 0.02

**Candidate Pool Design**:
- `n_persistent_base` Sobol points are pre-generated and persist across steps (consumed when selected)
- Local candidates are generated around the top-k current observations (`k_centers` clusters, bandwidth `local_h`, decaying via `local_h_decay`)
- Total pool size = `n_total_candidates` (persistent + local + new Sobol)

---

## 4. Evaluation Protocol

All tasks are evaluated using the unified `eval_scale_sweep.py` script.

### 4.1 Scale Sweep Design

The key evaluation paradigm is a **scale sweep**: test the trained policy on task variants with increasing transformation magnitude.

- **7 scales**: `[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]`
  - Scale 1.0 ≈ in-distribution (matches training variant distribution)
  - Scale < 1.0 = easier (smaller transformations, closer to base function)
  - Scale > 1.0 = out-of-distribution (larger transformations, harder generalization)
- **20 variants per scale**: each with independently sampled transformation parameters
- **3 runs per variant**: different random initializations
- **Total per method per scale**: 20 variants x 3 runs = 60 BO trajectories

### 4.2 Baseline Methods (10 total)

| Method | Type | Uses Own Surrogate? | Description |
|--------|------|-------------------|-------------|
| **Random** | Non-adaptive | No | Uniform random selection from candidate pool |
| **EI** | Acquisition function | No | Expected Improvement |
| **UCB** | Acquisition function | No | Upper Confidence Bound (kappa=2.0) |
| **PI** | Acquisition function | No | Probability of Improvement (xi=0.01) |
| **FunBO** | Acquisition function | No | EI + exploration bonus (beta=2.0) |
| **PFNs4BO** | Neural BO | Yes (pretrained PFN) | Prior-Fitted Network for BO (`hebo_plus_model`) |
| **TuRBO** | Trust-region BO | Yes (botorch SingleTaskGP) | Thompson Sampling in adaptive trust region |
| **TAF_me** | Transfer AF | No | Transfer Acquisition Function (mean ensemble) |
| **TAF_ranking** | Transfer AF | No | TAF with ranking-based aggregation (rho=1.0) |
| **CAP-PPO** | RL policy | No | Our method |

### 4.3 Surrogates

Each evaluation is run with **two surrogate models**:
- **GP**: Gaussian Process (scikit-learn)
- **TabPFN**: Tabular Prior-Fitted Network (`TabPFNRegressor`, 8 estimators)

This tests whether CAP-PPO's advantage holds regardless of the surrogate model providing predictions.

### 4.4 Evaluation Metric

- **Simple Regret**: `y_best_found - f_global_min` (lower is better, 0 is perfect)
- Reported as final regret at the end of the BO trajectory
- Aggregated as mean over variants and runs per scale
- Comparison tables and line plots generated via `plot_scale_sweep.py`

### 4.5 Output Artifacts

For each evaluation run:
- `scale_sweep_data.pkl` — full trajectory data (regret traces for all methods, variants, runs)
- `scale_sweep_results.json` — summary statistics (mean/std per method per scale)
- `scale_sweep_*.{png,pdf}` — retained final plots

---

## 5. Main Experiments

### 5.1 Hartmann-3D (3D Synthetic)

| Item | Value |
|------|-------|
| Location | `paper_experiments/hartmann3d/` |
| Model | `models/default/ppo_best.pt` |
| Budget | 20 evals (2 init + 18 steps) |
| Results | `results/gp/`, `results/tabpfn/` |
| Figures | `figures/comparison_tables/comparison_table_hartmann3d_{gp,tabpfn}.{png,pdf}` |

### 5.2 Hartmann-6D (6D Synthetic)

| Item | Value |
|------|-------|
| Location | `paper_experiments/hartmann6d/` |
| Model (paper) | `runs/ppo_hartmann_6d_family_znorm_rb_ep5000/ppo_best.pt` (znorm + regret_balanced) |
| Model (old) | `models/default/` (first-generation, no z-norm) |
| Budget | 50 evals (2 init + 48 steps) |
| Results | `results/znorm_rb_gp/`, `results/znorm_rb_tabpfn/` (paper version) |
| Figures | `figures/comparison_tables/comparison_table_hartmann6d_{gp,tabpfn}.{png,pdf}` |

### 5.3 HPLC (6D Chemistry Emulator)

| Item | Value |
|------|-------|
| Location | `paper_experiments/hplc/` |
| Model | `models/default/ppo_best.pt` |
| Objective Source | BNN (BNN_mean + RFF perturbation) |
| Budget | 50 evals (2 init + 48 steps) |
| Results | `results/gp/`, `results/tabpfn/` |
| Figures | `figures/comparison_tables/comparison_table_hplc_{gp,tabpfn}.{png,pdf}` |

### 5.4 Alkox (4D Chemistry Emulator)

| Item | Value |
|------|-------|
| Location | `paper_experiments/alkox/` |
| Model | `models/default/ppo_best.pt` |
| Objective Source | BNN |
| Budget | 30 evals (2 init + 28 steps) |
| Results | `results/gp/`, `results/tabpfn/` |

### 5.5 Branin (2D Synthetic)

Branin retains the final calibrated multi-scale RFF variants:

| Version | Variant Seed | Key Change | Location |
|---------|-------------|------------|----------|
| **calibrated_multiscale_rff_pool** | 2026 | Calibrated multi-scale RFF | `models/calibrated_multiscale_rff_pool/` |
| **calibrated_multiscale_rff_medium_pool** | 2026 | Calibrated multi-scale RFF, medium pool | `models/calibrated_multiscale_rff_medium_pool/` |

Results for each are in `results/<version>_{gp,tabpfn}/`.

---

## 6. Ablation Studies

All ablation experiments are in `paper_experiments/ablation/` and focus on **Hartmann-6D** and **Alkox** as representative tasks. Each ablation experiment follows the same evaluation protocol (7 scales, 20 variants, 3 runs, GP + TabPFN surrogates).

### 6.1 Component Ablation (`ablation/component/`)

Tests the contribution of individual architectural/training components by removing one at a time.

| Ablation | What's Removed | Implementation | Tasks |
|----------|---------------|----------------|-------|
| **Full (default_v2)** | Nothing (baseline) | Standard CAP-PPO | Hartmann-6D, Alkox |
| **w/o Synthetic Objective** | Synthetic objective generation (GP->RFF / BNN+RFF) | Hartmann-6D: `objective_source=direct`; Alkox: `bnn_rff_alpha=0` (BNN mean only, no RFF diversity) | Hartmann-6D, Alkox |
| **w/o Cross-Attention** | Cross-attention mechanism | `CAP_NO_CROSS_ATTN=1` env var → mean-pooling replaces cross-attention | Hartmann-6D, Alkox |

**Purpose**: Demonstrate that both the synthetic objective diversity and the cross-attention architecture are essential components.

Location: `ablation/component/{hartmann6d,alkox}/{default_v2,wo_crossattn,wo_synthetic}/`

Figures: `ablation/figures/ablation_component_comparison.{png,pdf}`

### 6.2 w/o RL Ablation (`ablation/wo_rl/`)

Tests whether the RL-learned selection strategy matters or whether the candidate pool design alone is sufficient.

**Design**: All methods (EI, UCB, PI, Random, TAF, CAP-PPO) see the **exact same** candidate pool at each BO step. The pool is generated using CAP-PPO's persistent + local strategy. This isolates the contribution of the *learned selection strategy* from the *candidate pool design*.

- TuRBO and PFNs4BO are excluded (they generate their own candidates internally)
- Implemented in `scripts/eval_shared_pool.py`

Location: `ablation/wo_rl/{hartmann6d,alkox}/`

**Purpose**: If CAP-PPO still outperforms EI/UCB on the same candidate set, the RL training itself is valuable (not just the pool construction).

### 6.3 Trajectory Quality Ablation (`ablation/trajectory/`)

Tests whether the quality of demonstration trajectories matters for training.

| Condition | Implementation | Description |
|-----------|---------------|-------------|
| **Default** | EI-based BO trajectories | Standard: 5 EI trajectories per variant |
| **Random Trajectories** | Random-search trajectories | Replace EI trajectories with pure random search |

Location: `ablation/trajectory/{hartmann6d,alkox}/random_traj/`

**Purpose**: Show that high-quality demonstration data (from EI-based BO) improves training vs. random exploration.

### 6.4 Sensitivity Analysis (`ablation/sensitivity/`)

Systematically varies 4 hyperparameter dimensions, each with 2 extreme settings plus the default baseline. All 18 experiments (4 dimensions x 2 settings x 2 tasks + 2 defaults) use the new reward/normalization configuration.

#### Dimension 1: Candidate Pool Size

| Setting | `n_persistent_base` | `n_total_candidates` |
|---------|--------------------|--------------------|
| Small | 64 | 128 (H6D) / 96 (Alkox) |
| **Default** | **128** | **256 (H6D) / 192 (Alkox)** |
| Large | 256 | 512 (H6D) / 384 (Alkox) |

**Question**: How large should the candidate pool be?

#### Dimension 2: Pool Construction (Local vs. Global)

| Setting | `k_centers` | Behavior |
|---------|------------|----------|
| No local (k=0) | 0 | Pure Sobol (global only, no local exploitation candidates) |
| **Default** | **2-3** | **Mixed: persistent Sobol + local candidates around best points** |
| Heavy local (k=6) | 6 | Many local clusters (more exploitation, less exploration) |

**Question**: Is the local candidate generation important, or is global coverage sufficient?

#### Dimension 3: K Variants (Training Diversity)

| Setting | K |
|---------|---|
| Small | 5 |
| **Default** | **10** |
| Large | 20 |

**Question**: How many training task variants are needed for good generalization?

#### Dimension 4: Optimization Budget

| Setting | Total Evals (Hartmann-6D) | Total Evals (Alkox) |
|---------|--------------------------|---------------------|
| Short | 20 (18 steps) | 15 (13 steps) |
| **Default** | **50 (48 steps)** | **30 (28 steps)** |
| Long | 80 (78 steps) | 50 (48 steps) |

**Question**: Does CAP-PPO maintain its advantage across different optimization horizons?

Location: `ablation/sensitivity/{hartmann6d,alkox}/{default,pool_size_small,pool_size_large,pool_construct_no_local,pool_construct_heavy_local,k_variants_5,k_variants_20,budget_short,budget_long}/`

Figures:
- `ablation/figures/ablation_lineplots_v{1,2,3}.{png,pdf}` — Line plots showing regret vs. hyperparameter setting
- `ablation/figures/ablation_normalized_regret_v{1,2,3}.{png,pdf}` — Normalized regret heatmaps

---

## 7. Experimental Infrastructure

### 7.1 Pipeline Scripts

Each experiment directory contains:

| Script | Purpose |
|--------|---------|
| `train_<task>.sh` | Full pipeline: data gen → TAF → train |
| `eval_<task>.sh` | Scale sweep evaluation (configurable surrogate) |

Ablation scripts:
| Script | Purpose |
|--------|---------|
| `scripts/train_ablation.sh` | Unified ablation training (env-var driven) |
| `scripts/eval_ablation.sh` | Unified ablation evaluation |
| `scripts/run_one_ablation.sh` | Single ablation: train + eval_gp + eval_tabpfn + plot |
| `scripts/run_all_ablations.sh` | Master orchestrator for all 18 sensitivity experiments |
| `scripts/eval_shared_pool.py` | w/o RL evaluation (shared candidate pool) |

### 7.2 Resume Support

All evaluation scripts support interruption and resumption:
- Per-variant results are checkpointed in `_resume/` subdirectories
- Format: `scale_<X>p<Y>_v<NNN>.pkl` (one file per variant per scale)
- On restart, completed variants are loaded from checkpoint; only missing ones are re-evaluated

### 7.3 Parallelism

- Training: `--n_workers` (default 8) for parallel episode collection
- Evaluation: `--n_workers` for parallel variant evaluation via `multiprocessing.Pool`
- Thread control: `OMP_NUM_THREADS`, `MKL_NUM_THREADS` automatically set to `n_cpus / n_workers`

### 7.4 Environment Requirements

```bash
export TF_USE_LEGACY_KERAS=1          # Required for Olympus emulators
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONPATH="${REPO_ROOT}/MYRL:${REPO_ROOT}/olympus/src"
export TABPFN_MODEL_CACHE_DIR="${REPO_ROOT}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
```

Python defaults to `python`; set `PYTHON_BIN=/path/to/python` to override shell scripts.

---

## 8. Key Result Files and How to Read Them

### 8.1 Training Outputs

| File | Content |
|------|---------|
| `config.json` | Full training hyperparameters (reproduces the run) |
| `ppo_best.pt` | Best model checkpoint (by training reward) |
| `ppo_final.pt` | Final model checkpoint (last episode) |
| `ppo_ep<N>.pt` | Intermediate checkpoints (every 500 episodes) |
| `metrics.json` | Training metrics (episode rewards, losses) |
| `tensorboard/` | TensorBoard event files for training curves |

### 8.2 Evaluation Outputs

| File | Content |
|------|---------|
| `scale_sweep_data.pkl` | **Master data file**: full regret traces for all methods x scales x variants x runs |
| `scale_sweep_results.json` | Summary: mean/std regret per method per scale |
| `replot/` | Auto-generated plots |
| `_resume/` | Per-variant checkpoint files |

### 8.3 Loading Results for Analysis

```python
import pickle
with open("scale_sweep_data.pkl", "rb") as f:
    data = pickle.load(f)

# data["method_names"]     → list of method names
# data["scales"]           → [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
# data["trajectories"]["1.0"]["CAP-PPO"]  → ndarray (20, 3, n_steps+1)
# data["results"]["1.0"]["CAP-PPO"]["mean"] → float (final regret mean)
```

---

## 9. Paper Writing Guide

### 9.1 Main Results (Section: Experiments)

For each of the 5 tasks, report:
1. **Regret vs. Scale curve**: CAP-PPO vs. all 9 baselines, both GP and TabPFN surrogates
2. **Final regret table**: at scale=1.0 (in-distribution) and scale=1.5 or 2.0 (OOD)
3. **Key finding**: CAP-PPO maintains competitive or superior performance across scales, especially on harder tasks (Alkox, Hartmann-6D)

Data sources:
- `hartmann3d/results/gp/scale_sweep_results.json` (and tabpfn)
- `hartmann6d/results/znorm_rb_gp/scale_sweep_results.json` (and tabpfn)
- `hplc/results/gp/scale_sweep_results.json` (and tabpfn)
- `alkox/results/gp/scale_sweep_results.json` (and tabpfn)
- `branin/results/v2_gp/scale_sweep_results.json` (and tabpfn, using v2_balanced model)

Figures: `figures/comparison_tables/comparison_table_<task>_{gp,tabpfn}.{png,pdf}`

### 9.2 Ablation Study (Section: Analysis)

#### Component Ablation (Table or Bar Chart)
- Compare Full vs. w/o Synthetic vs. w/o Cross-Attention on Hartmann-6D and Alkox
- Show that removing either component degrades performance
- Figure: `ablation/figures/ablation_component_comparison.{png,pdf}`

#### w/o RL (Shared Pool)
- Same candidate pool, different selection strategies → CAP-PPO still wins
- Proves the RL-learned strategy adds value beyond pool design
- Data: `ablation/wo_rl/{hartmann6d,alkox}/`

#### Trajectory Quality
- EI trajectories vs. random trajectories for training
- Data: `ablation/trajectory/{hartmann6d,alkox}/random_traj/`

### 9.3 Sensitivity Analysis (Section: Analysis or Appendix)

For each of the 4 hyperparameter dimensions:
- Line plot or grouped bar chart: default vs. small vs. large
- On both Hartmann-6D and Alkox
- Key message: CAP-PPO is robust to reasonable hyperparameter choices

Figures: `ablation/figures/ablation_lineplots_v3.{png,pdf}`, `ablation/figures/ablation_normalized_regret_v3.{png,pdf}`

### 9.4 Branin Analysis (Optional/Appendix)

The Branin task provides insights into:
- Training data distribution effects (symmetric vs. balanced variants)
- GP-policy feedback loop failure modes (`analysis_catastrophic_failures.md`)
- Objective source variants (oracle_gp, sqrt, gp_rff, multiscale_rff)

### 9.5 Figures Checklist

| Figure | Source | Section |
|--------|--------|---------|
| Regret vs. Scale (per task, GP) | `<task>/results/<version>_gp/` | Main experiments |
| Regret vs. Scale (per task, TabPFN) | `<task>/results/<version>_tabpfn/` | Main experiments |
| Comparison tables | `figures/comparison_tables/` | Main experiments |
| Component ablation | `ablation/figures/ablation_component_comparison.*` | Ablation |
| Sensitivity line plots | `ablation/figures/ablation_lineplots_v3.*` | Sensitivity |
| Normalized regret heatmap | `ablation/figures/ablation_normalized_regret_v3.*` | Sensitivity |

---

## 10. Directory Map

```
paper_experiments/
|-- README.md                          # Brief overview
|-- EXPERIMENT_REPORT.md               # This file
|
|-- branin/                            # Branin 2D experiments
|   |-- data/                          # Variant NPZ + trajectory NPZ + TAF PKL
|   |-- models/{original,v2_balanced,high_ent,k3,sqrt,gp_rff,...}/
|   |-- results/{original_gp,v2_gp,high_ent_gp,...}/
|   |-- runs/ppo_branin_family_*/      # Raw training outputs
|   |-- scripts/                       # train_branin*.sh, eval_branin*.sh
|   +-- analysis_catastrophic_failures.md
|
|-- hartmann3d/                        # Hartmann 3D experiments
|   |-- data/, models/, results/, runs/, scripts/
|
|-- hartmann6d/                        # Hartmann 6D experiments (main paper task)
|   |-- models/default/                # Old model
|   |-- runs/ppo_..._znorm_rb_ep5000/  # New model (paper version)
|   |-- results/{old_gp,znorm_rb_gp,znorm_rb_tabpfn,...}/
|   +-- scripts/
|
|-- hplc/                              # HPLC 6D chemistry emulator
|   |-- models/default/, results/{gp,tabpfn}/, scripts/
|
|-- alkox/                             # Alkox 4D chemistry emulator
|   |-- models/default/, results/{gp,tabpfn}/, scripts/
|
|-- ablation/                          # All ablation experiments
|   |-- component/                     # Component ablation (w/o synthetic, w/o cross-attn)
|   |   |-- hartmann6d/{default_v2, wo_crossattn, wo_synthetic}/
|   |   +-- alkox/{default_v2, wo_crossattn, wo_synthetic}/
|   |-- wo_rl/                         # w/o RL (shared candidate pool)
|   |   |-- hartmann6d/, alkox/
|   |-- trajectory/                    # Trajectory quality ablation
|   |   |-- hartmann6d/random_traj/, alkox/random_traj/
|   |-- sensitivity/                   # Hyperparameter sensitivity (18 experiments)
|   |   |-- hartmann6d/{default,pool_size_small,...,budget_long}/
|   |   +-- alkox/{default,pool_size_small,...,budget_long}/
|   |-- figures/                       # All ablation plots
|   |-- scripts/                       # Unified train/eval/orchestration scripts
|   |-- README.md                      # Ablation overview
|   +-- README_sensitivity.md          # Detailed sensitivity commands
|
|-- figures/                           # Cross-task comparison tables
|   +-- comparison_tables/
|
+-- archive/                           # Archived experiments and debug scripts
    |-- branin_debug_scripts/
    |-- branin_ablation_runs/
    |-- experiment_plan_20260413.md
    +-- technical_report_20260416.md
```
