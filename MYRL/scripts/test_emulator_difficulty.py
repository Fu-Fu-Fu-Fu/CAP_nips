"""
Test emulator difficulty for traditional BO (EI).

For each 4D dataset:
1. Load the deterministic NeuralNet emulator
2. Run GP+EI BO (20 evals) multiple times
3. Report final regret
4. Plot 2D cross-section contour maps

Usage:
    python MYRL/scripts/test_emulator_difficulty.py
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from itertools import combinations

# ---- olympus ----
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../olympus/src"))
from olympus.emulators import Emulator
from olympus.datasets import Dataset

# ---- GPy + EI ----
import GPy
from scipy.stats import norm
from scipy.stats.qmc import Sobol


# =============================================================================
# Dataset optimization directions (from olympus config)
# =============================================================================
# These datasets have maximization objectives; we negate y so the unified
# minimization code finds the optimum correctly.
MAXIMIZE_DATASETS = {"fullerenes", "alkox", "suzuki", "hplc"}


# =============================================================================
# Helpers
# =============================================================================

def load_emulator(name):
    """Load a deterministic NeuralNet emulator and return an evaluation function.

    Follows the compatibility patching pattern from benzylation_emulator.py.
    """
    emu = Emulator(dataset=name, model="NeuralNet")

    # Replace pickled Dataset with fresh one (old pickles lack modern attrs)
    emu.dataset = Dataset(kind=name)

    # Patch DataTransformers from old pickles (missing _stable_stddev etc.)
    for tf in (emu.feature_transformer, emu.target_transformer):
        if hasattr(tf, "_stddev") and not hasattr(tf, "_stable_stddev"):
            tf._stable_stddev = np.where(tf._stddev == 0.0, 1.0, tf._stddev)
        if hasattr(tf, "_min") and not hasattr(tf, "_stable_min"):
            tf._stable_min = tf._min
        if hasattr(tf, "_max") and not hasattr(tf, "_stable_max"):
            tf._stable_max = tf._max

    config_path = os.path.join(
        os.path.dirname(__file__),
        f"../../olympus/src/olympus/datasets/dataset_{name}/config.json",
    )
    with open(config_path) as f:
        config = json.load(f)
    params = config["parameters"]
    dim = len(params)
    lower = np.array([p["low"] for p in params], dtype=np.float64)
    upper = np.array([p["high"] for p in params], dtype=np.float64)
    param_names = [p["name"] for p in params]

    values = config.get("measurements", config.get("objectives", []))
    obj_name = values[0]["name"] if values else "?"

    def evaluate(X_norm):
        """Evaluate on [0,1]^d normalized input via emulator.run(). Returns shape (N,)."""
        X_norm = np.atleast_2d(X_norm)
        X_real = X_norm * (upper - lower) + lower
        # Use the batch interface: emulator.run(X_2d, num_samples=1)
        y_preds, _, _ = emu.run(X_real, num_samples=1)
        return np.asarray(y_preds, dtype=np.float64).reshape(-1)

    # For maximization datasets, negate so we can uniformly minimise
    maximize = name in MAXIMIZE_DATASETS

    def evaluate_for_bo(X_norm):
        y = evaluate(X_norm)
        return -y if maximize else y

    return evaluate, evaluate_for_bo, dim, lower, upper, param_names, obj_name, maximize


def run_ei_bo(evaluate, dim, n_init=2, n_steps=18, n_candidates=2048, seed=42):
    """Run GP+EI BO on a [0,1]^d function. Returns best_y trajectory."""
    rng = np.random.RandomState(seed)

    # Initial points
    X = rng.rand(n_init, dim)
    y = evaluate(X).reshape(-1, 1)

    best_ys = [float(y.min())]

    for step in range(n_steps):
        # Fit GP (GPy Matern52)
        kernel = GPy.kern.Matern52(input_dim=dim, ARD=True)
        gp = GPy.models.GPRegression(X, y, kernel)
        gp.Gaussian_noise.variance = 1e-6
        gp.Gaussian_noise.fix()
        try:
            gp.optimize_restarts(num_restarts=5, verbose=False, robust=True)
        except Exception:
            gp.optimize(messages=False)

        # Generate candidates (Sobol)
        sobol = Sobol(d=dim, scramble=True, seed=seed + step)
        X_cand = sobol.random(n_candidates)

        # EI (minimization)
        mu_pred, var_pred = gp.predict(X_cand)
        mu_pred = mu_pred.ravel()
        sigma_pred = np.sqrt(np.maximum(var_pred.ravel(), 1e-18))
        best_y = float(y.min())
        z = (best_y - mu_pred) / sigma_pred
        ei = (best_y - mu_pred) * norm.cdf(z) + sigma_pred * norm.pdf(z)

        # Select best EI
        idx = np.argmax(ei)
        x_new = X_cand[idx:idx+1]
        y_new = evaluate(x_new).reshape(-1, 1)

        X = np.vstack([X, x_new])
        y = np.vstack([y, y_new])
        best_ys.append(min(best_ys[-1], float(y_new[0, 0])))

    return np.array(best_ys), X, y.ravel()


def estimate_global_min(evaluate, dim, n_sobol=10000, seed=0):
    """Estimate global min using dense Sobol grid."""
    sobol = Sobol(d=dim, scramble=True, seed=seed)
    X = sobol.random(n_sobol)
    y = evaluate(X)
    return float(y.min()), X[y.argmin()], float(y.max()), float(y.mean()), float(y.std())


def plot_contour_maps(evaluate, dim, param_names, obj_name, dataset_name,
                      global_min_val, save_dir, resolution=80):
    """Plot 2D cross-section contour maps for all dim pairs."""
    n_pairs = dim * (dim - 1) // 2
    pairs = list(combinations(range(dim), 2))

    # Determine subplot grid
    n_cols = min(3, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    if n_pairs == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    # Center point for fixed dims
    center = np.full(dim, 0.5)

    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx // n_cols, idx % n_cols]

        # Create grid
        x_grid = np.linspace(0, 1, resolution)
        y_grid = np.linspace(0, 1, resolution)
        XX, YY = np.meshgrid(x_grid, y_grid)

        # Build evaluation points: fix other dims at center
        X_eval = np.tile(center, (resolution * resolution, 1))
        X_eval[:, i] = XX.ravel()
        X_eval[:, j] = YY.ravel()

        Z = evaluate(X_eval).reshape(resolution, resolution)

        # Contour plot
        levels = 20
        cs = ax.contourf(XX, YY, Z, levels=levels, cmap="RdYlBu_r")
        ax.contour(XX, YY, Z, levels=levels, colors="k", linewidths=0.3, alpha=0.5)
        plt.colorbar(cs, ax=ax, shrink=0.8)

        ax.set_xlabel(param_names[i])
        ax.set_ylabel(param_names[j])
        ax.set_title(f"{param_names[i]} vs {param_names[j]}")
        ax.set_aspect("equal")

    # Hide empty subplots
    for idx in range(n_pairs, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.suptitle(
        f"{dataset_name} — {obj_name} (global opt ≈ {global_min_val:.4f})\n"
        f"2D cross-sections at center (other dims = 0.5)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"contour_{dataset_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved contour plot: {path}")
    return path


# =============================================================================
# Main
# =============================================================================

def main():
    save_dir = "./results_policies/emulator_difficulty_test"
    os.makedirs(save_dir, exist_ok=True)

    # 4D datasets to test
    datasets_4d = ["alkox", "benzylation", "photo_pce10", "photo_wf3", "snar", "suzuki"]

    # Also do the 3D and 5D/6D for completeness
    all_datasets = ["colors_n9", "fullerenes",
                    "alkox", "benzylation", "photo_pce10", "photo_wf3", "snar", "suzuki",
                    "colors_bob", "hplc"]

    n_bo_runs = 5
    n_init = 2
    n_steps_map = {}  # dim -> steps
    # Budget: 20 evals for 3-4D, 30 for 5D, 50 for 6D
    for name in all_datasets:
        config_path = os.path.join(
            os.path.dirname(__file__),
            f"../../olympus/src/olympus/datasets/dataset_{name}/config.json",
        )
        with open(config_path) as f:
            config = json.load(f)
        dim = len(config["parameters"])
        if dim <= 4:
            n_steps_map[name] = 18  # 20 total evals
        elif dim == 5:
            n_steps_map[name] = 28  # 30 total evals
        else:
            n_steps_map[name] = 48  # 50 total evals

    results = {}

    print("=" * 70)
    print("Olympus Emulator Difficulty Test")
    print("=" * 70)

    for name in all_datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {name}")
        print(f"{'='*60}")

        try:
            evaluate_raw, evaluate_for_bo, dim, lower, upper, param_names, obj_name, maximize = load_emulator(name)
        except Exception as e:
            print(f"  FAILED to load emulator: {e}")
            results[name] = {"error": str(e)}
            continue

        direction = "MAX" if maximize else "MIN"
        # Estimate global optimum (using negated objective for BO)
        print(f"  dim={dim}, params={param_names}, obj={obj_name}, direction={direction}")
        print(f"  Estimating global optimum (10K Sobol)...")
        g_min, g_min_x, g_max, g_mean, g_std = estimate_global_min(evaluate_for_bo, dim, n_sobol=10000)
        print(f"  Negated-obj: min ≈ {g_min:.6f}, max ≈ {g_max:.6f}")
        if maximize:
            print(f"  Original-obj: best (max) ≈ {-g_min:.6f}, worst (min) ≈ {-g_max:.6f}")
        print(f"  y_range = {g_max - g_min:.4f}")

        # Run BO (on negated objective for max datasets)
        n_steps = n_steps_map[name]
        print(f"  Running GP+EI BO ({n_bo_runs} runs, {n_init}+{n_steps}={n_init+n_steps} evals)...")
        final_regrets = []
        for run in range(n_bo_runs):
            best_ys, _, _ = run_ei_bo(evaluate_for_bo, dim, n_init=n_init, n_steps=n_steps,
                                       seed=2026 + run)
            regret = best_ys[-1] - g_min
            final_regrets.append(regret)
            if maximize:
                print(f"    Run {run+1}: best_original={-best_ys[-1]:.6f}, regret={regret:.6f}")
            else:
                print(f"    Run {run+1}: final_best={best_ys[-1]:.6f}, regret={regret:.6f}")

        mean_regret = np.mean(final_regrets)
        std_regret = np.std(final_regrets)
        norm_regret = mean_regret / max(g_max - g_min, 1e-8)

        print(f"  EI regret: {mean_regret:.6f} ± {std_regret:.6f}")
        print(f"  Normalized regret (regret/y_range): {norm_regret:.6f}")

        if norm_regret < 0.01:
            difficulty = "EASY"
        elif norm_regret < 0.05:
            difficulty = "MODERATE"
        elif norm_regret < 0.15:
            difficulty = "HARD"
        else:
            difficulty = "VERY HARD"
        print(f"  Difficulty: {difficulty}")

        # Store results with original-scale values for clarity
        if maximize:
            orig_best = -g_min   # best in original (max) scale
            orig_worst = -g_max
        else:
            orig_best = g_min
            orig_worst = g_max

        results[name] = {
            "dim": dim,
            "obj_name": obj_name,
            "direction": direction,
            "param_names": param_names,
            "global_optimum": orig_best,
            "global_worst": orig_worst,
            "y_range": g_max - g_min,
            "y_mean": g_mean,
            "y_std": g_std,
            "ei_regret_mean": mean_regret,
            "ei_regret_std": std_regret,
            "ei_norm_regret": norm_regret,
            "difficulty": difficulty,
            "n_evals": n_init + n_steps,
        }

        # Plot contour maps (use original un-negated evaluate for interpretability)
        print(f"  Plotting 2D contour maps...")
        opt_label = f"global {'max' if maximize else 'min'} ≈ {orig_best:.4f}"
        try:
            plot_contour_maps(evaluate_raw, dim, param_names, obj_name, name,
                              orig_best, save_dir, resolution=80)
        except Exception as e:
            print(f"  FAILED to plot: {e}")

    # Summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Name':<16} {'Dir':>3} {'Dim':>3} {'Evals':>5} {'EI Regret':>12} {'Norm Regret':>12} {'y_range':>10} {'Difficulty':<12}")
    print("-" * 85)
    for name in all_datasets:
        r = results.get(name, {})
        if "error" in r:
            print(f"{name:<16} ERROR: {r['error'][:50]}")
            continue
        print(f"{name:<16} {r['direction']:>3} {r['dim']:>3} {r['n_evals']:>5} "
              f"{r['ei_regret_mean']:>8.4f}±{r['ei_regret_std']:.4f} "
              f"{r['ei_norm_regret']:>12.6f} "
              f"{r['y_range']:>10.4f} "
              f"{r['difficulty']:<12}")

    # Save results
    results_path = os.path.join(save_dir, "difficulty_results.json")
    # Convert numpy to python types
    save_results = {}
    for k, v in results.items():
        save_results[k] = {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv) for kk, vv in v.items()}
    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
