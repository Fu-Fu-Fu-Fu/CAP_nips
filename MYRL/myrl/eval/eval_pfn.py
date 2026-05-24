"""
Evaluate PFN (base vs finetuned) on a task's variant suite (ID + OOD levels if available).

For dim=2 tasks:
- evaluate on a 2D grid (and optionally plot per-variant heatmaps)

For dim!=2 tasks:
- evaluate on a Sobol point set (no per-variant contour plots)

For each variant, we compare:
- Base TabPFN
- Finetuned TabPFN
- GP regression baseline
"""

import argparse
import json
import os
import pickle
import warnings
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.exceptions import ConvergenceWarning
from scipy.stats.qmc import Sobol

from tabpfn import TabPFNRegressor

from ..tasks import get_task

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")
warnings.filterwarnings("ignore", message=".*ABNORMAL_TERMINATION_IN_LNSRCH.*")
warnings.filterwarnings("ignore", message=".*scale the data.*")
warnings.filterwarnings("ignore", message=".*optimal value found.*close to the specified.*bound.*")

FAMILY_TASKS = {"branin_family", "goldstein_price_family", "hartmann_3d_family"}

CENTER_X1 = 2.5
CENTER_X2 = 7.5


def _variant_key(variant_params: Dict[str, float], *, ndigits: int = 12) -> Tuple[Tuple[str, object], ...]:
    items: List[Tuple[str, object]] = []
    for k in sorted(variant_params.keys()):
        v = variant_params[k]
        if isinstance(v, (float, np.floating)):
            v = round(float(v), int(ndigits))
        items.append((str(k), v))
    return tuple(items)


def _load_variants_from_npz(path: str) -> List[Dict[str, float]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Train variants cache not found: {path}")
    data = np.load(path, allow_pickle=True)
    if "variants" not in data:
        raise KeyError(f"NPZ missing key 'variants': {path}")
    return data["variants"].tolist()


def _sample_variants_excluding(
    rng: np.random.Generator,
    *,
    task,
    group_name: str,
    spec: Dict[str, Any],
    n: int,
    forbidden: set,
    used_eval: set,
    max_tries_per_sample: int = 10_000,
) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for _ in range(int(n)):
        for _try in range(int(max_tries_per_sample)):
            seed = int(rng.integers(0, 1_000_000_000))
            suite = task.sample_eval_suite(n_per_group=1, seed=seed, suite_specs={str(group_name): spec})
            cand = suite.get(str(group_name), None)
            if not cand:
                continue
            v = cand[0]
            key = _variant_key(v)
            if key in forbidden or key in used_eval:
                continue
            used_eval.add(key)
            out.append(v)
            break
        else:
            raise RuntimeError(
                "Failed to sample a non-training evaluation variant (too many collisions). "
                "Check spec ranges and forbidden set size."
            )
    return out


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_boot: int = 2000,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    mean = float(np.mean(values)) if values.size else float("nan")
    if values.size <= 1 or int(n_boot) <= 0:
        return mean, mean, mean
    alpha = (1.0 - float(ci)) / 2.0
    idx = rng.integers(0, values.size, size=(int(n_boot), values.size))
    boot_means = values[idx].mean(axis=1)
    lo = float(np.quantile(boot_means, alpha))
    hi = float(np.quantile(boot_means, 1.0 - alpha))
    return mean, lo, hi


def _resolve_plot_modes(mode: str) -> List[str]:
    mode = str(mode)
    if mode == "both":
        return ["mean_bootstrap_95", "median_30_70"]
    return [mode]


def _split_csv(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    items = [x.strip() for x in str(s).split(",") if x.strip()]
    return items or None


def branin_variant_numpy(X: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    X = np.atleast_2d(X).astype(np.float64)
    x1 = X[:, 0]
    x2 = X[:, 1]

    dx1 = float(params.get("dx1", 0.0))
    dx2 = float(params.get("dx2", 0.0))
    sx1 = float(params.get("sx1", 1.0))
    sx2 = float(params.get("sx2", 1.0))
    rotation = float(params.get("rotation", 0.0))
    alpha = float(params.get("alpha", 1.0))
    beta = float(params.get("beta", 0.0))

    if abs(rotation) > 1e-12:
        theta = rotation * np.pi / 180.0
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        x1c = x1 - CENTER_X1
        x2c = x2 - CENTER_X2
        x1r = cos_t * x1c - sin_t * x2c + CENTER_X1
        x2r = sin_t * x1c + cos_t * x2c + CENTER_X2
    else:
        x1r = x1
        x2r = x2

    x1t = sx1 * x1r + dx1
    x2t = sx2 * x2r + dx2

    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    y = a * (x2t - b * x1t**2 + c * x1t - r) ** 2 + s * (1 - t) * np.cos(x1t) + s
    y = alpha * y + beta
    return y.astype(np.float32)


def build_grid(
    x1_range: Tuple[float, float],
    x2_range: Tuple[float, float],
    grid_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x1_lin = np.linspace(x1_range[0], x1_range[1], grid_size, dtype=np.float32)
    x2_lin = np.linspace(x2_range[0], x2_range[1], grid_size, dtype=np.float32)
    X1, X2 = np.meshgrid(x1_lin, x2_lin)
    X_grid = np.stack([X1.ravel(), X2.ravel()], axis=1).astype(np.float32)
    return X1, X2, X_grid


def sample_sobol_points(
    bounds: Tuple[np.ndarray, np.ndarray],
    *,
    n: int,
    seed: int,
) -> np.ndarray:
    lower, upper = bounds
    lower = np.asarray(lower, dtype=np.float64).reshape(-1)
    upper = np.asarray(upper, dtype=np.float64).reshape(-1)
    dim = int(lower.shape[0])
    sampler = Sobol(d=dim, scramble=True, seed=int(seed))
    X_unit = sampler.random(int(n))
    X = X_unit * (upper - lower) + lower
    return X.astype(np.float32)


# ==================== Variant suite (match eval_rl.py) ====================
_VARIANT_SUITE_SPECS = {
    "in_range": {
        "dx": [(-2.0, 2.0)],
        "rotation": [(-30.0, 30.0)],
        "sx": [(0.75, 1.25)],
    },
    "ood_level_1": {
        "dx": [(-3.0, -2.0), (2.0, 3.0)],
        "rotation": [(-45.0, -30.0), (30.0, 45.0)],
        "sx": [(0.60, 0.75), (1.25, 1.40)],
    },
    "ood_level_2": {
        "dx": [(-4.0, -3.0), (3.0, 4.0)],
        "rotation": [(-60.0, -45.0), (45.0, 60.0)],
        "sx": [(0.45, 0.60), (1.40, 1.55)],
    },
    "ood_level_3": {
        "dx": [(-5.0, -4.0), (4.0, 5.0)],
        "rotation": [(-75.0, -60.0), (60.0, 75.0)],
        "sx": [(0.30, 0.45), (1.55, 1.70)],
    },
}


def _validate_segments(name: str, segments: List[Tuple[float, float]]) -> None:
    for (lo, hi) in segments:
        if not (float(lo) < float(hi)):
            raise ValueError(f"Invalid segment for {name}: ({lo}, {hi})")


def _sample_from_segments(rng: np.random.Generator, segments: List[Tuple[float, float]]) -> float:
    _validate_segments("segments", segments)
    lengths = np.array([float(hi) - float(lo) for (lo, hi) in segments], dtype=np.float64)
    probs = lengths / lengths.sum()
    idx = int(rng.choice(len(segments), p=probs))
    lo, hi = segments[idx]
    return float(rng.uniform(float(lo), float(hi)))


def sample_variant_params_from_spec(rng: np.random.Generator, spec: Dict[str, List[Tuple[float, float]]]) -> Dict[str, float]:
    dx1 = _sample_from_segments(rng, spec["dx"])
    dx2 = _sample_from_segments(rng, spec["dx"])
    rotation = _sample_from_segments(rng, spec["rotation"])
    sx1 = _sample_from_segments(rng, spec["sx"])
    sx2 = _sample_from_segments(rng, spec["sx"])
    return {
        "dx1": float(dx1),
        "dx2": float(dx2),
        "rotation": float(rotation),
        "sx1": float(sx1),
        "sx2": float(sx2),
        "alpha": 1.0,
        "beta": 0.0,
    }


def load_or_generate_variant_suite(
    suite_cache: str,
    *,
    task_name: str,
    n_per_group: int,
    seed: int,
    suite_specs: Optional[Dict[str, Any]] = None,
    forbidden_variant_keys: Optional[set] = None,
    disallow_train_variants: bool = False,
    force_regen: bool = False,
) -> Dict[str, List[Dict[str, float]]]:
    task = get_task(task_name)
    suite_specs = suite_specs or task.default_variant_suite()
    forbidden_variant_keys = forbidden_variant_keys or set()

    if suite_cache and os.path.exists(suite_cache) and not bool(force_regen):
        data = np.load(suite_cache, allow_pickle=True)
        suite_obj = data.get("suite", None)
        if suite_obj is not None and len(suite_obj) > 0:
            suite = suite_obj[0]
            if isinstance(suite, dict):
                if bool(disallow_train_variants) and forbidden_variant_keys and task.task_name in FAMILY_TASKS:
                    n_collide = 0
                    for variants in suite.values():
                        for v in variants:
                            if _variant_key(v) in forbidden_variant_keys:
                                n_collide += 1
                    if n_collide == 0:
                        return suite
                else:
                    return suite

    rng = np.random.default_rng(int(seed))
    if bool(disallow_train_variants) and forbidden_variant_keys and task.task_name in FAMILY_TASKS:
        suite: Dict[str, List[Dict[str, float]]] = {}
        used_eval: set = set()
        for group_name, spec in suite_specs.items():
            suite[group_name] = _sample_variants_excluding(
                rng,
                task=task,
                group_name=str(group_name),
                spec=spec,
                n=int(n_per_group),
                forbidden=forbidden_variant_keys,
                used_eval=used_eval,
            )
    else:
        suite = task.sample_eval_suite(n_per_group=int(n_per_group), seed=int(seed), suite_specs=suite_specs)

    if suite_cache:
        os.makedirs(os.path.dirname(suite_cache) or ".", exist_ok=True)
        meta = {"task": str(task_name), "n_per_group": int(n_per_group), "seed": int(seed), "groups": list(suite.keys())}
        if bool(disallow_train_variants) and forbidden_variant_keys:
            meta["disallow_train_variants"] = True
            meta["n_train_variants"] = int(len(forbidden_variant_keys))
        np.savez(
            suite_cache,
            suite=np.array([suite], dtype=object),
            metadata=np.array([json.dumps(meta)], dtype=object),
        )

    return suite


def make_gp(*, input_dim: int) -> GaussianProcessRegressor:
    input_dim = int(input_dim)
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=[1.0] * input_dim,
        length_scale_bounds=(1e-5, 1e5),
        nu=2.5,
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=3,
    )


def eval_variant_metrics_only(
    *,
    group_name: str,
    variant_id: int,
    params: Dict[str, float],
    task_name: str,
    base_regressor: TabPFNRegressor,
    tuned_regressor: TabPFNRegressor,
    X_eval: np.ndarray,
    x_bounds: Tuple[np.ndarray, np.ndarray],
    context_sizes: List[int],
    ctx_pool_size: int,
    eval_seed: int,
) -> Dict[int, Dict[str, float]]:
    task = get_task(task_name)
    dim = int(task.dim)

    y_true = task.evaluate_numpy(X_eval, params).reshape(-1)

    rng = np.random.default_rng(int(eval_seed) + 1000 * int(variant_id))
    lower, upper = x_bounds
    X_ctx_pool = rng.uniform(lower, upper, size=(int(ctx_pool_size), int(dim))).astype(np.float32)
    y_ctx_pool = task.evaluate_numpy(X_ctx_pool, params).reshape(-1)

    metrics: Dict[int, Dict[str, float]] = {}
    for k in context_sizes:
        idx = rng.choice(np.arange(X_ctx_pool.shape[0]), size=int(k), replace=False)
        X_ctx = X_ctx_pool[idx]
        y_ctx = y_ctx_pool[idx]

        base_regressor.fit(X_ctx, y_ctx)
        tuned_regressor.fit(X_ctx, y_ctx)

        y_pred_base = base_regressor.predict(X_eval).reshape(-1)
        y_pred_tuned = tuned_regressor.predict(X_eval).reshape(-1)

        gp = make_gp(input_dim=int(dim))
        gp.fit(X_ctx, y_ctx)
        y_pred_gp, _ = gp.predict(X_eval, return_std=True)
        y_pred_gp = np.asarray(y_pred_gp, dtype=np.float64).reshape(-1)

        mse_base = mean_squared_error(y_true, y_pred_base)
        mse_tuned = mean_squared_error(y_true, y_pred_tuned)
        mse_gp = mean_squared_error(y_true, y_pred_gp)
        metrics[int(k)] = {"mse_base": float(mse_base), "mse_tuned": float(mse_tuned), "mse_gp": float(mse_gp)}

    return metrics


def eval_variant_and_plot(
    group_name: str,
    variant_id: int,
    params: Dict[str, float],
    task_name: str,
    base_regressor: TabPFNRegressor,
    tuned_regressor: TabPFNRegressor,
    X_grid: np.ndarray,
    X1: np.ndarray,
    X2: np.ndarray,
    x_bounds: Tuple[np.ndarray, np.ndarray],
    context_sizes: List[int],
    ctx_pool_size: int,
    eval_seed: int,
    save_dir: str,
    plot: bool = True,
) -> Dict[int, Dict[str, float]]:
    task = get_task(task_name)
    if int(task.dim) != 2 and bool(plot):
        raise ValueError(f"Per-variant plotting only supports dim=2, got dim={task.dim} ({task.task_name})")
    if bool(plot):
        os.makedirs(save_dir, exist_ok=True)
    grid_size = X1.shape[0]

    y_true_grid_flat = task.evaluate_numpy(X_grid, params)
    y_true_grid = y_true_grid_flat.reshape(grid_size, grid_size)

    rng = np.random.default_rng(eval_seed + 1000 * variant_id)
    lower, upper = x_bounds
    X_ctx_pool = rng.uniform(lower, upper, size=(ctx_pool_size, 2)).astype(np.float32)
    y_ctx_pool = task.evaluate_numpy(X_ctx_pool, params)

    metrics: Dict[int, Dict[str, float]] = {}
    for k in context_sizes:
        idx = rng.choice(np.arange(X_ctx_pool.shape[0]), size=k, replace=False)
        X_ctx = X_ctx_pool[idx]
        y_ctx = y_ctx_pool[idx]

        base_regressor.fit(X_ctx, y_ctx)
        tuned_regressor.fit(X_ctx, y_ctx)

        y_pred_base = base_regressor.predict(X_grid)
        y_pred_tuned = tuned_regressor.predict(X_grid)

        gp = make_gp(input_dim=int(task.dim))
        gp.fit(X_ctx, y_ctx)
        y_pred_gp, _ = gp.predict(X_grid, return_std=True)

        mse_base = mean_squared_error(y_true_grid_flat, y_pred_base)
        mse_tuned = mean_squared_error(y_true_grid_flat, y_pred_tuned)
        mse_gp = mean_squared_error(y_true_grid_flat, y_pred_gp)
        metrics[k] = {"mse_base": float(mse_base), "mse_tuned": float(mse_tuned), "mse_gp": float(mse_gp)}

        if not bool(plot):
            continue

        y_pred_base_grid = y_pred_base.reshape(grid_size, grid_size)
        y_pred_tuned_grid = y_pred_tuned.reshape(grid_size, grid_size)
        y_pred_gp_grid = y_pred_gp.reshape(grid_size, grid_size)

        vmin = min(
            float(y_true_grid.min()),
            float(y_pred_base_grid.min()),
            float(y_pred_tuned_grid.min()),
            float(y_pred_gp_grid.min()),
        )
        vmax = max(
            float(y_true_grid.max()),
            float(y_pred_base_grid.max()),
            float(y_pred_tuned_grid.max()),
            float(y_pred_gp_grid.max()),
        )

        fig, axes = plt.subplots(1, 4, figsize=(20, 4), constrained_layout=True)
        levels = 30

        cs0 = axes[0].contourf(X1, X2, y_true_grid, levels=levels, vmin=vmin, vmax=vmax)
        axes[0].scatter(X_ctx[:, 0], X_ctx[:, 1], c="white", edgecolors="black", s=30)
        axes[0].set_title(f"True ({group_name} v{variant_id})\n(k={k})")

        cs1 = axes[1].contourf(X1, X2, y_pred_base_grid, levels=levels, vmin=vmin, vmax=vmax)
        axes[1].scatter(X_ctx[:, 0], X_ctx[:, 1], c="white", edgecolors="black", s=30)
        axes[1].set_title(f"Base TabPFN\n(k={k}, MSE={mse_base:.3f})")

        cs2 = axes[2].contourf(X1, X2, y_pred_tuned_grid, levels=levels, vmin=vmin, vmax=vmax)
        axes[2].scatter(X_ctx[:, 0], X_ctx[:, 1], c="white", edgecolors="black", s=30)
        axes[2].set_title(f"Finetuned TabPFN\n(k={k}, MSE={mse_tuned:.3f})")

        cs3 = axes[3].contourf(X1, X2, y_pred_gp_grid, levels=levels, vmin=vmin, vmax=vmax)
        axes[3].scatter(X_ctx[:, 0], X_ctx[:, 1], c="white", edgecolors="black", s=30)
        axes[3].set_title(f"GP\n(k={k}, MSE={mse_gp:.3f})")

        for ax in axes:
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")

        fig.colorbar(cs3, ax=axes, orientation="vertical", fraction=0.03, pad=0.04)

        param_str = (
            f"dx1={params['dx1']:.2f}, dx2={params['dx2']:.2f}, "
            f"rot={params['rotation']:.1f}°, sx1={params['sx1']:.2f}, sx2={params['sx2']:.2f}"
        )
        plt.suptitle(param_str, fontsize=12)

        save_path = os.path.join(save_dir, f"variant_{variant_id:03d}_k{k}.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    return metrics


def plot_group_mse_summary(
    *,
    group_name: str,
    context_sizes: List[int],
    mses_by_model: Dict[str, np.ndarray],
    save_dir: str,
    plot_mode: str,
    seed: int,
):
    os.makedirs(save_dir, exist_ok=True)
    x = np.asarray([int(k) for k in context_sizes], dtype=np.int64)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    rng = np.random.default_rng(int(seed))

    colors = {"base": "tab:orange", "tuned": "tab:blue", "gp": "tab:green"}

    for model_name, mses in mses_by_model.items():
        mses = np.asarray(mses, dtype=np.float64)  # (n_variants, n_k)
        if plot_mode == "mean_bootstrap_95":
            center = []
            lo = []
            hi = []
            for j in range(mses.shape[1]):
                m, l, h = _bootstrap_mean_ci(mses[:, j], rng=rng, n_boot=2000, ci=0.95)
                center.append(m)
                lo.append(l)
                hi.append(h)
            center = np.asarray(center, dtype=np.float64)
            lo = np.asarray(lo, dtype=np.float64)
            hi = np.asarray(hi, dtype=np.float64)
            label = f"{model_name} (mean, 95% CI)"
        elif plot_mode == "median_30_70":
            center = np.median(mses, axis=0)
            lo = np.quantile(mses, 0.30, axis=0)
            hi = np.quantile(mses, 0.70, axis=0)
            label = f"{model_name} (median, 30–70%)"
        else:
            raise ValueError(f"Unknown plot_mode: {plot_mode}")

        ax.plot(x, center, label=label, color=colors.get(model_name, None), linewidth=2)
        ax.fill_between(x, lo, hi, color=colors.get(model_name, None), alpha=0.2)

    ax.set_xlabel("Context size (k)")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"PFN regression on {group_name}")
    ax.legend(fontsize=9)

    suffix = "" if plot_mode == "mean_bootstrap_95" else f"_{plot_mode}"
    out_png = os.path.join(save_dir, f"mse_summary_{group_name}{suffix}.png")
    out_pdf = os.path.join(save_dir, f"mse_summary_{group_name}{suffix}.pdf")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    plt.close(fig)
    return out_png


def plot_contours_from_saved_pfn_eval(
    pkl_path: str,
    *,
    save_dir: str,
    n_variants_per_group_to_plot: int,
    plot_seed: int,
    groups_csv: Optional[str],
    context_sizes: Optional[List[int]],
    grid_size: int,
    ctx_pool_size: int,
    eval_seed_override: Optional[int] = None,
):
    with open(pkl_path, "rb") as f:
        blob = pickle.load(f)

    blob_dim = blob.get("dim", None)
    if blob_dim is not None and int(blob_dim) != 2:
        raise ValueError(f"Contour plotting requires dim=2, got dim={blob_dim} from {pkl_path}")

    task_name = str(blob.get("task", "branin_family"))
    suite: Dict[str, List[Dict[str, float]]] = blob.get("suite", {})
    tuned_model_path = str(blob.get("tuned_model_path", ""))
    stored_context_sizes = [int(k) for k in blob.get("context_sizes", [])]
    stored_eval_seed = int(blob.get("eval_seed", 0))

    if not tuned_model_path:
        raise ValueError(f"Missing tuned_model_path in pkl: {pkl_path}")
    if not os.path.isabs(tuned_model_path) and not os.path.exists(tuned_model_path):
        repo_root_guess = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        cand = os.path.join(repo_root_guess, tuned_model_path)
        if os.path.exists(cand):
            tuned_model_path = cand
    if not os.path.exists(tuned_model_path):
        raise FileNotFoundError(f"Tuned TabPFN checkpoint not found: {tuned_model_path}")
    if not suite:
        raise ValueError(f"Missing suite in pkl: {pkl_path}")

    if context_sizes is None:
        if stored_context_sizes:
            context_sizes = [int(max(stored_context_sizes))]
        else:
            context_sizes = [8]

    eval_seed_base = int(eval_seed_override) if eval_seed_override is not None else int(stored_eval_seed)

    task = get_task(task_name)
    if int(task.dim) != 2:
        raise ValueError(f"eval_pfn contour plotting only supports dim=2 tasks, got dim={task.dim} ({task.task_name})")
    lower = np.asarray(task.bounds[0], dtype=np.float32)
    upper = np.asarray(task.bounds[1], dtype=np.float32)
    x1_range = (float(lower[0]), float(upper[0]))
    x2_range = (float(lower[1]), float(upper[1]))
    X1, X2, X_grid = build_grid(x1_range, x2_range, int(grid_size))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    regressor_config = {
        "ignore_pretraining_limits": True,
        "device": device,
        "n_estimators": 1,
        "random_state": 42,
        "inference_precision": torch.float32,
    }
    base_regressor = TabPFNRegressor(**regressor_config, differentiable_input=False)
    tuned_regressor = TabPFNRegressor(
        model_path=tuned_model_path,
        device=device,
        n_estimators=1,
        random_state=42,
        inference_precision=torch.float32,
        differentiable_input=False,
        ignore_pretraining_limits=True,
    )
    base_regressor._initialize_model_variables()

    group_names = _split_csv(groups_csv) or list(suite.keys())
    rng = np.random.default_rng(int(plot_seed))
    os.makedirs(save_dir, exist_ok=True)

    print(f"Plot contours from: {pkl_path}")
    print(f"Task: {task.task_name}")
    print(f"Tuned ckpt: {tuned_model_path}")
    print(f"Output dir: {save_dir}")
    print(f"Groups: {group_names}")
    print(f"Context sizes: {context_sizes}")
    print(f"Grid size: {grid_size}, ctx_pool_size: {ctx_pool_size}")

    for group_idx, group_name in enumerate(group_names):
        if group_name not in suite:
            print(f"[WARN] group not found in suite: {group_name}; skipping.")
            continue
        variants = suite.get(group_name, [])
        if not variants:
            print(f"[WARN] group {group_name} has no variants; skipping.")
            continue

        k = int(n_variants_per_group_to_plot)
        n_pick = min(int(k), len(variants))
        pick_idx = rng.choice(np.arange(len(variants)), size=n_pick, replace=False)

        group_dir = os.path.join(save_dir, group_name)
        os.makedirs(group_dir, exist_ok=True)
        print(f"\n=== Contours: {group_name} | pick {n_pick}/{len(variants)} variants ===")

        for idx in pick_idx.tolist():
            params = variants[int(idx)]
            variant_id = int(idx) + 1
            eval_variant_and_plot(
                group_name=group_name,
                variant_id=variant_id,
                params=params,
                task_name=task.task_name,
                base_regressor=base_regressor,
                tuned_regressor=tuned_regressor,
                X_grid=X_grid,
                X1=X1,
                X2=X2,
                x_bounds=(lower, upper),
                context_sizes=[int(kc) for kc in context_sizes],
                ctx_pool_size=int(ctx_pool_size),
                eval_seed=int(eval_seed_base) + 100000 * int(group_idx),
                save_dir=group_dir,
                plot=True,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="branin_family")
    parser.add_argument("--suite_cache", type=str, default="./data/variants_eval_suite_pfn.npz")
    parser.add_argument("--tuned_model_path", type=str, default="./model/finetuned_tabpfn_branin_family.ckpt")
    parser.add_argument(
        "--train_variants_path",
        type=str,
        default=None,
        help="Training variants cache (npz with key 'variants') to exclude from eval suite.",
    )
    parser.add_argument(
        "--disallow_train_variants",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Guarantee eval variants are not identical to training variants (branin_family only).",
    )
    parser.add_argument(
        "--force_regen_suite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ignore suite_cache and resample the evaluation suite.",
    )
    parser.add_argument("--n_variants_per_group", type=int, default=5)
    parser.add_argument("--suite_seed", type=int, default=2026)
    parser.add_argument("--eval_seed", type=int, default=123)
    parser.add_argument("--grid_size", type=int, default=20)
    parser.add_argument("--ctx_pool_size", type=int, default=2000)
    parser.add_argument("--save_dir", type=str, default="./figs/pfn_variant_suite")
    parser.add_argument("--context_sizes", type=int, nargs="+", default=[4, 8, 12, 16, 20])
    parser.add_argument(
        "--plot_per_variant",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to save per-variant contour plots (can be very large).",
    )
    parser.add_argument(
        "--mse_plot",
        type=str,
        choices=["mean_bootstrap_95", "median_30_70", "both"],
        default="both",
        help="How to plot MSE summary curves.",
    )
    parser.add_argument(
        "--save_eval_data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save raw MSE results to a .pkl for re-plotting without reruns.",
    )
    parser.add_argument(
        "--load_eval_data",
        type=str,
        default=None,
        help="Load a saved pfn_eval_data.pkl and only re-plot (no evaluation).",
    )
    parser.add_argument(
        "--plot_contours_from_pkl",
        type=str,
        default=None,
        help="Plot contour figures for a random subset of variants per group from a saved pfn_eval_data.pkl.",
    )
    parser.add_argument("--contour_k_per_group", type=int, default=3, help="How many variants to plot per group.")
    parser.add_argument("--contour_seed", type=int, default=0, help="Random seed for selecting variants to plot.")
    parser.add_argument("--contour_groups", type=str, default=None, help="Comma-separated groups to plot (default: all).")
    parser.add_argument(
        "--contour_context_sizes",
        type=int,
        nargs="+",
        default=None,
        help="Context sizes k to plot (default: max(context_sizes) from the saved pkl).",
    )
    parser.add_argument("--contour_grid_size", type=int, default=25, help="Grid size for contour plotting.")
    parser.add_argument("--contour_ctx_pool_size", type=int, default=2000, help="Context pool size for contour plotting.")
    parser.add_argument("--contour_eval_seed", type=int, default=None, help="Override eval seed base (default: use saved eval_seed).")
    parser.add_argument("--contour_save_dir", type=str, default=None, help="Output directory for contour plots (default: <save_dir>/contours_subset).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.load_eval_data:
        with open(args.load_eval_data, "rb") as f:
            blob = pickle.load(f)
        suite = blob["suite"]
        group_metrics = blob["group_metrics"]
        context_sizes = [int(k) for k in blob["context_sizes"]]
        os.makedirs(args.save_dir, exist_ok=True)
        for group_name, by_k in group_metrics.items():
            mses_base = np.asarray([by_k[int(k)]["mse_base"] for k in context_sizes], dtype=np.float64).T
            mses_tuned = np.asarray([by_k[int(k)]["mse_tuned"] for k in context_sizes], dtype=np.float64).T
            mses_gp = np.asarray([by_k[int(k)]["mse_gp"] for k in context_sizes], dtype=np.float64).T
            mses_by_model = {"base": mses_base, "tuned": mses_tuned, "gp": mses_gp}
            for mode in _resolve_plot_modes(str(args.mse_plot)):
                plot_group_mse_summary(
                    group_name=group_name,
                    context_sizes=context_sizes,
                    mses_by_model=mses_by_model,
                    save_dir=args.save_dir,
                    plot_mode=str(mode),
                    seed=int(blob.get("eval_seed", 0)),
                )
        print(f"Re-plot completed. Outputs in {args.save_dir}")
        return

    if args.plot_contours_from_pkl:
        out_dir = str(args.contour_save_dir or os.path.join(str(args.save_dir), "contours_subset"))
        plot_contours_from_saved_pfn_eval(
            str(args.plot_contours_from_pkl),
            save_dir=out_dir,
            n_variants_per_group_to_plot=int(args.contour_k_per_group),
            plot_seed=int(args.contour_seed),
            groups_csv=args.contour_groups,
            context_sizes=args.contour_context_sizes,
            grid_size=int(args.contour_grid_size),
            ctx_pool_size=int(args.contour_ctx_pool_size),
            eval_seed_override=args.contour_eval_seed,
        )
        return

    task = get_task(args.task)
    lower = np.asarray(task.bounds[0], dtype=np.float32)
    upper = np.asarray(task.bounds[1], dtype=np.float32)
    dim = int(task.dim)

    if bool(args.plot_per_variant) and dim != 2:
        print(f"[WARN] --plot_per_variant only supports dim=2; got dim={dim} ({task.task_name}). Disabling.")
        args.plot_per_variant = False

    forbidden = set()
    train_variants: List[Dict[str, float]] = []
    if bool(args.disallow_train_variants) and task.task_name in FAMILY_TASKS:
        if not args.train_variants_path:
            raise ValueError("--train_variants_path is required when --disallow_train_variants is enabled")
        train_variants = _load_variants_from_npz(str(args.train_variants_path))
        forbidden = {_variant_key(v) for v in train_variants}
        print(f"Loaded {len(train_variants)} training variants from {args.train_variants_path} (will exclude from eval)")

    suite = load_or_generate_variant_suite(
        args.suite_cache,
        task_name=task.task_name,
        n_per_group=args.n_variants_per_group,
        seed=args.suite_seed,
        forbidden_variant_keys=forbidden,
        disallow_train_variants=bool(args.disallow_train_variants),
        force_regen=bool(args.force_regen_suite),
    )

    # Evaluation points:
    # - dim=2: use a grid (enables per-variant heatmaps)
    # - dim!=2: use Sobol points (no per-variant contours)
    X1 = X2 = X_grid = None
    if dim == 2:
        x1_range = (float(lower[0]), float(upper[0]))
        x2_range = (float(lower[1]), float(upper[1]))
        X1, X2, X_grid = build_grid(x1_range, x2_range, args.grid_size)

    regressor_config = {
        "ignore_pretraining_limits": True,
        "device": device,
        "n_estimators": 1,
        "random_state": 42,
        "inference_precision": torch.float32,
    }
    base_regressor = TabPFNRegressor(**regressor_config, differentiable_input=False)
    tuned_regressor = TabPFNRegressor(
        model_path=args.tuned_model_path,
        device=device,
        n_estimators=1,
        random_state=42,
        inference_precision=torch.float32,
        differentiable_input=False,
        ignore_pretraining_limits=True,
    )

    base_regressor._initialize_model_variables()
    print(f"Device: {device}")
    print(
        f"Variant suite groups: {', '.join(list(suite.keys()))} "
        f"(n_per_group={args.n_variants_per_group}, cache={args.suite_cache})"
    )

    os.makedirs(args.save_dir, exist_ok=True)
    summary: Dict[str, Dict[int, Dict[str, float]]] = {}
    raw_group_metrics: Dict[str, Dict[int, Dict[str, List[float]]]] = {}

    group_names = list(suite.keys())
    for group_idx, group_name in enumerate(group_names):
        variants = suite.get(group_name, [])
        if not variants:
            print(f"[WARN] group {group_name} has no variants; skipping.")
            continue

        print(f"\n=== Group: {group_name} ({len(variants)} variants) ===")
        group_metrics: Dict[int, List[Dict[str, float]]] = {k: [] for k in args.context_sizes}
        raw_group_metrics[group_name] = {int(k): {"mse_base": [], "mse_tuned": [], "mse_gp": []} for k in args.context_sizes}
        group_dir = os.path.join(args.save_dir, group_name)
        if bool(args.plot_per_variant):
            os.makedirs(group_dir, exist_ok=True)

        X_eval = None
        if dim != 2:
            n_eval = int(args.grid_size) ** 2
            X_eval = sample_sobol_points(
                (lower, upper),
                n=n_eval,
                seed=int(args.eval_seed) + 100000 * int(group_idx),
            )

        for i, params in enumerate(variants, start=1):
            print(f"Variant {i}/{len(variants)}: {params}")
            if dim == 2:
                assert X_grid is not None and X1 is not None and X2 is not None
                metrics = eval_variant_and_plot(
                    group_name=group_name,
                    variant_id=i,
                    params=params,
                    task_name=task.task_name,
                    base_regressor=base_regressor,
                    tuned_regressor=tuned_regressor,
                    X_grid=X_grid,
                    X1=X1,
                    X2=X2,
                    x_bounds=(lower, upper),
                    context_sizes=args.context_sizes,
                    ctx_pool_size=args.ctx_pool_size,
                    eval_seed=int(args.eval_seed) + 100000 * int(group_idx),
                    save_dir=group_dir,
                    plot=bool(args.plot_per_variant),
                )
            else:
                assert X_eval is not None
                metrics = eval_variant_metrics_only(
                    group_name=group_name,
                    variant_id=i,
                    params=params,
                    task_name=task.task_name,
                    base_regressor=base_regressor,
                    tuned_regressor=tuned_regressor,
                    X_eval=X_eval,
                    x_bounds=(lower, upper),
                    context_sizes=[int(k) for k in args.context_sizes],
                    ctx_pool_size=int(args.ctx_pool_size),
                    eval_seed=int(args.eval_seed) + 100000 * int(group_idx),
                )
            for k, m in metrics.items():
                group_metrics[k].append(m)
                raw_group_metrics[group_name][int(k)]["mse_base"].append(float(m["mse_base"]))
                raw_group_metrics[group_name][int(k)]["mse_tuned"].append(float(m["mse_tuned"]))
                raw_group_metrics[group_name][int(k)]["mse_gp"].append(float(m["mse_gp"]))

        print(f"\n--- Summary: {group_name} (mean MSE over variants) ---")
        summary[group_name] = {}
        for k in args.context_sizes:
            mses = group_metrics[k]
            mean_base = float(np.mean([x["mse_base"] for x in mses]))
            mean_tuned = float(np.mean([x["mse_tuned"] for x in mses]))
            mean_gp = float(np.mean([x["mse_gp"] for x in mses]))
            summary[group_name][int(k)] = {"base": mean_base, "tuned": mean_tuned, "gp": mean_gp}
            print(f"k={k:2d} | Base: {mean_base:.4f} | Tuned: {mean_tuned:.4f} | GP: {mean_gp:.4f}")

        mses_base = np.asarray([raw_group_metrics[group_name][int(k)]["mse_base"] for k in args.context_sizes], dtype=np.float64).T
        mses_tuned = np.asarray([raw_group_metrics[group_name][int(k)]["mse_tuned"] for k in args.context_sizes], dtype=np.float64).T
        mses_gp = np.asarray([raw_group_metrics[group_name][int(k)]["mse_gp"] for k in args.context_sizes], dtype=np.float64).T
        mses_by_model = {"base": mses_base, "tuned": mses_tuned, "gp": mses_gp}
        for mode in _resolve_plot_modes(str(args.mse_plot)):
            out = plot_group_mse_summary(
                group_name=group_name,
                context_sizes=[int(k) for k in args.context_sizes],
                mses_by_model=mses_by_model,
                save_dir=args.save_dir,
                plot_mode=str(mode),
                seed=int(args.eval_seed),
            )
            print(f"Saved PFN MSE summary plot to {out}")

    summary_path = os.path.join(args.save_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    if bool(args.save_eval_data):
        pkl_path = os.path.join(args.save_dir, "pfn_eval_data.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(
                {
                    "task": str(task.task_name),
                    "dim": int(task.dim),
                    "bounds_lower": np.asarray(task.bounds[0], dtype=np.float32),
                    "bounds_upper": np.asarray(task.bounds[1], dtype=np.float32),
                    "suite_seed": int(args.suite_seed),
                    "eval_seed": int(args.eval_seed),
                    "context_sizes": [int(k) for k in args.context_sizes],
                    "suite": suite,
                    "group_metrics": raw_group_metrics,
                    "tuned_model_path": str(args.tuned_model_path),
                    "train_variants_path": str(args.train_variants_path),
                    "disallow_train_variants": bool(args.disallow_train_variants),
                },
                f,
            )
        print(f"Saved PFN eval data to {pkl_path}")


if __name__ == "__main__":
    main()
