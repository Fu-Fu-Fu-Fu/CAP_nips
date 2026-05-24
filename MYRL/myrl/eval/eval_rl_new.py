"""
评估不同策略在GP和TabPFN代理模型上的性能
对比方法：Random, EI, UCB, PI, TAF(me), TAF(ranking), CAP-PPO

实验设置（方案A）：
实验1 - 公平对比 (experiment_mode='fair'):
    - 所有方法使用128个候选点（与RL训练时一致）
    - 目的：在相同条件下对比所有方法

实验2 - 上限对比 (experiment_mode='optimal'):
    - CAP-PPO: 128个候选点（训练配置）
    - Baselines (EI/UCB/PI/TAF): 2048个候选点（纯Sobol）
    - Random: 128个候选点（候选点数量对Random无影响）
    - 目的：展示各方法在各自最优设置下的性能

- 在4种Branin变体类型上评估（in_range, ood_level_1/2/3）
- 每种类型10个变体，每个变体5次运行
- 输出：2张对比图（GP版和TabPFN版），每张图左边是Rank，右边是Regret曲线
"""

import argparse
import os
import numpy as np
import torch
import pickle
import json
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple, Optional
from scipy.stats import rankdata
from scipy.stats.qmc import Sobol
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

# MYRL modules
from ..tasks import get_task
from ..rl.train_rl import TaskVariantObjectiveFunction
from ..bo.select_candidates import compute_std_from_tabpfn_output, predict_tabpfn_with_normalization
from tabpfn import TabPFNRegressor

# Import policies
from ..policies.policies import EI, UCB, PI, TAF, RandomPolicy, RLPolicy, MetaBOPolicy, FunBO, PFNs4BOPolicy, TuRBOPolicy

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")
warnings.filterwarnings("ignore", message=".*ABNORMAL_TERMINATION_IN_LNSRCH.*")
warnings.filterwarnings("ignore", message=".*scale the data.*")
warnings.filterwarnings("ignore", message=".*optimal value found.*close to the specified.*bound.*")

CAP_PPO_NAME = "CAP-PPO"
LEGACY_RL_NAME = "RL"
FAMILY_TASKS = {
    "alkox_emulator",
    "ackley_5d_family",
    "ackley_10d_family",
    "branin_family",
    "goldstein_price_family",
    "hartmann_3d_family",
    "hartmann_6d_family",
    "benzylation_emulator",
    "hplc_emulator",
}

class _SklearnGPModelTarget:
    def __init__(self, gp: GaussianProcessRegressor):
        self._gp = gp

    def predict_noiseless(self, X: np.ndarray):
        mean, std = self._gp.predict(np.asarray(X, dtype=np.float64), return_std=True)
        mean = np.asarray(mean, dtype=np.float64).reshape(-1, 1)
        var = (np.asarray(std, dtype=np.float64) ** 2).reshape(-1, 1)
        return mean, var


class _TabPFNModelTarget:
    def __init__(self, regressor: TabPFNRegressor, y_mean: float = 0.0, y_std: float = 1.0):
        self._regressor = regressor
        self._y_mean = y_mean
        self._y_std = y_std

    def predict_noiseless(self, X: np.ndarray):
        full_out = self._regressor.predict(np.asarray(X, dtype=np.float32), output_type="full")
        mean_norm = np.asarray(full_out.get("mean", None), dtype=np.float64).reshape(-1, 1)
        std_norm = np.asarray(compute_std_from_tabpfn_output(full_out), dtype=np.float64).reshape(-1, 1)
        # 反归一化回 raw 空间
        mean = mean_norm * self._y_std + self._y_mean
        var = (std_norm * self._y_std) ** 2
        return mean, var


def _sanitize_regrets_for_plot(unit_curves: np.ndarray) -> np.ndarray:
    unit_curves = np.asarray(unit_curves, dtype=np.float64)
    return np.maximum(unit_curves, 1e-12)

def _resolve_regret_plot_modes(regret_plot: str) -> List[str]:
    regret_plot = str(regret_plot)
    if regret_plot == "both":
        return ["mean_bootstrap_95", "median_30_70"]
    return [regret_plot]

def _normalize_method_name(name: str) -> str:
    name = str(name).strip()
    if name == LEGACY_RL_NAME:
        return CAP_PPO_NAME
    return name


def _normalize_method_list(names: List[str]) -> List[str]:
    return [_normalize_method_name(x) for x in names]

def _normalize_group_results_keys(group_results: Dict) -> Dict:
    """
    Backward-compat: older saved results may use method key 'RL'.
    Normalize it to CAP_PPO_NAME for plotting and re-saving.
    """
    if LEGACY_RL_NAME in group_results and CAP_PPO_NAME not in group_results:
        group_results = dict(group_results)
        group_results[CAP_PPO_NAME] = group_results.pop(LEGACY_RL_NAME)
    return group_results


def _normalize_all_group_results_keys(all_group_results: Dict) -> Dict:
    return {g: _normalize_group_results_keys(r) for (g, r) in all_group_results.items()}

def _is_meta_key(name: str) -> bool:
    return str(name).startswith("__")

def _resolve_metabo_logpath(path: str) -> str:
    """
    Accept either:
    - a MetaBO run directory containing weights_<iter>/params_<iter>
    - an env directory containing a LATEST file pointing to a run directory
    - an env directory containing timestamp subdirectories (pick newest lexicographically)
    """
    p = str(path)
    if not os.path.isdir(p):
        raise FileNotFoundError(f"MetaBO logpath not found (not a directory): {p}")

    def _has_ckpts(d: str) -> bool:
        try:
            names = os.listdir(d)
        except Exception:
            return False
        return any(n.startswith("weights_") for n in names) and any(n.startswith("params_") for n in names)

    if _has_ckpts(p):
        return p

    latest_file = os.path.join(p, "LATEST")
    if os.path.isfile(latest_file):
        with open(latest_file, "r", encoding="utf-8") as f:
            cand = f.read().strip()
        if cand and os.path.isdir(cand) and _has_ckpts(cand):
            return cand

    subdirs = sorted([n for n in os.listdir(p) if os.path.isdir(os.path.join(p, n))])
    for name in reversed(subdirs):
        cand = os.path.join(p, name)
        if _has_ckpts(cand):
            return cand

    raise FileNotFoundError(
        f"MetaBO logpath directory does not contain checkpoints (weights_*/params_*): {p}"
    )


def _resolve_metabo_load_iter(logpath: str, load_iter: Optional[int]) -> int:
    if load_iter is not None:
        return int(load_iter)

    iters_w = set()
    iters_p = set()
    for name in os.listdir(logpath):
        if name.startswith("weights_"):
            try:
                iters_w.add(int(name.split("_", 1)[1]))
            except Exception:
                pass
        elif name.startswith("params_"):
            try:
                iters_p.add(int(name.split("_", 1)[1]))
            except Exception:
                pass

    iters = sorted(iters_w.intersection(iters_p))
    if not iters:
        raise FileNotFoundError(
            f"No matching MetaBO checkpoints found under {logpath} (need weights_<iter> and params_<iter>)."
        )
    return int(iters[-1])


def _iter_method_items(group_results: Dict) -> List[Tuple[str, Dict]]:
    return [(k, v) for (k, v) in group_results.items() if not _is_meta_key(k)]


def _iter_method_names(group_results: Dict) -> List[str]:
    return [k for (k, _) in _iter_method_items(group_results)]


def _variant_key(variant_params: Dict, *, ndigits: int = 12) -> Tuple[Tuple[str, object], ...]:
    """
    Canonical key for comparing whether two variants are the "same function".
    We round floats to avoid tiny serialization differences.
    """
    items: List[Tuple[str, object]] = []
    for k in sorted(variant_params.keys()):
        v = variant_params[k]
        if isinstance(v, (float, np.floating)):
            v = round(float(v), int(ndigits))
        items.append((str(k), v))
    return tuple(items)


def _load_variants_from_npz(path: str) -> List[Dict]:
    if path is None:
        return []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Train variants cache not found: {path}")
    data = np.load(path, allow_pickle=True)
    if "variants" not in data:
        raise KeyError(f"NPZ missing key 'variants': {path}")
    return data["variants"].tolist()


def _sample_branin_variant_params_from_spec(rng: np.random.Generator, spec: Dict) -> Dict[str, float]:
    """
    Wrapper around sample_variant_params_from_spec with a stable return type.
    """
    return dict(sample_variant_params_from_spec(rng, spec))


def _sample_branin_variants_excluding(
    rng: np.random.Generator,
    *,
    spec: Dict,
    n: int,
    forbidden: set,
    max_tries_per_sample: int = 10_000,
) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for _ in range(int(n)):
        for _try in range(int(max_tries_per_sample)):
            v = _sample_branin_variant_params_from_spec(rng, spec)
            if _variant_key(v) not in forbidden:
                out.append(v)
                break
        else:
            raise RuntimeError(
                "Failed to sample a non-training evaluation variant (too many collisions). "
                "Check the spec ranges and forbidden set size."
            )
    return out


def _plot_rank_and_regret(
    unit_curves_by_method: Dict[str, np.ndarray],
    *,
    max_steps: int,
    save_dir: str,
    filename_prefix: str,
    title: str,
    colors: Dict[str, str],
    rng_seed: int = 0,
    n_boot: int = 2000,
    regret_plot: str = "mean_bootstrap_95",
):
    os.makedirs(save_dir, exist_ok=True)

    x = np.arange(max_steps + 1)
    methods = list(unit_curves_by_method.keys())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: rank (lower is better)
    mean_ranks = _compute_mean_rank_over_units(unit_curves_by_method)
    for method, r in mean_ranks.items():
        axes[0].plot(x, r, label=method, color=colors.get(method, None), linewidth=2)
    axes[0].set_xlabel("Number of Function Evaluations", fontsize=12)
    axes[0].set_ylabel("Mean Rank (lower is better)", fontsize=12)
    axes[0].set_title("Rank (mean over units)", fontsize=13)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(1.0, float(len(methods)) + 0.1)
    axes[0].legend(fontsize=10)

    # Right: regret curve
    rng = np.random.default_rng(int(rng_seed))
    for method, unit_curves in unit_curves_by_method.items():
        unit_curves = _sanitize_regrets_for_plot(unit_curves)
        if str(regret_plot) == "mean_bootstrap_95":
            center, lower, upper = _bootstrap_mean_ci_over_units(unit_curves, rng=rng, n_boot=int(n_boot), ci=0.95)
            ylabel = "Simple Regret"
            title_right = "Regret (bootstrap 95% CI)"
        elif str(regret_plot) == "median_30_70":
            center = np.median(unit_curves, axis=0)
            lower = np.quantile(unit_curves, 0.30, axis=0)
            upper = np.quantile(unit_curves, 0.70, axis=0)
            ylabel = "Simple Regret"
            title_right = "Regret (median, 30–70% band)"
        else:
            raise ValueError(f"Unknown regret_plot: {regret_plot}")

        center = _sanitize_regrets_for_plot(center)
        lower = _sanitize_regrets_for_plot(lower)
        upper = _sanitize_regrets_for_plot(upper)

        axes[1].plot(x, center, label=method, color=colors.get(method, None), linewidth=2)
        axes[1].fill_between(x, lower, upper, color=colors.get(method, None), alpha=0.2)
    axes[1].set_xlabel("Number of Function Evaluations", fontsize=12)
    axes[1].set_ylabel(ylabel, fontsize=12)
    axes[1].set_title(title_right, fontsize=13)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=10)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path_png = os.path.join(save_dir, f"{filename_prefix}.png")
    save_path_pdf = os.path.join(save_dir, f"{filename_prefix}.pdf")
    fig.savefig(save_path_png, dpi=150)
    fig.savefig(save_path_pdf)
    plt.close(fig)

    return save_path_png


def plot_group_comparison(
    group_name: str,
    group_results: Dict,
    *,
    max_steps: int,
    save_dir: str,
    surrogate_type: str,
    experiment_mode: str,
    regret_plot: str = "mean_bootstrap_95",
):
    colors = {
        'Random': 'tab:gray',
        'EI': 'tab:green',
        'UCB': 'tab:cyan',
        'PI': 'tab:olive',
        'TAF_me': 'tab:orange',
        'TAF_ranking': 'tab:pink',
        CAP_PPO_NAME: 'tab:blue',
        LEGACY_RL_NAME: 'tab:blue',
        'MetaBO': 'tab:purple',
    }

    unit_curves_by_method: Dict[str, np.ndarray] = {}
    for method, data in _iter_method_items(group_results):
        unit_curves = []
        for v_runs in data['regrets_by_variant']:
            arr = np.asarray(v_runs, dtype=np.float64)  # (n_runs, T)
            unit_curves.append(arr.mean(axis=0))
        unit_curves_by_method[method] = np.asarray(unit_curves, dtype=np.float64)

    if experiment_mode == "fair":
        mode_label = "Fair (all 128 cand.)"
    else:
        mode_label = f"Optimal ({CAP_PPO_NAME} 128, others 2048)"
    title = f"{group_name}: {surrogate_type.upper()} surrogate, {mode_label}"

    group_dir = os.path.join(save_dir, "groups", group_name)
    suffix = "" if str(regret_plot) == "mean_bootstrap_95" else f"_{regret_plot}"
    return _plot_rank_and_regret(
        unit_curves_by_method,
        max_steps=max_steps,
        save_dir=group_dir,
        filename_prefix=f"comparison_{surrogate_type}_{experiment_mode}_{group_name}{suffix}",
        title=title,
        colors=colors,
        regret_plot=str(regret_plot),
    )


def plot_trajectories(
    func,
    trajectories_by_method: Dict[str, np.ndarray],
    *,
    save_dir: str,
    filename_prefix: str,
    n_init: int,
    methods: List[str],
    grid_size: int = 80,
):
    if int(getattr(func, "dim", 2)) != 2:
        print(f"[WARN] plot_trajectories only supports dim=2; got dim={getattr(func, 'dim', None)}. Skipping.")
        return None

    os.makedirs(save_dir, exist_ok=True)

    lower, upper = func.bounds
    x1 = np.linspace(lower[0], upper[0], int(grid_size))
    x2 = np.linspace(lower[1], upper[1], int(grid_size))
    X1, X2 = np.meshgrid(x1, x2)
    grid_points = np.stack([X1.reshape(-1), X2.reshape(-1)], axis=1).astype(np.float32)
    Z = func(grid_points).reshape(X1.shape)

    n_methods = len(methods)
    ncols = min(4, n_methods)
    nrows = int(np.ceil(n_methods / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

    for idx, method in enumerate(methods):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]

        ax.contourf(X1, X2, Z, levels=50, cmap='viridis', alpha=0.8)
        ax.contour(X1, X2, Z, levels=20, colors='white', alpha=0.3, linewidths=0.5)

        traj = trajectories_by_method[method]
        ax.scatter(traj[:n_init, 0], traj[:n_init, 1], c='white', s=80,
                   marker='o', edgecolors='black', linewidths=1.2, label='Init', zorder=5)
        ax.scatter(traj[n_init:, 0], traj[n_init:, 1], c='red', s=45,
                   marker='x', linewidths=1.2, label='Query', zorder=5)
        ax.plot(traj[:, 0], traj[:, 1], 'r--', alpha=0.5, linewidth=1)

        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_xlabel("$x_1$", fontsize=11)
        ax.set_ylabel("$x_2$", fontsize=11)
        ax.set_title(method, fontsize=12)
        ax.legend(loc='upper right', fontsize=8)

    # Hide unused axes
    for idx in range(n_methods, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r][c].axis("off")

    plt.tight_layout()
    save_path_png = os.path.join(save_dir, f"{filename_prefix}.png")
    save_path_pdf = os.path.join(save_dir, f"{filename_prefix}.pdf")
    plt.savefig(save_path_png, dpi=150)
    plt.savefig(save_path_pdf)
    plt.close(fig)

    return save_path_png


def _pick_representative_run_index(
    group_results: Dict,
    *,
    variant_index: int,
    n_runs: int,
    pick: str,
    pick_method: str,
) -> int:
    if n_runs <= 0:
        return 0

    pick = str(pick)
    if pick == "first":
        return 0

    method = _normalize_method_name(str(pick_method))
    if method not in group_results:
        return 0

    runs = group_results[method]["regrets_by_variant"][variant_index]
    finals = [float(curve[-1]) for curve in runs[:n_runs]]
    if not finals:
        return 0

    finals_arr = np.asarray(finals, dtype=np.float64)
    if pick == "best":
        return int(np.argmin(finals_arr))
    if pick == "median":
        return int(np.argsort(finals_arr)[len(finals_arr) // 2])

    return 0


def _save_trajectory_npz(
    out_path: str,
    *,
    variant_params: Dict,
    run_index: int,
    run_seed: int,
    n_init: int,
    trajectories_by_method: Dict[str, np.ndarray],
):
    payload = {
        "variant_params": np.array(variant_params, dtype=object),
        "run_index": np.array(int(run_index), dtype=np.int64),
        "run_seed": np.array(int(run_seed), dtype=np.int64),
        "n_init": np.array(int(n_init), dtype=np.int64),
    }
    for method, traj in trajectories_by_method.items():
        payload[f"traj_{method}"] = np.asarray(traj, dtype=np.float32)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **payload)

# ==================== Variant Suite Specifications ====================
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


def _sample_from_segments(rng: np.random.Generator, segments: List[Tuple[float, float]]) -> float:
    """Sample a value from a list of segments"""
    lengths = np.array([hi - lo for (lo, hi) in segments], dtype=np.float64)
    probs = lengths / lengths.sum()
    idx = int(rng.choice(len(segments), p=probs))
    lo, hi = segments[idx]
    return float(rng.uniform(lo, hi))


def sample_variant_params_from_spec(rng: np.random.Generator, spec: Dict) -> Dict[str, float]:
    """Sample Branin variant parameters from specification"""
    dx1 = _sample_from_segments(rng, spec["dx"])
    dx2 = _sample_from_segments(rng, spec["dx"])
    rotation = _sample_from_segments(rng, spec["rotation"])
    sx1 = _sample_from_segments(rng, spec["sx"])
    sx2 = _sample_from_segments(rng, spec["sx"])

    return {
        "dx1": dx1,
        "dx2": dx2,
        "rotation": rotation,
        "sx1": sx1,
        "sx2": sx2,
        "alpha": 1.0,
        "beta": 0.0,
    }


# ==================== TAF Data Preparation ====================
def prepare_taf_data(bo_trajs_path: str, output_pickle_path: str):
    """
    将bo_trajs_train.npz转换为TAF期望的格式

    TAF期望的数据格式:
    {
        'D': <dim>,
        'M': 10,
        'X': [X1, X2, ..., X10],
        'Y': [Y1, Y2, ..., Y10],
        'kernel_lengthscale': [...],
        'kernel_variance': [...],
        'noise_variance': [...],
        'use_prior_mean_function': [...]
    }
    """
    print(f"\nPreparing TAF source data from {bo_trajs_path}...")

    data = np.load(bo_trajs_path, allow_pickle=True)
    X_trajs = np.asarray(data['X_trajs'])  # (n_traj, T, D)
    y_trajs = np.asarray(data['y_trajs'])  # (n_traj, T)
    if "variant_indices" not in data:
        raise KeyError(f"bo_trajs_path missing key 'variant_indices': {bo_trajs_path}")
    variant_indices = np.asarray(data["variant_indices"], dtype=np.int64).reshape(-1)

    if X_trajs.ndim != 3:
        raise ValueError(f"Invalid X_trajs shape: {X_trajs.shape}")
    D = int(X_trajs.shape[2])

    # 提取每个训练变体的“第一条轨迹”（finetune.py 写入时：每个变体先写 best-of BO 轨迹，再写合成轨迹）
    variant_ids = np.sort(np.unique(variant_indices))
    M = int(len(variant_ids))
    traj_indices = []
    for v in variant_ids.tolist():
        idxs = np.where(variant_indices == int(v))[0]
        if idxs.size == 0:
            continue
        traj_indices.append(int(idxs.min()))

    taf_data = {
        'D': D,
        'M': M,
        'X': [],
        'Y': [],
        'kernel_lengthscale': [],
        'kernel_variance': [],
        'noise_variance': [],
        'use_prior_mean_function': []
    }

    for traj_idx in traj_indices:
        X_i = X_trajs[traj_idx].astype(np.float64)  # (T, D)
        y_i = y_trajs[traj_idx].reshape(-1, 1).astype(np.float64)  # (T, 1)

        # 用GP拟合这条轨迹,获取kernel参数
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=[1.0] * int(D),
            length_scale_bounds=(1e-5, 1e5),
            nu=2.5,
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=3,
        )
        gp.fit(X_i, y_i.ravel())

        taf_data['X'].append(X_i)
        taf_data['Y'].append(y_i)
        taf_data['kernel_lengthscale'].append(gp.kernel_.k2.length_scale)
        taf_data['kernel_variance'].append(gp.kernel_.k1.constant_value)
        taf_data['noise_variance'].append(gp.alpha)
        taf_data['use_prior_mean_function'].append(False)

    # 保存为pickle
    os.makedirs(os.path.dirname(output_pickle_path), exist_ok=True)
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(taf_data, f)

    print(f"TAF source data saved to {output_pickle_path}")
    print(f"  M = {M} source tasks")


def _sample_variants_excluding(
    task,
    *,
    rng: np.random.Generator,
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
                f"Failed to sample a non-training evaluation variant for group={group_name} (too many collisions)."
            )
    return out
    print(f"  D = {D} dimensions")




# ==================== Candidate Generation ====================
def generate_mixed_candidates(
    lower: np.ndarray,
    upper: np.ndarray,
    X_context: np.ndarray,
    y_context: np.ndarray,
    step: int,
    rng: np.random.Generator,
    n_candidates: int = 128,
    n_global: int = 32,
    k_centers: int = 3,
    local_h: float = 2.25,
    local_h_decay: float = 0.9,
) -> np.ndarray:
    """
    生成混合候选点（与RL训练时一致）
    - n_global个Sobol点
    - 剩余点在当前最优点附近采样
    """
    X_context = np.asarray(X_context, dtype=np.float32)
    y_context = np.asarray(y_context, dtype=np.float32).reshape(-1)
    lower = np.asarray(lower, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)

    dim = int(lower.shape[0])
    n_total = int(n_candidates)
    n_global = int(max(0, min(int(n_global), n_total)))
    n_local = int(n_total - n_global)

    local_h_now = float(local_h) * (float(local_h_decay) ** int(step))

    # Global candidates (Sobol)
    X_global = np.empty((0, dim), dtype=np.float32)
    if n_global > 0:
        sobol_seed = int(rng.integers(0, 100000))
        sobol_sampler = Sobol(d=dim, scramble=True, seed=sobol_seed)
        sobol_samples = sobol_sampler.random(n_global)
        X_global = (sobol_samples * (upper - lower) + lower).astype(np.float32)

    # Local candidates (around best points)
    X_local = np.empty((0, dim), dtype=np.float32)
    if n_local > 0 and int(k_centers) > 0 and X_context.size > 0:
        k = int(k_centers)
        order = np.argsort(y_context)
        min_center_dist = max(1e-12, 2.0 * local_h_now)
        centers = []
        for idx in order:
            cand = X_context[idx]
            if all(np.max(np.abs(cand - c)) >= min_center_dist for c in centers):
                centers.append(cand)
            if len(centers) >= k:
                break
        if not centers:
            centers = [X_context[order[0]]]
        centers_available = np.asarray(centers, dtype=np.float32)

        base = n_local // k
        rem = n_local % k
        chunks = []
        for i in range(k):
            m = base + (1 if i < rem else 0)
            if m <= 0:
                continue
            center = centers_available[i] if i < centers_available.shape[0] else centers_available[0]
            deltas = rng.uniform(-local_h_now, local_h_now, size=(m, dim)).astype(np.float32)
            pts = center.reshape(1, dim) + deltas
            pts = np.clip(pts, lower, upper).astype(np.float32)
            chunks.append(pts)
        if chunks:
            X_local = np.vstack(chunks).astype(np.float32)

    X_candidates = np.vstack([X_global, X_local]).astype(np.float32)

    # Fill or trim
    if X_candidates.shape[0] < n_total:
        n_missing = n_total - X_candidates.shape[0]
        fill = rng.uniform(lower, upper, size=(n_missing, dim)).astype(np.float32)
        X_candidates = np.vstack([X_candidates, fill]).astype(np.float32)
    elif X_candidates.shape[0] > n_total:
        X_candidates = X_candidates[:n_total].astype(np.float32)

    # Shuffle
    perm = rng.permutation(n_total)
    return X_candidates[perm]


def generate_persistent_adaptive_candidates(
    lower: np.ndarray,
    upper: np.ndarray,
    X_context: np.ndarray,
    y_context: np.ndarray,
    step: int,
    rng: np.random.Generator,
    persistent_pool: np.ndarray,
    persistent_available: np.ndarray,
    n_total_candidates: int = 192,
    k_centers: int = 2,
    local_h: float = 1.5,
    local_h_decay: float = 0.9,
    explore_fraction: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    生成持久基底 + 自适应局部候选点（给 CAP-PPO 评估使用）。

    Args:
        explore_fraction: fraction of fresh candidates allocated to exploration
            around the point in persistent_pool farthest from all observations.
            0.0 = pure exploitation (backward compatible).

    Returns:
        X_candidates: (n_total_candidates, dim) 候选点
        is_persistent: (n_total_candidates,) float32 标记 (1=persistent, 0=fresh)
        n_persistent_in_candidates: 候选集中持久点的数量
    """
    X_context = np.asarray(X_context, dtype=np.float32)
    y_context = np.asarray(y_context, dtype=np.float32).reshape(-1)
    lower = np.asarray(lower, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)

    dim = int(lower.shape[0])

    # 获取可用持久点
    X_persistent = persistent_pool[persistent_available]
    n_pers = X_persistent.shape[0]

    # 计算局部补充数量
    n_fresh = n_total_candidates - n_pers
    explore_fraction = float(max(0.0, min(1.0, explore_fraction)))
    n_explore = int(n_fresh * explore_fraction)
    n_exploit = n_fresh - n_explore

    # 生成局部候选（exploitation: 围绕最佳观测点）
    local_h_now = float(local_h) * (float(local_h_decay) ** int(step))
    X_exploit = np.empty((0, dim), dtype=np.float32)
    if n_exploit > 0 and int(k_centers) > 0 and X_context.size > 0:
        k = int(k_centers)
        order = np.argsort(y_context)
        min_center_dist = max(1e-12, 2.0 * local_h_now)
        centers = []
        for idx in order:
            cand = X_context[idx]
            if all(np.max(np.abs(cand - c)) >= min_center_dist for c in centers):
                centers.append(cand)
            if len(centers) >= k:
                break
        if not centers:
            centers = [X_context[order[0]]]
        centers_available = np.asarray(centers, dtype=np.float32)

        base = n_exploit // k
        rem = n_exploit % k
        chunks = []
        for i in range(k):
            m = base + (1 if i < rem else 0)
            if m <= 0:
                continue
            center = centers_available[i] if i < centers_available.shape[0] else centers_available[0]
            deltas = rng.uniform(-local_h_now, local_h_now, size=(m, dim)).astype(np.float32)
            pts = center.reshape(1, dim) + deltas
            pts = np.clip(pts, lower, upper).astype(np.float32)
            chunks.append(pts)
        if chunks:
            X_exploit = np.vstack(chunks).astype(np.float32)

    # Fill or trim exploit to n_exploit
    if n_exploit > 0:
        if X_exploit.shape[0] < n_exploit:
            n_missing = n_exploit - X_exploit.shape[0]
            fill = rng.uniform(lower, upper, size=(n_missing, dim)).astype(np.float32)
            X_exploit = np.vstack([X_exploit, fill]).astype(np.float32) if X_exploit.shape[0] > 0 else fill
        elif X_exploit.shape[0] > n_exploit:
            X_exploit = X_exploit[:n_exploit].astype(np.float32)

    # 生成探索候选（exploration: 围绕距观测最远的持久点）
    X_explore = np.empty((0, dim), dtype=np.float32)
    if n_explore > 0 and X_context.size > 0:
        # 从整个持久池（包括已消耗的）中找距所有观测最远的点
        dists = np.abs(persistent_pool[:, None, :] - X_context[None, :, :]).max(axis=-1)  # (n_pool, n_ctx) L-inf
        min_dists = dists.min(axis=1)  # (n_pool,) 每个池点到最近观测的距离
        explore_center = persistent_pool[np.argmax(min_dists)].copy()

        deltas = rng.uniform(-local_h_now, local_h_now, size=(n_explore, dim)).astype(np.float32)
        X_explore = (explore_center.reshape(1, dim) + deltas).astype(np.float32)
        X_explore = np.clip(X_explore, lower, upper).astype(np.float32)

    # 合并
    X_local = np.vstack([X_exploit, X_explore]).astype(np.float32) if (X_exploit.shape[0] + X_explore.shape[0]) > 0 else np.empty((0, dim), dtype=np.float32)
    X_candidates = np.vstack([X_persistent, X_local]).astype(np.float32)

    # 来源标记
    is_persistent = np.concatenate([
        np.ones(n_pers, dtype=np.float32),
        np.zeros(n_fresh, dtype=np.float32),
    ])

    return X_candidates, is_persistent, n_pers


# ==================== Surrogate Models ====================
def get_gp_predictions(X_context: np.ndarray, y_context: np.ndarray,
                       X_candidates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, _SklearnGPModelTarget]:
    """使用GP进行预测"""
    input_dim = int(X_context.shape[1])
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=[1.0] * input_dim,
        length_scale_bounds=(1e-5, 1e5),
        nu=2.5,
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=3,
    )
    gp.fit(X_context, y_context)
    mean, std = gp.predict(X_candidates, return_std=True)
    return mean, std, _SklearnGPModelTarget(gp)


def get_tabpfn_predictions(regressor, X_context: np.ndarray, y_context: np.ndarray,
                           X_candidates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, _TabPFNModelTarget]:
    """使用TabPFN进行预测（带 Y 归一化）"""
    pred_mean, pred_std, full_out = predict_tabpfn_with_normalization(
        regressor, X_context, y_context, X_candidates
    )
    # 计算归一化统计量供 ModelTarget 后续预测使用
    y_ctx = np.asarray(y_context, dtype=np.float32).reshape(-1)
    y_mean = float(y_ctx.mean())
    y_std_val = float(max(y_ctx.std(), 1e-8))
    return pred_mean, pred_std, _TabPFNModelTarget(regressor, y_mean=y_mean, y_std=y_std_val)


# ==================== State Building ====================
def build_state_for_policies(X_candidates: np.ndarray, pred_mean: np.ndarray,
                             pred_std: np.ndarray, y_context: np.ndarray,
                             current_step: int, max_steps: int) -> torch.Tensor:
    """
    构建policies期望的state格式

    State格式: [posterior_mean, posterior_std, x1..xd, incumbent, timestep, budget]
    Feature order: ["posterior_mean", "posterior_std", x1..xd, "incumbent", "timestep", "budget"]
    """
    incumbent = y_context.min()
    n_candidates = len(X_candidates)
    dim = X_candidates.shape[1]

    # state shape: (n_candidates, 2 + dim + 1 + 2)
    state = np.zeros((n_candidates, 2 + dim + 3), dtype=np.float32)
    state[:, 0] = pred_mean  # posterior_mean
    state[:, 1] = pred_std   # posterior_std
    state[:, 2:2+dim] = X_candidates  # x coordinates
    state[:, 2+dim] = incumbent  # incumbent
    state[:, 2+dim+1] = current_step  # timestep
    state[:, 2+dim+2] = max_steps  # budget

    return torch.FloatTensor(state)




# ==================== Single BO Run ====================
def run_bo_with_policy(
    func,
    global_min: float,
    policy,
    policy_name: str,
    X_init: np.ndarray,
    y_init: np.ndarray,
    max_steps: int,
    surrogate_type: str,
    tabpfn_regressor=None,
    rng: np.random.Generator = None,
    n_candidates: int = 128,
    n_global: int = 32,
    k_centers: int = 3,
    local_h: float = 2.25,
    local_h_decay: float = 0.9,
    device: str = "cpu",
    return_trajectory: bool = False,
    return_trace: bool = False,
    n_persistent_base: int = 0,
    n_total_candidates: int = 0,
    taf_for_rl: "TAF | None" = None,
    explore_fraction: float = 0.0,
) -> List[float] | Tuple[List[float], np.ndarray] | Dict[str, np.ndarray | float]:
    """
    使用指定policy运行一次BO

    Args:
        func: 目标函数
        policy: 策略对象
        policy_name: 策略名称
        X_init, y_init: 初始观测
        max_steps: 最大步数
        surrogate_type: 'gp' or 'tabpfn'
        tabpfn_regressor: TabPFN回归器（如果使用TabPFN）
        rng: 随机数生成器
        n_candidates: 候选点数量（baseline 使用）
        n_persistent_base: 持久 Sobol 基底数量（CAP-PPO 使用）
        n_total_candidates: 总候选点数量（CAP-PPO 使用）
        其他: 候选点生成参数

    Returns:
        默认: regrets 列表
        若 return_trajectory=True: (regrets, trajectory_X)
        若 return_trace=True: 完整 run trace dict
    """
    # TuRBO runs its own complete BO loop (own GP + trust region + Thompson sampling)
    if isinstance(policy, TuRBOPolicy):
        return policy.run_full_bo(
            func=func,
            global_min=global_min,
            X_init=X_init,
            y_init=y_init,
            max_steps=max_steps,
            rng=rng,
            return_trajectory=return_trajectory,
            return_trace=return_trace,
        )

    X_context = X_init.copy()
    y_context = y_init.copy()
    lower, upper = func.bounds
    bounds = (lower, upper)

    use_persistent = isinstance(policy, RLPolicy) and n_persistent_base > 0 and n_total_candidates > 0

    # 对 CAP-PPO 初始化持久池
    if use_persistent:
        policy.reset_persistent_pool(lower, upper, rng)

    regrets = [float(y_context.min() - global_min)]
    trajectory = [X_context.copy()] if return_trajectory else None
    selected_xs = [] if return_trace else None
    selected_ys = [] if return_trace else None
    best_y_trace = [float(y_context.min())] if return_trace else None

    for step in range(max_steps):
        # 1. 生成候选点
        is_persistent = None
        n_pers_in_cand = 0
        if use_persistent:
            X_candidates, is_persistent, n_pers_in_cand = generate_persistent_adaptive_candidates(
                lower=lower,
                upper=upper,
                X_context=X_context,
                y_context=y_context,
                step=step,
                rng=rng,
                persistent_pool=policy.persistent_pool,
                persistent_available=policy.persistent_available,
                n_total_candidates=n_total_candidates,
                k_centers=k_centers,
                local_h=local_h,
                local_h_decay=local_h_decay,
                explore_fraction=explore_fraction,
            )
        else:
            X_candidates = generate_mixed_candidates(
                lower=lower,
                upper=upper,
                X_context=X_context,
                y_context=y_context,
                step=step,
                rng=rng,
                n_candidates=n_candidates,
                n_global=n_global,
                k_centers=k_centers,
                local_h=local_h,
                local_h_decay=local_h_decay,
            )

        # 2. 用surrogate预测
        if surrogate_type == 'gp':
            pred_mean, pred_std, model_target = get_gp_predictions(X_context, y_context, X_candidates)
        elif surrogate_type in {'tabpfn', 'tabpfn_base', 'tabpfn_tuned'}:
            pred_mean, pred_std, model_target = get_tabpfn_predictions(
                tabpfn_regressor, X_context, y_context, X_candidates
            )
        else:
            raise ValueError(f"Unknown surrogate_type: {surrogate_type}")

        # 3. 构建state
        state = build_state_for_policies(X_candidates, pred_mean, pred_std, y_context, step, max_steps)

        # 4. Policy选择动作
        if isinstance(policy, RLPolicy):
            # 如果模型需要 TAF 特征，计算 TAF ranking score
            taf_rank_norm = None
            if policy.use_taf_feature and taf_for_rl is not None:
                taf_scores = taf_for_rl.af(state.numpy(), X_context, model_target)
                ranks = np.argsort(np.argsort(-taf_scores)).astype(np.float32)
                n_cand = len(taf_scores)
                taf_rank_norm = ranks / max(n_cand - 1, 1)
            # RL需要额外的context信息（含 is_persistent 和 taf_rank_norm）
            policy.set_context(X_context, y_context, X_candidates, pred_mean, pred_std, bounds, step,
                             is_persistent=is_persistent, taf_rank_norm=taf_rank_norm)
            action, _ = policy.act(state)
        elif isinstance(policy, PFNs4BOPolicy):
            # PFNs4BO has its own surrogate; needs raw (X, y, candidates)
            policy.set_context(X_context, y_context, X_candidates, bounds)
            action, _ = policy.act(state, rng=rng)
        elif isinstance(policy, TAF):
            # TAF需要X_target和model_target
            action, _ = policy.act(state, X_context, model_target)
        else:
            # 其他policies (EI, UCB, PI, FunBO, Random)
            action, _ = policy.act(state, rng=rng)

        action = int(action.item())

        # 4.5 标记持久点已消耗
        if use_persistent:
            policy.consume_persistent_point(action, n_pers_in_cand)

        # 5. 评估真实函数值
        x_new = X_candidates[action:action+1]
        y_new = func(x_new)[0]

        # 6. 更新context
        X_context = np.vstack([X_context, x_new])
        y_context = np.concatenate([y_context, [y_new]])
        if trajectory is not None:
            trajectory.append(x_new.copy())
        if return_trace:
            selected_xs.append(x_new.copy())
            selected_ys.append(float(y_new))

        # 7. 记录regret
        best_y = y_context.min()
        regret = float(best_y - global_min)
        regret = max(regret, 0.0)  # 避免负值
        regrets.append(regret)
        if return_trace:
            best_y_trace.append(float(best_y))

    if return_trace:
        X_selected = (
            np.vstack(selected_xs).astype(np.float32)
            if selected_xs else np.empty((0, X_context.shape[1]), dtype=np.float32)
        )
        y_selected = np.asarray(selected_ys, dtype=np.float32) if selected_ys else np.empty((0,), dtype=np.float32)
        trace = {
            "X_init": np.asarray(X_init, dtype=np.float32),
            "y_init": np.asarray(y_init, dtype=np.float32).reshape(-1),
            "X_selected": X_selected,
            "y_selected": y_selected,
            "best_y_trace": np.asarray(best_y_trace, dtype=np.float32),
            "regret_trace": np.asarray(regrets, dtype=np.float32),
            "X_all": np.asarray(X_context, dtype=np.float32),
            "y_all": np.asarray(y_context, dtype=np.float32).reshape(-1),
            "global_min": float(global_min),
        }
        if trajectory is not None:
            trace["trajectory_X"] = np.vstack(trajectory).astype(np.float32)
        return trace
    if trajectory is not None:
        return regrets, np.vstack(trajectory)
    return regrets




# ==================== Evaluation Function ====================
def evaluate_policies_with_surrogate(
    surrogate_type: str,
    rl_model_path: str,
    tabpfn_model_path: str = None,
    taf_data_path: str = None,
    train_variants_path: Optional[str] = None,
    disallow_train_variants: bool = True,
    taf_rho: float = 1.0,
    task_name: str = "branin_family",
    n_variants_per_group: int = 10,
    n_runs_per_variant: int = 5,
    max_steps: int = 20,
    n_init: int = 2,
    n_candidates: int = 128,
    n_candidates_baseline: int = 2048,
    n_global: int = 32,
    k_centers: int = 3,
    local_h: float = 2.25,
    local_h_decay: float = 0.9,
    n_persistent_base: int = 128,
    n_total_candidates: int = 192,
    seed: int = 42,
    device: str = "cpu",
    save_dir: str = "./results_policies",
    hidden_dim: int = 128,
    n_self_attn_layers: int = 3,
    n_cross_attn_layers: int = 3,
    n_heads: int = 8,
    experiment_mode: str = "fair",
    plot_per_group: bool = True,
    plot_trajectories_enabled: bool = True,
    traj_grid_size: int = 80,
    traj_methods_csv: str = "CAP-PPO,EI,UCB,PI,TAF_me,TAF_ranking,Random",
    print_all_runs: bool = False,
    traj_variant_mode: str = "group_representative",
    traj_pick: str = "first",
    traj_pick_method: str = "CAP-PPO",
    save_trajectory_data: bool = True,
    regret_plot: str = "mean_bootstrap_95",
    save_eval_data: bool = True,
    metabo_logpath: Optional[str] = None,
    metabo_load_iter: Optional[int] = None,
):
    """
    使用指定的surrogate评估所有policies

    Args:
        surrogate_type: 'gp' | 'tabpfn_base' | 'tabpfn_tuned' (tabpfn is alias for tuned)
        rl_model_path: RL模型路径
        tabpfn_model_path: TabPFN模型路径
        taf_data_path: TAF历史数据路径
        experiment_mode: 'fair' (公平对比) or 'optimal' (上限对比)
        n_candidates: RL和Random的候选点数量
        n_candidates_baseline: baseline方法的候选点数量（仅在optimal模式下使用）
        其他参数: 实验配置

    Returns:
        all_group_results: 每个variant group的结果
    """
    os.makedirs(save_dir, exist_ok=True)

    task = get_task(task_name)
    dim = int(task.dim)
    # For dim>2, inline 2D trajectory plots are skipped (plot_trajectories returns None),
    # but trajectory capture + NPZ saving still works. Use plot_nd_trajectories.py to visualize.
    _plot_2d_trajectories = bool(plot_trajectories_enabled) and dim == 2

    # 根据experiment_mode配置每个策略的候选点数量
    # CAP-PPO 使用持久基底 + 自适应补充，其他 baseline 使用 generate_mixed_candidates
    _cap_ppo_config = {
        'n_candidates': n_total_candidates, 'n_global': 0, 'k_centers': k_centers,
        'n_persistent_base': n_persistent_base, 'n_total_candidates': n_total_candidates,
    }
    if experiment_mode == "fair":
        # 实验1：公平对比 - baseline 使用与 CAP-PPO 相同的总候选点数
        policy_configs = {
            'Random': {'n_candidates': n_total_candidates, 'n_global': n_global, 'k_centers': k_centers},
            'EI': {'n_candidates': n_total_candidates, 'n_global': n_global, 'k_centers': k_centers},
            'UCB': {'n_candidates': n_total_candidates, 'n_global': n_global, 'k_centers': k_centers},
            'PI': {'n_candidates': n_total_candidates, 'n_global': n_global, 'k_centers': k_centers},
            'TAF_me': {'n_candidates': n_total_candidates, 'n_global': n_global, 'k_centers': k_centers},
            'TAF_ranking': {'n_candidates': n_total_candidates, 'n_global': n_global, 'k_centers': k_centers},
            CAP_PPO_NAME: _cap_ppo_config,
        }
        mode_desc = f"Fair Comparison (all methods: {n_total_candidates} candidates)"
    elif experiment_mode == "optimal":
        # 实验2：上限对比 - CAP-PPO用持久+自适应, baseline用2048纯Sobol
        policy_configs = {
            'Random': {'n_candidates': n_total_candidates, 'n_global': n_global, 'k_centers': k_centers},
            'EI': {'n_candidates': n_candidates_baseline, 'n_global': n_candidates_baseline, 'k_centers': 0},
            'UCB': {'n_candidates': n_candidates_baseline, 'n_global': n_candidates_baseline, 'k_centers': 0},
            'PI': {'n_candidates': n_candidates_baseline, 'n_global': n_candidates_baseline, 'k_centers': 0},
            'TAF_me': {'n_candidates': n_candidates_baseline, 'n_global': n_candidates_baseline, 'k_centers': 0},
            'TAF_ranking': {'n_candidates': n_candidates_baseline, 'n_global': n_candidates_baseline, 'k_centers': 0},
            CAP_PPO_NAME: _cap_ppo_config,
        }
        mode_desc = f"Optimal Comparison ({CAP_PPO_NAME}: {n_total_candidates} persistent+adaptive, baselines: {n_candidates_baseline} pure Sobol)"
    else:
        raise ValueError(f"Unknown experiment_mode: {experiment_mode}. Must be 'fair' or 'optimal'.")

    use_metabo = bool(metabo_logpath)
    resolved_metabo_logpath: Optional[str] = None
    resolved_metabo_iter: Optional[int] = None
    if use_metabo:
        resolved_metabo_logpath = _resolve_metabo_logpath(str(metabo_logpath))
        resolved_metabo_iter = _resolve_metabo_load_iter(resolved_metabo_logpath, metabo_load_iter)
        policy_configs['MetaBO'] = {'n_candidates': n_candidates, 'n_global': n_global, 'k_centers': k_centers}

    print(f"\n{'='*80}")
    print(f"Evaluating policies with {surrogate_type.upper()} surrogate")
    print(f"Experiment Mode: {mode_desc}")
    print(f"{'='*80}")
    if task.task_name in FAMILY_TASKS:
        print(f"Variant groups: {list(task.default_variant_suite().keys())}")
    else:
        print("Variant groups: ['in_range']")
    print(f"Variants per group: {n_variants_per_group}")
    print(f"Runs per variant: {n_runs_per_variant}")
    print(f"Max steps: {max_steps}, Init: {n_init}")
    print(f"{CAP_PPO_NAME} model: {rl_model_path}")
    if surrogate_type in {"tabpfn", "tabpfn_base", "tabpfn_tuned"}:
        if surrogate_type == "tabpfn_base":
            print("TabPFN surrogate: base (pretrained)")
        else:
            print(f"TabPFN surrogate: tuned ckpt={tabpfn_model_path}")
    print(f"TAF data: {taf_data_path}")
    print(f"TAF ranking rho: {taf_rho}")
    print(f"\nPolicy Configurations:")
    for policy_name, config in policy_configs.items():
        extra = ""
        if 'n_persistent_base' in config:
            extra = f", persistent_base={config['n_persistent_base']}"
        print(f"  {policy_name}: n_candidates={config['n_candidates']}, "
              f"n_global={config.get('n_global', 0)}, k_centers={config['k_centers']}{extra}")
    print(f"{'='*80}\n")

    # 1. 初始化policies
    # Match state layout: [posterior_mean, posterior_std, x1..xd, incumbent, timestep, budget]
    x_feature_names = [f"x{i}" for i in range(int(task.dim))]
    feature_order = ["posterior_mean", "posterior_std"] + x_feature_names + ["incumbent", "timestep", "budget"]

    policies = {
        'Random': RandomPolicy(),
        'EI': EI(feature_order),
        'UCB': UCB(feature_order, kappa=2.0, D=int(task.dim), delta=0.1),
        'PI': PI(feature_order, xi=0.01),
        'TAF_me': TAF(taf_data_path, mode='me'),
        'TAF_ranking': TAF(taf_data_path, mode='ranking', rho=taf_rho),
        CAP_PPO_NAME: RLPolicy(
            model_path=rl_model_path,
            coord_dim=int(task.dim),
            hidden_dim=hidden_dim,
            n_self_attn_layers=n_self_attn_layers,
            n_cross_attn_layers=n_cross_attn_layers,
            n_heads=n_heads,
            max_steps=max_steps,
            device=device,
            n_persistent_base=n_persistent_base,
            n_total_candidates=n_total_candidates,
            k_centers=k_centers,
            local_h=local_h,
            local_h_decay=local_h_decay,
        ),
    }

    if use_metabo:
        policies['MetaBO'] = MetaBOPolicy(
            logpath=str(resolved_metabo_logpath),
            load_iter=int(resolved_metabo_iter),
            device=device,
            n_features=2 + int(task.dim) + 3,
        )

    print(f"Policies: {list(policies.keys())}\n")

    # 2. 初始化TabPFN regressor (如果使用TabPFN)
    tabpfn_regressor = None
    if surrogate_type in {"tabpfn", "tabpfn_base", "tabpfn_tuned"}:
        if surrogate_type == "tabpfn_base":
            tabpfn_regressor = TabPFNRegressor(
                device=device,
                n_estimators=1,
                random_state=42,
                inference_precision=torch.float32,
                ignore_pretraining_limits=True,
            )
        else:
            if tabpfn_model_path is None:
                raise ValueError("tabpfn_model_path is required for tabpfn_tuned/tabpfn")
            tabpfn_regressor = TabPFNRegressor(
                device=device,
                n_estimators=1,
                random_state=42,
                inference_precision=torch.float32,
                ignore_pretraining_limits=True,
                model_path=tabpfn_model_path,
            )
        print("TabPFN regressor initialized\n")

    # 3. 对每个variant group进行评估
    suite_rng = np.random.default_rng(seed)
    all_group_results = {}
    train_variant_keys: set = set()
    train_variants: List[Dict] = []
    if bool(disallow_train_variants) and task.task_name in FAMILY_TASKS:
        if not train_variants_path:
            raise ValueError(
                "train_variants_path is required when disallow_train_variants=True for variant-family tasks "
                f"({', '.join(sorted(FAMILY_TASKS))})"
            )
        train_variants = _load_variants_from_npz(str(train_variants_path))
        train_variant_keys = {_variant_key(v) for v in train_variants}
        print(f"Loaded {len(train_variants)} training variants from {train_variants_path} (will exclude from eval)")

    traj_methods = _normalize_method_list([m.strip() for m in str(traj_methods_csv).split(",") if m.strip()])
    if int(n_runs_per_variant) > 1 and not bool(print_all_runs):
        print("Note: only printing Run 1 per variant; other runs are executed and included in summaries/JSON. "
              "Use --print_all_runs to print every run.")

    suite_items: List[Tuple[str, List[Dict]]] = []
    if task.task_name in FAMILY_TASKS:
        suite_specs = task.default_variant_suite()
        used_eval_keys: set = set()
        for group_name, spec in suite_specs.items():
            if bool(disallow_train_variants) and train_variant_keys:
                variants = _sample_variants_excluding(
                    task,
                    rng=suite_rng,
                    group_name=str(group_name),
                    spec=spec,
                    n=int(n_variants_per_group),
                    forbidden=train_variant_keys,
                    used_eval=used_eval_keys,
                )
            else:
                variants = task.sample_eval_suite(
                    n_per_group=int(n_variants_per_group),
                    seed=int(suite_rng.integers(0, 1_000_000_000)),
                    suite_specs={str(group_name): spec},
                )[str(group_name)]
            suite_items.append((group_name, variants))
    else:
        suite = task.sample_eval_suite(n_per_group=int(n_variants_per_group), seed=int(seed))
        suite_items = list(suite.items())

    for item in suite_items:
        group_name, variants = item
        print(f"\n{'-'*80}")
        print(f"Group: {group_name}")
        if task.task_name in FAMILY_TASKS:
            spec = task.default_variant_suite()[group_name]
            print(f"Spec: {spec}")
            if bool(disallow_train_variants) and train_variant_keys:
                n_collide = sum(1 for v in variants if _variant_key(v) in train_variant_keys)
                if n_collide > 0:
                    raise RuntimeError(f"Eval suite contains {n_collide} training variants; exclusion failed.")
        print(f"{'-'*80}")

        n_variants = len(variants)

        # 初始化结果存储
        results = {
            policy_name: {
                'regrets': [],  # All runs flattened
                'regrets_by_variant': [[] for _ in range(n_variants)],
            }
            for policy_name in policies.keys()
        }
        results["__meta__"] = {
            "group_name": str(group_name),
            "variants": [dict(v) for v in variants],
            "global_mins": [None for _ in range(n_variants)],
            "run_seeds_by_variant": [[] for _ in range(n_variants)],
        }

        # 对每个变体进行评估
        group_rep_variant_params: Optional[Dict] = None
        group_rep_trajs: Optional[Dict[str, np.ndarray]] = None
        group_rep_run_index: Optional[int] = None
        group_rep_run_seed: Optional[int] = None
        for v_idx, variant_params in enumerate(variants):
            func = TaskVariantObjectiveFunction(task_name=task.task_name, variant_params=variant_params)
            global_min = float(task.estimate_global_min(variant_params))
            results["__meta__"]["global_mins"][v_idx] = float(global_min)
            lower, upper = func.bounds

            if task.task_name in {"branin_family", "goldstein_price_family"}:
                print(
                    f"\n  Variant {v_idx+1}/{n_variants}: "
                    f"dx1={variant_params['dx1']:.2f}, dx2={variant_params['dx2']:.2f}, "
                    f"rot={variant_params['rotation']:.1f}, "
                    f"sx1={variant_params['sx1']:.2f}, sx2={variant_params['sx2']:.2f}, "
                    f"global_min={global_min:.6f}"
                )
            elif task.task_name == "hartmann_3d_family":
                print(
                    f"\n  Variant {v_idx+1}/{n_variants}: "
                    f"dx=({variant_params['dx1']:.3f},{variant_params['dx2']:.3f},{variant_params['dx3']:.3f}), "
                    f"r=({variant_params['rx']:.1f},{variant_params['ry']:.1f},{variant_params['rz']:.1f}), "
                    f"s=({variant_params['sx1']:.2f},{variant_params['sx2']:.2f},{variant_params['sx3']:.2f}), "
                    f"global_min={global_min:.6f}"
                )
            else:
                print(f"\n  Variant {v_idx+1}/{n_variants}: params={variant_params}, global_min={global_min:.6f}")

            # 对每个变体运行多次
            capture_scope = bool(plot_trajectories_enabled) and (
                str(traj_variant_mode) == "each_variant"
                or (str(traj_variant_mode) == "group_representative" and v_idx == 0)
            )
            variant_trajs_by_run: List[Dict[str, np.ndarray]] = []
            variant_run_seeds: List[int] = []

            for run in range(n_runs_per_variant):
                run_seed = int(suite_rng.integers(0, 100000))
                results["__meta__"]["run_seeds_by_variant"][v_idx].append(int(run_seed))
                init_rng = np.random.default_rng(run_seed)

                # 生成相同的初始点
                X_init = init_rng.uniform(lower, upper, size=(n_init, func.dim)).astype(np.float32)
                y_init = func(X_init)

                run_trajs: Dict[str, np.ndarray] = {}

                # 对每个policy运行
                for policy_name, policy in policies.items():
                    run_rng = np.random.default_rng(run_seed)

                    # 使用该策略配置的候选点参数
                    policy_config = policy_configs[policy_name]

                    capture_traj = bool(capture_scope) and (policy_name in traj_methods)
                    # 如果是 CAP-PPO 且需要 TAF 特征，传入 TAF ranking 对象
                    _taf_for_rl = None
                    if isinstance(policy, RLPolicy) and policy.use_taf_feature:
                        _taf_for_rl = policies.get('TAF_ranking', None)
                    out = run_bo_with_policy(
                        func=func,
                        global_min=global_min,
                        policy=policy,
                        policy_name=policy_name,
                        X_init=X_init,
                        y_init=y_init,
                        max_steps=max_steps,
                        surrogate_type=surrogate_type,
                        tabpfn_regressor=tabpfn_regressor,
                        rng=run_rng,
                        n_candidates=policy_config['n_candidates'],
                        n_global=policy_config.get('n_global', 0),
                        k_centers=policy_config['k_centers'],
                        local_h=local_h,
                        local_h_decay=local_h_decay,
                        device=device,
                        return_trajectory=capture_traj,
                        n_persistent_base=policy_config.get('n_persistent_base', 0),
                        n_total_candidates=policy_config.get('n_total_candidates', 0),
                        taf_for_rl=_taf_for_rl,
                    )

                    if capture_traj:
                        regrets, traj = out
                        run_trajs[policy_name] = traj
                    else:
                        regrets = out

                    results[policy_name]['regrets'].append(regrets)
                    results[policy_name]['regrets_by_variant'][v_idx].append(regrets)

                if bool(print_all_runs) or run == 0:
                    print(f"    Run {run+1}: ", end="")
                    for policy_name in policies.keys():
                        final_regret = results[policy_name]['regrets_by_variant'][v_idx][run][-1]
                        print(f"{policy_name}={final_regret:.4f} ", end="")
                    print()

                if capture_scope:
                    variant_trajs_by_run.append(run_trajs)
                    variant_run_seeds.append(run_seed)

            if capture_scope and variant_trajs_by_run:
                chosen_run = _pick_representative_run_index(
                    results,
                    variant_index=v_idx,
                    n_runs=int(n_runs_per_variant),
                    pick=str(traj_pick),
                    pick_method=str(traj_pick_method),
                )
                chosen_run = int(max(0, min(chosen_run, len(variant_trajs_by_run) - 1)))
                chosen_trajs = variant_trajs_by_run[chosen_run]
                chosen_seed = variant_run_seeds[chosen_run] if chosen_run < len(variant_run_seeds) else -1

                if str(traj_variant_mode) == "each_variant":
                    variant_dir = os.path.join(save_dir, "groups", group_name, "variants", f"variant_{v_idx+1:02d}")
                    methods = [m for m in traj_methods if m in chosen_trajs]
                    if methods and _plot_2d_trajectories:
                        rep_func = TaskVariantObjectiveFunction(task_name=task.task_name, variant_params=variant_params)
                        traj_path = plot_trajectories(
                            rep_func,
                            chosen_trajs,
                            save_dir=variant_dir,
                            filename_prefix=(
                                f"trajectories_{surrogate_type}_{experiment_mode}_{group_name}_v{v_idx+1:02d}_run{chosen_run+1}"
                            ),
                            n_init=n_init,
                            methods=methods,
                            grid_size=traj_grid_size,
                        )
                        if traj_path:
                            print(f"Saved trajectories plot to {traj_path}")

                    if bool(save_trajectory_data) and chosen_trajs:
                        npz_path = os.path.join(
                            variant_dir,
                            f"trajectory_{surrogate_type}_{experiment_mode}_{group_name}_v{v_idx+1:02d}_run{chosen_run+1}.npz",
                        )
                        _save_trajectory_npz(
                            npz_path,
                            variant_params=variant_params,
                            run_index=chosen_run,
                            run_seed=chosen_seed,
                            n_init=n_init,
                            trajectories_by_method=chosen_trajs,
                        )
                else:
                    if v_idx == 0:
                        group_rep_variant_params = variant_params
                        group_rep_trajs = chosen_trajs
                        group_rep_run_index = chosen_run
                        group_rep_run_seed = chosen_seed
        # 保存group结果
        all_group_results[group_name] = results

        group_dir = os.path.join(save_dir, "groups", group_name)

        if plot_per_group:
            for rp in _resolve_regret_plot_modes(str(regret_plot)):
                plot_path = plot_group_comparison(
                    group_name,
                    results,
                    max_steps=max_steps,
                    save_dir=save_dir,
                    surrogate_type=surrogate_type,
                    experiment_mode=experiment_mode,
                    regret_plot=str(rp),
                )
                print(f"Saved group comparison plot to {plot_path}")

        if plot_trajectories_enabled and group_rep_variant_params is not None and group_rep_trajs is not None:
            rep_func = TaskVariantObjectiveFunction(task_name=task.task_name, variant_params=group_rep_variant_params)
            methods = [m for m in traj_methods if m in group_rep_trajs]
            if methods and _plot_2d_trajectories:
                traj_path = plot_trajectories(
                    rep_func,
                    group_rep_trajs,
                    save_dir=group_dir,
                    filename_prefix=f"trajectories_{surrogate_type}_{experiment_mode}_{group_name}_run{(group_rep_run_index or 0)+1}",
                    n_init=n_init,
                    methods=methods,
                    grid_size=traj_grid_size,
                )
                if traj_path:
                    print(f"Saved trajectories plot to {traj_path}")
            if bool(save_trajectory_data) and group_rep_trajs:
                npz_path = os.path.join(group_dir, f"trajectory_{surrogate_type}_{experiment_mode}_{group_name}.npz")
                _save_trajectory_npz(
                    npz_path,
                    variant_params=group_rep_variant_params,
                    run_index=int(group_rep_run_index or 0),
                    run_seed=int(group_rep_run_seed or -1),
                    n_init=n_init,
                    trajectories_by_method=group_rep_trajs,
                )

        # 打印group总结
        print(f"\n  Summary for {group_name}:")
        for policy_name, data in _iter_method_items(results):
            # 计算per-variant的平均最终regret
            per_variant_final = []
            for v_runs in data['regrets_by_variant']:
                arr = np.array(v_runs)
                per_variant_final.append(arr[:, -1].mean())

            mean_regret = np.mean(per_variant_final)
            std_regret = np.std(per_variant_final)
            print(f"    {policy_name}: {mean_regret:.6f} ± {std_regret:.6f}")

    # 4. 绘制总体对比图
    for rp in _resolve_regret_plot_modes(str(regret_plot)):
        plot_overall_comparison(
            all_group_results,
            max_steps,
            save_dir,
            surrogate_type,
            experiment_mode,
            regret_plot=str(rp),
        )

    # 5. 保存结果
    save_results(
        all_group_results,
        save_dir,
        surrogate_type,
        experiment_mode,
        extra_meta={
            "task": str(task.task_name),
            "surrogate_type": str(surrogate_type),
            "experiment_mode": str(experiment_mode),
            "seed": int(seed),
            "n_variants_per_group": int(n_variants_per_group),
            "n_runs_per_variant": int(n_runs_per_variant),
            "max_steps": int(max_steps),
            "n_init": int(n_init),
            "policy_configs": policy_configs,
            "tabpfn_model_path": tabpfn_model_path,
            "rl_model_path": rl_model_path,
            "taf_data_path": taf_data_path,
            "train_variants_path": train_variants_path,
            "disallow_train_variants": bool(disallow_train_variants),
            "regret_plot": str(regret_plot),
        },
        save_eval_data=bool(save_eval_data),
    )

    return all_group_results




# ==================== Plotting Functions ====================
def _compute_mean_rank_over_units(unit_curves_by_method: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """计算平均rank"""
    methods = list(unit_curves_by_method.keys())
    if not methods:
        raise ValueError("Empty unit_curves_by_method")

    first_shape = unit_curves_by_method[methods[0]].shape
    n_units, T = first_shape

    mean_ranks = {m: np.zeros(T, dtype=np.float64) for m in methods}
    for t in range(T):
        vals = np.stack([unit_curves_by_method[m][:, t] for m in methods], axis=1)
        ranks = np.apply_along_axis(lambda row: rankdata(row, method="average"), 1, vals)
        ranks_mean = ranks.mean(axis=0)
        for j, m in enumerate(methods):
            mean_ranks[m][t] = ranks_mean[j]

    return mean_ranks


def _bootstrap_mean_ci_over_units(
    unit_curves: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 2000,
    ci: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap置信区间"""
    unit_curves = np.asarray(unit_curves, dtype=np.float64)
    n_units, _ = unit_curves.shape
    mean = unit_curves.mean(axis=0)

    if n_units <= 1 or n_boot <= 0:
        return mean, mean, mean

    alpha = (1.0 - ci) / 2.0
    idx = rng.integers(0, n_units, size=(n_boot, n_units))
    boot_means = unit_curves[idx].mean(axis=1)
    lower = np.quantile(boot_means, alpha, axis=0)
    upper = np.quantile(boot_means, 1.0 - alpha, axis=0)

    return mean, lower, upper


def plot_overall_comparison(
    all_group_results: Dict,
    max_steps: int,
    save_dir: str,
    surrogate_type: str,
    experiment_mode: str = "fair",
    regret_plot: str = "mean_bootstrap_95",
):
    """
    绘制总体对比图
    左图: Mean Rank
    右图: Simple Regret (aggregated over all groups)
    """
    print(f"\nPlotting overall comparison for {surrogate_type} ({experiment_mode} mode)...")

    # 定义颜色
    colors = {
        'Random': 'tab:gray',
        'EI': 'tab:green',
        'UCB': 'tab:cyan',
        'PI': 'tab:olive',
        'TAF_me': 'tab:orange',
        'TAF_ranking': 'tab:pink',
        CAP_PPO_NAME: 'tab:blue',
        LEGACY_RL_NAME: 'tab:blue',
        'MetaBO': 'tab:purple',
    }

    # 聚合所有group的结果
    # 将每个group的每个variant视为一个"unit"
    first_group = list(all_group_results.values())[0]
    all_methods = _iter_method_names(first_group)
    unit_curves_by_method = {m: [] for m in all_methods}

    for group_name, results in all_group_results.items():
        for method, data in _iter_method_items(results):
            # regrets_by_variant: list of lists
            for v_runs in data['regrets_by_variant']:
                # v_runs: list of regret curves for this variant
                arr = np.array(v_runs)  # (n_runs, T)
                # 对runs取平均作为这个variant的unit curve
                unit_curve = arr.mean(axis=0)
                unit_curves_by_method[method].append(unit_curve)

    # 转换为numpy array
    for method in all_methods:
        unit_curves_by_method[method] = np.array(unit_curves_by_method[method])

    # 根据experiment_mode调整标题
    if experiment_mode == "fair":
        mode_label = "Fair (all 128 cand.)"
    else:
        mode_label = f"Optimal ({CAP_PPO_NAME} 128, others 2048)"
    title = f"Policy Comparison (all groups, {surrogate_type.upper()} surrogate, {mode_label})"
    suffix = "" if str(regret_plot) == "mean_bootstrap_95" else f"_{regret_plot}"
    save_path_png = _plot_rank_and_regret(
        unit_curves_by_method,
        max_steps=max_steps,
        save_dir=save_dir,
        filename_prefix=f"comparison_{surrogate_type}_{experiment_mode}{suffix}",
        title=title,
        colors=colors,
        rng_seed=0,
        n_boot=2000,
        regret_plot=str(regret_plot),
    )

    print(f"Saved comparison plots to {save_path_png}")


def _collect_capppo_unit_curves(all_group_results: Dict) -> np.ndarray:
    all_group_results = _normalize_all_group_results_keys(all_group_results)
    unit_curves: List[np.ndarray] = []
    for _, results in all_group_results.items():
        data = results.get(CAP_PPO_NAME, None)
        if data is None:
            raise KeyError(f"Missing {CAP_PPO_NAME} in results.")
        for v_runs in data["regrets_by_variant"]:
            arr = np.array(v_runs)
            unit_curves.append(arr.mean(axis=0))
    if not unit_curves:
        raise ValueError("No CAP-PPO curves found to plot.")
    return np.array(unit_curves)


def plot_capppo_surrogate_comparison(
    all_group_results_gp: Dict,
    all_group_results_tab: Dict,
    max_steps: int,
    save_dir: str,
    experiment_mode: str = "fair",
    regret_plot: str = "mean_bootstrap_95",
):
    """Compare CAP-PPO regret under GP vs tuned TabPFN surrogates."""
    os.makedirs(save_dir, exist_ok=True)
    unit_curves_gp = _collect_capppo_unit_curves(all_group_results_gp)
    unit_curves_tab = _collect_capppo_unit_curves(all_group_results_tab)

    if experiment_mode == "fair":
        mode_label = "Fair (all 128 cand.)"
    else:
        mode_label = f"Optimal ({CAP_PPO_NAME} 128, others 2048)"

    x = np.arange(max_steps + 1)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    rng = np.random.default_rng(0)

    def _plot_curve(unit_curves: np.ndarray, label: str, color: str):
        unit_curves = _sanitize_regrets_for_plot(unit_curves)
        if str(regret_plot) == "mean_bootstrap_95":
            center, lower, upper = _bootstrap_mean_ci_over_units(unit_curves, rng=rng, n_boot=2000, ci=0.95)
        elif str(regret_plot) == "median_30_70":
            center = np.median(unit_curves, axis=0)
            lower = np.quantile(unit_curves, 0.3, axis=0)
            upper = np.quantile(unit_curves, 0.7, axis=0)
        else:
            raise ValueError(f"Unknown regret_plot: {regret_plot}")
        center = _sanitize_regrets_for_plot(center)
        lower = _sanitize_regrets_for_plot(lower)
        upper = _sanitize_regrets_for_plot(upper)
        ax.plot(x, center, label=label, color=color, linewidth=2)
        ax.fill_between(x, lower, upper, color=color, alpha=0.2)

    _plot_curve(unit_curves_gp, f"{CAP_PPO_NAME} + GP", "tab:blue")
    _plot_curve(unit_curves_tab, f"{CAP_PPO_NAME} + TabPFN-tuned", "tab:orange")

    title = f"{CAP_PPO_NAME} Regret: GP vs TabPFN-tuned ({mode_label})"
    ax.set_title(title)
    ax.set_xlabel("Evaluation Steps")
    ax.set_ylabel("Simple Regret")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    suffix = "" if str(regret_plot) == "mean_bootstrap_95" else f"_{regret_plot}"
    save_path = os.path.join(save_dir, f"comparison_capppo_gp_vs_tabpfn_tuned_{experiment_mode}{suffix}.png")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved CAP-PPO surrogate comparison plot to {save_path}")


def save_results(
    all_group_results: Dict,
    save_dir: str,
    surrogate_type: str,
    experiment_mode: str = "fair",
    *,
    extra_meta: Optional[Dict] = None,
    save_eval_data: bool = True,
):
    """保存结果（含可复用的原始测试数据，用于后续重画图而无需重新评估）"""
    print(f"\nSaving results for {surrogate_type} ({experiment_mode} mode)...")

    meta = dict(extra_meta or {})
    meta.update({"surrogate_type": str(surrogate_type), "experiment_mode": str(experiment_mode)})

    # 保存为JSON（便于快速查看）
    results_json: Dict[str, object] = {"__meta__": meta, "groups": {}}
    for group_name, results in all_group_results.items():
        group_blob: Dict[str, object] = {"__meta__": results.get("__meta__", {})}
        methods_blob: Dict[str, object] = {}
        for method, data in _iter_method_items(results):
            methods_blob[method] = {
                "regrets": [list(r) for r in data["regrets"]],
                "regrets_by_variant": [[list(r) for r in v_runs] for v_runs in data["regrets_by_variant"]],
            }
        group_blob["methods"] = methods_blob
        results_json["groups"][group_name] = group_blob

    json_path = os.path.join(save_dir, f"results_{surrogate_type}_{experiment_mode}.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Results JSON saved to {json_path}")

    # 保存为 pkl（便于后续直接 load 后重画图/做二次分析）
    if bool(save_eval_data):
        pkl_path = os.path.join(save_dir, f"eval_data_{surrogate_type}_{experiment_mode}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({"__meta__": meta, "all_group_results": all_group_results}, f)
        print(f"Eval data (pkl) saved to {pkl_path}")




# ==================== Main Function ====================
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate different policies with GP and TabPFN surrogates"
    )

    parser.add_argument("--task", type=str, default="branin_family", help="Task name (see myrl.tasks.list_tasks())")

    # 模型路径
    parser.add_argument("--rl_model_path", type=str, required=True,
                       help=f"{CAP_PPO_NAME} model path (ppo_*.pt)")
    parser.add_argument("--tabpfn_model_path", type=str,
                       default=None,
                       help="TabPFN model path (required for tabpfn_tuned surrogate)")
    parser.add_argument("--taf_data_path", type=str,
                       default="./data/taf_source_data.pkl",
                       help="TAF source data path (will be created if not exists)")
    parser.add_argument("--bo_trajs_path", type=str,
                       default="./data/bo_trajs_train.npz",
                       help="BO trajectories for TAF preparation")
    parser.add_argument(
        "--train_variants_path",
        type=str,
        default=None,
        help="Path to the training variants cache (npz with key 'variants'); default: --bo_trajs_path",
    )
    parser.add_argument(
        "--disallow_train_variants",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Guarantee eval variants are not identical to training variants (branin_family only).",
    )
    parser.add_argument("--taf_rho", type=float, default=1.0,
                       help="TAF ranking-mode bandwidth rho (>0)")

    # 实验配置
    parser.add_argument("--n_variants_per_group", type=int, default=10,
                       help="Number of variants per group")
    parser.add_argument("--n_runs_per_variant", type=int, default=5,
                       help="Number of runs per variant")
    parser.add_argument("--max_steps", type=int, default=20,
                       help="Max BO steps")
    parser.add_argument("--n_init", type=int, default=2,
                       help="Number of initial points")

    # 候选点配置
    parser.add_argument("--experiment_mode", type=str, choices=['fair', 'optimal', 'both'],
                       default='both',
                       help=f"Experiment mode: 'fair' (all same candidates), 'optimal' ({CAP_PPO_NAME} persistent+adaptive, others 2048 Sobol), or 'both'")
    parser.add_argument("--n_persistent_base", type=int, default=128,
                       help=f"Number of persistent Sobol base candidates for {CAP_PPO_NAME}")
    parser.add_argument("--n_total_candidates", type=int, default=192,
                       help=f"Total candidates per step for {CAP_PPO_NAME} (persistent + adaptive local)")
    parser.add_argument("--n_candidates_baseline", type=int, default=2048,
                       help="Number of candidates for baselines in 'optimal' mode")
    parser.add_argument("--n_global", type=int, default=32,
                       help="Number of global Sobol candidates (for baselines)")
    parser.add_argument("--k_centers", type=int, default=2,
                       help="Number of local centers")
    parser.add_argument("--local_h", type=float, default=1.5,
                       help="Local sampling radius")
    parser.add_argument("--local_h_decay", type=float, default=0.9,
                       help="Local radius decay")

    # RL网络参数
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_self_attn_layers", type=int, default=3)
    parser.add_argument("--n_cross_attn_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=8)

    # 其他
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./results_policies",
                       help="Save directory")
    parser.add_argument(
        "--surrogate",
        type=str,
        choices=["gp", "tabpfn", "tabpfn_base", "tabpfn_tuned", "both", "compare"],
        default="both",
        help="Which surrogate to use: gp, base TabPFN, tuned TabPFN, or run multiple.",
    )
    parser.add_argument(
        "--regret_plot",
        type=str,
        choices=["mean_bootstrap_95", "median_30_70", "both"],
        default="mean_bootstrap_95",
        help="How to plot regret curves: bootstrap mean CI, median with 30–70% band, or both.",
    )
    parser.add_argument(
        "--save_eval_data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a .pkl with raw eval data for later re-plotting without reruns.",
    )
    parser.add_argument(
        "--load_eval_data",
        type=str,
        default=None,
        help="Load a saved eval_data_*.pkl and only re-plot (no evaluation).",
    )
    parser.add_argument("--plot_per_group", action=argparse.BooleanOptionalAction, default=True,
                       help="Whether to save per-group comparison plots")
    parser.add_argument("--plot_trajectories", action=argparse.BooleanOptionalAction, default=True,
                       help="Whether to save a representative trajectory plot per group")
    parser.add_argument("--traj_grid_size", type=int, default=80,
                       help="Grid size for trajectory contour plot (higher is slower)")
    parser.add_argument("--traj_methods", type=str, default="CAP-PPO,EI,UCB,PI,TAF_me,TAF_ranking,Random",
                       help="Comma-separated methods to show in trajectory plot")
    parser.add_argument("--print_all_runs", action=argparse.BooleanOptionalAction, default=False,
                       help="Print final regret for every run (otherwise only prints Run 1 per variant)")
    parser.add_argument("--traj_variant_mode", type=str, choices=["group_representative", "each_variant"],
                       default="group_representative",
                       help="Save trajectories for one representative variant per group, or for each variant")
    parser.add_argument("--traj_pick", type=str, choices=["first", "best", "median"],
                       default="first",
                       help="How to pick 1 run (among n_runs_per_variant) to save a trajectory for plotting")
    parser.add_argument("--traj_pick_method", type=str, default="CAP-PPO",
                       help=f"Method used to rank runs when traj_pick is 'best' or 'median' (e.g., {CAP_PPO_NAME}, EI)")
    parser.add_argument("--save_trajectory_data", action=argparse.BooleanOptionalAction, default=True,
                       help="Whether to also save selected trajectories as .npz files")
    parser.add_argument(
        "--metabo_logpath",
        type=str,
        default=None,
        help="Optional: MetaBO log directory. Can be a run dir (contains weights_*/params_*), or an env dir with LATEST.",
    )
    parser.add_argument(
        "--metabo_load_iter",
        type=int,
        default=None,
        help="Optional: Which PPO iteration to load for MetaBO policy (default: auto-pick latest available).",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.load_eval_data:
        with open(args.load_eval_data, "rb") as f:
            blob = pickle.load(f)
        all_group_results = _normalize_all_group_results_keys(blob.get("all_group_results", {}) or {})
        meta = blob.get("__meta__", {}) or {}
        surrogate_type = str(meta.get("surrogate_type", "loaded"))
        experiment_mode = str(meta.get("experiment_mode", "loaded"))
        save_dir = str(args.save_dir)
        os.makedirs(save_dir, exist_ok=True)
        max_steps = int(meta.get("max_steps", args.max_steps))
        for rp in _resolve_regret_plot_modes(str(args.regret_plot)):
            if bool(args.plot_per_group):
                for group_name, results in all_group_results.items():
                    plot_group_comparison(
                        group_name,
                        results,
                        max_steps=max_steps,
                        save_dir=save_dir,
                        surrogate_type=surrogate_type,
                        experiment_mode=experiment_mode,
                        regret_plot=str(rp),
                    )
            plot_overall_comparison(
                all_group_results,
                max_steps,
                save_dir,
                surrogate_type,
                experiment_mode,
                regret_plot=str(rp),
            )
        print(f"Re-plot completed. Outputs in {save_dir}")
        return

    # 准备TAF数据
    if not os.path.exists(args.taf_data_path):
        prepare_taf_data(args.bo_trajs_path, args.taf_data_path)

    # 确定要运行的实验模式
    if args.experiment_mode == 'both':
        experiment_modes = ['fair', 'optimal']
    else:
        experiment_modes = [args.experiment_mode]

    # 确定要运行的代理模型
    surrogate_choice = str(args.surrogate)
    if surrogate_choice == "both":
        surrogates = ["gp", "tabpfn_tuned"]
    elif surrogate_choice == "compare":
        surrogates = ["gp", "tabpfn_base", "tabpfn_tuned"]
    elif surrogate_choice == "tabpfn":
        surrogates = ["tabpfn_tuned"]
    else:
        surrogates = [surrogate_choice]

    # 运行评估
    results_by_mode: Dict[str, Dict[str, Dict]] = {mode: {} for mode in experiment_modes}
    for surrogate_type in surrogates:
        for exp_mode in experiment_modes:
            save_dir = os.path.join(args.save_dir, surrogate_type, exp_mode)

            all_group_results = evaluate_policies_with_surrogate(
                surrogate_type=surrogate_type,
                rl_model_path=args.rl_model_path,
                tabpfn_model_path=args.tabpfn_model_path if surrogate_type in {"tabpfn", "tabpfn_tuned"} else None,
                taf_data_path=args.taf_data_path,
                train_variants_path=str(args.train_variants_path or args.bo_trajs_path),
                disallow_train_variants=bool(args.disallow_train_variants),
                taf_rho=args.taf_rho,
                task_name=args.task,
                n_variants_per_group=args.n_variants_per_group,
                n_runs_per_variant=args.n_runs_per_variant,
                max_steps=args.max_steps,
                n_init=args.n_init,
                n_candidates=args.n_total_candidates,
                n_candidates_baseline=args.n_candidates_baseline,
                n_global=args.n_global,
                k_centers=args.k_centers,
                local_h=args.local_h,
                local_h_decay=args.local_h_decay,
                n_persistent_base=args.n_persistent_base,
                n_total_candidates=args.n_total_candidates,
                seed=args.seed,
                device=device,
                save_dir=save_dir,
                hidden_dim=args.hidden_dim,
                n_self_attn_layers=args.n_self_attn_layers,
                n_cross_attn_layers=args.n_cross_attn_layers,
                n_heads=args.n_heads,
                experiment_mode=exp_mode,
                plot_per_group=args.plot_per_group,
                plot_trajectories_enabled=args.plot_trajectories,
                traj_grid_size=args.traj_grid_size,
                traj_methods_csv=args.traj_methods,
                print_all_runs=args.print_all_runs,
                traj_variant_mode=args.traj_variant_mode,
                traj_pick=args.traj_pick,
                traj_pick_method=args.traj_pick_method,
                save_trajectory_data=args.save_trajectory_data,
                regret_plot=str(args.regret_plot),
                save_eval_data=bool(args.save_eval_data),
                metabo_logpath=args.metabo_logpath,
                metabo_load_iter=args.metabo_load_iter,
            )
            results_by_mode[str(exp_mode)][str(surrogate_type)] = all_group_results

    # Cross-surrogate CAP-PPO regret comparison (GP vs tuned TabPFN)
    for exp_mode in experiment_modes:
        mode_results = results_by_mode.get(str(exp_mode), {})
        if "gp" in mode_results and "tabpfn_tuned" in mode_results:
            compare_dir = os.path.join(args.save_dir, "compare_surrogates", str(exp_mode))
            for rp in _resolve_regret_plot_modes(str(args.regret_plot)):
                plot_capppo_surrogate_comparison(
                    mode_results["gp"],
                    mode_results["tabpfn_tuned"],
                    max_steps=int(args.max_steps),
                    save_dir=compare_dir,
                    experiment_mode=str(exp_mode),
                    regret_plot=str(rp),
                )

    print("\n" + "="*80)
    print("Evaluation completed!")
    print(f"Results saved to {args.save_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
