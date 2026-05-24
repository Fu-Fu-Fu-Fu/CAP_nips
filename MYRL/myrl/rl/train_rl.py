"""
改进版 PPO 训练 - 解决 RL 学不好的问题

改进点:
1. 更丰富的特征：加入 TAF_me、当前步数等
2. 更好的奖励设计：加入 shaping reward
3. 更大的训练规模
4. 更好的探索策略
5. 默认使用 finetune.py 预生成的 BO 轨迹缓存重构 oracle GP，与微调数据源一致

本项目中该策略在论文/图表中记为：CAP-PPO（Candidate Acquisition Policy trained with PPO）。
"""
import os
import math
import json
import warnings
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import logging
from torch.distributions import Categorical
from typing import Dict, Any, Optional, Tuple, List
from scipy.stats.qmc import Sobol
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

warnings.filterwarnings('ignore', message='The balance properties of Sobol')
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")
warnings.filterwarnings("ignore", message=".*ABNORMAL_TERMINATION_IN_LNSRCH.*")
warnings.filterwarnings("ignore", message=".*scale the data.*")
warnings.filterwarnings("ignore", message=".*optimal value found.*close to the specified.*bound.*")
logging.getLogger("posthog").setLevel(logging.CRITICAL)
logging.getLogger("analytics").setLevel(logging.CRITICAL)
logging.getLogger("posthog.analytics").setLevel(logging.CRITICAL)

from ..bo.select_candidates import (
    SelectionConfig,
    ObjectiveFunction,
    compute_std_from_tabpfn_output,
    predict_tabpfn_with_normalization,
)
from ..tasks import get_task
from ..policies.policies import TAF
from tabpfn import TabPFNRegressor
from scipy.optimize import minimize


# 定义域中心（用于旋转变换）
CENTER_X1 = 2.5  # (-5 + 10) / 2
CENTER_X2 = 7.5  # (0 + 15) / 2


# ==================== Branin 变体函数 (与 finetune.py 保持一致) ====================
def branin_family_torch(
    x: torch.Tensor,
    dx1: float = 0.0, dx2: float = 0.0,
    sx1: float = 1.0, sx2: float = 1.0,
    alpha: float = 1.0, beta: float = 0.0,
    rotation: float = 0.0,
    a: float = 1.0, b: float = 5.1 / (4.0 * np.pi**2),
    c: float = 5.0 / np.pi, r: float = 6.0,
    s: float = 10.0, t: float = 1.0 / (8.0 * np.pi),
) -> torch.Tensor:
    x1 = x[..., 0]
    x2 = x[..., 1]

    # 旋转变换（围绕定义域中心）
    if abs(rotation) > 1e-8:
        cx1 = CENTER_X1  # 2.5
        cx2 = CENTER_X2  # 7.5
        theta = rotation * np.pi / 180.0
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        x1_centered = x1 - cx1
        x2_centered = x2 - cx2
        x1_rot = cos_theta * x1_centered - sin_theta * x2_centered + cx1
        x2_rot = sin_theta * x1_centered + cos_theta * x2_centered + cx2
    else:
        x1_rot = x1
        x2_rot = x2

    x1_t = sx1 * x1_rot + dx1
    x2_t = sx2 * x2_rot + dx2
    y = a * (x2_t - b * x1_t**2 + c * x1_t - r) ** 2 + s * (1.0 - t) * torch.cos(x1_t) + s
    return alpha * y + beta


def branin_family_numpy(X: np.ndarray, variant_params: dict, device: str = "cpu") -> np.ndarray:
    """计算 Branin 变体函数值"""
    x_tensor = torch.from_numpy(X).float().to(device)
    with torch.no_grad():
        y = branin_family_torch(x_tensor, **variant_params).cpu().numpy()
    return y.astype(np.float32)


# ==================== TAF (me) helper ====================
def prepare_taf_data(bo_trajs_path: str, output_pickle_path: str):
    """
    将 bo_trajs 缓存转换为 TAF 期望的格式（与 eval_rl_new.py 对齐）。
    """
    data = np.load(bo_trajs_path, allow_pickle=True)
    X_trajs = np.asarray(data["X_trajs"])  # (n_traj, T, D)
    y_trajs = np.asarray(data["y_trajs"])  # (n_traj, T)
    if "variant_indices" not in data:
        raise KeyError(f"bo_trajs_path missing key 'variant_indices': {bo_trajs_path}")
    variant_indices = np.asarray(data["variant_indices"], dtype=np.int64).reshape(-1)

    if X_trajs.ndim != 3:
        raise ValueError(f"Invalid X_trajs shape: {X_trajs.shape}")
    D = int(X_trajs.shape[2])

    variant_ids = np.sort(np.unique(variant_indices))
    M = int(len(variant_ids))
    traj_indices = []
    for v in variant_ids.tolist():
        idxs = np.where(variant_indices == int(v))[0]
        if idxs.size == 0:
            continue
        traj_indices.append(int(idxs.min()))

    taf_data = {
        "D": D,
        "M": M,
        "X": [],
        "Y": [],
        "kernel_lengthscale": [],
        "kernel_variance": [],
        "noise_variance": [],
        "use_prior_mean_function": [],
    }

    for traj_idx in traj_indices:
        X_i = X_trajs[traj_idx].astype(np.float64)  # (T, D)
        y_i = y_trajs[traj_idx].reshape(-1, 1).astype(np.float64)  # (T, 1)

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

        taf_data["X"].append(X_i)
        taf_data["Y"].append(y_i)
        taf_data["kernel_lengthscale"].append(gp.kernel_.k2.length_scale)
        taf_data["kernel_variance"].append(gp.kernel_.k1.constant_value)
        taf_data["noise_variance"].append(gp.alpha)
        taf_data["use_prior_mean_function"].append(False)

    os.makedirs(os.path.dirname(output_pickle_path), exist_ok=True)
    with open(output_pickle_path, "wb") as f:
        pickle.dump(taf_data, f)


def build_state_for_policies(
    X_candidates: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    y_context: np.ndarray,
    current_step: int,
    max_steps: int,
) -> np.ndarray:
    """
    构建 policies 期望的 state（与 eval_rl_new.py 对齐）。
    State: [posterior_mean, posterior_std, x..., incumbent, timestep, budget]
    """
    incumbent = y_context.min()
    n_candidates = len(X_candidates)
    dim = X_candidates.shape[1]

    state = np.zeros((n_candidates, 2 + dim + 3), dtype=np.float32)
    state[:, 0] = pred_mean
    state[:, 1] = pred_std
    state[:, 2:2 + dim] = X_candidates
    state[:, 2 + dim] = incumbent
    state[:, 2 + dim + 1] = current_step
    state[:, 2 + dim + 2] = max_steps
    return state


class TAFMeHelper:
    def __init__(self, taf_data_path: str, max_steps: int):
        self.taf_policy = TAF(taf_data_path, mode="me")
        self.max_steps = int(max_steps)

    def compute(
        self,
        X_candidates: np.ndarray,
        pred_mean: np.ndarray,
        pred_std: np.ndarray,
        X_context: np.ndarray,
        y_context: np.ndarray,
        current_step: int,
    ) -> np.ndarray:
        state = build_state_for_policies(
            X_candidates, pred_mean, pred_std, y_context, current_step, self.max_steps
        )
        taf_values = self.taf_policy.af(state, X_context, model_target=None)
        return np.asarray(taf_values, dtype=np.float32)


class BraninVariantFunction(ObjectiveFunction):
    """Branin 变体函数"""

    def __init__(self, variant_params: dict, device: str = "cpu"):
        self.variant_params = variant_params
        self._device = device
        self._lower = np.array([-5.0, 0.0], dtype=np.float32)
        self._upper = np.array([10.0, 15.0], dtype=np.float32)
        self._global_min = None

    @property
    def dim(self) -> int:
        return 2

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._lower, self._upper

    @property
    def optimal_value(self) -> Optional[float]:
        if self._global_min is None:
            self._global_min = self._find_global_min()
        return self._global_min

    def _find_global_min(self) -> float:
        """搜索全局最小值"""
        x1 = np.linspace(-5, 10, 100)
        x2 = np.linspace(0, 15, 100)
        X1, X2 = np.meshgrid(x1, x2)
        grid_points = np.stack([X1.flatten(), X2.flatten()], axis=1).astype(np.float32)
        y_grid = self(grid_points)

        sorted_indices = np.argsort(y_grid)
        n_starts = 20
        start_points = grid_points[sorted_indices[:n_starts]]

        def func(x):
            x = np.array(x, dtype=np.float32).reshape(1, 2)
            return float(self(x)[0])

        bounds = [(-5, 10), (0, 15)]
        best_min = float('inf')

        for x0 in start_points:
            try:
                res = minimize(func, x0, bounds=bounds, method="L-BFGS-B")
                if res.fun < best_min:
                    best_min = res.fun
            except:
                pass

        best_min = min(best_min, float(y_grid.min()))
        return best_min

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X).astype(np.float32)
        return branin_family_numpy(X, self.variant_params, self._device)


# ==================== Oracle GP objective (fit from finetune.py BO trajectories) ====================
def make_oracle_gp_model(
    input_dim: int = 2,
    n_restarts_optimizer: int = 3,
    fixed_length_scale: Optional[float] = None,
    alpha: float = 1e-6,
) -> GaussianProcessRegressor:
    """
    Create the same GP configuration as finetune.py uses for BO + oracle fitting.
    """
    if fixed_length_scale is not None and float(fixed_length_scale) > 0:
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
            length_scale=[float(fixed_length_scale)] * input_dim,
            length_scale_bounds="fixed",
            nu=2.5,
        )
        optimizer = None
        n_restarts_optimizer = 0
    else:
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=[1.0] * input_dim,
            length_scale_bounds=(1e-5, 1e5),
            nu=2.5,
        )
        optimizer = "fmin_l_bfgs_b"
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=float(alpha),
        normalize_y=True,
        optimizer=optimizer,
        n_restarts_optimizer=n_restarts_optimizer,
    )


def _sample_from_oracle_gp(
    oracle_gp: GaussianProcessRegressor,
    X: np.ndarray,
    rng: np.random.Generator,
    *,
    min_std: float = 1e-12,
) -> np.ndarray:
    mu, std = oracle_gp.predict(np.asarray(X, dtype=np.float64), return_std=True)
    std = np.maximum(np.asarray(std, dtype=np.float64), float(min_std))
    y = rng.normal(loc=np.asarray(mu, dtype=np.float64), scale=std)
    return np.asarray(y, dtype=np.float32).reshape(-1)


def _apply_y_transform(y: np.ndarray, transform: str) -> np.ndarray:
    """Apply a monotonic transform to y values before GP fitting.

    Supported transforms:
      - "none": identity (no transform)
      - "sqrt": sqrt(y - y_min + 1.0)  — compresses large y-ranges
      - "log":  log(y - y_min + 1.0)   — stronger compression
    """
    if transform == "none":
        return y
    y = np.asarray(y, dtype=np.float64).ravel()
    y_min = float(y.min())
    y_shifted = y - y_min + 1.0
    if transform == "sqrt":
        return np.sqrt(y_shifted)
    elif transform == "log":
        return np.log(y_shifted)
    else:
        raise ValueError(f"Unknown y_transform={transform!r}. Use 'none', 'sqrt', or 'log'.")


def load_oracle_gps_from_trajectories_cache(
    trajectories_path: str,
    *,
    n_restarts_optimizer: int = 3,
    y_transform: str = "none",
    fixed_length_scale: Optional[float] = None,
    alpha: float = 1e-6,
) -> Tuple[List[GaussianProcessRegressor], List[Dict[str, float]]]:
    """
    Reconstruct the per-variant oracle GPs from finetune.py trajectory cache.

    Expected cache format: the npz saved by finetune.py:load_or_generate_bo_trajectories,
    containing keys: X_trajs, y_trajs, variant_indices, variants.

    For each variant, we fit an oracle GP on the first trajectory index of that variant
    (finetune.py writes the selected best-of BO trajectory first, then synthetic ones).
    """
    if not os.path.exists(trajectories_path):
        raise FileNotFoundError(
            f"Trajectory cache not found: {trajectories_path}. "
            "Run finetune.py with --stage generate/all to create it."
        )

    data = np.load(trajectories_path, allow_pickle=True)
    for k in ("X_trajs", "y_trajs", "variant_indices"):
        if k not in data:
            raise KeyError(f"Invalid trajectories cache (missing '{k}'): {trajectories_path}")

    X_trajs = np.asarray(data["X_trajs"])
    y_trajs = np.asarray(data["y_trajs"])
    variant_indices = np.asarray(data["variant_indices"], dtype=np.int64)

    variants: List[Dict[str, float]] = []
    if "variants" in data:
        variants = data["variants"].tolist()

    variant_ids = np.unique(variant_indices)
    variant_ids = np.sort(variant_ids)

    oracle_gps: List[GaussianProcessRegressor] = []
    for v in variant_ids:
        idxs = np.where(variant_indices == v)[0]
        if idxs.size == 0:
            raise ValueError(f"Variant {v} has no trajectories in cache: {trajectories_path}")

        traj_idx = int(idxs.min())
        X = np.asarray(X_trajs[traj_idx], dtype=np.float64)
        y = np.asarray(y_trajs[traj_idx], dtype=np.float64).reshape(-1)

        y_fit = _apply_y_transform(y, y_transform)
        gp = make_oracle_gp_model(
            input_dim=X.shape[1],
            n_restarts_optimizer=n_restarts_optimizer,
            fixed_length_scale=fixed_length_scale,
            alpha=alpha,
        )
        gp.fit(X, y_fit)
        oracle_gps.append(gp)

    if y_transform != "none":
        print(f"  [oracle_gp] Applied y_transform={y_transform!r} before GP fitting")
    if fixed_length_scale is not None and float(fixed_length_scale) > 0:
        print(
            f"  [oracle_gp] Fixed length_scale={float(fixed_length_scale):.6g}, "
            f"alpha={float(alpha):.3g}, optimizer=None"
        )

    if variants and len(variants) != len(oracle_gps):
        print(
            f"[WARN] variants count ({len(variants)}) != oracle_gps ({len(oracle_gps)}). "
            "Proceeding with oracle_gps only."
        )
        variants = []

    return oracle_gps, variants


def estimate_gp_mean_min_on_grid(
    oracle_gp: GaussianProcessRegressor,
    bounds: Tuple[np.ndarray, np.ndarray],
    *,
    grid_size: int = 80,
) -> float:
    lower, upper = bounds
    dim = int(np.asarray(lower).shape[0])
    if dim == 2:
        x1 = np.linspace(float(lower[0]), float(upper[0]), int(grid_size), dtype=np.float64)
        x2 = np.linspace(float(lower[1]), float(upper[1]), int(grid_size), dtype=np.float64)
        X1, X2 = np.meshgrid(x1, x2)
        X_cand = np.stack([X1.ravel(), X2.ravel()], axis=1).astype(np.float64)
    else:
        # Fall back to a Sobol-based search for dim != 2.
        # Keep the total point budget roughly comparable to a 2D grid.
        n = int(grid_size) ** 2
        sobol_sampler = Sobol(d=dim, scramble=True, seed=0)
        X_unit = sobol_sampler.random(n)
        X_cand = (X_unit * (upper - lower) + lower).astype(np.float64)

    mu = oracle_gp.predict(X_cand).reshape(-1)
    best_min = float(np.min(mu))

    # Multi-start local refinement on the mean surface mu(x).
    n_starts = int(min(25, len(mu)))
    start_points = X_cand[np.argsort(mu)[:n_starts]]
    bounds_list = [(float(lower[i]), float(upper[i])) for i in range(dim)]

    def func(x):
        x = np.asarray(x, dtype=np.float64).reshape(1, dim)
        return float(oracle_gp.predict(x).reshape(-1)[0])

    for x0 in start_points:
        try:
            res = minimize(func, np.asarray(x0, dtype=np.float64), bounds=bounds_list, method="L-BFGS-B")
            best_min = min(best_min, float(res.fun))
        except Exception:
            pass

    return float(best_min)


def estimate_objective_min_on_grid(
    objective: ObjectiveFunction,
    bounds: Tuple[np.ndarray, np.ndarray],
    *,
    grid_size: int = 80,
    extra_points: Optional[np.ndarray] = None,
    sobol_seed: int = 0,
    n_lbfgs_starts: int = 25,
) -> float:
    """
    Estimate the (approximate) global minimum value of a deterministic objective function.

    - dim==2: dense grid + multi-start local refinement
    - dim!=2: Sobol search (grid_size^2 points) + multi-start local refinement

    extra_points (e.g., initial context points) are included to ensure the estimate is
    no worse than values we have already observed in the episode.
    """
    lower, upper = bounds
    lower = np.asarray(lower, dtype=np.float64).reshape(-1)
    upper = np.asarray(upper, dtype=np.float64).reshape(-1)
    dim = int(lower.shape[0])

    if dim == 2:
        x1 = np.linspace(float(lower[0]), float(upper[0]), int(grid_size), dtype=np.float64)
        x2 = np.linspace(float(lower[1]), float(upper[1]), int(grid_size), dtype=np.float64)
        X1, X2 = np.meshgrid(x1, x2)
        X_cand = np.stack([X1.ravel(), X2.ravel()], axis=1).astype(np.float64)
    else:
        n = int(grid_size) ** 2
        sobol_sampler = Sobol(d=dim, scramble=True, seed=int(sobol_seed))
        X_unit = sobol_sampler.random(n)
        X_cand = (X_unit * (upper - lower) + lower).astype(np.float64)

    if extra_points is not None:
        extra_points = np.asarray(extra_points, dtype=np.float64).reshape(-1, dim)
        if extra_points.size > 0:
            X_cand = np.vstack([X_cand, extra_points]).astype(np.float64)

    y = np.asarray(objective(X_cand), dtype=np.float64).reshape(-1)
    best_min = float(np.min(y))

    n_starts = int(min(n_lbfgs_starts, len(y)))
    start_points = X_cand[np.argsort(y)[:n_starts]]
    bounds_list = [(float(lower[i]), float(upper[i])) for i in range(dim)]

    def func(x):
        x = np.asarray(x, dtype=np.float64).reshape(1, dim)
        return float(np.asarray(objective(x), dtype=np.float64).reshape(-1)[0])

    for x0 in start_points:
        try:
            res = minimize(func, np.asarray(x0, dtype=np.float64), bounds=bounds_list, method="L-BFGS-B")
            best_min = min(best_min, float(res.fun))
        except Exception:
            pass

    return float(best_min)


class OracleGPObjectiveFunction(ObjectiveFunction):
    """
    A stochastic black-box objective backed by a fitted sklearn GP.

    Each query returns a sample y ~ N(mu(x), std(x)) (independent per query),
    matching finetune.py's synthetic trajectory sampling logic.
    """

    def __init__(
        self,
        oracle_gp: GaussianProcessRegressor,
        rng: np.random.Generator,
        *,
        bounds: Tuple[np.ndarray, np.ndarray],
        min_std: float = 1e-12,
    ):
        self.oracle_gp = oracle_gp
        self.rng = rng
        self._lower = np.asarray(bounds[0], dtype=np.float32).copy()
        self._upper = np.asarray(bounds[1], dtype=np.float32).copy()
        self._min_std = float(min_std)

    @property
    def dim(self) -> int:
        return int(self._lower.shape[0])

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._lower, self._upper

    @property
    def optimal_value(self) -> Optional[float]:
        return None

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X).astype(np.float32)
        return _sample_from_oracle_gp(self.oracle_gp, X, self.rng, min_std=self._min_std)


def _get_gp_product_kernel_params(gp: GaussianProcessRegressor) -> Tuple[float, np.ndarray, float]:
    """
    Extract (kernel_variance, length_scale, nu) from a fitted sklearn GP kernel_.

    Expected kernel form: ConstantKernel * Matern(ARD, nu=2.5)
    """
    kernel = getattr(gp, "kernel_", None)
    if kernel is None:
        raise ValueError("GP is not fitted (missing kernel_)")

    k1 = getattr(kernel, "k1", None)
    k2 = getattr(kernel, "k2", None)
    if k1 is None or k2 is None:
        raise ValueError(f"Unsupported kernel type (expected Product): {type(kernel)}")

    const = None
    matern = None
    if hasattr(k1, "constant_value"):
        const = k1
        matern = k2
    elif hasattr(k2, "constant_value"):
        const = k2
        matern = k1
    else:
        raise ValueError(f"Kernel product has no ConstantKernel: {type(k1)}, {type(k2)}")

    if not hasattr(matern, "length_scale"):
        raise ValueError(f"Kernel product has no Matern(length_scale): {type(matern)}")

    kernel_variance = float(getattr(const, "constant_value"))
    length_scale = np.asarray(getattr(matern, "length_scale"), dtype=np.float64).reshape(-1)
    nu = float(getattr(matern, "nu", 2.5))

    if length_scale.size <= 0:
        raise ValueError("Invalid Matern length_scale")
    return kernel_variance, length_scale, nu


def _get_gp_y_normalization(gp: GaussianProcessRegressor) -> Tuple[float, float]:
    """
    Return (y_mean, y_std) used by sklearn when normalize_y=True; otherwise (0,1).
    """
    if not bool(getattr(gp, "normalize_y", False)):
        return 0.0, 1.0

    y_mean = getattr(gp, "_y_train_mean", None)
    y_std = getattr(gp, "_y_train_std", None)
    if y_mean is None or y_std is None:
        # Fallback for older/newer sklearn attribute names (best-effort).
        y_mean = getattr(gp, "y_train_mean_", 0.0)
        y_std = getattr(gp, "y_train_std_", 1.0)

    y_mean = float(np.asarray(y_mean, dtype=np.float64).reshape(-1)[0])
    y_std = float(np.asarray(y_std, dtype=np.float64).reshape(-1)[0])
    if not np.isfinite(y_std) or y_std <= 0:
        y_std = 1.0
    return y_mean, y_std


def _sample_matern_rff_frequencies(
    rng: np.random.Generator,
    *,
    n_features: int,
    dim: int,
    length_scale: np.ndarray,
    nu: float,
) -> np.ndarray:
    """
    Sample random frequencies for Matern kernel using its spectral density.

    For Matern with parameter nu, the spectral density corresponds to a multivariate Student-t:
      omega' ~ t_df(0, I), where df = 2*nu
      omega = diag(1/length_scale) * omega'
    """
    df = float(2.0 * float(nu))
    if df <= 0:
        raise ValueError(f"Invalid nu={nu} (df=2*nu must be > 0)")
    length_scale = np.asarray(length_scale, dtype=np.float64).reshape(-1)
    if length_scale.size == 1:
        length_scale = np.full(int(dim), float(length_scale[0]), dtype=np.float64)
    if length_scale.size != int(dim):
        raise ValueError(f"length_scale dim mismatch: got {length_scale.size}, expected {dim}")

    z = rng.normal(size=(int(n_features), int(dim))).astype(np.float64)
    chi2 = rng.chisquare(df, size=(int(n_features), 1)).astype(np.float64)
    omega_prime = z * np.sqrt(df / chi2)
    omega = omega_prime / length_scale.reshape(1, int(dim))
    return omega.astype(np.float64)


class SampledRFFOracleFunction(ObjectiveFunction):
    """
    Deterministic function sample from an oracle GP posterior using Random Fourier Features.

    Episode protocol (scheme 2):
    - choose an oracle GP (per training variant)
    - sample one deterministic function f from the GP posterior
    - interact with f deterministically within the episode
    """

    def __init__(
        self,
        oracle_gp: GaussianProcessRegressor,
        rng: np.random.Generator,
        *,
        bounds: Tuple[np.ndarray, np.ndarray],
        n_features: int = 512,
        jitter: float = 1e-8,
        normalize: bool = False,
        n_sobol_probe: int = 512,
    ):
        self.oracle_gp = oracle_gp
        self.rng = rng
        self._lower = np.asarray(bounds[0], dtype=np.float32).copy()
        self._upper = np.asarray(bounds[1], dtype=np.float32).copy()
        self._dim = int(self._lower.reshape(-1).shape[0])
        self._n_features = int(n_features)
        self._jitter = float(jitter)
        self._normalize = bool(normalize)

        kernel_variance, length_scale, nu = _get_gp_product_kernel_params(self.oracle_gp)
        self._kernel_variance = float(kernel_variance)
        self._length_scale = np.asarray(length_scale, dtype=np.float64).reshape(-1)
        self._nu = float(nu)
        self._y_mean, self._y_std = _get_gp_y_normalization(self.oracle_gp)

        self._omega = _sample_matern_rff_frequencies(
            self.rng,
            n_features=self._n_features,
            dim=self._dim,
            length_scale=self._length_scale,
            nu=self._nu,
        )  # (m, d)
        self._phase = self.rng.uniform(0.0, 2.0 * np.pi, size=(self._n_features,)).astype(np.float64)

        # Fit weight posterior on the GP's training data (in normalized-y space).
        X_train = np.asarray(getattr(self.oracle_gp, "X_train_", None), dtype=np.float64)
        y_train = np.asarray(getattr(self.oracle_gp, "y_train_", None), dtype=np.float64).reshape(-1)
        if X_train is None or X_train.size == 0:
            raise ValueError("oracle_gp is missing X_train_ (not fitted?)")
        if y_train is None or y_train.size == 0:
            raise ValueError("oracle_gp is missing y_train_ (not fitted?)")
        if X_train.shape[1] != self._dim:
            raise ValueError(f"X_train_ dim mismatch: {X_train.shape[1]} vs bounds dim {self._dim}")

        noise = getattr(self.oracle_gp, "alpha", 1e-6)
        noise_var = float(np.mean(np.asarray(noise, dtype=np.float64)))
        noise_var = max(noise_var, 1e-12)

        Phi = self._phi(X_train)  # (n, m)
        m = int(self._n_features)
        A = np.eye(m, dtype=np.float64) + (Phi.T @ Phi) / noise_var
        A = A + self._jitter * np.eye(m, dtype=np.float64)
        L = np.linalg.cholesky(A)  # lower
        rhs = (Phi.T @ y_train) / noise_var
        mu_w = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
        z = self.rng.normal(size=(m,)).astype(np.float64)
        w = mu_w + np.linalg.solve(L.T, z)  # sample with cov=A^{-1}
        self._w = w.astype(np.float64)

        # Precompute z-normalization statistics (if requested)
        self._znorm_mu = 0.0
        self._znorm_sigma = 1.0
        if self._normalize:
            from scipy.stats.qmc import Sobol as _Sobol
            sobol = _Sobol(d=self._dim, scramble=True,
                           seed=int(self.rng.integers(0, 2**31 - 1)))
            X_probe = sobol.random(n_sobol_probe).astype(np.float32)
            X_probe = X_probe * (self._upper - self._lower) + self._lower
            y_probe = self._eval_raw(X_probe)
            self._znorm_mu = float(y_probe.mean())
            self._znorm_sigma = float(max(y_probe.std(), 1e-8))

    @property
    def dim(self) -> int:
        return int(self._dim)

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._lower, self._upper

    def _phi(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X).astype(np.float64)
        proj = X @ self._omega.T + self._phase.reshape(1, -1)
        scale = np.sqrt(2.0 * float(self._kernel_variance) / float(self._n_features))
        return (scale * np.cos(proj)).astype(np.float64)

    def _eval_raw(self, X: np.ndarray) -> np.ndarray:
        """Evaluate in original y-space (before z-normalization)."""
        X = np.atleast_2d(X).astype(np.float64)
        Phi = self._phi(X)
        f_norm = Phi @ self._w  # normalized-y space if gp.normalize_y=True
        f = float(self._y_mean) + float(self._y_std) * np.asarray(f_norm, dtype=np.float64).reshape(-1)
        return np.asarray(f, dtype=np.float64).reshape(-1)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        f = self._eval_raw(X)
        if self._normalize:
            f = (f - self._znorm_mu) / self._znorm_sigma
        return np.asarray(f, dtype=np.float32).reshape(-1)


class SampledBNNFunction(ObjectiveFunction):
    """
    Deterministic function sampled from a BNN variational posterior.

    Analogous to SampledRFFOracleFunction, but uses NN weight sampling
    instead of RFF. Each call to __init__ draws one set of deterministic
    weights from the BNN posterior, yielding a deterministic function.
    """

    def __init__(self, bnn_params: dict, rng: np.random.Generator, *, bounds):
        # bnn_params: {'layers': [(loc, sigma, bias), ...], 'y_mean': float, 'y_std': float}
        self._lower = np.asarray(bounds[0], dtype=np.float32)
        self._upper = np.asarray(bounds[1], dtype=np.float32)
        self._dim = len(self._lower)
        self._y_mean = float(bnn_params['y_mean'])
        self._y_std = float(bnn_params['y_std'])

        # Sample deterministic weights from the variational posterior
        self._weights = []
        for loc, sigma, bias in bnn_params['layers']:
            eps = rng.normal(size=loc.shape)
            w = loc.astype(np.float64) + np.abs(sigma.astype(np.float64)) * eps
            self._weights.append((w, bias.astype(np.float64)))

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._lower, self._upper

    def __call__(self, X: np.ndarray) -> np.ndarray:
        h = np.atleast_2d(X).astype(np.float64)
        for w, b in self._weights[:-1]:
            h = h @ w + b
            h = np.where(h > 0, h, 0.2 * h)  # LeakyReLU (alpha=0.2)
        w, b = self._weights[-1]
        y_norm = (h @ w + b).flatten()
        return (self._y_mean + self._y_std * y_norm).astype(np.float32)


class RFFPriorFunction:
    """
    Random Fourier Feature function sampled from a GP prior (Matern 2.5).

    NOT fitted to any data — this is a pure prior sample used as a smooth
    random perturbation.  Evaluation is a fast matrix multiply.
    """

    def __init__(self, dim: int, length_scale: float, n_features: int = 256,
                 seed: int = 0, normalize_x: bool = False,
                 bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        rng = np.random.default_rng(seed)
        self._normalize_x = bool(normalize_x)
        if self._normalize_x:
            if bounds is None:
                raise ValueError("bounds is required when normalize_x=True for RFFPriorFunction")
            self._lower = np.asarray(bounds[0], dtype=np.float64)
            self._upper = np.asarray(bounds[1], dtype=np.float64)
            self._span = np.maximum(self._upper - self._lower, 1e-12)
        else:
            self._lower = None
            self._span = None
        nu = 2.5
        # Sample spectral frequencies from Matern 2.5 spectral density
        # (equivalent to scaled Student-t)
        z = rng.normal(size=(n_features, dim))
        v = rng.chisquare(df=2 * nu, size=(n_features, 1))
        self._W = z / (length_scale * np.sqrt(v / (2 * nu)))  # (n_features, dim)
        self._b = rng.uniform(0, 2 * np.pi, size=n_features)  # (n_features,)
        self._scale = np.sqrt(2.0 / n_features)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X).astype(np.float64)
        if self._normalize_x:
            X = (X - self._lower) / self._span
        Z = np.cos(X @ self._W.T + self._b[None, :])  # (n, n_features)
        return (self._scale * Z.sum(axis=1)).astype(np.float64)  # (n,)


class MultiScaleRFFPerturbation:
    """Sum of RFF samples at different length scales.

    In low-dimensional spaces (e.g. 2D Branin), a single smooth RFF cannot
    create multi-modal structure.  Superimposing several RFFs with decreasing
    length scales introduces higher-frequency components that break the
    single-basin topology, matching the multi-modal landscape of the real
    objective function.
    """

    def __init__(self, dim: int, length_scales: List[float],
                 alphas: List[float], n_features: int = 256,
                 seed: int = 0, normalize_x: bool = False,
                 bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        if len(alphas) != len(length_scales):
            raise ValueError(
                f"MultiScaleRFFPerturbation expects one alpha per length scale "
                f"(got {len(alphas)} alphas for {len(length_scales)} length scales)."
            )
        rng = np.random.default_rng(seed)
        self._rffs: List[RFFPriorFunction] = []
        self._alphas = list(alphas)
        for ls in length_scales:
            s = int(rng.integers(0, 2**31 - 1))
            self._rffs.append(
                RFFPriorFunction(dim=dim, length_scale=ls,
                                 n_features=n_features, seed=s,
                                 normalize_x=normalize_x, bounds=bounds)
            )

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        result = np.zeros(X.shape[0], dtype=np.float64)
        for rff, alpha in zip(self._rffs, self._alphas):
            result += alpha * rff(X)
        return result


class BNNMeanWithRFFPerturbation(ObjectiveFunction):
    """
    Deterministic training function: BNN posterior mean + α · RFF prior sample.

    f(x) = BNN_mean(x) + alpha * g(x)

    - BNN_mean: accurate surrogate of a historical experiment (from Olympus-aligned BNN)
    - g(x): smooth random function from GP prior (Matern 2.5, no data fitting)
    - alpha: perturbation strength controlling diversity vs fidelity

    Each instantiation draws a new RFF → a new deterministic function per episode.
    """

    def __init__(self, bnn_params: dict, rng: np.random.Generator, *,
                 bounds, alpha: float = 5.0, rff_length_scale: float = 0.3,
                 rff_n_features: int = 256, normalize_bnn: bool = False,
                 n_sobol_probe: int = 512):
        self._lower = np.asarray(bounds[0], dtype=np.float32)
        self._upper = np.asarray(bounds[1], dtype=np.float32)
        self._dim = len(self._lower)
        self._y_mean = float(bnn_params['y_mean'])
        self._y_std = float(bnn_params['y_std'])
        self._alpha = float(alpha)
        self._normalize_bnn = normalize_bnn

        # Store posterior mean weights (loc only, no sampling)
        self._weights = []
        for loc, sigma, bias in bnn_params['layers']:
            self._weights.append((loc.astype(np.float64), bias.astype(np.float64)))

        # Sample a fresh RFF perturbation function
        rff_seed = int(rng.integers(0, 2**31 - 1))
        self._rff = RFFPriorFunction(
            dim=self._dim, length_scale=rff_length_scale,
            n_features=rff_n_features, seed=rff_seed,
        )

        # Precompute BNN mean statistics for z-normalization (if requested)
        if normalize_bnn:
            from scipy.stats.qmc import Sobol as _Sobol
            sobol = _Sobol(d=self._dim, scramble=True,
                           seed=int(rng.integers(0, 2**31 - 1)))
            X_probe = sobol.random(n_sobol_probe).astype(np.float32)
            X_probe = X_probe * (self._upper - self._lower) + self._lower
            y_bnn = self._bnn_mean(X_probe)
            self._bnn_mu = float(y_bnn.mean())
            self._bnn_sigma = float(max(y_bnn.std(), 1e-8))

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._lower, self._upper

    def _bnn_mean(self, X: np.ndarray) -> np.ndarray:
        """Forward pass using BNN posterior mean (loc weights only)."""
        h = np.atleast_2d(X).astype(np.float64)
        for w, b in self._weights[:-1]:
            h = h @ w + b
            h = np.where(h > 0, h, 0.2 * h)  # LeakyReLU (alpha=0.2)
        w, b = self._weights[-1]
        y_norm = (h @ w + b).flatten()
        return self._y_mean + self._y_std * y_norm

    def __call__(self, X: np.ndarray) -> np.ndarray:
        y_base = self._bnn_mean(X)
        if self._normalize_bnn:
            y_base = (y_base - self._bnn_mu) / self._bnn_sigma
        y_perturb = self._rff(X)
        return (y_base + self._alpha * y_perturb).astype(np.float32)


class GPMeanWithRFFPerturbation(ObjectiveFunction):
    """
    Deterministic training function: GP posterior mean + alpha * RFF prior sample.

    f(x) = GP_mean(x) + alpha * g(x)

    - GP_mean: accurate surrogate from oracle GP (well-fitted, high fidelity)
    - g(x): smooth random function from GP prior (Matern 2.5, no data fitting)
    - alpha: perturbation strength controlling diversity vs fidelity

    Solves the oracle_gp RFF diversity problem: when GP posterior variance is
    tiny (e.g. Branin with large y-range), SampledRFFOracleFunction produces
    near-identical samples. This class decouples diversity (RFF prior) from
    fidelity (GP mean), matching the BNNMeanWithRFFPerturbation pattern.
    """

    def __init__(self, oracle_gp: GaussianProcessRegressor,
                 rng: np.random.Generator, *,
                 bounds: Tuple[np.ndarray, np.ndarray],
                 alpha: float = 1.5,
                 rff_length_scale: float = 0.3,
                 rff_n_features: int = 256,
                 normalize_gp_mean: bool = False,
                 n_sobol_probe: int = 512,
                 rff_multiscale_length_scales: Optional[List[float]] = None,
                 rff_multiscale_alphas: Optional[List[float]] = None,
                 rff_normalize_x: bool = False):
        self._oracle_gp = oracle_gp
        self._lower = np.asarray(bounds[0], dtype=np.float32)
        self._upper = np.asarray(bounds[1], dtype=np.float32)
        self._dim = len(self._lower)
        self._normalize_gp_mean = normalize_gp_mean
        self._y_mean, self._y_std = _get_gp_y_normalization(oracle_gp)

        # Sample a fresh RFF perturbation function from GP prior
        rff_seed = int(rng.integers(0, 2**31 - 1))
        if rff_multiscale_length_scales:
            ms_alphas = rff_multiscale_alphas or [alpha] * len(rff_multiscale_length_scales)
            self._rff = MultiScaleRFFPerturbation(
                dim=self._dim, length_scales=rff_multiscale_length_scales,
                alphas=ms_alphas, n_features=rff_n_features, seed=rff_seed,
                normalize_x=rff_normalize_x, bounds=(self._lower, self._upper),
            )
            self._alpha = 1.0  # alphas baked into MultiScaleRFF
        else:
            self._rff = RFFPriorFunction(
                dim=self._dim, length_scale=rff_length_scale,
                n_features=rff_n_features, seed=rff_seed,
                normalize_x=rff_normalize_x, bounds=(self._lower, self._upper),
            )
            self._alpha = float(alpha)

        # Precompute GP mean statistics for z-normalization (if requested)
        self._gp_mu = 0.0
        self._gp_sigma = 1.0
        if normalize_gp_mean:
            from scipy.stats.qmc import Sobol as _Sobol
            sobol = _Sobol(d=self._dim, scramble=True,
                           seed=int(rng.integers(0, 2**31 - 1)))
            X_probe = sobol.random(n_sobol_probe).astype(np.float32)
            X_probe = X_probe * (self._upper - self._lower) + self._lower
            y_gp = self._gp_mean(X_probe)
            self._gp_mu = float(y_gp.mean())
            self._gp_sigma = float(max(y_gp.std(), 1e-8))

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._lower, self._upper

    def _gp_mean(self, X: np.ndarray) -> np.ndarray:
        """GP posterior mean in original y-space."""
        X = np.atleast_2d(X).astype(np.float64)
        y_pred = self._oracle_gp.predict(X).reshape(-1)
        return y_pred

    def __call__(self, X: np.ndarray) -> np.ndarray:
        y_base = self._gp_mean(X)
        if self._normalize_gp_mean:
            y_base = (y_base - self._gp_mu) / self._gp_sigma
        y_perturb = self._rff(X)
        return (y_base + self._alpha * y_perturb).astype(np.float32)


def load_bnn_params(bnn_params_path: str) -> List[dict]:
    """Load BNN variational parameters from .npz file."""
    data = np.load(bnn_params_path, allow_pickle=True)
    n_variants = int(data['n_variants'])
    params_list = []
    for i in range(n_variants):
        layers = []
        l = 0
        while f'layer_{i}_{l}_loc' in data:
            layers.append((
                data[f'layer_{i}_{l}_loc'],
                data[f'layer_{i}_{l}_sigma'],
                data[f'layer_{i}_{l}_bias'],
            ))
            l += 1
        params_list.append({
            'layers': layers,
            'y_mean': float(data[f'y_mean_{i}']),
            'y_std': float(data[f'y_std_{i}']),
        })
    return params_list


class TaskVariantObjectiveFunction(ObjectiveFunction):
    """
    Direct objective backed by a registered TaskSpec's numpy evaluator.
    """

    def __init__(self, *, task_name: str, variant_params: Optional[Dict[str, float]] = None):
        self._task = get_task(str(task_name))
        self._variant_params = variant_params or {}
        self._lower = np.asarray(self._task.bounds[0], dtype=np.float32).copy()
        self._upper = np.asarray(self._task.bounds[1], dtype=np.float32).copy()

    @property
    def dim(self) -> int:
        return int(self._task.dim)

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._lower, self._upper

    @property
    def optimal_value(self) -> Optional[float]:
        return self._task.optimal_value(self._variant_params)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self._task.evaluate_numpy(X, self._variant_params)


# ==================== 注意力模块 (从 v2 独立复制) ====================
class MultiHeadSelfAttention(nn.Module):
    """多头自注意力"""

    def __init__(self, hidden_dim, n_heads=4, dropout=0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) optional padding mask (1=valid, 0=padding)
        Returns:
            out: (batch, seq_len, hidden_dim)
        """
        batch, n, d = x.shape

        q = self.q_proj(x).view(batch, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, n, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, n, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            # mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, n, d)
        out = self.out_proj(out)

        return out


class MultiHeadCrossAttention(nn.Module):
    """多头交叉注意力: Query 来自一个序列，Key/Value 来自另一个序列"""

    def __init__(self, hidden_dim, n_heads=4, dropout=0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, query, key_value, kv_mask=None):
        """
        Args:
            query: (batch, n_query, hidden_dim) - 候选点
            key_value: (batch, n_kv, hidden_dim) - 上下文点
            kv_mask: (batch, n_kv) optional mask for key/value (1=valid, 0=padding)
        Returns:
            out: (batch, n_query, hidden_dim)
        """
        batch, n_q, d = query.shape
        n_kv = key_value.shape[1]

        q = self.q_proj(query).view(batch, n_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(batch, n_kv, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(batch, n_kv, self.n_heads, self.head_dim).transpose(1, 2)

        # (batch, n_heads, n_q, n_kv)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if kv_mask is not None:
            # kv_mask: (batch, n_kv) -> (batch, 1, 1, n_kv)
            kv_mask = kv_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(kv_mask == 0, float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (batch, n_heads, n_q, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch, n_q, d)
        out = self.out_proj(out)

        return out


class TransformerBlock(nn.Module):
    """Transformer 块: Self-Attention + FFN"""

    def __init__(self, hidden_dim, n_heads=4, ffn_dim=None, dropout=0.1):
        super().__init__()
        ffn_dim = ffn_dim or hidden_dim * 2

        self.attn = MultiHeadSelfAttention(hidden_dim, n_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    """Cross-Attention 块: Cross-Attention + FFN"""

    def __init__(self, hidden_dim, n_heads=4, ffn_dim=None, dropout=0.1):
        super().__init__()
        ffn_dim = ffn_dim or hidden_dim * 2

        self.cross_attn = MultiHeadCrossAttention(hidden_dim, n_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, query, key_value, kv_mask=None):
        # Cross-Attention + Residual
        query = query + self.cross_attn(self.norm1(query), self.norm_kv(key_value), kv_mask)
        # FFN + Residual
        query = query + self.ffn(self.norm2(query))
        return query


class AttentionPooling(nn.Module):
    """Attention-based Pooling"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) optional (1=valid, 0=padding)
        Returns:
            pooled: (batch, hidden_dim)
        """
        attn_weights = self.attention(x)  # (batch, n, 1)

        if mask is not None:
            mask = mask.unsqueeze(-1)  # (batch, n, 1)
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(attn_weights, dim=1)
        pooled = (x * attn_weights).sum(dim=1)
        return pooled


# ==================== 改进的双塔网络 ====================
class ImprovedDualTowerSelector(nn.Module):
    """
    改进的双塔 Cross-Attention 候选点选择网络
    
    改进点:
    1. Candidate 特征增加一个 acquisition 分数（TAF_me 或 EI）
    2. 加入当前步数信息
    3. 更深的网络
    """
    
    def __init__(
        self,
        coord_dim: int = 2,
        hidden_dim: int = 128,  # 增大隐藏维度
        n_self_attn_layers: int = 3,
        n_cross_attn_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_steps: int = 20,
        use_taf_feature: bool = False,
    ):
        super().__init__()

        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        self.use_taf_feature = use_taf_feature

        # Context 特征: [x, y_rank] -> (coord_dim + 1)
        context_input_dim = coord_dim + 1

        # Candidate 特征: [x, μ, σ, is_persistent] -> (coord_dim + 3)
        # 加 TAF rank 特征时: [x, μ, σ, is_persistent, taf_rank] -> (coord_dim + 4)
        candidate_input_dim = coord_dim + (4 if use_taf_feature else 3)
        
        # Embedding layers
        self.context_embed = nn.Sequential(
            nn.Linear(context_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        self.candidate_embed = nn.Sequential(
            nn.Linear(candidate_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Step embedding
        self.step_embed = nn.Embedding(max_steps + 1, hidden_dim)
        
        # Context 塔: Self-Attention
        self.context_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, dropout=dropout)
            for _ in range(n_self_attn_layers)
        ])
        
        # Candidate 塔: Self-Attention + Cross-Attention
        self.candidate_self_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, dropout=dropout)
            for _ in range(n_self_attn_layers)
        ])
        
        self.cross_layers = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, n_heads, dropout=dropout)
            for _ in range(n_cross_attn_layers)
        ])
        
        # Actor Head
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Critic Head (使用 attention pooling)
        self.context_pool = AttentionPooling(hidden_dim)
        self.candidate_pool = AttentionPooling(hidden_dim)
        
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, context, candidates, step, context_mask=None):
        """
        Args:
            context: (batch, n_context, context_dim)
            candidates: (batch, n_candidates, candidate_dim)
            step: (batch,) 当前步数
            context_mask: optional
        """
        batch = context.shape[0]
        
        # Embedding
        ctx_emb = self.context_embed(context)
        cand_emb = self.candidate_embed(candidates)
        
        # 加入步数信息
        step_emb = self.step_embed(step)  # (batch, hidden_dim)
        ctx_emb = ctx_emb + step_emb.unsqueeze(1)
        cand_emb = cand_emb + step_emb.unsqueeze(1)
        
        # Context 塔
        for layer in self.context_layers:
            ctx_emb = layer(ctx_emb, mask=context_mask)
        
        # Candidate Self-Attention
        for layer in self.candidate_self_layers:
            cand_emb = layer(cand_emb)
        
        # Cross-Attention
        for layer in self.cross_layers:
            cand_emb = layer(cand_emb, ctx_emb, kv_mask=context_mask)
        
        # Actor: 每个候选点的 logit
        logits = self.actor_head(cand_emb).squeeze(-1)  # (batch, n_candidates)
        
        # Critic: pooled features
        ctx_pooled = self.context_pool(ctx_emb, mask=context_mask)
        cand_pooled = self.candidate_pool(cand_emb)
        
        combined = torch.cat([ctx_pooled, cand_pooled], dim=-1)
        value = self.critic_head(combined).squeeze(-1)
        
        return logits, value
    
    def get_action(self, context, candidates, step, context_mask=None):
        logits, value = self.forward(context, candidates, step, context_mask)
        
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob, value
    
    def evaluate(self, context, candidates, step, actions, context_mask=None):
        logits, value = self.forward(context, candidates, step, context_mask)
        
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_prob, entropy, value


# ==================== 改进的 PPO ====================
class ImprovedPPO:
    """改进的 PPO"""
    
    def __init__(
        self,
        coord_dim: int = 2,
        hidden_dim: int = 128,
        n_self_attn_layers: int = 3,
        n_cross_attn_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_steps: int = 20,
        lr: float = 1e-4,  # 降低学习率
        gamma: float = 0.99,  # 接近无折扣，对齐 regret AUC 目标（48步场景）
        lam: float = 0.9,  # GAE lambda，降低以减少方差
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.02,  # backward-compat alias for ent_coef_start
        ent_coef_end: float | None = None,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
        use_taf_feature: bool = False,
    ):
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef_start = float(ent_coef)
        self.ent_coef_end = float(ent_coef if ent_coef_end is None else ent_coef_end)
        self.max_grad_norm = max_grad_norm
        self.max_steps = max_steps
        self.use_taf_feature = use_taf_feature
        self.total_updates = 1
        self.update_step = 0

        self.policy = ImprovedDualTowerSelector(
            coord_dim=coord_dim,
            hidden_dim=hidden_dim,
            n_self_attn_layers=n_self_attn_layers,
            n_cross_attn_layers=n_cross_attn_layers,
            n_heads=n_heads,
            dropout=dropout,
            max_steps=max_steps,
            use_taf_feature=use_taf_feature,
        ).to(device)
        
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=lr,
            weight_decay=1e-4,
        )

        # Learning rate scheduler: 线性 warmup + cosine 衰减（不再使用 WarmRestarts）
        # WarmRestarts 会在 restart 时把 LR 弹回初始值，破坏已学到的策略
        self.scheduler = None  # 在 setup_scheduler() 中设置

    def setup_scheduler(self, total_updates: int, warmup_fraction: float = 0.05):
        """设置线性 warmup + cosine 衰减 LR 调度器"""
        self.total_updates = max(1, int(total_updates))
        warmup_steps = max(1, int(total_updates * warmup_fraction))

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(warmup_steps)
            progress = float(current_step - warmup_steps) / float(max(1, total_updates - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def get_current_ent_coef(self) -> float:
        """Cosine-annealed entropy coefficient."""
        if self.total_updates <= 1:
            return self.ent_coef_end
        progress = min(max(self.update_step / max(self.total_updates - 1, 1), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.ent_coef_end + (self.ent_coef_start - self.ent_coef_end) * cosine

    def select_candidate(
        self,
        X_context: np.ndarray,
        y_context: np.ndarray,
        X_candidates: np.ndarray,
        pred_mean: np.ndarray,
        pred_std: np.ndarray,
        bounds: Tuple[np.ndarray, np.ndarray],
        current_step: int,
        is_persistent: np.ndarray = None,
        taf_rank_norm: np.ndarray = None,
    ) -> Tuple[int, float, float]:
        """选择候选点"""
        context_feat, candidate_feat = self._build_features(
            X_context, y_context, X_candidates, pred_mean, pred_std, bounds, current_step,
            is_persistent=is_persistent,
            taf_rank_norm=taf_rank_norm,
        )

        context_feat = context_feat.unsqueeze(0).to(self.device)
        candidate_feat = candidate_feat.unsqueeze(0).to(self.device)
        step_tensor = torch.tensor([current_step], device=self.device)

        action, log_prob, value = self.policy.get_action(
            context_feat, candidate_feat, step_tensor
        )
        return action.item(), log_prob.item(), value.item()

    def _build_features(
        self,
        X_context: np.ndarray,
        y_context: np.ndarray,
        X_candidates: np.ndarray,
        pred_mean: np.ndarray,
        pred_std: np.ndarray,
        bounds: Tuple[np.ndarray, np.ndarray],
        current_step: int = 0,
        is_persistent: np.ndarray = None,
        taf_rank_norm: np.ndarray = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建特征

        Context: [x_norm, y_rank]
        Candidate: [x_norm, μ_norm, σ_norm, is_persistent(, taf_rank_norm)]
        """
        lower, upper = bounds

        # ========== Context 特征 ==========
        X_ctx_norm = (X_context - lower) / (upper - lower + 1e-8)

        # Rank-based encoding: 0 = 最好, 1 = 最差（稳定、尺度无关）
        n_ctx = len(y_context)
        if n_ctx > 1:
            ranks = np.argsort(np.argsort(y_context)).astype(np.float32)
            y_rank = ranks / (n_ctx - 1)
        else:
            y_rank = np.zeros(n_ctx, dtype=np.float32)

        context_feat = np.concatenate([
            X_ctx_norm,
            y_rank.reshape(-1, 1)
        ], axis=-1)

        # ========== Candidate 特征 ==========
        X_cand_norm = (X_candidates - lower) / (upper - lower + 1e-8)

        y_best = y_context.min()
        y_range = float(y_context.max() - y_context.min())
        y_range = max(y_range, 1e-6)

        # 均值：相对于当前最优值，按观测范围归一化
        mean_norm = (pred_mean - y_best) / y_range

        # 标准差：按观测范围归一化
        std_norm = pred_std / y_range

        n_candidates = len(X_candidates)

        # is_persistent 标记
        if is_persistent is None:
            is_persistent_arr = np.zeros(n_candidates, dtype=np.float32)
        else:
            is_persistent_arr = np.asarray(is_persistent, dtype=np.float32)

        parts = [
            X_cand_norm,
            mean_norm.reshape(-1, 1),
            std_norm.reshape(-1, 1),
            is_persistent_arr.reshape(-1, 1),
        ]

        # TAF ranking 特征
        if self.use_taf_feature:
            if taf_rank_norm is not None:
                parts.append(np.asarray(taf_rank_norm, dtype=np.float32).reshape(-1, 1))
            else:
                parts.append(np.zeros((n_candidates, 1), dtype=np.float32))

        candidate_feat = np.concatenate(parts, axis=-1)

        return (
            torch.FloatTensor(context_feat),
            torch.FloatTensor(candidate_feat)
        )

    def compute_gae(self, rewards, values, dones, last_value):
        advantages = []
        gae = 0
        values = values + [last_value]
        
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns
    
    def update(self, rollout, n_epochs=4, batch_size=64):
        context_feats = rollout["context_feats"]
        candidate_feats = rollout["candidate_feats"]
        steps = rollout["steps"]
        actions = torch.LongTensor(rollout["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(rollout["log_probs"]).to(self.device)
        advantages = torch.FloatTensor(rollout["advantages"]).to(self.device)
        returns = torch.FloatTensor(rollout["returns"]).to(self.device)
        
        # 标准化 advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        ent_coef = self.get_current_ent_coef()

        n_samples = len(actions)
        indices = np.arange(n_samples)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        for epoch in range(n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]
                
                # Pad sequences
                batch_ctx = self._pad_sequences([context_feats[i] for i in batch_indices])
                batch_cand = self._pad_sequences([candidate_feats[i] for i in batch_indices])
                batch_steps = torch.LongTensor([steps[i] for i in batch_indices]).to(self.device)
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # 计算 context mask (1=valid, 0=padding)
                # padding 的位置所有特征都是 0，所以 sum(dim=-1) == 0
                batch_ctx_mask = (batch_ctx.abs().sum(dim=-1) > 1e-8).float()

                # Forward
                log_probs, entropy, values = self.policy.evaluate(
                    batch_ctx, batch_cand, batch_steps, batch_actions, batch_ctx_mask
                )
                
                # Policy loss
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = ((values - batch_returns) ** 2).mean()
                
                # Total loss
                loss = policy_loss + self.vf_coef * value_loss - ent_coef * entropy.mean()
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        
        if self.scheduler is not None:
            self.scheduler.step()
        self.update_step += 1

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "ent_coef": ent_coef,
            "lr": self.optimizer.param_groups[0]['lr'],
        }
    
    def _pad_sequences(self, sequences):
        max_len = max(s.shape[0] for s in sequences)
        feat_dim = sequences[0].shape[1]
        batch = len(sequences)
        
        padded = torch.zeros(batch, max_len, feat_dim, device=self.device)
        for i, seq in enumerate(sequences):
            padded[i, :seq.shape[0], :] = seq.to(self.device)
        
        return padded


# ==================== TabPFN → TAF 接口适配 ====================
class _TabPFNModelTargetForTAF:
    """包装 TabPFN regressor 以支持 TAF 需要的 predict_noiseless(X) 接口。"""

    def __init__(self, regressor, y_mean: float = 0.0, y_std: float = 1.0):
        self._regressor = regressor
        self._y_mean = y_mean
        self._y_std = y_std

    def predict_noiseless(self, X: np.ndarray):
        full_out = self._regressor.predict(np.asarray(X, dtype=np.float32), output_type="full")
        mean_norm = np.asarray(full_out.get("mean", None), dtype=np.float64).reshape(-1, 1)
        std_norm = np.asarray(compute_std_from_tabpfn_output(full_out), dtype=np.float64).reshape(-1, 1)
        mean = mean_norm * self._y_std + self._y_mean
        var = (std_norm * self._y_std) ** 2
        return mean, var


# ==================== BO 环境 ====================
class ImprovedBraninBOEnv:
    """
    改进的贝叶斯优化环境

    默认使用 finetune.py 的轨迹缓存（bo_trajs_train.npz）重构 oracle GP；
    也支持直接使用任务的真实目标函数（objective_source="direct"，兼容别名 "branin"）。
    """

    def __init__(
        self,
        task_name: str = "branin_family",
        variants_path: str = "./data/variants_train.npz",
        trajectories_path: str = "./data/bo_trajs_train.npz",
        objective_source: str = "oracle_gp",  # "oracle_gp", "direct", or "bnn"
        bnn_params_path: str = None,
        bnn_rff_alpha: float = 5.0,
        bnn_rff_length_scale: float = 0.3,
        normalize_bnn: bool = False,
        normalize_oracle_gp: bool = False,
        oracle_gp_y_transform: str = "none",
        oracle_gp_rff_alpha: float = 0.0,
        oracle_gp_rff_length_scale: float = 0.3,
        oracle_gp_rff_multiscale: str = "",
        oracle_gp_rff_multiscale_alphas: str = "",
        oracle_gp_rff_normalize_x: bool = False,
        model_path: str = None,
        taf_data_path: str = None,
        max_steps: int = 18,
        n_init_context: int = 2,
        n_persistent_base: int = 128,
        n_total_candidates: int = 192,
        k_centers: int = 2,
        local_h: float = 1.5,
        local_h_decay: float = 0.9,
        explore_fraction: float = 0.0,
        oracle_gp_min_grid_size: int = 80,
        oracle_gp_n_restarts_optimizer: int = 3,
        oracle_gp_fixed_length_scale: float = 0.0,
        oracle_gp_alpha: float = 1e-6,
        oracle_gp_min_n_lbfgs_starts: int = 25,
        reward_mode: str = "auc",
        reward_mixed_lambda: float = 0.3,
        reward_terminal_weight: float = 1.0,
        reward_frontload_power: float = 1.0,
        reward_stage_midpoint: float = 0.4,
        reward_regret_auc_weight: float = 0.2,
        reward_regret_delta_weight: float = 1.0,
        reward_regret_early_power: float = 0.5,
        reward_regret_terminal_power: float = 3.0,
        reward_regret_scale_floor_ratio: float = 0.02,
        variant_sampling: str = "random",
        inference_precision: str = "float32",
        device: str = "cpu",
        seed: int = 42,
    ):
        self.task_name = str(task_name)
        self.task = get_task(self.task_name)

        self.objective_source = str(objective_source)
        if self.objective_source == "branin":
            self.objective_source = "direct"
        if self.objective_source not in {"oracle_gp", "direct", "bnn"}:
            raise ValueError(f"Invalid objective_source={objective_source!r}. Use 'oracle_gp', 'direct', or 'bnn'.")

        self.model_path = model_path
        self.max_steps = max_steps
        self.n_init_context = n_init_context
        self.n_persistent_base = n_persistent_base
        self.n_total_candidates = n_total_candidates
        self.k_centers = k_centers
        self.local_h = local_h
        self.local_h_decay = local_h_decay
        self.explore_fraction = float(max(0.0, min(1.0, explore_fraction)))
        self.oracle_gp_min_grid_size = int(oracle_gp_min_grid_size)
        self.oracle_gp_n_restarts_optimizer = int(oracle_gp_n_restarts_optimizer)
        self.oracle_gp_fixed_length_scale = float(oracle_gp_fixed_length_scale)
        self.oracle_gp_alpha = float(oracle_gp_alpha)
        self.oracle_gp_min_n_lbfgs_starts = int(oracle_gp_min_n_lbfgs_starts)
        self.reward_mode = str(reward_mode)
        if self.reward_mode not in {
            "auc",
            "delta",
            "delta_terminal",
            "mixed",
            "frontload_mixed",
            "staged_mixed",
            "regret_balanced",
        }:
            raise ValueError(
                f"Invalid reward_mode={reward_mode!r}. "
                "Use 'auc', 'delta', 'delta_terminal', 'mixed', 'frontload_mixed', 'staged_mixed', "
                "or 'regret_balanced'."
            )
        self.reward_mixed_lambda = float(reward_mixed_lambda)
        self.reward_terminal_weight = float(reward_terminal_weight)
        self.reward_frontload_power = float(reward_frontload_power)
        self.reward_stage_midpoint = float(reward_stage_midpoint)
        self.reward_regret_auc_weight = float(reward_regret_auc_weight)
        self.reward_regret_delta_weight = float(reward_regret_delta_weight)
        self.reward_regret_early_power = float(reward_regret_early_power)
        self.reward_regret_terminal_power = float(reward_regret_terminal_power)
        self.reward_regret_scale_floor_ratio = float(reward_regret_scale_floor_ratio)
        self.variant_sampling = str(variant_sampling)
        if self.variant_sampling not in {"random", "shuffled_cycle"}:
            raise ValueError(
                f"Invalid variant_sampling={variant_sampling!r}. "
                "Use 'random' or 'shuffled_cycle'."
            )
        # TabPFN inference precision: float16 is ~1.5-2x faster on GPU
        self._inference_precision = torch.float16 if inference_precision == "float16" else torch.float32
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.bnn_rff_alpha = float(bnn_rff_alpha)
        self.bnn_rff_length_scale = float(bnn_rff_length_scale)
        self.normalize_bnn = bool(normalize_bnn)
        self.normalize_oracle_gp = bool(normalize_oracle_gp)
        self.oracle_gp_y_transform = str(oracle_gp_y_transform)
        self.oracle_gp_rff_alpha = float(oracle_gp_rff_alpha)
        self.oracle_gp_rff_length_scale = float(oracle_gp_rff_length_scale)
        self.oracle_gp_rff_multiscale_ls = (
            [float(x) for x in oracle_gp_rff_multiscale.split(",") if x.strip()]
            if oracle_gp_rff_multiscale else []
        )
        self.oracle_gp_rff_multiscale_alphas = (
            [float(x) for x in oracle_gp_rff_multiscale_alphas.split(",") if x.strip()]
            if oracle_gp_rff_multiscale_alphas else []
        )
        if self.oracle_gp_rff_multiscale_ls and self.oracle_gp_rff_multiscale_alphas:
            if len(self.oracle_gp_rff_multiscale_ls) != len(self.oracle_gp_rff_multiscale_alphas):
                raise ValueError(
                    "--oracle_gp_rff_multiscale_alphas must have the same number of entries as "
                    "--oracle_gp_rff_multiscale "
                    f"(got {len(self.oracle_gp_rff_multiscale_alphas)} vs "
                    f"{len(self.oracle_gp_rff_multiscale_ls)})."
                )
        self.oracle_gp_rff_normalize_x = bool(oracle_gp_rff_normalize_x)

        self.oracle_gps: List[GaussianProcessRegressor] = []
        self.bnn_params_list: List[dict] = []
        self.variants: List[Dict[str, float]] = []
        self.num_variants = 0

        if self.objective_source == "bnn":
            if bnn_params_path is None:
                raise ValueError("bnn_params_path is required when objective_source='bnn'")
            self.bnn_params_list = load_bnn_params(bnn_params_path)
            self.num_variants = len(self.bnn_params_list)
            # Load variants from variants_path for reference
            data = np.load(variants_path, allow_pickle=True)
            if "variants" in data:
                self.variants = data["variants"].tolist()
            print(f"从 {bnn_params_path} 加载 {self.num_variants} 个 BNN 变分参数")
            if self.num_variants <= 0:
                raise ValueError(f"No BNN params loaded from {bnn_params_path}")
            # BNN mode: like oracle_gp, sample a deterministic function per episode,
            # so global_min and reward_scale are computed per episode in reset().
            self.variant_global_mins = None
            self.variant_y_ranges = None
        elif self.objective_source == "oracle_gp":
            self.oracle_gps, cache_variants = load_oracle_gps_from_trajectories_cache(
                trajectories_path,
                n_restarts_optimizer=self.oracle_gp_n_restarts_optimizer,
                y_transform=self.oracle_gp_y_transform,
                fixed_length_scale=(
                    self.oracle_gp_fixed_length_scale
                    if self.oracle_gp_fixed_length_scale > 0
                    else None
                ),
                alpha=self.oracle_gp_alpha,
            )
            self.num_variants = len(self.oracle_gps)
            self.variants = cache_variants
            print(f"从 {trajectories_path} 重构 {self.num_variants} 个 oracle GP（每个来自一个训练变体的 best-of BO 轨迹）")
            if self.num_variants <= 0:
                raise ValueError(f"No oracle GPs reconstructed from {trajectories_path}")
            # For oracle_gp objective, we sample a deterministic function per episode (scheme 2),
            # so global_min is computed per episode on that sampled function.
            self.variant_global_mins = None
            # oracle_gp 模式下，reward 归一化使用每个 episode 的 RFF 函数 y_range（在 reset 中计算），
            # 不需要预计算 variant_y_ranges
            self.variant_y_ranges = None
        else:
            # 从预生成的训练变体文件加载（与 finetune.py 训练数据来源一致）
            data = np.load(variants_path, allow_pickle=True)
            if "variants" not in data:
                raise KeyError(f"variants_path must contain key 'variants': {variants_path}")
            self.variants = data["variants"].tolist()
            self.num_variants = len(self.variants)
            print(f"从 {variants_path} 加载 {self.num_variants} 个变体")
            if self.num_variants <= 0:
                raise ValueError(f"No variants loaded from {variants_path}")
            # 预计算每个变体的全局最小值（特征/日志需要；避免每个 episode reset 重算）
            self.variant_global_mins = self._precompute_variant_global_mins()
            # 预计算每个变体的 y_range，用于 reward 归一化
            self.variant_y_ranges = self._precompute_variant_y_ranges()
        
        self.regressor = None
        self._init_regressor()
        
        self.current_func = None
        self.X_context = None
        self.y_context = None
        self.best_y = None
        self.best_y_0 = None       # episode 初始 best_y，用于 reward 计算
        self.prev_best_y = None    # 上一步 best_y，用于 delta reward
        self.reward_scale = None    # reward 归一化系数：oracle_gp 模式用 RFF y_range，direct 模式用 variant y_range
        # Backward-compat: keep best_mu field for logging, but under deterministic objectives it equals best_y.
        self.best_mu = None
        self.global_min = None
        self.step_count = 0
        self.initial_regret = None
        self.current_variant_idx = None
        self._variant_cycle_order = None
        self._variant_cycle_pos = 0

        # 缓存当前候选点，避免 step() 时重新生成
        self._cached_candidates = None
        self._cached_pred_mean = None
        self._cached_pred_std = None

        # TAF ranking 特征：加载 GPy source GPs，用于每步计算 TAF score
        self.taf_data_path = taf_data_path
        self.taf_obj = None
        if taf_data_path is not None:
            from myrl.policies.policies import TAF
            self.taf_obj = TAF(taf_data_path, mode="ranking", rho=1.0)
            print(f"TAF ranking 已加载: {taf_data_path} (M={self.taf_obj.M} source GPs)")

    def _precompute_oracle_gp_mean_mins(self):
        mins = []
        bounds = (np.asarray(self.task.bounds[0], dtype=np.float32), np.asarray(self.task.bounds[1], dtype=np.float32))
        print("预计算每个 oracle GP 的 mean-min（legacy，仅用于对照/调试）...")
        for i, gp in enumerate(self.oracle_gps):
            mins.append(
                estimate_gp_mean_min_on_grid(
                    gp,
                    bounds,
                    grid_size=self.oracle_gp_min_grid_size,
                )
            )
            if (i + 1) % 5 == 0 or (i + 1) == len(self.oracle_gps):
                print(f"  computed {i+1}/{len(self.oracle_gps)}")
        return mins

    def _precompute_variant_global_mins(self):
        mins = []
        print("预计算每个变体的 global_min（仅一次）...")
        for i, params in enumerate(self.variants):
            mins.append(float(self.task.estimate_global_min(params)))
            if (i + 1) % 5 == 0 or (i + 1) == self.num_variants:
                print(f"  computed {i+1}/{self.num_variants}")
        return mins

    def _precompute_oracle_gp_y_ranges(self):
        """预计算每个 oracle GP 训练数据的 y_range，用于 reward 归一化（C_gp）。"""
        ranges = []
        for gp in self.oracle_gps:
            y_train = np.asarray(gp.y_train_, dtype=np.float64).reshape(-1)
            y_range = float(np.max(y_train) - np.min(y_train))
            ranges.append(max(y_range, 1e-6))
        print(f"Oracle GP y_ranges (reward scale): {[f'{r:.3f}' for r in ranges]}")
        return ranges

    def _precompute_variant_y_ranges(self):
        """预计算每个变体的 y_range（Sobol 探测），用于 reward 归一化（C_gp）。"""
        ranges = []
        lower = np.asarray(self.task.bounds[0], dtype=np.float32)
        upper = np.asarray(self.task.bounds[1], dtype=np.float32)
        dim = int(self.task.dim)
        sobol = Sobol(d=dim, scramble=True, seed=0)
        X_probe = (sobol.random(2000) * (upper - lower) + lower).astype(np.float32)
        for params in self.variants:
            y = self.task.evaluate_numpy(X_probe, params).reshape(-1)
            y_range = float(np.max(y) - np.min(y))
            ranges.append(max(y_range, 1e-6))
        print(f"Variant y_ranges (reward scale): {[f'{r:.3f}' for r in ranges]}")
        return ranges

    def _init_regressor(self):
        regressor_kwargs = {
            "device": self.device,
            "n_estimators": 1,
            "random_state": 42,
            "inference_precision": self._inference_precision,
            "ignore_pretraining_limits": True,
        }
        if self.model_path is not None:
            regressor_kwargs["model_path"] = self.model_path
        self.regressor = TabPFNRegressor(**regressor_kwargs)
        self._cache_tabpfn_model()

    def _cache_tabpfn_model(self):
        """Warmup fit + cache the model to avoid rebuilding on every fit().

        TabPFN v2 rebuilds the Transformer architecture and reloads weights
        from state_dict on every fit() call (~7 s overhead).  After the first
        fit we replace ``model_path`` with a ``RegressorModelSpecs`` that
        holds the already-loaded model so subsequent fits skip the rebuild.
        """
        from tabpfn.base import RegressorModelSpecs

        dim = int(self.task.dim)
        dummy_X = np.random.rand(3, dim).astype(np.float32)
        dummy_y = np.random.randn(3).astype(np.float32)
        self.regressor.fit(dummy_X, dummy_y)
        # TabPFN API varies across versions:
        #   older: RegressorModelSpecs(model, config, norm_criterion)
        #   7.1.1: RegressorModelSpecs(model, architecture_config, inference_config, norm_criterion)
        import inspect
        sig = inspect.signature(RegressorModelSpecs.__init__)
        if 'inference_config' in sig.parameters:
            # TabPFN >= 7.x: configs_ is a list, take first element
            arch_cfg = getattr(self.regressor, 'configs_', [None])[0]
            inf_cfg = getattr(self.regressor, 'inference_config_', None)
            self.regressor.model_path = RegressorModelSpecs(
                model=self.regressor.model_,
                architecture_config=arch_cfg,
                inference_config=inf_cfg,
                norm_criterion=self.regressor.znorm_space_bardist_,
            )
        else:
            # Older TabPFN
            config_attr = getattr(self.regressor, 'config_', None) or getattr(self.regressor, 'configs_', None)
            self.regressor.model_path = RegressorModelSpecs(
                model=self.regressor.model_,
                config=config_attr,
                norm_criterion=self.regressor.znorm_space_bardist_,
            )
        print(f"TabPFN model cached — subsequent fit() calls will skip model rebuild")

    def _sample_variant_idx(self) -> int:
        """Sample a training variant according to the configured scheduling mode."""
        if self.variant_sampling == "random":
            return int(self.rng.integers(0, self.num_variants))

        if self._variant_cycle_order is None or self._variant_cycle_pos >= self.num_variants:
            self._variant_cycle_order = self.rng.permutation(self.num_variants)
            self._variant_cycle_pos = 0

        variant_idx = int(self._variant_cycle_order[self._variant_cycle_pos])
        self._variant_cycle_pos += 1
        return variant_idx
    
    def reset(self, seed=None, variant_idx=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._variant_cycle_order = None
            self._variant_cycle_pos = 0

        # 从训练集合中选择一个任务（一个变体 / 一个 oracle GP）
        if variant_idx is None:
            variant_idx = self._sample_variant_idx()
        self.current_variant_idx = variant_idx

        if self.objective_source == "bnn":
            bnn_params = self.bnn_params_list[variant_idx]
            self.current_func = BNNMeanWithRFFPerturbation(
                bnn_params=bnn_params,
                rng=self.rng,
                bounds=(
                    np.asarray(self.task.bounds[0], dtype=np.float32),
                    np.asarray(self.task.bounds[1], dtype=np.float32),
                ),
                alpha=self.bnn_rff_alpha,
                rff_length_scale=self.bnn_rff_length_scale,
                normalize_bnn=self.normalize_bnn,
            )
        elif self.objective_source == "oracle_gp":
            oracle_gp = self.oracle_gps[variant_idx]
            if self.oracle_gp_rff_alpha > 0:
                self.current_func = GPMeanWithRFFPerturbation(
                    oracle_gp=oracle_gp,
                    rng=self.rng,
                    bounds=(
                        np.asarray(self.task.bounds[0], dtype=np.float32),
                        np.asarray(self.task.bounds[1], dtype=np.float32),
                    ),
                    alpha=self.oracle_gp_rff_alpha,
                    rff_length_scale=self.oracle_gp_rff_length_scale,
                    normalize_gp_mean=self.normalize_oracle_gp,
                    rff_multiscale_length_scales=self.oracle_gp_rff_multiscale_ls or None,
                    rff_multiscale_alphas=self.oracle_gp_rff_multiscale_alphas or None,
                    rff_normalize_x=self.oracle_gp_rff_normalize_x,
                )
            else:
                self.current_func = SampledRFFOracleFunction(
                    oracle_gp=oracle_gp,
                    rng=self.rng,
                    bounds=(
                        np.asarray(self.task.bounds[0], dtype=np.float32),
                        np.asarray(self.task.bounds[1], dtype=np.float32),
                    ),
                    normalize=self.normalize_oracle_gp,
                )
        else:
            variant_params = self.variants[variant_idx]
            self.current_func = TaskVariantObjectiveFunction(task_name=self.task_name, variant_params=variant_params)
            self.global_min = float(self.variant_global_mins[variant_idx])
        
        lower, upper = self.current_func.bounds
        self.X_context = self.rng.uniform(
            lower, upper, 
            size=(self.n_init_context, self.current_func.dim)
        ).astype(np.float32)
        self.y_context = self.current_func(self.X_context)

        self.best_y = float(np.min(self.y_context))
        self.best_y_0 = float(self.best_y)  # 记录初始 best，用于 reward
        self.prev_best_y = float(self.best_y)
        self.best_mu = float(self.best_y)

        if self.objective_source in ("oracle_gp", "bnn"):
            # Sobol 探测当前采样函数 (RFF/BNN)，用于：
            #   1. reward 归一化：C_func = 函数的 y_range（替代 C_gp）
            #   2. global_min 估计：仅用于日志中的 regret 监控（不影响 reward 和网络输入）
            n_sobol = int(self.oracle_gp_min_grid_size) ** 2
            sobol_sampler = Sobol(d=int(self.current_func.dim), scramble=True,
                                 seed=int(self.rng.integers(0, 2**31 - 1)))
            X_probe = sobol_sampler.random(n_sobol).astype(np.float32)
            X_probe = X_probe * (upper - lower) + lower
            X_probe = np.vstack([X_probe, self.X_context])
            y_probe = np.asarray(self.current_func(X_probe), dtype=np.float64).reshape(-1)
            self.global_min = float(np.min(y_probe))
            # C_func: 当前采样函数的实际 y_range，使不同 episode 的 reward scale 一致
            y_range_func = float(np.max(y_probe) - np.min(y_probe))
            self.reward_scale = max(y_range_func, 1e-6)
        else:
            self.reward_scale = float(self.variant_y_ranges[variant_idx])

        self.initial_regret = max(1e-8, float(self.best_y - self.global_min))
        self.step_count = 0

        # 生成持久化 Sobol 候选基底
        dim = int(self.current_func.dim)
        sobol = Sobol(d=dim, scramble=True, seed=int(self.rng.integers(0, 100000)))
        sobol_unit = sobol.random(self.n_persistent_base)
        self.persistent_pool = (sobol_unit * (upper - lower) + lower).astype(np.float32)
        self.persistent_available = np.ones(self.n_persistent_base, dtype=bool)

        return self._get_observation()
    
    def _get_observation(self):
        """生成观测（持久基底 + 自适应补充），并缓存候选点供 step() 使用"""
        lower, upper = self.current_func.bounds

        # 获取可用持久点
        X_persistent = self.persistent_pool[self.persistent_available]
        n_pers = X_persistent.shape[0]

        # 计算局部补充数量
        n_fresh = self.n_total_candidates - n_pers
        n_explore = int(n_fresh * self.explore_fraction)
        n_exploit = n_fresh - n_explore

        # 生成局部候选（exploitation）
        X_exploit = self._generate_local_candidates(n_exploit, lower, upper)

        # 生成探索候选（exploration: 围绕距观测最远的持久点）
        dim = int(lower.shape[0])
        X_explore = np.empty((0, dim), dtype=np.float32)
        if n_explore > 0 and self.X_context is not None and len(self.X_context) > 0:
            dists = np.abs(self.persistent_pool[:, None, :] - self.X_context[None, :, :]).max(axis=-1)
            min_dists = dists.min(axis=1)
            explore_center = self.persistent_pool[np.argmax(min_dists)].copy()
            local_h_now = float(self.local_h) * (float(self.local_h_decay) ** int(self.step_count))
            deltas = self.rng.uniform(-local_h_now, local_h_now, size=(n_explore, dim)).astype(np.float32)
            X_explore = np.clip(explore_center.reshape(1, dim) + deltas, lower, upper).astype(np.float32)

        # 合并
        parts = [X_persistent, X_exploit]
        if X_explore.shape[0] > 0:
            parts.append(X_explore)
        X_candidates = np.vstack(parts).astype(np.float32)

        # 记录来源标记
        self._is_persistent = np.concatenate([
            np.ones(n_pers, dtype=np.float32),
            np.zeros(n_fresh, dtype=np.float32),
        ])
        self._n_persistent_in_candidates = n_pers

        pred_mean, pred_std, full_out = predict_tabpfn_with_normalization(
            self.regressor, self.X_context, self.y_context, X_candidates
        )

        # 缓存候选点，供 step() 使用
        self._cached_candidates = X_candidates
        self._cached_pred_mean = pred_mean
        self._cached_pred_std = pred_std

        # 计算 TAF ranking score 作为候选点特征
        taf_rank_norm = None
        if self.taf_obj is not None:
            taf_rank_norm = self._compute_taf_rank_norm(
                X_candidates, pred_mean, pred_std, self.X_context, self.y_context,
            )

        return {
            "X_context": self.X_context.copy(),
            "y_context": self.y_context.copy(),
            "X_candidates": X_candidates,
            "pred_mean": pred_mean,
            "pred_std": pred_std,
            "step": self.step_count,
            "is_persistent": self._is_persistent.copy(),
            "taf_rank_norm": taf_rank_norm,
        }

    def _compute_taf_rank_norm(
        self,
        X_candidates: np.ndarray,
        pred_mean: np.ndarray,
        pred_std: np.ndarray,
        X_context: np.ndarray,
        y_context: np.ndarray,
    ) -> np.ndarray:
        """
        计算 TAF ranking score 并 rank 归一化到 [0, 1]。

        构造 TAF af() 需要的 state 格式，调用 self.taf_obj.af()，
        然后将 raw scores 转换为 rank：0 = 最好, 1 = 最差。
        """
        n_candidates = len(X_candidates)
        dim = X_candidates.shape[1]

        # 构造 TAF 期望的 state: [posterior_mean, posterior_std, x1..xD, incumbent, timestep, budget]
        incumbent = float(y_context.min())
        state = np.zeros((n_candidates, 2 + dim + 3), dtype=np.float32)
        state[:, 0] = pred_mean
        state[:, 1] = pred_std
        state[:, 2:2 + dim] = X_candidates
        state[:, 2 + dim] = incumbent
        state[:, 2 + dim + 1] = float(self.step_count)
        state[:, 2 + dim + 2] = float(self.max_steps)

        # 构造 model_target：包装 TabPFN 以支持 predict_noiseless 接口
        y_ctx = np.asarray(y_context, dtype=np.float32).reshape(-1)
        y_mean_val = float(y_ctx.mean())
        y_std_val = float(max(y_ctx.std(), 1e-8))
        model_target = _TabPFNModelTargetForTAF(self.regressor, y_mean=y_mean_val, y_std=y_std_val)

        # 调用 TAF ranking
        taf_scores = self.taf_obj.af(state, X_context, model_target)

        # rank 归一化: 0 = 最好（TAF score 最高）, 1 = 最差
        ranks = np.argsort(np.argsort(-taf_scores)).astype(np.float32)
        if n_candidates > 1:
            taf_rank_norm = ranks / (n_candidates - 1)
        else:
            taf_rank_norm = np.zeros(n_candidates, dtype=np.float32)

        return taf_rank_norm

    def _generate_local_candidates(self, n_fresh: int, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """
        生成局部候选点：围绕 top-k 上下文点的局部盒邻域均匀采样（h 随 step 衰减）。
        n_fresh 从外部传入（= n_total_candidates - 可用持久点数）。
        """
        dim = int(self.current_func.dim)
        n_fresh = max(0, int(n_fresh))
        if n_fresh == 0:
            return np.empty((0, dim), dtype=np.float32)

        local_h = float(self.local_h) * (float(self.local_h_decay) ** int(self.step_count))

        X_local = np.empty((0, dim), dtype=np.float32)
        if self.k_centers > 0 and self.X_context is not None and len(self.X_context) > 0:
            k = int(self.k_centers)
            # centers: top-k by y (smaller is better)
            order = np.argsort(self.y_context)
            min_center_dist = max(1e-12, 2.0 * local_h)
            centers = []
            for idx in order:
                cand = self.X_context[idx]
                if all(np.max(np.abs(cand - c)) >= min_center_dist for c in centers):
                    centers.append(cand)
                if len(centers) >= k:
                    break
            if not centers:
                centers = [self.X_context[order[0]]]
            centers_available = np.asarray(centers, dtype=np.float32)

            base = n_fresh // k
            rem = n_fresh % k
            chunks = []
            for i in range(k):
                m = base + (1 if i < rem else 0)
                if m <= 0:
                    continue
                center = centers_available[i] if i < centers_available.shape[0] else centers_available[0]
                deltas = self.rng.uniform(-local_h, local_h, size=(m, dim)).astype(np.float32)
                pts = center.reshape(1, dim) + deltas
                pts = np.clip(pts, lower, upper).astype(np.float32)
                chunks.append(pts)
            if chunks:
                X_local = np.vstack(chunks).astype(np.float32)

        # Safety: fill or trim to n_fresh
        if X_local.shape[0] < n_fresh:
            n_missing = n_fresh - X_local.shape[0]
            fill = self.rng.uniform(lower, upper, size=(n_missing, dim)).astype(np.float32)
            X_local = np.vstack([X_local, fill]).astype(np.float32)
        elif X_local.shape[0] > n_fresh:
            X_local = X_local[:n_fresh].astype(np.float32)

        return X_local
    
    def step(self, action_idx: int):
        """
        执行动作：选择第 action_idx 个候选点进行评估

        重要：使用缓存的候选点，而不是重新生成！
        这确保了 RL 选择的 action 对应的点就是实际评估的点。
        """
        # 使用缓存的候选点（由之前的 _get_observation() 生成）
        if self._cached_candidates is None:
            raise RuntimeError("step() 调用前必须先调用 reset() 或确保有缓存的候选点")

        X_candidates = self._cached_candidates
        pred_mean = self._cached_pred_mean
        pred_std = self._cached_pred_std

        x_new = X_candidates[action_idx:action_idx+1]
        y_new = self.current_func(x_new)[0]

        # 标记持久点已消耗
        n_pers = self._n_persistent_in_candidates
        if action_idx < n_pers:
            original_indices = np.where(self.persistent_available)[0]
            self.persistent_available[original_indices[action_idx]] = False

        best_before = float(self.best_y)
        self.best_y = min(float(self.best_y), float(y_new))

        self.best_mu = float(self.best_y)

        self.X_context = np.vstack([self.X_context, x_new])
        self.y_context = np.concatenate([self.y_context, [y_new]])

        self.step_count += 1
        done = self.step_count >= self.max_steps

        cumulative_improvement = max(0.0, self.best_y_0 - self.best_y) / self.reward_scale
        delta_improvement = max(0.0, best_before - self.best_y) / self.reward_scale
        best_regret_before = max(0.0, best_before - float(self.global_min))
        best_regret_now = max(0.0, float(self.best_y) - float(self.global_min))

        if self.reward_mode == "auc":
            reward = cumulative_improvement
        elif self.reward_mode == "delta":
            reward = delta_improvement
        elif self.reward_mode == "delta_terminal":
            reward = delta_improvement
            if done:
                reward += self.reward_terminal_weight * cumulative_improvement
        elif self.reward_mode == "mixed":
            lam = self.reward_mixed_lambda
            reward = lam * cumulative_improvement + (1.0 - lam) * delta_improvement
        elif self.reward_mode == "frontload_mixed":
            lam = self.reward_mixed_lambda
            step_progress = float(self.step_count) / max(float(self.max_steps), 1.0)
            frontload_weight = max(0.0, 1.0 - step_progress + (1.0 / max(float(self.max_steps), 1.0)))
            frontload_weight = frontload_weight ** self.reward_frontload_power
            reward = frontload_weight * (
                lam * cumulative_improvement + (1.0 - lam) * delta_improvement
            )
            if done:
                reward += self.reward_terminal_weight * cumulative_improvement
        elif self.reward_mode == "staged_mixed":
            lam = self.reward_mixed_lambda
            step_progress = float(self.step_count) / max(float(self.max_steps), 1.0)
            stage_midpoint = min(max(self.reward_stage_midpoint, 1e-6), 1.0)
            early_weight = max(0.0, 1.0 - step_progress / stage_midpoint)
            early_weight = early_weight ** self.reward_frontload_power
            dense_reward = lam * cumulative_improvement + (1.0 - lam) * delta_improvement
            reward = early_weight * dense_reward + (1.0 - early_weight) * delta_improvement
            if done:
                reward += self.reward_terminal_weight * cumulative_improvement
        elif self.reward_mode == "regret_balanced":
            norm = max(
                float(self.initial_regret),
                float(self.reward_regret_scale_floor_ratio) * float(self.reward_scale),
                1e-8,
            )
            prev_progress = np.clip((float(self.initial_regret) - best_regret_before) / norm, 0.0, 1.0)
            progress = np.clip((float(self.initial_regret) - best_regret_now) / norm, 0.0, 1.0)
            delta_progress = max(0.0, float(progress) - float(prev_progress))
            early_weight = (
                (float(self.max_steps) - float(self.step_count) + 1.0) / max(float(self.max_steps), 1.0)
            ) ** float(self.reward_regret_early_power)
            reward = (
                float(self.reward_regret_auc_weight) * (float(progress) / max(float(self.max_steps), 1.0))
                + float(self.reward_regret_delta_weight) * float(early_weight) * float(delta_progress)
            )
            if done:
                reward += float(self.reward_terminal_weight) * (
                    float(progress) ** float(self.reward_regret_terminal_power)
                )
        else:
            raise RuntimeError(f"Unsupported reward_mode: {self.reward_mode}")

        self.prev_best_y = float(self.best_y)

        # 生成下一个观测（会缓存新的候选点）
        next_obs = None if done else self._get_observation()
        
        info = {
            "y_new": y_new,
            "best_y": self.best_y,
            "best_mu": self.best_mu,
            "global_min": self.global_min,
            "regret": float(self.best_y - self.global_min),
            "regret_y": float(self.best_y - self.global_min),
            "step": self.step_count,
        }
        
        return next_obs, reward, done, info
    


# ==================== Rollout Buffer ====================
class ImprovedRolloutBuffer:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.context_feats = []
        self.candidate_feats = []
        self.steps = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def add(self, context_feat, candidate_feat, step, action, reward, value, log_prob, done):
        self.context_feats.append(context_feat)
        self.candidate_feats.append(candidate_feat)
        self.steps.append(step)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(float(done))
    
    def get(self, last_value=0.0, gamma=0.99, lam=0.95):
        # GAE
        advantages = []
        gae = 0
        values = self.values + [last_value]
        
        for t in reversed(range(len(self.rewards))):
            mask = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * values[t + 1] * mask - values[t]
            gae = delta + gamma * lam * mask * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        
        return {
            "context_feats": self.context_feats,
            "candidate_feats": self.candidate_feats,
            "steps": self.steps,
            "actions": self.actions,
            "log_probs": self.log_probs,
            "advantages": advantages,
            "returns": returns,
        }
    
    def __len__(self):
        return len(self.actions)


# ==================== Parallel Rollout Collection ====================

# Per-worker globals (set by _train_worker_init, used by _train_worker_collect)
_w_env = None   # ImprovedBraninBOEnv
_w_ppo = None   # ImprovedPPO (inference only)


def _train_worker_init(env_kwargs: dict, ppo_kwargs: dict):
    """Initializer for each worker process (called once per worker via Pool)."""
    global _w_env, _w_ppo

    import torch as _torch
    tpw = os.environ.get("OMP_NUM_THREADS")
    if tpw is not None:
        _torch.set_num_threads(int(tpw))

    import warnings as _w
    _w.filterwarnings("ignore")

    _w_env = ImprovedBraninBOEnv(**env_kwargs)
    _w_ppo = ImprovedPPO(**ppo_kwargs)
    _w_ppo.policy.eval()


def _train_worker_collect(work_item):
    """Run episodes and return rollouts + metrics.

    work_item: (state_dict_bytes, episode_specs, max_steps)
        state_dict_bytes: bytes from torch.save(state_dict, BytesIO)
        episode_specs: list of (variant_idx, ep_seed)
        max_steps: int
    Returns: list of (transitions, episode_reward, final_regret, best_y)
    """
    import io
    import torch as _torch

    state_dict_bytes, episode_specs, max_steps = work_item

    # Load policy weights
    buf = io.BytesIO(state_dict_bytes)
    sd = _torch.load(buf, map_location=_w_ppo.device, weights_only=True)
    _w_ppo.policy.load_state_dict(sd)
    _w_ppo.policy.eval()

    results = []
    for variant_idx, ep_seed in episode_specs:
        obs = _w_env.reset(seed=ep_seed, variant_idx=variant_idx)
        bounds = _w_env.current_func.bounds
        episode_reward = 0.0
        transitions = []

        for step in range(max_steps):
            X_context = obs["X_context"]
            y_context = obs["y_context"]
            X_candidates = obs["X_candidates"]
            pred_mean = obs["pred_mean"]
            pred_std = obs["pred_std"]
            current_step = obs["step"]
            is_persistent = obs["is_persistent"]
            taf_rank_norm = obs.get("taf_rank_norm", None)

            action, log_prob, value = _w_ppo.select_candidate(
                X_context, y_context, X_candidates, pred_mean, pred_std, bounds, current_step,
                is_persistent=is_persistent, taf_rank_norm=taf_rank_norm,
            )

            next_obs, reward, done, info = _w_env.step(action)
            episode_reward += reward

            context_feat, candidate_feat = _w_ppo._build_features(
                X_context, y_context, X_candidates, pred_mean, pred_std, bounds, current_step,
                is_persistent=is_persistent, taf_rank_norm=taf_rank_norm,
            )
            transitions.append((
                context_feat, candidate_feat, current_step,
                action, reward, value, log_prob, done,
            ))

            if done:
                break
            obs = next_obs

        results.append((transitions, episode_reward, info["regret"], info["best_y"]))
    return results


# ==================== 训练函数 ====================
def train_improved(
    task_name: str = "branin_family",
    variants_path: str = "./data/variants_train.npz",
    trajectories_path: str = "./data/bo_trajs_train.npz",
    objective_source: str = "oracle_gp",
    bnn_params_path: str = None,
    bnn_rff_alpha: float = 5.0,
    bnn_rff_length_scale: float = 0.3,
    normalize_bnn: bool = False,
    normalize_oracle_gp: bool = False,
    oracle_gp_y_transform: str = "none",
    oracle_gp_rff_alpha: float = 0.0,
    oracle_gp_rff_length_scale: float = 0.3,
    oracle_gp_rff_multiscale: str = "",
    oracle_gp_rff_multiscale_alphas: str = "",
    oracle_gp_rff_normalize_x: bool = False,
    model_path: str = None,
    taf_data_path: str = None,
    total_episodes: int = 5000,  # 增加训练量
    max_steps: int = 18,
    n_init_context: int = 2,
    n_persistent_base: int = 128,
    n_total_candidates: int = 192,
    k_centers: int = 2,
    local_h: float = 1.5,
    local_h_decay: float = 0.9,
    explore_fraction: float = 0.0,
    oracle_gp_min_grid_size: int = 80,
    oracle_gp_fixed_length_scale: float = 0.0,
    oracle_gp_alpha: float = 1e-6,
    oracle_gp_min_n_lbfgs_starts: int = 25,
    reward_mode: str = "auc",
    reward_mixed_lambda: float = 0.3,
    reward_terminal_weight: float = 1.0,
    reward_frontload_power: float = 1.0,
    reward_stage_midpoint: float = 0.4,
    reward_regret_auc_weight: float = 0.2,
    reward_regret_delta_weight: float = 1.0,
    reward_regret_early_power: float = 0.5,
    reward_regret_terminal_power: float = 3.0,
    reward_regret_scale_floor_ratio: float = 0.02,
    variant_sampling: str = "random",
    inference_precision: str = "float32",
    update_every: int = 20,  # 更频繁的更新
    save_every: int = 500,
    save_dir: str = "./runs/ppo_bo_v3_improved",
    seed: int = 42,
    use_tensorboard: bool = True,  # TensorBoard 开关
    # 网络超参数
    hidden_dim: int = 128,
    n_self_attn_layers: int = 3,
    n_cross_attn_layers: int = 3,
    n_heads: int = 8,
    ent_coef_start: float = 0.02,
    ent_coef_end: float = 0.02,
    n_workers: int = 1,
):
    os.makedirs(save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    task_spec = get_task(str(task_name))
    coord_dim = int(task_spec.dim)
    print(f"\n{'='*70}")
    print(f"Improved PPO Training for Bayesian Optimization (v3)")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Total Episodes: {total_episodes}")
    print(f"Init points per episode: {n_init_context}")
    print(f"Steps per Episode (actions): {max_steps} (total evals = {n_init_context + max_steps})")
    print(f"Persistent base: {n_persistent_base}, Total candidates: {n_total_candidates}")
    print(f"Update Every: {update_every} episodes")
    print(f"网络: hidden={hidden_dim}, self_attn={n_self_attn_layers}, cross_attn={n_cross_attn_layers}, heads={n_heads}")
    print(f"Objective source: {objective_source}")
    if objective_source == "bnn":
        print(f"BNN RFF perturbation: alpha={bnn_rff_alpha}, length_scale={bnn_rff_length_scale}")
    if objective_source == "oracle_gp" and oracle_gp_fixed_length_scale > 0:
        print(
            f"Oracle GP fitting: fixed_length_scale={oracle_gp_fixed_length_scale}, "
            f"alpha={oracle_gp_alpha}"
        )
    if objective_source == "oracle_gp" and oracle_gp_rff_alpha > 0:
        if oracle_gp_rff_multiscale:
            print(
                f"Oracle GP mode: GP_mean + MultiScale-RFF "
                f"(ls={oracle_gp_rff_multiscale}, "
                f"alphas={oracle_gp_rff_multiscale_alphas or oracle_gp_rff_alpha}, "
                f"normalize_x={oracle_gp_rff_normalize_x})"
            )
        else:
            print(
                f"Oracle GP mode: GP_mean + alpha*RFF_prior "
                f"(alpha={oracle_gp_rff_alpha}, ls={oracle_gp_rff_length_scale}, "
                f"normalize_x={oracle_gp_rff_normalize_x})"
            )
    elif objective_source == "oracle_gp":
        print(f"Oracle GP mode: posterior RFF sampling (classic)")
    print(
        f"Reward: mode={reward_mode}, mixed_lambda={reward_mixed_lambda}, "
        f"terminal_weight={reward_terminal_weight}, frontload_power={reward_frontload_power}, "
        f"stage_midpoint={reward_stage_midpoint}, regret_auc_weight={reward_regret_auc_weight}, "
        f"regret_delta_weight={reward_regret_delta_weight}, regret_early_power={reward_regret_early_power}, "
        f"regret_terminal_power={reward_regret_terminal_power}, "
        f"regret_scale_floor_ratio={reward_regret_scale_floor_ratio}"
    )
    print(f"Variant sampling: {variant_sampling}")
    print(f"Entropy coef: {ent_coef_start} -> {ent_coef_end}")
    print(f"Task: {task_name}")
    print(f"Task dim: {coord_dim}")
    print(f"TabPFN inference precision: {inference_precision}")
    print(f"变体文件: {variants_path}")
    print(f"轨迹缓存: {trajectories_path}")
    print(f"{'='*70}\n")
    
    # 保存配置
    config = {
        "task_name": task_name,
        "coord_dim": coord_dim,
        "variants_path": variants_path,
        "trajectories_path": trajectories_path,
        "objective_source": objective_source,
        "bnn_params_path": bnn_params_path,
        "bnn_rff_alpha": bnn_rff_alpha,
        "bnn_rff_length_scale": bnn_rff_length_scale,
        "normalize_bnn": normalize_bnn,
        "normalize_oracle_gp": normalize_oracle_gp,
        "oracle_gp_y_transform": oracle_gp_y_transform,
        "oracle_gp_rff_alpha": oracle_gp_rff_alpha,
        "oracle_gp_rff_length_scale": oracle_gp_rff_length_scale,
        "oracle_gp_rff_multiscale": oracle_gp_rff_multiscale,
        "oracle_gp_rff_multiscale_alphas": oracle_gp_rff_multiscale_alphas,
        "oracle_gp_rff_normalize_x": oracle_gp_rff_normalize_x,
        "model_path": model_path,
        "total_episodes": total_episodes,
        "max_steps": max_steps,
        "n_init_context": n_init_context,
        "n_persistent_base": n_persistent_base,
        "n_total_candidates": n_total_candidates,
        "k_centers": k_centers,
        "local_h": local_h,
        "local_h_decay": local_h_decay,
        "explore_fraction": explore_fraction,
        "n_workers": n_workers,
        "oracle_gp_min_grid_size": oracle_gp_min_grid_size,
        "oracle_gp_fixed_length_scale": oracle_gp_fixed_length_scale,
        "oracle_gp_alpha": oracle_gp_alpha,
        "reward_mode": reward_mode,
        "reward_mixed_lambda": reward_mixed_lambda,
        "reward_terminal_weight": reward_terminal_weight,
        "reward_frontload_power": reward_frontload_power,
        "reward_stage_midpoint": reward_stage_midpoint,
        "reward_regret_auc_weight": reward_regret_auc_weight,
        "reward_regret_delta_weight": reward_regret_delta_weight,
        "reward_regret_early_power": reward_regret_early_power,
        "reward_regret_terminal_power": reward_regret_terminal_power,
        "reward_regret_scale_floor_ratio": reward_regret_scale_floor_ratio,
        "variant_sampling": variant_sampling,
        "hidden_dim": hidden_dim,
        "n_self_attn_layers": n_self_attn_layers,
        "n_cross_attn_layers": n_cross_attn_layers,
        "n_heads": n_heads,
        "ent_coef_start": ent_coef_start,
        "ent_coef_end": ent_coef_end,
        "taf_data_path": taf_data_path,
        "seed": seed,
        "version": "v3_improved",
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # 初始化 TensorBoard
    writer = None
    if use_tensorboard:
        tb_dir = os.path.join(save_dir, "tensorboard")
        writer = SummaryWriter(log_dir=tb_dir)
        print(f"TensorBoard 日志目录: {tb_dir}")
    
    # 创建环境
    env = ImprovedBraninBOEnv(
        task_name=task_name,
        variants_path=variants_path,
        trajectories_path=trajectories_path,
        objective_source=objective_source,
        bnn_params_path=bnn_params_path,
        bnn_rff_alpha=bnn_rff_alpha,
        bnn_rff_length_scale=bnn_rff_length_scale,
        normalize_bnn=normalize_bnn,
        normalize_oracle_gp=normalize_oracle_gp,
        oracle_gp_y_transform=oracle_gp_y_transform,
        oracle_gp_rff_alpha=oracle_gp_rff_alpha,
        oracle_gp_rff_length_scale=oracle_gp_rff_length_scale,
        oracle_gp_rff_multiscale=oracle_gp_rff_multiscale,
        oracle_gp_rff_multiscale_alphas=oracle_gp_rff_multiscale_alphas,
        oracle_gp_rff_normalize_x=oracle_gp_rff_normalize_x,
        model_path=model_path,
        taf_data_path=taf_data_path,
        max_steps=max_steps,
        n_init_context=n_init_context,
        n_persistent_base=n_persistent_base,
        n_total_candidates=n_total_candidates,
        k_centers=k_centers,
        local_h=local_h,
        local_h_decay=local_h_decay,
        explore_fraction=explore_fraction,
        oracle_gp_min_grid_size=oracle_gp_min_grid_size,
        oracle_gp_fixed_length_scale=oracle_gp_fixed_length_scale,
        oracle_gp_alpha=oracle_gp_alpha,
        oracle_gp_min_n_lbfgs_starts=oracle_gp_min_n_lbfgs_starts,
        reward_mode=reward_mode,
        reward_mixed_lambda=reward_mixed_lambda,
        reward_terminal_weight=reward_terminal_weight,
        reward_frontload_power=reward_frontload_power,
        reward_stage_midpoint=reward_stage_midpoint,
        reward_regret_auc_weight=reward_regret_auc_weight,
        reward_regret_delta_weight=reward_regret_delta_weight,
        reward_regret_early_power=reward_regret_early_power,
        reward_regret_terminal_power=reward_regret_terminal_power,
        reward_regret_scale_floor_ratio=reward_regret_scale_floor_ratio,
        variant_sampling=variant_sampling,
        inference_precision=inference_precision,
        device=device,
        seed=seed,
    )
    print(f"环境已创建: {env.num_variants} 个训练任务 (objective_source={env.objective_source})")

    # 创建 PPO
    use_taf_feature = (taf_data_path is not None)
    ppo = ImprovedPPO(
        coord_dim=coord_dim,
        hidden_dim=hidden_dim,
        n_self_attn_layers=n_self_attn_layers,
        n_cross_attn_layers=n_cross_attn_layers,
        n_heads=n_heads,
        max_steps=max_steps,
        device=device,
        use_taf_feature=use_taf_feature,
        ent_coef=ent_coef_start,
        ent_coef_end=ent_coef_end,
    )
    print(f"PPO 网络参数量: {sum(p.numel() for p in ppo.policy.parameters()):,}")

    # 设置 LR 调度器：线性 warmup (5%) + cosine 衰减
    total_updates = total_episodes // update_every
    ppo.setup_scheduler(total_updates, warmup_fraction=0.05)
    print(f"LR 调度: warmup {int(total_updates * 0.05)} updates + cosine decay, total {total_updates} updates")

    buffer = ImprovedRolloutBuffer()
    
    # 训练记录
    metrics = {
        "episode_reward": [],
        "episode_regret": [],
        "best_regret": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
    }
    
    best_avg_regret = float('inf')
    recent_regrets = deque(maxlen=100)
    
    # ── Helper: process one batch of episode results (shared by serial / parallel) ──
    def _process_episode_results(episode_num, episode_reward, final_regret, best_y):
        """Update metrics, best model, tensorboard for a single episode."""
        nonlocal best_avg_regret
        metrics["episode_reward"].append(episode_reward)
        metrics["episode_regret"].append(final_regret)
        recent_regrets.append(final_regret)
        if writer is not None:
            writer.add_scalar("Episode/Reward", episode_reward, episode_num)
            writer.add_scalar("Episode/Regret", final_regret, episode_num)
            writer.add_scalar("Episode/BestY", best_y, episode_num)
        avg = float(np.mean(recent_regrets))
        if len(recent_regrets) == recent_regrets.maxlen and avg < best_avg_regret:
            best_avg_regret = avg
            torch.save(ppo.policy.state_dict(), os.path.join(save_dir, "ppo_best.pt"))
        metrics["best_regret"].append(best_avg_regret)

    def _do_ppo_update(episode_num):
        """PPO update + logging + checkpoint."""
        if len(buffer) == 0:
            return
        rollout = buffer.get(last_value=0.0, gamma=ppo.gamma, lam=ppo.lam)
        losses = ppo.update(rollout, n_epochs=4, batch_size=64)
        buffer.reset()
        metrics["policy_loss"].append(losses["policy_loss"])
        metrics["value_loss"].append(losses["value_loss"])
        metrics["entropy"].append(losses["entropy"])
        if writer is not None:
            writer.add_scalar("Train/PolicyLoss", losses["policy_loss"], episode_num)
            writer.add_scalar("Train/ValueLoss", losses["value_loss"], episode_num)
            writer.add_scalar("Train/Entropy", losses["entropy"], episode_num)
            writer.add_scalar("Train/EntropyCoef", losses["ent_coef"], episode_num)
            writer.add_scalar("Train/LearningRate", losses["lr"], episode_num)
            writer.add_scalar("Train/AvgRegret_100ep", np.mean(list(recent_regrets)), episode_num)
            writer.add_scalar("Train/BestAvgRegret", best_avg_regret, episode_num)
        if episode_num % 10 == 0:
            avg_reward = np.mean(metrics["episode_reward"][-100:])
            avg_regret = np.mean(metrics["episode_regret"][-100:])
            print(f"Episode {episode_num:5d} | Reward: {avg_reward:6.2f} | Regret: {avg_regret:.4f} | "
                  f"Best: {best_avg_regret:.4f} | LR: {losses['lr']:.2e}")
        if episode_num % save_every == 0:
            sp = os.path.join(save_dir, f"ppo_ep{episode_num}.pt")
            torch.save(ppo.policy.state_dict(), sp)
            with open(os.path.join(save_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f)
            print(f"\n模型已保存: {sp}\n")

    print(f"\n开始训练... (n_workers={n_workers})\n")

    if n_workers <= 1:
        # ──────── Serial path (original) ────────
        for episode in range(1, total_episodes + 1):
            obs = env.reset()
            episode_reward = 0
            bounds = env.current_func.bounds

            for step in range(max_steps):
                X_context = obs["X_context"]
                y_context = obs["y_context"]
                X_candidates = obs["X_candidates"]
                pred_mean = obs["pred_mean"]
                pred_std = obs["pred_std"]
                current_step = obs["step"]
                is_persistent = obs["is_persistent"]
                taf_rank_norm = obs.get("taf_rank_norm", None)

                action, log_prob, value = ppo.select_candidate(
                    X_context, y_context, X_candidates, pred_mean, pred_std, bounds, current_step,
                    is_persistent=is_persistent, taf_rank_norm=taf_rank_norm,
                )
                next_obs, reward, done, info = env.step(action)
                episode_reward += reward

                context_feat, candidate_feat = ppo._build_features(
                    X_context, y_context, X_candidates, pred_mean, pred_std, bounds, current_step,
                    is_persistent=is_persistent, taf_rank_norm=taf_rank_norm,
                )
                buffer.add(context_feat, candidate_feat, current_step, action, reward, value, log_prob, done)
                if done:
                    break
                obs = next_obs

            _process_episode_results(episode, episode_reward, info["regret"], info["best_y"])
            if episode % update_every == 0:
                _do_ppo_update(episode)

    else:
        # ──────── Parallel path ────────
        import io as _io
        import multiprocessing as _mp

        n_cpus = os.cpu_count() or 1
        threads_per_worker = max(1, n_cpus // n_workers)
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                     "NUMEXPR_MAX_THREADS"):
            os.environ[var] = str(threads_per_worker)
        print(f"Parallel: {n_workers} workers, {threads_per_worker} threads/worker ({n_cpus} CPUs)")

        # Build kwargs dicts for worker init (must be picklable)
        env_kwargs = dict(
            task_name=task_name, variants_path=variants_path,
            trajectories_path=trajectories_path, objective_source=objective_source,
            bnn_params_path=bnn_params_path, bnn_rff_alpha=bnn_rff_alpha,
            bnn_rff_length_scale=bnn_rff_length_scale, normalize_bnn=normalize_bnn,
            normalize_oracle_gp=normalize_oracle_gp,
            oracle_gp_y_transform=oracle_gp_y_transform,
            oracle_gp_rff_alpha=oracle_gp_rff_alpha,
            oracle_gp_rff_length_scale=oracle_gp_rff_length_scale,
            oracle_gp_rff_multiscale=oracle_gp_rff_multiscale,
            oracle_gp_rff_multiscale_alphas=oracle_gp_rff_multiscale_alphas,
            oracle_gp_rff_normalize_x=oracle_gp_rff_normalize_x,
            model_path=model_path,
            taf_data_path=taf_data_path, max_steps=max_steps,
            n_init_context=n_init_context, n_persistent_base=n_persistent_base,
            n_total_candidates=n_total_candidates, k_centers=k_centers,
            local_h=local_h, local_h_decay=local_h_decay,
            explore_fraction=explore_fraction,
            oracle_gp_min_grid_size=oracle_gp_min_grid_size,
            oracle_gp_fixed_length_scale=oracle_gp_fixed_length_scale,
            oracle_gp_alpha=oracle_gp_alpha,
            oracle_gp_min_n_lbfgs_starts=oracle_gp_min_n_lbfgs_starts,
            reward_mode=reward_mode, reward_mixed_lambda=reward_mixed_lambda,
            reward_terminal_weight=reward_terminal_weight,
            reward_frontload_power=reward_frontload_power,
            reward_stage_midpoint=reward_stage_midpoint,
            reward_regret_auc_weight=reward_regret_auc_weight,
            reward_regret_delta_weight=reward_regret_delta_weight,
            reward_regret_early_power=reward_regret_early_power,
            reward_regret_terminal_power=reward_regret_terminal_power,
            reward_regret_scale_floor_ratio=reward_regret_scale_floor_ratio,
            variant_sampling="random",  # workers don't manage cycle; main controls it
            inference_precision=inference_precision,
            device=device, seed=seed,
        )
        ppo_kwargs = dict(
            coord_dim=coord_dim, hidden_dim=hidden_dim,
            n_self_attn_layers=n_self_attn_layers,
            n_cross_attn_layers=n_cross_attn_layers,
            n_heads=n_heads, max_steps=max_steps, device=device,
            use_taf_feature=(taf_data_path is not None),
            ent_coef=ent_coef_start, ent_coef_end=ent_coef_end,
        )

        ctx = _mp.get_context("spawn")
        pool = ctx.Pool(processes=n_workers, initializer=_train_worker_init,
                        initargs=(env_kwargs, ppo_kwargs))

        try:
            n_batches = total_episodes // update_every
            rng_main = np.random.default_rng(seed)

            for batch_idx in range(n_batches):
                batch_start_ep = batch_idx * update_every + 1

                # Pre-generate variant indices and seeds for this batch
                episode_specs_all = []
                for _ in range(update_every):
                    vi = env._sample_variant_idx()
                    es = int(rng_main.integers(0, 2**31))
                    episode_specs_all.append((vi, es))

                # Serialize current policy weights once
                buf_io = _io.BytesIO()
                torch.save(ppo.policy.state_dict(), buf_io)
                sd_bytes = buf_io.getvalue()

                # Split across workers
                work_items = []
                eps_per_worker = update_every // n_workers
                remainder = update_every % n_workers
                offset = 0
                for w in range(n_workers):
                    n_ep = eps_per_worker + (1 if w < remainder else 0)
                    specs = episode_specs_all[offset:offset + n_ep]
                    offset += n_ep
                    work_items.append((sd_bytes, specs, max_steps))

                # Parallel collection
                all_results = pool.map(_train_worker_collect, work_items)

                # Process results
                ep_counter = batch_start_ep
                for worker_results in all_results:
                    for transitions, ep_reward, ep_regret, ep_best_y in worker_results:
                        for t in transitions:
                            ctx_feat, cand_feat, step_t, act, rew, val, lp, dn = t
                            buffer.add(ctx_feat, cand_feat, step_t, act, rew, val, lp, dn)
                        _process_episode_results(ep_counter, ep_reward, ep_regret, ep_best_y)
                        ep_counter += 1

                # PPO update at end of batch
                _do_ppo_update(batch_start_ep + update_every - 1)

        finally:
            pool.close()
            pool.join()

    # ── Save final model ──
    final_path = os.path.join(save_dir, "ppo_final.pt")
    torch.save(ppo.policy.state_dict(), final_path)
    print(f"\n最终模型已保存: {final_path}")

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)

    if writer is not None:
        writer.close()
        print("TensorBoard writer 已关闭")

    return ppo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="branin_family")
    parser.add_argument("--variants_path", type=str, default="./data/variants_train.npz")
    parser.add_argument("--trajectories_path", type=str, default="./data/bo_trajs_train.npz")
    parser.add_argument("--objective_source", type=str, choices=["oracle_gp", "direct", "branin", "bnn"], default="oracle_gp")
    parser.add_argument("--bnn_params_path", type=str, default=None,
                       help="Path to BNN variational parameters .npz (required when objective_source='bnn').")
    parser.add_argument("--bnn_rff_alpha", type=float, default=5.0,
                       help="RFF perturbation strength for BNN mode. Controls diversity vs fidelity.")
    parser.add_argument("--bnn_rff_length_scale", type=float, default=0.3,
                       help="RFF length scale (Matern 2.5) for BNN mode. Smaller = rougher perturbation.")
    parser.add_argument("--normalize_bnn", action="store_true", default=False,
                       help="Z-normalize BNN mean output before adding RFF perturbation. "
                            "Fixes diversity collapse when BNN y_range >> RFF range (e.g. HPLC). "
                            "Use with --bnn_rff_alpha 1.5.")
    parser.add_argument("--normalize_oracle_gp", action="store_true", default=False,
                       help="Z-normalize oracle GP RFF posterior sample output. "
                            "Stabilizes reward scale across episodes for tasks with large y-range "
                            "(e.g. Branin). Recommended with --reward_mode regret_balanced.")
    parser.add_argument("--oracle_gp_y_transform", type=str, choices=["none", "sqrt", "log"],
                       default="none",
                       help="Monotonic transform applied to y before oracle GP fitting. "
                            "'sqrt' compresses large y-ranges (e.g. Branin [0.4,300] -> [1,17]), "
                            "improving GP length-scale estimation and RFF sample diversity. "
                            "Default 'none' preserves existing behavior.")
    parser.add_argument("--oracle_gp_fixed_length_scale", type=float, default=0.0,
                       help="When > 0, fit oracle GP with this fixed Matern length scale and optimizer=None. "
                            "Default 0 preserves hyperparameter optimization.")
    parser.add_argument("--oracle_gp_alpha", type=float, default=1e-6,
                       help="Oracle GP observation noise alpha used when fitting trajectory GPs. "
                            "Default preserves existing behavior.")
    parser.add_argument("--oracle_gp_rff_alpha", type=float, default=0.0,
                       help="When > 0, use GP_mean + alpha*RFF_prior instead of posterior RFF sampling. "
                            "Decouples diversity from GP posterior variance. "
                            "Recommended for tasks with low RFF diversity (e.g. Branin). "
                            "0 = classic posterior sampling (default).")
    parser.add_argument("--oracle_gp_rff_length_scale", type=float, default=0.3,
                       help="RFF prior length scale (Matern 2.5) for oracle_gp mode. "
                            "Only used when --oracle_gp_rff_alpha > 0.")
    parser.add_argument("--oracle_gp_rff_multiscale", type=str, default="",
                       help="Comma-separated length scales for multi-scale RFF perturbation. "
                            "E.g. '0.01,0.025,0.061,0.15'. Creates multi-modal training "
                            "functions for low-dim tasks. Requires --oracle_gp_rff_alpha > 0.")
    parser.add_argument("--oracle_gp_rff_multiscale_alphas", type=str, default="",
                       help="Comma-separated per-scale alphas for multi-scale RFF. "
                            "If empty, uses --oracle_gp_rff_alpha for all scales.")
    parser.add_argument("--oracle_gp_rff_normalize_x", action="store_true", default=False,
                       help="Normalize X to [0,1]^d inside oracle-GP RFF perturbations before applying "
                            "length scales. Default false preserves existing raw-coordinate behavior.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to finetuned TabPFN checkpoint. None = use base model.")
    parser.add_argument("--taf_data_path", type=str, default=None,
                       help="Path to TAF source data pickle. When provided, TAF ranking score is added as candidate feature.")
    parser.add_argument("--total_episodes", type=int, default=5000)
    parser.add_argument("--max_steps", type=int, default=18, help="Number of BO actions per episode (total evals = n_init_context + max_steps).")
    parser.add_argument("--n_init_context", type=int, default=2, help="Number of initial random context points per episode.")
    parser.add_argument("--n_persistent_base", type=int, default=128, help="Number of persistent Sobol base candidates per episode")
    parser.add_argument("--n_total_candidates", type=int, default=192, help="Total candidates per step (persistent + adaptive local)")
    parser.add_argument("--k_centers", type=int, default=2)
    parser.add_argument("--local_h", type=float, default=1.5)
    parser.add_argument("--local_h_decay", type=float, default=0.9)
    parser.add_argument("--explore_fraction", type=float, default=0.0,
                        help="Fraction of fresh candidates for exploration (0=off)")
    parser.add_argument(
        "--oracle_gp_min_grid_size",
        type=int,
        default=80,
        help="Grid size for estimating per-episode sampled-function global_min (dim==2: grid; dim!=2: Sobol(grid_size^2)).",
    )
    parser.add_argument(
        "--oracle_gp_min_n_lbfgs_starts",
        type=int,
        default=25,
        help="Number of L-BFGS-B multi-start for per-episode global_min estimation. Reduce to speed up.",
    )
    parser.add_argument(
        "--inference_precision",
        type=str,
        choices=["float32", "float16"],
        default="float32",
        help="TabPFN inference precision. float16 is ~1.5-2x faster on GPU with minimal quality loss.",
    )
    parser.add_argument(
        "--reward_mode",
        type=str,
        choices=["auc", "delta", "delta_terminal", "mixed", "frontload_mixed", "staged_mixed", "regret_balanced"],
        default="auc",
        help="Per-step reward design for PPO training.",
    )
    parser.add_argument(
        "--reward_mixed_lambda",
        type=float,
        default=0.3,
        help="Lambda for mixed reward: lambda * cumulative + (1-lambda) * delta.",
    )
    parser.add_argument(
        "--reward_terminal_weight",
        type=float,
        default=1.0,
        help="Terminal bonus weight for delta_terminal reward.",
    )
    parser.add_argument(
        "--reward_frontload_power",
        type=float,
        default=1.0,
        help="Front-load exponent for frontload_mixed reward. Larger values emphasize earlier steps more strongly.",
    )
    parser.add_argument(
        "--reward_stage_midpoint",
        type=float,
        default=0.4,
        help="For staged_mixed reward, fraction of rollout reserved for early dense shaping before fading toward delta reward.",
    )
    parser.add_argument(
        "--reward_regret_auc_weight",
        type=float,
        default=0.2,
        help="For regret_balanced reward, weight on dense best-regret progress term.",
    )
    parser.add_argument(
        "--reward_regret_delta_weight",
        type=float,
        default=1.0,
        help="For regret_balanced reward, weight on early delta-progress term.",
    )
    parser.add_argument(
        "--reward_regret_early_power",
        type=float,
        default=0.5,
        help="For regret_balanced reward, exponent controlling how strongly earlier progress is emphasized.",
    )
    parser.add_argument(
        "--reward_regret_terminal_power",
        type=float,
        default=3.0,
        help="For regret_balanced reward, exponent on terminal best-regret progress bonus.",
    )
    parser.add_argument(
        "--reward_regret_scale_floor_ratio",
        type=float,
        default=0.02,
        help="For regret_balanced reward, floor ratio multiplying reward_scale to stabilize normalization across episodes.",
    )
    parser.add_argument(
        "--variant_sampling",
        type=str,
        choices=["random", "shuffled_cycle"],
        default="random",
        help="How to sample training variants across episodes. "
             "shuffled_cycle guarantees balanced coverage inside each cycle of K variants.",
    )
    parser.add_argument("--n_workers", type=int, default=1,
                        help="Number of parallel workers for rollout collection. "
                             "Each worker has its own env (TabPFN + TAF GPs). "
                             "CPU threads auto-divided across workers.")
    parser.add_argument("--update_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--save_dir", type=str, default="./runs/ppo_bo_v3_improved")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_tensorboard", action="store_true", default=True)
    parser.add_argument("--no_tensorboard", dest="use_tensorboard", action="store_false")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_self_attn_layers", type=int, default=3)
    parser.add_argument("--n_cross_attn_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--ent_coef_start", type=float, default=0.02)
    parser.add_argument("--ent_coef_end", type=float, default=0.02)
    args = parser.parse_args()

    train_improved(
        task_name=args.task,
        variants_path=args.variants_path,
        trajectories_path=args.trajectories_path,
        objective_source=args.objective_source,
        bnn_params_path=args.bnn_params_path,
        bnn_rff_alpha=args.bnn_rff_alpha,
        bnn_rff_length_scale=args.bnn_rff_length_scale,
        normalize_bnn=args.normalize_bnn,
        normalize_oracle_gp=args.normalize_oracle_gp,
        oracle_gp_y_transform=args.oracle_gp_y_transform,
        oracle_gp_fixed_length_scale=args.oracle_gp_fixed_length_scale,
        oracle_gp_alpha=args.oracle_gp_alpha,
        oracle_gp_rff_alpha=args.oracle_gp_rff_alpha,
        oracle_gp_rff_length_scale=args.oracle_gp_rff_length_scale,
        oracle_gp_rff_multiscale=args.oracle_gp_rff_multiscale,
        oracle_gp_rff_multiscale_alphas=args.oracle_gp_rff_multiscale_alphas,
        oracle_gp_rff_normalize_x=args.oracle_gp_rff_normalize_x,
        model_path=args.model_path,
        taf_data_path=args.taf_data_path,
        total_episodes=args.total_episodes,
        max_steps=args.max_steps,
        n_init_context=args.n_init_context,
        n_persistent_base=args.n_persistent_base,
        n_total_candidates=args.n_total_candidates,
        k_centers=args.k_centers,
        local_h=args.local_h,
        local_h_decay=args.local_h_decay,
        explore_fraction=getattr(args, 'explore_fraction', 0.0),
        oracle_gp_min_grid_size=args.oracle_gp_min_grid_size,
        oracle_gp_min_n_lbfgs_starts=args.oracle_gp_min_n_lbfgs_starts,
        reward_mode=args.reward_mode,
        reward_mixed_lambda=args.reward_mixed_lambda,
        reward_terminal_weight=args.reward_terminal_weight,
        reward_frontload_power=args.reward_frontload_power,
        reward_stage_midpoint=args.reward_stage_midpoint,
        reward_regret_auc_weight=args.reward_regret_auc_weight,
        reward_regret_delta_weight=args.reward_regret_delta_weight,
        reward_regret_early_power=args.reward_regret_early_power,
        reward_regret_terminal_power=args.reward_regret_terminal_power,
        reward_regret_scale_floor_ratio=args.reward_regret_scale_floor_ratio,
        variant_sampling=args.variant_sampling,
        inference_precision=args.inference_precision,
        update_every=args.update_every,
        save_every=args.save_every,
        save_dir=args.save_dir,
        seed=args.seed,
        use_tensorboard=args.use_tensorboard,
        hidden_dim=args.hidden_dim,
        n_self_attn_layers=args.n_self_attn_layers,
        n_cross_attn_layers=args.n_cross_attn_layers,
        n_heads=args.n_heads,
        ent_coef_start=args.ent_coef_start,
        ent_coef_end=args.ent_coef_end,
        n_workers=getattr(args, 'n_workers', 1),
    )


if __name__ == "__main__":
    main()
