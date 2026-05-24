"""
Branin 函数族微调 TabPFN

更新 (2026-01):
本脚本支持"工业设定"的轨迹驱动微调数据构造：
- 每个变体只保留 1 条"真实"BO轨迹：先对该变体运行多次 GP+EI（默认 5 次），选择最终找到的最优值最好的那条轨迹。
- 用这条轨迹拟合一个高斯过程（作为"黑箱函数"的可采样数据源）。
- 对每个变体的该高斯过程：每条合成轨迹开始时从 GP 后验采样一条确定性函数 f，轨迹内始终返回 f(x)（默认 100 条；每条仍为 20 次评估；不运行 BO）。
- 将每个变体的 1 条真实 BO 轨迹 + 100 条合成轨迹合并，用于 TabPFN 微调；训练 task 仍按 prefix/suffix 切分。
"""
import os
import json
import argparse
from typing import Callable, List, Tuple, Dict, Optional

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats.qmc import Sobol

from tabpfn import TabPFNRegressor
from tabpfn.finetune_utils import clone_model_for_evaluation
from tabpfn.utils import meta_dataset_collator
from tabpfn.model_loading import save_tabpfn_model
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")
warnings.filterwarnings("ignore", message=".*ABNORMAL_TERMINATION_IN_LNSRCH.*")
warnings.filterwarnings("ignore", message=".*scale the data.*")
warnings.filterwarnings("ignore", message=".*optimal value found.*close to the specified.*bound.*")

from ..tasks import get_task
from ..rl.train_rl import SampledRFFOracleFunction

# ============================================================
# 0. 定义域中心（用于旋转变换）
# ============================================================
CENTER_X1 = 2.5  # (-5 + 10) / 2
CENTER_X2 = 7.5  # (0 + 15) / 2

# ============================================================
# 2.x 轨迹驱动微调：变体采样 + GP+EI 轨迹生成
# ============================================================
def _branin_family_numpy_fast(
    X: np.ndarray,
    variant_params: Dict[str, float],
) -> np.ndarray:
    """
    纯 numpy 实现 Branin 变体族（用于 BO 轨迹生成，避免 torch 小 batch 开销）。

    变换顺序: 旋转(绕中心) -> 缩放 -> 平移
    """
    X = np.atleast_2d(X).astype(np.float64)
    x1 = X[:, 0]
    x2 = X[:, 1]

    dx1 = float(variant_params.get("dx1", 0.0))
    dx2 = float(variant_params.get("dx2", 0.0))
    sx1 = float(variant_params.get("sx1", 1.0))
    sx2 = float(variant_params.get("sx2", 1.0))
    rotation = float(variant_params.get("rotation", 0.0))
    alpha = float(variant_params.get("alpha", 1.0))
    beta = float(variant_params.get("beta", 0.0))

    if abs(rotation) > 1e-12:
        cx1 = CENTER_X1
        cx2 = CENTER_X2
        theta = rotation * np.pi / 180.0
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        x1c = x1 - cx1
        x2c = x2 - cx2
        x1r = cos_t * x1c - sin_t * x2c + cx1
        x2r = sin_t * x1c + cos_t * x2c + cx2
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
    y = a * (x2t - b * x1t**2 + c * x1t - r) ** 2 + s * (1.0 - t) * np.cos(x1t) + s
    y = alpha * y + beta
    return y.astype(np.float32)


def _compute_ei_numpy(mean: np.ndarray, std: np.ndarray, y_best: float, xi: float = 0.01) -> np.ndarray:
    """最小化问题的 Expected Improvement (EI)，与 src/select_candidates.py:186 一致。"""
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    std = np.asarray(std, dtype=np.float64).reshape(-1)

    imp = y_best - mean - xi
    ei = np.zeros_like(mean, dtype=np.float64)
    mask = std > 1e-12
    if np.any(mask):
        z = np.zeros_like(mean, dtype=np.float64)
        z[mask] = imp[mask] / std[mask]
        ei[mask] = imp[mask] * norm.cdf(z[mask]) + std[mask] * norm.pdf(z[mask])
    ei[ei < 0.0] = 0.0
    return ei.astype(np.float64)


def sample_variants_uniform_in_ranges(
    k: int,
    dx_range: Tuple[float, float],
    rotation_range: Tuple[float, float],
    sx_range: Tuple[float, float],
    seed: int = 42,
) -> List[Dict[str, float]]:
    """在给定范围内独立均匀采样 k 个变体（dx1/dx2 独立，sx1/sx2 独立）。"""
    rng = np.random.default_rng(seed)
    variants: List[Dict[str, float]] = []
    for _ in range(k):
        variants.append({
            "dx1": float(rng.uniform(*dx_range)),
            "dx2": float(rng.uniform(*dx_range)),
            "rotation": float(rng.uniform(*rotation_range)),
            "sx1": float(rng.uniform(*sx_range)),
            "sx2": float(rng.uniform(*sx_range)),
            # 固定输出变换（本次需求只做平移/旋转/缩放）
            "alpha": 1.0,
            "beta": 0.0,
        })
    return variants


def _make_gp_model(input_dim: int = 2, n_restarts_optimizer: int = 3) -> GaussianProcessRegressor:
    """按用户指定的标准 BO 配置创建 GP。"""
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=[1.0] * input_dim,
        length_scale_bounds=(1e-5, 1e5),
        nu=2.5,
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=n_restarts_optimizer,
    )


def _sample_sobol_points(
    bounds: Tuple[np.ndarray, np.ndarray],
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    lower, upper = bounds
    sobol_seed = int(rng.integers(0, 100000))
    sampler = Sobol(d=lower.shape[0], scramble=True, seed=sobol_seed)
    u = sampler.random(n)
    return (u * (upper - lower) + lower).astype(np.float64)


def _propose_next_point_gp_ei(
    gp: GaussianProcessRegressor,
    y_best: float,
    bounds: Tuple[np.ndarray, np.ndarray],
    rng: np.random.Generator,
    xi: float = 0.01,
    n_sobol_candidates: int = 512,
    n_start_points: int = 10,
) -> np.ndarray:
    """
    标准做法：连续优化 EI（multi-start L-BFGS-B），起点来自 Sobol 上 EI 高的点。
    """
    lower, upper = bounds
    box_bounds = list(zip(lower.tolist(), upper.tolist()))

    X0 = _sample_sobol_points(bounds, n_sobol_candidates, rng)
    mu0, std0 = gp.predict(X0, return_std=True)
    ei0 = _compute_ei_numpy(mu0, std0, y_best, xi=xi)

    # 若 EI 全 0（常见于 std 很小），退化为选预测均值最小的点作为起点
    if float(ei0.max()) <= 1e-18 or not np.isfinite(ei0).all():
        best0 = int(np.argmin(mu0))
        return X0[best0].astype(np.float64)

    top_idx = np.argsort(ei0)[-n_start_points:][::-1]
    start_points = X0[top_idx]

    def neg_ei(x: np.ndarray) -> float:
        x2d = np.asarray(x, dtype=np.float64).reshape(1, -1)
        mu, std = gp.predict(x2d, return_std=True)
        ei = _compute_ei_numpy(mu, std, y_best, xi=xi)[0]
        if not np.isfinite(ei):
            return 1e9
        return float(-ei)

    best_x = start_points[0]
    mu_best, std_best = gp.predict(best_x.reshape(1, -1), return_std=True)
    best_ei = float(_compute_ei_numpy(mu_best, std_best, y_best, xi=xi)[0])

    for x0 in start_points:
        try:
            res = minimize(neg_ei, x0=x0, bounds=box_bounds, method="L-BFGS-B")
            x_cand = np.asarray(res.x, dtype=np.float64)
            mu, std = gp.predict(x_cand.reshape(1, -1), return_std=True)
            ei = float(_compute_ei_numpy(mu, std, y_best, xi=xi)[0])
            if ei > best_ei:
                best_ei = ei
                best_x = x_cand
        except Exception:
            continue

    return best_x.astype(np.float64)


def run_gp_ei_bo_trajectory(
    evaluate_numpy: Callable[[np.ndarray, Dict[str, float]], np.ndarray],
    variant_params: Dict[str, float],
    total_evals: int,
    n_init: int,
    bounds: Tuple[np.ndarray, np.ndarray],
    seed: int,
    xi: float = 0.01,
    n_restarts_optimizer: int = 3,
    n_sobol_candidates: int = 512,
    n_start_points: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在单个变体上运行一次 GP+EI 贝叶斯优化，返回固定长度轨迹。
    total_evals 包含初始点，总长度固定为 total_evals。
    """
    assert 1 <= n_init < total_evals
    rng = np.random.default_rng(seed)
    lower, upper = bounds

    X = rng.uniform(lower, upper, size=(n_init, lower.shape[0])).astype(np.float64)
    y = np.asarray(evaluate_numpy(X, variant_params), dtype=np.float64).reshape(-1)

    while X.shape[0] < total_evals:
        gp = _make_gp_model(input_dim=lower.shape[0], n_restarts_optimizer=n_restarts_optimizer)
        gp.fit(X, y)

        y_best = float(np.min(y))
        x_next = _propose_next_point_gp_ei(
            gp,
            y_best=y_best,
            bounds=bounds,
            rng=rng,
            xi=xi,
            n_sobol_candidates=n_sobol_candidates,
            n_start_points=n_start_points,
        )

        y_next = float(np.asarray(evaluate_numpy(x_next.reshape(1, -1), variant_params), dtype=np.float64).reshape(-1)[0])
        X = np.vstack([X, x_next.reshape(1, -1)])
        y = np.concatenate([y, [y_next]])

    return X.astype(np.float32), y.astype(np.float32)


def _fit_oracle_gp_from_trajectory(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_restarts_optimizer: int = 3,
) -> GaussianProcessRegressor:
    gp = _make_gp_model(input_dim=X.shape[1], n_restarts_optimizer=n_restarts_optimizer)
    gp.fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64))
    return gp


def _sample_from_oracle_gp(
    oracle_gp: GaussianProcessRegressor,
    X: np.ndarray,
    rng: np.random.Generator,
    *,
    min_std: float = 1e-12,
) -> np.ndarray:
    mu, std = oracle_gp.predict(np.asarray(X, dtype=np.float64), return_std=True)
    std = np.maximum(np.asarray(std, dtype=np.float64), float(min_std))
    return rng.normal(loc=np.asarray(mu, dtype=np.float64), scale=std).astype(np.float64)


def sample_random_trajectory_from_oracle_gp(
    oracle_gp: GaussianProcessRegressor,
    total_evals: int,
    bounds: Tuple[np.ndarray, np.ndarray],
    seed: int,
    n_init: int = 2,
    exploit_prob: float = 0.7,
    local_std_ratio: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从"oracle GP"采样一条模拟 BO 行为的轨迹。

    前 n_init 个点随机均匀采样，之后的点以 exploit_prob 概率在当前最优点
    附近局部采样（模拟 BO 的 exploitation），否则随机采样（exploration）。
    这样生成的 context 分布更接近 RL/BO 下游使用时的真实分布。
    """
    rng = np.random.default_rng(seed)
    lower, upper = bounds
    dim = lower.shape[0]
    span = upper - lower

    sampled_f = SampledRFFOracleFunction(
        oracle_gp=oracle_gp,
        rng=rng,
        bounds=(np.asarray(lower, dtype=np.float32), np.asarray(upper, dtype=np.float32)),
    )

    X_list = []
    y_list = []

    # Phase 1: random init
    X_init = rng.uniform(lower, upper, size=(n_init, dim)).astype(np.float64)
    y_init = np.asarray(sampled_f(X_init), dtype=np.float64).reshape(-1)
    X_list.append(X_init)
    y_list.append(y_init)

    # Phase 2: pseudo-BO (exploit near best / explore random)
    for _ in range(total_evals - n_init):
        all_y = np.concatenate(y_list)
        all_X = np.concatenate(X_list, axis=0)
        best_idx = int(np.argmin(all_y))
        best_x = all_X[best_idx]

        if rng.random() < exploit_prob:
            # Local sampling near current best
            local_std = local_std_ratio * span
            x_new = rng.normal(loc=best_x, scale=local_std)
            x_new = np.clip(x_new, lower, upper)
        else:
            # Random exploration
            x_new = rng.uniform(lower, upper, size=(dim,))

        x_new = x_new.astype(np.float64).reshape(1, -1)
        y_new = np.asarray(sampled_f(x_new), dtype=np.float64).reshape(-1)
        X_list.append(x_new)
        y_list.append(y_new)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list)
    return X.astype(np.float32), y.astype(np.float32)


def load_or_generate_training_variants(
    cache_path: str,
    *,
    task_name: str,
    k: int,
    seed: int,
    allow_generate: bool = True,
    force_regen: bool = False,
) -> List[Dict[str, float]]:
    if os.path.exists(cache_path) and not force_regen:
        data = np.load(cache_path, allow_pickle=True)
        variants = data["variants"].tolist()
        return variants

    if not allow_generate:
        raise FileNotFoundError(f"Variants cache not found: {cache_path}")

    task = get_task(task_name)
    variants = task.sample_train_variants(k=int(k), seed=int(seed))
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    metadata = {
        "task": str(task_name),
        "k": int(k),
        "seed": int(seed),
    }
    np.savez(cache_path, variants=np.array(variants, dtype=object), metadata=np.array([json.dumps(metadata)], dtype=object))
    return variants


def load_or_generate_bo_trajectories(
    cache_path: str,
    *,
    task_name: str,
    variants: List[Dict[str, float]],
    n_trials_per_variant: int,
    n_synth_trajectories_per_variant: int,
    total_evals: int,
    n_init: int,
    bounds: Tuple[np.ndarray, np.ndarray],
    seed: int,
    xi: float = 0.01,
    gp_n_restarts_optimizer: int = 3,
    n_sobol_candidates: int = 512,
    n_start_points: int = 10,
    allow_generate: bool = True,
    force_regen: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回：
      X_trajs: (n_traj, total_evals, dim)
      y_trajs: (n_traj, total_evals)
      variant_indices: (n_traj,)
    """
    if os.path.exists(cache_path) and not force_regen:
        data = np.load(cache_path, allow_pickle=True)
        return data["X_trajs"], data["y_trajs"], data["variant_indices"]

    if not allow_generate:
        raise FileNotFoundError(f"Trajectory cache not found: {cache_path}")

    task = get_task(task_name)
    evaluate_numpy = lambda X, params: task.evaluate_numpy(X, params)  # noqa: E731

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    rng = np.random.default_rng(seed)

    k = len(variants)
    n_traj = k * (1 + n_synth_trajectories_per_variant)
    dim = bounds[0].shape[0]
    X_trajs = np.zeros((n_traj, total_evals, dim), dtype=np.float32)
    y_trajs = np.zeros((n_traj, total_evals), dtype=np.float32)
    variant_indices = np.zeros((n_traj,), dtype=np.int64)

    variant_infos: List[Dict[str, object]] = []
    pbar = tqdm(total=n_traj, desc="Generate industrial trajectories (best-of BO + oracle GP sampling)")
    t_idx = 0
    for v_idx, params in enumerate(variants):
        best = None
        trial_summaries = []
        for trial in range(int(n_trials_per_variant)):
            run_seed = int(rng.integers(0, 2**31 - 1))
            X_trial, y_trial = run_gp_ei_bo_trajectory(
                evaluate_numpy,
                variant_params=params,
                total_evals=total_evals,
                n_init=n_init,
                bounds=bounds,
                seed=run_seed,
                xi=xi,
                n_restarts_optimizer=gp_n_restarts_optimizer,
                n_sobol_candidates=n_sobol_candidates,
                n_start_points=n_start_points,
            )
            best_y = float(np.min(y_trial))
            trial_summaries.append({"trial": trial, "seed": run_seed, "best_y": best_y})
            if best is None or best_y < float(best["best_y"]):
                best = {
                    "trial": trial,
                    "seed": run_seed,
                    "best_y": best_y,
                    "X": X_trial,
                    "y": y_trial,
                }

        assert best is not None
        X_best = best["X"]
        y_best_traj = best["y"]

        oracle_gp = _fit_oracle_gp_from_trajectory(
            X_best,
            y_best_traj,
            n_restarts_optimizer=gp_n_restarts_optimizer,
        )

        variant_infos.append(
            {
                "variant_index": int(v_idx),
                "selected_trial": int(best["trial"]),
                "selected_seed": int(best["seed"]),
                "selected_best_y": float(best["best_y"]),
                "trials": trial_summaries,
            }
        )

        # 真实轨迹（best-of-5 BO）
        X_trajs[t_idx] = X_best
        y_trajs[t_idx] = y_best_traj
        variant_indices[t_idx] = v_idx
        t_idx += 1
        pbar.update(1)

        # 合成轨迹：在 oracle GP 上重新跑 BO
        for _ in range(int(n_synth_trajectories_per_variant)):
            synth_seed = int(rng.integers(0, 2**31 - 1))
            X_syn, y_syn = sample_random_trajectory_from_oracle_gp(
                oracle_gp=oracle_gp,
                total_evals=total_evals,
                bounds=bounds,
                seed=synth_seed,
            )
            X_trajs[t_idx] = X_syn
            y_trajs[t_idx] = y_syn
            variant_indices[t_idx] = v_idx
            t_idx += 1
            pbar.update(1)
    pbar.close()

    metadata = {
        "k": k,
        "n_trials_per_variant": n_trials_per_variant,
        "n_synth_trajectories_per_variant": n_synth_trajectories_per_variant,
        "total_evals": total_evals,
        "n_init": n_init,
        "xi": xi,
        "gp_n_restarts_optimizer": gp_n_restarts_optimizer,
        "n_sobol_candidates": n_sobol_candidates,
        "n_start_points": n_start_points,
        "seed": seed,
    }
    np.savez(
        cache_path,
        variants=np.array(variants, dtype=object),
        X_trajs=X_trajs,
        y_trajs=y_trajs,
        variant_indices=variant_indices,
        variant_infos=np.array(variant_infos, dtype=object),
        metadata=np.array([json.dumps(metadata)], dtype=object),
    )
    return X_trajs, y_trajs, variant_indices


def make_train_val_split_by_variant(
    variant_indices: np.ndarray,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    按变体分层划分：每个变体的轨迹中留出 val_ratio 做验证。
    """
    rng = np.random.default_rng(seed)
    variant_indices = np.asarray(variant_indices, dtype=np.int64)
    train_idx: List[int] = []
    val_idx: List[int] = []

    for v in np.unique(variant_indices):
        idx = np.where(variant_indices == v)[0]
        idx = idx.copy()
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_ratio)))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())

    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64)


def augment_real_trajectories(X_trajs, y_trajs, variant_indices, n_augment=20, seed=42):
    """对每条真实 trajectory (variant 内第 0 条) 生成 n_augment 份随机排列副本"""
    rng = np.random.default_rng(seed)
    aug_X, aug_y, aug_vi = [], [], []
    n_synth_per_variant = None

    for vi in np.unique(variant_indices):
        mask = variant_indices == vi
        idxs = np.where(mask)[0]
        if n_synth_per_variant is None:
            n_synth_per_variant = len(idxs) - 1
        real_idx = idxs[0]  # 第 0 条是真实 BO trajectory

        X_real = X_trajs[real_idx]  # (total_evals, dim)
        y_real = y_trajs[real_idx]  # (total_evals,)

        for _ in range(n_augment):
            perm = rng.permutation(len(X_real))
            aug_X.append(X_real[perm])
            aug_y.append(y_real[perm])
            aug_vi.append(vi)

    # 拼接到原始数据后面
    X_all = np.concatenate([X_trajs, np.array(aug_X)], axis=0)
    y_all = np.concatenate([y_trajs, np.array(aug_y)], axis=0)
    vi_all = np.concatenate([variant_indices, np.array(aug_vi)])
    return X_all, y_all, vi_all


def create_finetuning_dataloader_from_trajectories(
    regressor: TabPFNRegressor,
    X_trajs: np.ndarray,
    y_trajs: np.ndarray,
    train_indices: np.ndarray,
    config: dict,
) -> DataLoader:
    print("--- Build finetuning datasets & dataloader (trajectory-based) ---")

    # Each dataset item is exactly one trajectory (length=total_evals=20).
    # This avoids mixing points from different variants/trajectories inside one raw dataset.
    train_indices = np.asarray(train_indices, dtype=np.int64)
    X_list = [X_trajs[i].astype(np.float32) for i in train_indices]
    y_list = [y_trajs[i].astype(np.float32) for i in train_indices]

    # Splitter now splits *within* the given trajectory dataset (prefix -> ctx, suffix -> test).
    rng = np.random.default_rng(config["random_seed"])
    total_evals = X_trajs.shape[1]
    max_context = min(config["max_context"], total_evals - 1)
    min_context = config["min_context"]
    if max_context < min_context:
        raise ValueError(
            f"Invalid context range: min_context={min_context}, max_context={max_context}, total_evals={total_evals}"
        )

    def splitter(X_one: np.ndarray, y_one: np.ndarray):
        m = int(rng.integers(min_context, max_context + 1))
        X_ctx = X_one[:m]
        y_ctx = y_one[:m]
        X_test = X_one[m:]
        y_test = y_one[m:]
        return X_ctx, X_test, y_ctx, y_test

    # max_data_size=None => do not split raw datasets (each trajectory is already small).
    training_datasets = regressor.get_preprocessed_datasets(
        X_list,
        y_list,
        splitter,
        max_data_size=None,
    )
    print(f"Number of meta-datasets from get_preprocessed_datasets: {len(training_datasets)}")

    finetuning_dataloader = DataLoader(
        training_datasets,
        batch_size=config["finetuning"]["meta_batch_size"],
        collate_fn=meta_dataset_collator,
        shuffle=True,
    )

    first_batch = next(iter(finetuning_dataloader))
    (
        X_trains_preprocessed,
        X_tests_preprocessed,
        y_trains_znorm,
        y_test_znorm,
        _cat_ixs,
        _confs,
        _raw_space_bardist_,
        _znorm_space_bardist_,
        _,
        _y_test_raw,
    ) = first_batch
    print("Inspect one preprocessed task:")
    print("  X_trains_preprocessed[0].shape:", X_trains_preprocessed[0].shape)
    print("  X_tests_preprocessed[0].shape :", X_tests_preprocessed[0].shape)
    print("  y_trains_znorm[0].shape       :", y_trains_znorm[0].shape)
    print("  y_test_znorm[0].shape         :", y_test_znorm[0].shape)
    print("---------------------------\n")
    return finetuning_dataloader


def evaluate_regressor_on_val_trajectories(
    regressor: TabPFNRegressor,
    regressor_config: dict,
    X_trajs: np.ndarray,
    y_trajs: np.ndarray,
    val_indices: np.ndarray,
    config: dict,
    n_tasks: int = 200,
) -> Tuple[float, float]:
    """
    在验证轨迹上评估：随机切 prefix/suffix，统计 suffix 上的 MSE/MAE（按点累计）。
    """
    eval_regressor = clone_model_for_evaluation(
        regressor,
        {
            **regressor_config,
            "inference_config": {
                "SUBSAMPLE_SAMPLES": config["n_inference_context_samples"],
            },
        },
        TabPFNRegressor,
    )

    rng = np.random.default_rng(config["random_seed"] + 12345)
    val_indices = np.asarray(val_indices, dtype=np.int64)
    total_evals = X_trajs.shape[1]

    max_context = min(config["max_context"], total_evals - 1)
    min_context = config["min_context"]

    se_sum = 0.0
    ae_sum = 0.0
    n_points = 0

    for _ in range(n_tasks):
        t_idx = int(rng.choice(val_indices))
        X = X_trajs[t_idx]
        y = y_trajs[t_idx]
        m = int(rng.integers(min_context, max_context + 1))

        X_ctx = X[:m].astype(np.float32)
        y_ctx = y[:m].astype(np.float32)
        X_test = X[m:].astype(np.float32)
        y_test = y[m:].astype(np.float32)
        if X_test.shape[0] == 0:
            continue

        eval_regressor.fit(X_ctx, y_ctx)
        preds = eval_regressor.predict(X_test).astype(np.float64)
        y_true = y_test.astype(np.float64)

        se_sum += float(np.sum((preds - y_true) ** 2))
        ae_sum += float(np.sum(np.abs(preds - y_true)))
        n_points += int(y_true.shape[0])

    mse = se_sum / max(1, n_points)
    mae = ae_sum / max(1, n_points)
    return float(mse), float(mae)


# ============================================================
# 8. TabPFNRegressor 初始化 & 多变体 splitter + dataloader
# ============================================================
def setup_regressor(config: dict) -> Tuple[TabPFNRegressor, dict]:
    print("--- TabPFNRegressor Setup ---")
    regressor_config = {
        "ignore_pretraining_limits": True,
        "device": config["device"],
        "n_estimators": 1,
        "random_state": config["random_seed"],
        "inference_precision": torch.float32,
    }
    regressor = TabPFNRegressor(
        **regressor_config,
        fit_mode="batched",
        differentiable_input=False,
    )
    print(f"Using device: {config['device']}")
    print("---------------------------\n")
    return regressor, regressor_config


def run_finetuning_trajectory_based(
    regressor: TabPFNRegressor,
    regressor_config: dict,
    finetuning_dataloader: DataLoader,
    X_trajs: np.ndarray,
    y_trajs: np.ndarray,
    val_indices: np.ndarray,
    config: dict,
) -> None:
    """
    轨迹驱动的 TabPFN 微调：
    - task 来自 BO 轨迹的 prefix/suffix 切分
    - 每个 epoch 在验证轨迹上评估 suffix 预测误差
    """
    # 确保只有一个底层模型可微调
    if hasattr(regressor, "models_"):
        if len(regressor.models_) > 1:
            raise ValueError(
                f"Your TabPFNRegressor uses multiple models ({len(regressor.models_)}). "
                "Finetuning is only supported for a single model."
            )
        model = regressor.models_[0]
    else:
        model = regressor.model_

    optimizer = Adam(model.parameters(), lr=config["finetuning"]["learning_rate"])

    num_epochs = config["finetuning"]["epochs"]
    warmup_epochs = max(1, num_epochs // 10)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    print(f"--- Optimizer: Adam, LR={config['finetuning']['learning_rate']} ---")
    print(f"--- Warmup={warmup_epochs} epochs, cosine decay to 1e-7 ---\n")

    val_tasks = config["finetuning"].get("val_tasks", 200)

    save_path = str(config.get("output_model_path") or "./model/finetuned_tabpfn_branin_family.ckpt")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    print("--- Start finetuning on BO trajectories ---")
    for epoch in range(num_epochs + 1):
        mse, mae = evaluate_regressor_on_val_trajectories(
            regressor=regressor,
            regressor_config=regressor_config,
            X_trajs=X_trajs,
            y_trajs=y_trajs,
            val_indices=val_indices,
            config=config,
            n_tasks=val_tasks,
        )
        status = "Initial" if epoch == 0 else f"Epoch {epoch}"
        print(f"{status} Validation | MSE: {mse:.4f}, MAE: {mae:.4f}")

        if epoch == 0:
            print("---------------------------")
            continue

        progress_bar = tqdm(finetuning_dataloader, desc=f"Finetuning Epoch {epoch}")
        for data_batch in progress_bar:
            optimizer.zero_grad()

            (
                X_trains_preprocessed,
                X_tests_preprocessed,
                y_trains_znorm,
                y_test_znorm,
                cat_ixs,
                confs,
                raw_space_bardist_,
                znorm_space_bardist_,
                _,
                _y_test_raw,
            ) = data_batch

            regressor.raw_space_bardist_ = raw_space_bardist_[0]
            regressor.znorm_space_bardist_ = znorm_space_bardist_[0]

            regressor.fit_from_preprocessed(
                X_trains_preprocessed,
                y_trains_znorm,
                cat_ixs,
                confs,
            )

            logits, _, _ = regressor.forward(X_tests_preprocessed)
            loss_fn = znorm_space_bardist_[0]
            y_target = y_test_znorm
            loss = loss_fn(logits, y_target.to(config["device"])).mean()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  LR after epoch {epoch}: {current_lr:.2e}")
        print("---------------------------")

    save_tabpfn_model(regressor, save_path)
    print(f"Saved fine-tuned TabPFNRegressor to: {save_path}")
    print("--- Finetuning Finished ---")


# ============================================================
# 11. main：串起所有步骤
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Trajectory-based TabPFN finetuning (task-adaptable)")
    parser.add_argument("--task", type=str, default="branin_family", help="Task name (see myrl.tasks.list_tasks())")
    parser.add_argument(
        "--stage",
        choices=["all", "generate", "finetune"],
        default="all",
        help="Run all steps, only generate cached data, or only finetune from caches.",
    )
    parser.add_argument(
        "--force_regen",
        action="store_true",
        help="Force regenerate caches (only meaningful for --stage all/generate).",
    )
    parser.add_argument("--variants_cache", type=str, default="./data/variants_train.npz")
    parser.add_argument("--trajectories_cache", type=str, default="./data/bo_trajs_train.npz")
    parser.add_argument(
        "--k_variants",
        type=int,
        default=None,
        help="Number of training variants (default: branin_family=10, other tasks=1).",
    )
    parser.add_argument("--variant_seed", type=int, default=123, help="Seed for sampling training variants.")
    parser.add_argument("--bo_seed", type=int, default=2025, help="Seed for generating BO/synthetic trajectories.")
    parser.add_argument("--random_seed", type=int, default=42, help="Seed for train/val split and task sampling.")

    # Trajectory generation knobs
    parser.add_argument("--n_trials_per_variant", type=int, default=5)
    parser.add_argument("--n_synth_trajectories_per_variant", type=int, default=100)
    parser.add_argument("--total_evals", type=int, default=20, help="Total evaluations per trajectory (includes init).")
    parser.add_argument("--n_init", type=int, default=2, help="Number of initial random points per trajectory.")
    parser.add_argument("--xi", type=float, default=0.01, help="EI exploration parameter.")
    parser.add_argument("--gp_n_restarts_optimizer", type=int, default=3)
    parser.add_argument("--n_sobol_candidates", type=int, default=512)
    parser.add_argument("--n_start_points", type=int, default=10)

    # Dataset splitting / context sampling
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio per variant.")
    parser.add_argument("--min_context", type=int, default=2)
    parser.add_argument("--max_context", type=int, default=19)
    parser.add_argument("--n_inference_context_samples", type=int, default=20)

    # Data augmentation
    parser.add_argument("--n_real_augment", type=int, default=20,
                        help="Number of random-permutation augmentations per real BO trajectory (0 to disable).")

    # Finetuning hyperparameters
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1.5e-6)
    parser.add_argument("--meta_batch_size", type=int, default=1)
    parser.add_argument("--num_tasks", type=int, default=20)
    parser.add_argument("--val_tasks", type=int, default=200)

    parser.add_argument(
        "--output_model_path",
        type=str,
        default=None,
        help="Where to save the finetuned TabPFN checkpoint (default: ./model/finetuned_tabpfn_<task>.ckpt).",
    )
    args = parser.parse_args()

    task = get_task(args.task)

    if int(args.total_evals) <= int(args.n_init):
        raise ValueError(f"--total_evals must be > --n_init (got {args.total_evals} <= {args.n_init})")
    if int(args.max_context) >= int(args.total_evals):
        raise ValueError(
            f"--max_context must be <= total_evals-1 (got max_context={args.max_context}, total_evals={args.total_evals})"
        )
    if int(args.min_context) < 1 or int(args.min_context) > int(args.max_context):
        raise ValueError(f"Invalid context range: min_context={args.min_context}, max_context={args.max_context}")
    if not (0.0 <= float(args.val_ratio) < 1.0):
        raise ValueError(f"--val_ratio must be in [0, 1) (got {args.val_ratio})")

    # 全局配置（轨迹驱动微调）
    default_k_variants = 10 if task.task_name in {
        "ackley_5d_family",
        "ackley_10d_family",
        "branin_family",
        "goldstein_price_family",
        "hartmann_3d_family",
        "hartmann_6d_family",
    } else 1
    k_variants = int(args.k_variants) if args.k_variants is not None else int(default_k_variants)
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "random_seed": int(args.random_seed),
        "variant_seed": int(args.variant_seed),
        "bo_seed": int(args.bo_seed),

        "task": str(args.task),

        # 训练变体数（默认沿用 Branin 设定：k=10；对无变体任务默认=1）
        "k_variants": int(k_variants),

        # BO 轨迹生成（GP+EI）：先对每个变体做多次 trial，选择 best-of 作为"真实"工业轨迹
        "bo": {
            "n_trials_per_variant": int(args.n_trials_per_variant),
            "total_evals": int(args.total_evals),     # 总函数评估次数（包含初始点）
            "n_init": int(args.n_init),
            "xi": float(args.xi),
            "gp_n_restarts_optimizer": int(args.gp_n_restarts_optimizer),
            "n_sobol_candidates": int(args.n_sobol_candidates),
            "n_start_points": int(args.n_start_points),
        },

        # 合成轨迹：对每个变体拟合得到的 oracle GP，随机采样多少条轨迹（不跑 BO）
        "synthetic": {
            "n_trajectories_per_variant": int(args.n_synth_trajectories_per_variant),
        },

        # 轨迹验证集比例（按变体分层）
        "val_ratio": float(args.val_ratio),

        # 轨迹切分上下文范围（suffix 必须非空，因此 max_context <= total_evals-1）
        "min_context": int(args.min_context),
        "max_context": int(args.max_context),
        "n_inference_context_samples": int(args.n_inference_context_samples),

        # 数据增强
        "n_real_augment": int(args.n_real_augment),

        # 微调超参数
        "finetuning": {
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "meta_batch_size": int(args.meta_batch_size),
            # 生成多少个随机 task（每个 task 来自随机轨迹 + 随机 m）
            "num_tasks": int(args.num_tasks),
            # 每个 epoch 的验证 task 数
            "val_tasks": int(args.val_tasks),
        },

        # 缓存路径
        "variants_cache": args.variants_cache,
        "trajectories_cache": args.trajectories_cache,

        # 输出模型路径
        "output_model_path": str(args.output_model_path or f"./model/finetuned_tabpfn_{task.task_name}.ckpt"),
    }

    print("=" * 60)
    print(f"Task={task.task_name} - BO Trajectory Based TabPFN Finetuning")
    print("=" * 60)
    print(f"Device: {config['device']}")
    print(
        "Seeds: "
        f"random_seed={config['random_seed']}, "
        f"variant_seed={config['variant_seed']}, "
        f"bo_seed={config['bo_seed']}"
    )
    print(f"k_variants: {config['k_variants']}")
    lower_b, upper_b = task.bounds
    print(f"Bounds: lower={lower_b.tolist()}, upper={upper_b.tolist()}")
    print(f"variants_cache: {config['variants_cache']}")
    print(f"trajectories_cache: {config['trajectories_cache']}")
    print(f"output_model_path: {config['output_model_path']}")
    print("BO config:")
    print(f"  n_trials_per_variant (best-of): {config['bo']['n_trials_per_variant']}")
    print(f"  total_evals: {config['bo']['total_evals']} (includes init)")
    print(f"  n_init: {config['bo']['n_init']}")
    print(f"  xi: {config['bo']['xi']}")
    print(f"  gp_n_restarts_optimizer: {config['bo']['gp_n_restarts_optimizer']}")
    print(f"Synthetic config:")
    print(f"  n_trajectories_per_variant: {config['synthetic']['n_trajectories_per_variant']}")
    print("=" * 60 + "\n")

    # Bounds
    lower = np.asarray(task.bounds[0], dtype=np.float64)
    upper = np.asarray(task.bounds[1], dtype=np.float64)
    bounds = (lower, upper)

    allow_generate = args.stage in {"all", "generate"}

    # --- Step 1/2: variants + trajectories (generate or load caches) ---
    variants = load_or_generate_training_variants(
        cache_path=config["variants_cache"],
        task_name=task.task_name,
        k=config["k_variants"],
        seed=config["variant_seed"],
        allow_generate=allow_generate,
        force_regen=args.force_regen,
    )
    print(f"Training variants: {len(variants)} ({config['variants_cache']})")

    X_trajs, y_trajs, variant_indices = load_or_generate_bo_trajectories(
        cache_path=config["trajectories_cache"],
        task_name=task.task_name,
        variants=variants,
        n_trials_per_variant=config["bo"]["n_trials_per_variant"],
        n_synth_trajectories_per_variant=config["synthetic"]["n_trajectories_per_variant"],
        total_evals=config["bo"]["total_evals"],
        n_init=config["bo"]["n_init"],
        bounds=bounds,
        seed=config["bo_seed"],
        xi=config["bo"]["xi"],
        gp_n_restarts_optimizer=config["bo"]["gp_n_restarts_optimizer"],
        n_sobol_candidates=config["bo"]["n_sobol_candidates"],
        n_start_points=config["bo"]["n_start_points"],
        allow_generate=allow_generate,
        force_regen=args.force_regen,
    )
    print(f"BO trajectories: {X_trajs.shape[0]} (each length={X_trajs.shape[1]}) ({config['trajectories_cache']})")

    if args.stage == "generate":
        print("Stage=generate: caches are ready; skipping finetuning.")
        return

    # --- Step 2.5: Data augmentation (random permutations of real trajectories) ---
    n_real_augment = config["n_real_augment"]
    if n_real_augment > 0:
        print(f"\nAugmenting real trajectories: {n_real_augment} permutations per real trajectory...")
        orig_count = X_trajs.shape[0]
        X_trajs, y_trajs, variant_indices = augment_real_trajectories(
            X_trajs, y_trajs, variant_indices,
            n_augment=n_real_augment,
            seed=config["random_seed"] + 999,
        )
        print(f"  Trajectories: {orig_count} -> {X_trajs.shape[0]} (+{X_trajs.shape[0] - orig_count} augmented)")

    # --- Step 3: train/val split (10% per variant) ---
    train_indices, val_indices = make_train_val_split_by_variant(
        variant_indices=variant_indices,
        val_ratio=config["val_ratio"],
        seed=config["random_seed"],
    )
    print(f"Train trajectories: {len(train_indices)}, Val trajectories: {len(val_indices)}")

    # --- Step 4: TabPFN init + finetuning dataloader + finetune ---
    regressor, regressor_config = setup_regressor(config)

    finetuning_dataloader = create_finetuning_dataloader_from_trajectories(
        regressor,
        X_trajs,
        y_trajs,
        train_indices,
        config,
    )

    run_finetuning_trajectory_based(
        regressor,
        regressor_config,
        finetuning_dataloader,
        X_trajs,
        y_trajs,
        val_indices,
        config,
    )


if __name__ == "__main__":
    main()
