"""
诊断测试：分析不同训练函数来源的质量

核心问题：训练函数之间是否有结构相似性？
- oracle_gp（GP→RFF）：从同一 GP 后验采样的 RFF 函数是否"长得像一家人"？
- BNN：采样函数之间是否有结构？
- 直接测量：在 Sobol 网格上计算函数值的 rank correlation

测试任务：alkox_emulator（identity variant，无仿射变换）

使用方式：
  export PYTHONPATH="MYRL:olympus/src"
  export TF_USE_LEGACY_KERAS=1
  python MYRL/scripts/test_training_function_quality.py
"""

from __future__ import annotations

import os
import sys
import numpy as np
from scipy.stats import spearmanr
from scipy.stats.qmc import Sobol

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

# Bootstrap
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _bootstrap import bootstrap_project_root
bootstrap_project_root()

from myrl.tasks.registry import get_task
from myrl.rl.train_rl import (
    make_oracle_gp_model,
    SampledRFFOracleFunction,
)


def run_bo_trajectory(task, variant_params, n_init=2, n_steps=28, seed=42):
    """用 GP+EI 在 task 上跑一条 BO 轨迹，返回 (X, y)"""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel

    rng = np.random.default_rng(seed)
    lower, upper = task.bounds
    dim = task.dim

    # 初始随机点
    X = rng.uniform(lower, upper, size=(n_init, dim)).astype(np.float64)
    y = task.evaluate_numpy(X.astype(np.float32), variant_params).astype(np.float64)

    for step in range(n_steps):
        # 拟合 GP
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=[1.0] * dim, length_scale_bounds=(1e-5, 1e5), nu=2.5
        )
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=3)
        gp.fit(X, y)

        # EI 选点
        n_cand = 2048
        sobol = Sobol(d=dim, scramble=True, seed=int(rng.integers(0, 2**31)))
        X_cand = (sobol.random(n_cand) * (upper - lower) + lower).astype(np.float64)
        mu, sigma = gp.predict(X_cand, return_std=True)
        best_y = np.min(y)

        from scipy.stats import norm
        imp = best_y - mu
        with np.errstate(divide="ignore", invalid="ignore"):
            Z = imp / (sigma + 1e-12)
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma < 1e-12] = 0.0

        idx = np.argmax(ei)
        x_new = X_cand[idx:idx+1]
        y_new = task.evaluate_numpy(x_new.astype(np.float32), variant_params).astype(np.float64)

        X = np.vstack([X, x_new])
        y = np.concatenate([y, y_new])

    return X, y


def test_gp_fit_quality(task, variant_params, X_traj, y_traj):
    """测试 GP 拟合质量：length_scale、预测精度"""
    gp = make_oracle_gp_model(input_dim=task.dim, n_restarts_optimizer=5)
    gp.fit(X_traj, y_traj)

    # 提取 kernel 参数
    kernel = gp.kernel_
    print(f"\n  拟合后 kernel: {kernel}")

    # 提取 length_scale
    params = kernel.get_params()
    for key, val in sorted(params.items()):
        if "length_scale" in key and "bounds" not in key:
            ls = np.atleast_1d(val)
            print(f"  length_scale: {ls}")
            print(f"  length_scale 范围: [{ls.min():.3f}, {ls.max():.3f}]")

    # 在测试网格上评估预测精度
    lower, upper = task.bounds
    sobol = Sobol(d=task.dim, scramble=True, seed=999)
    X_test = (sobol.random(2048) * (upper - lower) + lower).astype(np.float64)
    y_true = task.evaluate_numpy(X_test.astype(np.float32), variant_params).astype(np.float64)
    mu_pred, sigma_pred = gp.predict(X_test, return_std=True)

    rmse = np.sqrt(np.mean((y_true - mu_pred) ** 2))
    rank_corr, _ = spearmanr(y_true, mu_pred)
    print(f"  GP 预测 RMSE: {rmse:.4f} (y_range={y_true.max()-y_true.min():.2f})")
    print(f"  GP 预测 Spearman rank correlation: {rank_corr:.4f}")
    print(f"  GP 后验 sigma mean: {sigma_pred.mean():.4f}, max: {sigma_pred.max():.4f}")

    return gp


def test_rff_similarity(gp, task, n_samples=20, n_grid=4096):
    """测试 RFF 采样函数之间的相似性"""
    lower, upper = task.bounds
    bounds = (lower.astype(np.float32), upper.astype(np.float32))

    # 固定测试网格
    sobol = Sobol(d=task.dim, scramble=True, seed=888)
    X_grid = (sobol.random(n_grid) * (upper - lower) + lower).astype(np.float32)

    # 采样多个 RFF 函数并在网格上求值
    Y_all = []
    for i in range(n_samples):
        rng = np.random.default_rng(i * 1000 + 42)
        func = SampledRFFOracleFunction(oracle_gp=gp, rng=rng, bounds=bounds)
        y_i = func(X_grid).astype(np.float64)
        Y_all.append(y_i)
    Y_all = np.array(Y_all)  # (n_samples, n_grid)

    # 计算 pairwise Spearman rank correlation
    rank_corrs = []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            rc, _ = spearmanr(Y_all[i], Y_all[j])
            rank_corrs.append(rc)
    rank_corrs = np.array(rank_corrs)

    # 计算 pairwise Pearson correlation
    pearson_corrs = []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            pc = np.corrcoef(Y_all[i], Y_all[j])[0, 1]
            pearson_corrs.append(pc)
    pearson_corrs = np.array(pearson_corrs)

    print(f"\n  RFF 采样函数间相似性 ({n_samples} 个函数, {n_grid} 网格点):")
    print(f"  Spearman rank correlation:  mean={rank_corrs.mean():.4f}, std={rank_corrs.std():.4f}, "
          f"min={rank_corrs.min():.4f}, max={rank_corrs.max():.4f}")
    print(f"  Pearson correlation:        mean={pearson_corrs.mean():.4f}, std={pearson_corrs.std():.4f}, "
          f"min={pearson_corrs.min():.4f}, max={pearson_corrs.max():.4f}")

    # 计算每个 RFF 函数与真实函数的相似度
    return Y_all, rank_corrs, pearson_corrs


def test_rff_vs_true(task, variant_params, gp, n_samples=20, n_grid=4096):
    """测试 RFF 采样函数 vs 真实仿真器的相似性"""
    lower, upper = task.bounds
    bounds = (lower.astype(np.float32), upper.astype(np.float32))

    sobol = Sobol(d=task.dim, scramble=True, seed=888)
    X_grid = (sobol.random(n_grid) * (upper - lower) + lower).astype(np.float32)
    y_true = task.evaluate_numpy(X_grid, variant_params).astype(np.float64)

    rff_vs_true_rank = []
    rff_vs_true_pearson = []
    for i in range(n_samples):
        rng = np.random.default_rng(i * 1000 + 42)
        func = SampledRFFOracleFunction(oracle_gp=gp, rng=rng, bounds=bounds)
        y_rff = func(X_grid).astype(np.float64)
        rc, _ = spearmanr(y_true, y_rff)
        pc = np.corrcoef(y_true, y_rff)[0, 1]
        rff_vs_true_rank.append(rc)
        rff_vs_true_pearson.append(pc)

    rff_vs_true_rank = np.array(rff_vs_true_rank)
    rff_vs_true_pearson = np.array(rff_vs_true_pearson)

    print(f"\n  RFF vs 真实仿真器 ({n_samples} 个 RFF):")
    print(f"  Spearman rank:  mean={rff_vs_true_rank.mean():.4f}, std={rff_vs_true_rank.std():.4f}")
    print(f"  Pearson:        mean={rff_vs_true_pearson.mean():.4f}, std={rff_vs_true_pearson.std():.4f}")


def test_rff_optimization_difficulty(gp, task, n_samples=10, n_init=2, n_steps=28):
    """测试 RFF 函数上 GP+EI 的优化难度"""
    lower, upper = task.bounds
    bounds = (lower.astype(np.float32), upper.astype(np.float32))

    # 每个 RFF 函数上用随机搜索 vs Sobol 探测
    regrets = []
    for i in range(n_samples):
        rng = np.random.default_rng(i * 1000 + 42)
        func = SampledRFFOracleFunction(oracle_gp=gp, rng=rng, bounds=bounds)

        # Sobol 估计 global min
        sobol = Sobol(d=task.dim, scramble=True, seed=0)
        X_probe = (sobol.random(8192) * (upper - lower) + lower).astype(np.float32)
        y_probe = func(X_probe).astype(np.float64)
        global_min = float(np.min(y_probe))

        # 随机搜索 n_init+n_steps 个点
        n_total = n_init + n_steps
        X_rand = rng.uniform(lower, upper, size=(n_total, task.dim)).astype(np.float32)
        y_rand = func(X_rand).astype(np.float64)
        best_rand = float(np.min(y_rand))
        regret = best_rand - global_min
        regrets.append(regret)

    regrets = np.array(regrets)
    print(f"\n  RFF 函数优化难度 ({n_samples} 个函数, {n_init+n_steps} 随机点):")
    print(f"  随机搜索 regret: mean={regrets.mean():.4f}, std={regrets.std():.4f}, "
          f"min={regrets.min():.4f}, max={regrets.max():.4f}")
    if regrets.mean() < 0.01:
        print(f"  ⚠️  RFF 函数太平滑！随机搜索即可接近最优，RL 无学习信号")
    else:
        print(f"  ✓  RFF 函数有实质优化难度")


def main():
    print("=" * 70)
    print("训练函数质量诊断: alkox_emulator")
    print("=" * 70)

    task = get_task("alkox_emulator")
    variant_params = {}  # identity variant (no transform)
    print(f"Task: {task.task_name}, dim={task.dim}")

    # --- Step 1: 在 alkox 上跑 BO 轨迹 ---
    print("\n" + "=" * 70)
    print("Step 1: 生成 BO 轨迹 (GP+EI, 30 evals)")
    print("=" * 70)
    X_traj, y_traj = run_bo_trajectory(task, variant_params, n_init=2, n_steps=28, seed=42)
    print(f"  轨迹: {X_traj.shape[0]} 点")
    print(f"  y range: [{y_traj.min():.4f}, {y_traj.max():.4f}]")
    print(f"  best y: {y_traj.min():.4f}")

    # 估计 global min
    lower, upper = task.bounds
    sobol = Sobol(d=task.dim, scramble=True, seed=0)
    X_sobol = (sobol.random(16384) * (upper - lower) + lower).astype(np.float32)
    y_sobol = task.evaluate_numpy(X_sobol, variant_params).astype(np.float64)
    global_min_est = float(np.min(y_sobol))
    print(f"  global min 估计 (16K Sobol): {global_min_est:.4f}")
    print(f"  BO regret: {y_traj.min() - global_min_est:.4f}")

    # --- Step 2: GP 拟合质量 ---
    print("\n" + "=" * 70)
    print("Step 2: Oracle GP 拟合质量")
    print("=" * 70)
    gp = test_gp_fit_quality(task, variant_params, X_traj, y_traj)

    # --- Step 3: RFF 采样函数间相似性 ---
    print("\n" + "=" * 70)
    print("Step 3: RFF 采样函数间相似性")
    print("=" * 70)
    Y_all, rank_corrs, pearson_corrs = test_rff_similarity(gp, task, n_samples=20)

    # --- Step 4: RFF vs 真实仿真器 ---
    print("\n" + "=" * 70)
    print("Step 4: RFF 采样函数 vs 真实仿真器")
    print("=" * 70)
    test_rff_vs_true(task, variant_params, gp, n_samples=20)

    # --- Step 5: RFF 函数优化难度 ---
    print("\n" + "=" * 70)
    print("Step 5: RFF 函数优化难度")
    print("=" * 70)
    test_rff_optimization_difficulty(gp, task, n_samples=20)

    # --- 对比: benzylation ---
    print("\n" + "=" * 70)
    print("===== 对比: benzylation_emulator =====")
    print("=" * 70)

    task_benz = get_task("benzylation_emulator")
    variant_params_benz = {}

    print(f"\nTask: {task_benz.task_name}, dim={task_benz.dim}")
    print("\nStep 1b: 生成 BO 轨迹 (GP+EI, 30 evals)")
    X_traj_b, y_traj_b = run_bo_trajectory(task_benz, variant_params_benz, n_init=2, n_steps=28, seed=42)
    print(f"  轨迹: {X_traj_b.shape[0]} 点, y range: [{y_traj_b.min():.4f}, {y_traj_b.max():.4f}]")

    print("\nStep 2b: Oracle GP 拟合质量")
    gp_benz = test_gp_fit_quality(task_benz, variant_params_benz, X_traj_b, y_traj_b)

    print("\nStep 3b: RFF 采样函数间相似性")
    test_rff_similarity(gp_benz, task_benz, n_samples=20)

    print("\nStep 4b: RFF vs 真实仿真器")
    test_rff_vs_true(task_benz, variant_params_benz, gp_benz, n_samples=20)

    print("\nStep 5b: RFF 函数优化难度")
    test_rff_optimization_difficulty(gp_benz, task_benz, n_samples=20)

    print("\n" + "=" * 70)
    print("诊断完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
