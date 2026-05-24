"""
对比测试：BNN vs Oracle GP 作为训练函数生成器

使用现有管线代码：
- BNN: train_bnn_surrogates.build_and_train_bnn + train_rl.SampledBNNFunction
- GP:  train_rl.load_oracle_gps + SampledRFFOracleFunction

测试任务：
- Hartmann 6D (oracle_gp 的成功参照)
- Alkox (需要找到合适的代理模型)

度量：
- inter: 采样函数间 Spearman（结构相似性，目标 ~0.5-0.9）
- vs_true: 采样函数 vs 真实函数 Spearman（与真实的一致性）
- y_range: 采样函数值域宽度
- CV: 采样函数在网格点上的变异系数（后验宽度）

使用方式：
  export PYTHONPATH="MYRL:olympus/src"
  export TF_USE_LEGACY_KERAS=1
  python MYRL/scripts/test_bnn_vs_gp_surrogate.py
"""
from __future__ import annotations

import os, sys
import numpy as np
from scipy.stats import spearmanr
from scipy.stats.qmc import Sobol

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _bootstrap import bootstrap_project_root
bootstrap_project_root()

from myrl.rl.train_rl import (
    load_oracle_gps_from_trajectories_cache,
    SampledRFFOracleFunction,
    SampledBNNFunction,
)
from scripts.train_bnn_surrogates import build_and_train_bnn


def evaluate_surrogate(name, sample_funcs, y_true, X_grid):
    """
    sample_funcs: list of callables, each is a sampled deterministic function
    """
    n_samples = len(sample_funcs)
    Y_all = []
    for f in sample_funcs:
        y_i = np.asarray(f(X_grid), dtype=np.float64).reshape(-1)
        Y_all.append(y_i)
    Y_all = np.array(Y_all)

    # 检查 NaN/Inf
    valid = np.all(np.isfinite(Y_all), axis=1)
    if valid.sum() < 3:
        print(f"  {name:45s}  *** 太多 NaN/Inf，跳过 ***")
        return

    Y_valid = Y_all[valid]
    n_valid = len(Y_valid)

    # inter-function Spearman
    rcs_inter = []
    for i in range(n_valid):
        for j in range(i + 1, n_valid):
            r, _ = spearmanr(Y_valid[i], Y_valid[j])
            if np.isfinite(r):
                rcs_inter.append(r)
    mean_inter = np.mean(rcs_inter) if rcs_inter else float('nan')

    # vs true Spearman
    rcs_true = []
    for i in range(n_valid):
        r, _ = spearmanr(y_true, Y_valid[i])
        if np.isfinite(r):
            rcs_true.append(r)
    mean_vs_true = np.mean(rcs_true) if rcs_true else float('nan')

    # y_range
    y_ranges = [float(Y_valid[i].max() - Y_valid[i].min()) for i in range(n_valid)]
    mean_y_range = np.mean(y_ranges)

    # CV
    y_stds = Y_valid.std(axis=0)
    y_means_abs = np.abs(Y_valid.mean(axis=0)) + 1e-8
    cv = float(np.median(y_stds / y_means_abs))

    print(f"  {name:45s}  inter={mean_inter:+.3f}  vs_true={mean_vs_true:+.3f}  "
          f"y_range={mean_y_range:7.1f}  CV={cv:.3f}")


def make_rff_samples(gp, bounds, n_samples=20):
    """从 Oracle GP 后验采样 n 个 RFF 函数"""
    funcs = []
    for i in range(n_samples):
        rng = np.random.default_rng(i * 1000 + 42)
        funcs.append(SampledRFFOracleFunction(oracle_gp=gp, rng=rng, bounds=bounds))
    return funcs


def make_bnn_samples(bnn_params, bounds, n_samples=20, temp=1.0):
    """从 BNN 后验采样 n 个函数，支持温度控制"""
    funcs = []
    for i in range(n_samples):
        rng = np.random.default_rng(i * 1000 + 42)
        # 用 temp 缩放 sigma
        modified_params = {
            'layers': [(loc, sigma * temp, bias)
                       for loc, sigma, bias in bnn_params['layers']],
            'y_mean': bnn_params['y_mean'],
            'y_std': bnn_params['y_std'],
        }
        funcs.append(SampledBNNFunction(
            bnn_params=modified_params, rng=rng, bounds=bounds
        ))
    return funcs


def test_on_task(task_name, trajs_path, test_variant_indices=None):
    from myrl.tasks.registry import get_task

    task = get_task(task_name)
    dim = task.dim
    bounds = (np.zeros(dim, dtype=np.float32), np.ones(dim, dtype=np.float32))

    oracle_gps, _ = load_oracle_gps_from_trajectories_cache(trajs_path)
    data = np.load(trajs_path, allow_pickle=True)
    X_trajs = data['X_trajs']
    y_trajs = data['y_trajs']
    variant_indices = data['variant_indices']
    all_variants = data['variants'].tolist()
    n_variants = len(oracle_gps)

    if test_variant_indices is None:
        test_variant_indices = list(range(min(3, n_variants)))

    n_grid = 4096
    n_samples = 20

    print(f"\n{'='*95}")
    print(f"任务: {task_name} (dim={dim}), {n_variants} 个变体, 测试 variant {test_variant_indices}")
    print(f"{'='*95}")

    for vi in test_variant_indices:
        vparams = all_variants[vi]
        gp = oracle_gps[vi]

        # 训练数据
        idxs = np.where(variant_indices == vi)[0]
        traj_idx = int(idxs.min())
        X_train = X_trajs[traj_idx].astype(np.float64)
        y_train = y_trajs[traj_idx].astype(np.float64).reshape(-1)

        # 测试网格
        sobol = Sobol(d=dim, scramble=True, seed=888 + vi)
        X_grid = sobol.random(n_grid).astype(np.float32)
        y_true = task.evaluate_numpy(X_grid, vparams).astype(np.float64).reshape(-1)

        # GP kernel info
        params = gp.kernel_.get_params()
        ls = None
        for key, val in params.items():
            if 'length_scale' in key and 'bounds' not in key:
                ls = np.atleast_1d(val)
        ls_str = ','.join([f'{v:.2g}' for v in ls]) if ls is not None else '?'

        print(f"\n--- Variant {vi}: y_train=[{y_train.min():.1f},{y_train.max():.1f}], "
              f"n={len(y_train)}, GP ls=[{ls_str}] ---")

        # === Oracle GP → RFF ===
        rff_funcs = make_rff_samples(gp, bounds, n_samples)
        evaluate_surrogate("Oracle GP → RFF", rff_funcs, y_true, X_grid)

        # === BNN (现有管线: 48×3, kl=0.001) ===
        for hidden, depth, label in [(48, 3, "48×3"), (16, 2, "16×2")]:
            print(f"  [训练 BNN {label}, kl=0.001 ...]", end="", flush=True)
            bnn = build_and_train_bnn(
                X_train.astype(np.float32), y_train.astype(np.float32),
                hidden_nodes=hidden, hidden_depth=depth,
                max_epochs=15000, kl_weight=0.001, patience=50,
                seed=42 + vi
            )
            print(" done")

            for temp in [1.0, 0.3, 0.1]:
                funcs = make_bnn_samples(bnn, bounds, n_samples, temp=temp)
                evaluate_surrogate(
                    f"BNN {label} kl=0.001 temp={temp}",
                    funcs, y_true, X_grid
                )

        # === BNN 更小 kl_weight (0.0001) — 放宽后验 ===
        for hidden, depth, label in [(48, 3, "48×3"), (16, 2, "16×2")]:
            print(f"  [训练 BNN {label}, kl=0.0001 ...]", end="", flush=True)
            bnn_loose = build_and_train_bnn(
                X_train.astype(np.float32), y_train.astype(np.float32),
                hidden_nodes=hidden, hidden_depth=depth,
                max_epochs=15000, kl_weight=0.0001, patience=50,
                seed=42 + vi
            )
            print(" done")

            for temp in [1.0, 0.3]:
                funcs = make_bnn_samples(bnn_loose, bounds, n_samples, temp=temp)
                evaluate_surrogate(
                    f"BNN {label} kl=0.0001 temp={temp}",
                    funcs, y_true, X_grid
                )

        # === BNN 更大 kl_weight (0.01) — 收紧后验增加多样性 ===
        for hidden, depth, label in [(48, 3, "48×3")]:
            print(f"  [训练 BNN {label}, kl=0.01 ...]", end="", flush=True)
            bnn_tight = build_and_train_bnn(
                X_train.astype(np.float32), y_train.astype(np.float32),
                hidden_nodes=hidden, hidden_depth=depth,
                max_epochs=15000, kl_weight=0.01, patience=50,
                seed=42 + vi
            )
            print(" done")

            for temp in [1.0, 0.3]:
                funcs = make_bnn_samples(bnn_tight, bounds, n_samples, temp=temp)
                evaluate_surrogate(
                    f"BNN {label} kl=0.01 temp={temp}",
                    funcs, y_true, X_grid
                )


def main():
    print("=" * 95)
    print("BNN vs Oracle GP 代理模型对比测试（使用现有管线代码）")
    print("=" * 95)
    print()
    print("度量说明:")
    print("  inter   = 采样函数间 Spearman rank correlation (理想: 0.5~0.9，太高=退化，太低=随机)")
    print("  vs_true = 采样函数 vs 真实函数 Spearman (越高越好)")
    print("  y_range = 采样函数的平均值域宽度")
    print("  CV      = 函数值在网格点上跨采样的变异系数中位数 (后验宽度)")

    # === Hartmann 6D: oracle_gp 参照（已知成功） ===
    h6d_trajs = './data/hartmann_6d_family_bo_trajs_k10_boSeed2026.npz'
    if os.path.exists(h6d_trajs):
        test_on_task("hartmann_6d_family", h6d_trajs, test_variant_indices=[0, 1, 2])
    else:
        print(f"\n[跳过 Hartmann 6D] 轨迹文件不存在: {h6d_trajs}")

    # === Alkox: 需要找到合适的代理模型 ===
    alkox_trajs = './data/alkox_emulator_bo_trajs_k10_boSeed2026_transform.npz'
    if os.path.exists(alkox_trajs):
        test_on_task("alkox_emulator", alkox_trajs, test_variant_indices=[0, 1, 3])
    else:
        print(f"\n[跳过 alkox] 轨迹文件不存在: {alkox_trajs}")

    print("\n" + "=" * 95)
    print("测试完成")
    print("=" * 95)


if __name__ == "__main__":
    main()
