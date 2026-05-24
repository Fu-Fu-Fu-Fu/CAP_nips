"""
Surrogate 预测质量评估脚本

对比 Base TabPFN / Finetuned TabPFN / GP 在 eval variants 上的预测精度，
包括：
  1. Mean 预测的 MSE / MAE / R²
  2. Std 校准（覆盖率 calibration）
  3. NLL (Negative Log-Likelihood)
  4. 逐 step（context size）的精度变化
  5. 可视化：散点图、校准曲线、bin 分布诊断

用法：
  cd <repo-root>
  python MYRL/scripts/eval_surrogate_quality.py \
    --task branin_family \
    --tabpfn_tuned_path ./model/finetuned_tabpfn_branin_family.ckpt \
    --variants_path ./data/branin_family_variants_k10_seed2026.npz \
    --n_variants_per_group 10 \
    --n_test_points 500 \
    --save_dir ./results_fast/surrogate_quality
"""
from __future__ import annotations

import os
import sys
import argparse
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats.qmc import Sobol

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")
warnings.filterwarnings("ignore", message=".*ABNORMAL_TERMINATION_IN_LNSRCH.*")
warnings.filterwarnings("ignore", message=".*scale the data.*")
warnings.filterwarnings("ignore", message=".*optimal value found.*close to the specified.*bound.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# -- bootstrap --
from _bootstrap import bootstrap_project_root
bootstrap_project_root()

from tabpfn import TabPFNRegressor
from myrl.tasks import get_task
from myrl.bo.select_candidates import compute_std_from_tabpfn_output, predict_tabpfn_with_normalization


# ========================================================================
# Helper: surrogates
# ========================================================================
def make_gp(input_dim: int) -> GaussianProcessRegressor:
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=[1.0] * input_dim,
        length_scale_bounds=(1e-5, 1e5),
        nu=2.5,
    )
    return GaussianProcessRegressor(
        kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=3,
    )


def make_tabpfn(model_path: Optional[str] = None, device: str = "cuda") -> TabPFNRegressor:
    kwargs = dict(
        device=device,
        n_estimators=1,
        random_state=42,
        inference_precision=torch.float32,
        ignore_pretraining_limits=True,
    )
    if model_path is not None:
        kwargs["model_path"] = model_path
    return TabPFNRegressor(**kwargs)


def predict_gp(gp: GaussianProcessRegressor, X_ctx, y_ctx, X_test):
    gp.fit(X_ctx, y_ctx)
    mean, std = gp.predict(X_test, return_std=True)
    return np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)


def predict_tabpfn(reg: TabPFNRegressor, X_ctx, y_ctx, X_test):
    mean, std, full_out = predict_tabpfn_with_normalization(reg, X_ctx, y_ctx, X_test)
    return mean, std


def predict_tabpfn_with_bin_info(reg: TabPFNRegressor, X_ctx, y_ctx, X_test):
    """返回 mean, std 以及 bin 边界和 logits（用于诊断）"""
    mean, std, full_out = predict_tabpfn_with_normalization(reg, X_ctx, y_ctx, X_test)

    criterion = full_out.get("criterion", None)
    logits = full_out.get("logits", None)
    borders = criterion.borders.cpu().numpy() if criterion is not None else None
    logits_np = logits.detach().cpu().numpy() if logits is not None else None
    return mean, std, borders, logits_np


# ========================================================================
# Metrics
# ========================================================================
def compute_metrics(y_true, pred_mean, pred_std):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred_mean = np.asarray(pred_mean, dtype=np.float64).reshape(-1)
    pred_std = np.asarray(pred_std, dtype=np.float64).reshape(-1)
    pred_std = np.maximum(pred_std, 1e-12)

    residual = y_true - pred_mean
    mse = float(np.mean(residual ** 2))
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(mse))

    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))

    # NLL (Gaussian)
    nll = float(np.mean(0.5 * np.log(2 * np.pi * pred_std ** 2) + residual ** 2 / (2 * pred_std ** 2)))

    # Calibration: coverage at different confidence levels
    z_scores = np.abs(residual) / pred_std
    coverages = {}
    for p in [0.5, 0.8, 0.9, 0.95, 0.99]:
        z_thresh = norm.ppf(0.5 + p / 2)
        actual_cov = float(np.mean(z_scores <= z_thresh))
        coverages[f"cov_{int(p*100)}"] = actual_cov

    # Median absolute calibration error
    expected_covs = np.array([0.5, 0.8, 0.9, 0.95, 0.99])
    actual_covs = np.array([coverages[f"cov_{int(p*100)}"] for p in expected_covs])
    mace = float(np.mean(np.abs(expected_covs - actual_covs)))

    return {
        "mse": mse, "rmse": rmse, "mae": mae, "r2": r2,
        "nll": nll, "mace": mace,
        **coverages,
    }


# ========================================================================
# Variant sampling (reuse task logic, exclude training variants)
# ========================================================================
def load_training_variant_keys(variants_path: str) -> set:
    if not os.path.exists(variants_path):
        return set()
    data = np.load(variants_path, allow_pickle=True)
    variants = data["variants"].tolist()
    keys = set()
    for v in variants:
        key = tuple(sorted((k, round(float(val), 6)) for k, val in v.items()))
        keys.add(key)
    return keys


def sample_eval_variants(task, group_name, spec, n, forbidden_keys, seed=12345):
    rng = np.random.default_rng(seed)
    used = set()
    out = []
    for _ in range(n):
        for _ in range(50000):
            s = int(rng.integers(0, 1_000_000_000))
            suite = task.sample_eval_suite(n_per_group=1, seed=s, suite_specs={group_name: spec})
            cands = suite.get(group_name, [])
            if not cands:
                continue
            v = cands[0]
            key = tuple(sorted((k, round(float(val), 6)) for k, val in v.items()))
            if key in forbidden_keys or key in used:
                continue
            used.add(key)
            out.append(v)
            break
    return out


# ========================================================================
# Core evaluation
# ========================================================================
def evaluate_surrogates_on_variant(
    task, variant_params, surrogates, context_sizes, n_test_points, rng,
):
    """
    对一个 variant 在多个 context size 下评估各 surrogate 的预测质量。

    流程：
    1. 生成一组密集测试点，计算真实 y
    2. 对每个 context_size：
       a. 随机采样 context 点，计算真实 y
       b. 各 surrogate 在 test 点上预测 mean/std
       c. 计算 metrics
    """
    lower, upper = task.bounds
    dim = task.dim

    # 生成固定的测试点集
    sobol_seed = int(rng.integers(0, 100000))
    sobol = Sobol(d=dim, scramble=True, seed=sobol_seed)
    X_test = (sobol.random(n_test_points) * (upper - lower) + lower).astype(np.float32)
    y_test = task.evaluate_numpy(X_test, variant_params).astype(np.float64)

    results_by_ctx = {}
    for n_ctx in context_sizes:
        # 随机采样 context 点
        X_ctx = rng.uniform(lower, upper, size=(n_ctx, dim)).astype(np.float32)
        y_ctx = task.evaluate_numpy(X_ctx, variant_params).astype(np.float64)

        ctx_results = {}
        for name, predict_fn in surrogates.items():
            try:
                pred_mean, pred_std = predict_fn(X_ctx, y_ctx, X_test)
                metrics = compute_metrics(y_test, pred_mean, pred_std)
                ctx_results[name] = metrics
            except Exception as e:
                ctx_results[name] = {"error": str(e)}

        results_by_ctx[n_ctx] = ctx_results

    return results_by_ctx, X_test, y_test


def collect_bin_diagnostics(tabpfn_tuned, tabpfn_base, task, variant_params, rng, n_ctx=5):
    """收集一次 TabPFN 的 bin 诊断信息"""
    lower, upper = task.bounds
    dim = task.dim
    X_ctx = rng.uniform(lower, upper, size=(n_ctx, dim)).astype(np.float32)
    y_ctx = task.evaluate_numpy(X_ctx, variant_params).astype(np.float64)

    sobol_seed = int(rng.integers(0, 100000))
    sobol = Sobol(d=dim, scramble=True, seed=sobol_seed)
    X_test = (sobol.random(200) * (upper - lower) + lower).astype(np.float32)
    y_test = task.evaluate_numpy(X_test, variant_params).astype(np.float64)

    diag = {}
    for name, reg in [("TabPFN-tuned", tabpfn_tuned), ("TabPFN-base", tabpfn_base)]:
        if reg is None:
            continue
        mean, std, borders, logits = predict_tabpfn_with_bin_info(reg, X_ctx, y_ctx, X_test)
        diag[name] = {
            "mean": mean, "std": std, "borders": borders, "logits": logits,
            "y_test": y_test, "y_ctx": y_ctx,
        }
    return diag


# ========================================================================
# Plotting
# ========================================================================
def plot_scatter(all_results, save_dir, group_name):
    """各 surrogate 的 pred_mean vs y_true 散点图（汇总所有 variant 和 context size）"""
    # 这里使用 metrics 做 bar chart 更合适
    pass


def plot_metrics_by_context_size(group_results, context_sizes, save_dir, group_name):
    """
    对每个 group，画 metric 随 context_size 变化的折线图。
    group_results: list of (variant_results_by_ctx,)
    """
    surrogate_names = None
    metric_names = ["rmse", "mae", "r2", "nll", "mace"]
    metric_labels = ["RMSE", "MAE", "R²", "NLL (Gaussian)", "MACE (calibration)"]

    # 聚合
    # agg[metric][surrogate][ctx_idx] = list of values across variants
    agg = {m: {} for m in metric_names}

    for variant_result in group_results:
        for ctx_idx, n_ctx in enumerate(context_sizes):
            ctx_res = variant_result.get(n_ctx, {})
            for sname, metrics in ctx_res.items():
                if "error" in metrics:
                    continue
                if surrogate_names is None:
                    surrogate_names = list(ctx_res.keys())
                for m in metric_names:
                    if m not in metrics:
                        continue
                    if sname not in agg[m]:
                        agg[m][sname] = [[] for _ in context_sizes]
                    agg[m][sname][ctx_idx].append(metrics[m])

    if surrogate_names is None:
        return

    colors = {"GP": "tab:blue", "TabPFN-base": "tab:orange", "TabPFN-tuned": "tab:green"}
    markers = {"GP": "o", "TabPFN-base": "s", "TabPFN-tuned": "^"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, (m, label) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[i]
        for sname in surrogate_names:
            if sname not in agg[m]:
                continue
            means = [np.mean(vals) if vals else np.nan for vals in agg[m][sname]]
            stds = [np.std(vals) / max(1, np.sqrt(len(vals))) if vals else 0 for vals in agg[m][sname]]
            ax.errorbar(context_sizes, means, yerr=stds,
                       label=sname, color=colors.get(sname, "gray"),
                       marker=markers.get(sname, "x"), capsize=3, linewidth=2)
        ax.set_xlabel("Context Size (n_ctx)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 6th subplot: summary bar chart at final context size
    ax = axes[5]
    final_ctx = context_sizes[-1]
    x_pos = np.arange(len(surrogate_names))
    bar_width = 0.25
    for j, m in enumerate(["rmse", "nll", "mace"]):
        vals = []
        for sname in surrogate_names:
            if sname in agg[m] and agg[m][sname][-1]:
                vals.append(np.mean(agg[m][sname][-1]))
            else:
                vals.append(0)
        ax.bar(x_pos + j * bar_width, vals, bar_width, label=m.upper())
    ax.set_xticks(x_pos + bar_width)
    ax.set_xticklabels(surrogate_names)
    ax.set_title(f"Summary at n_ctx={final_ctx}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Surrogate Quality: {group_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"metrics_vs_context_{group_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_calibration_curves(group_results, context_sizes, save_dir, group_name):
    """画校准曲线：expected coverage vs actual coverage"""
    expected_ps = np.array([0.5, 0.8, 0.9, 0.95, 0.99])
    cov_keys = [f"cov_{int(p*100)}" for p in expected_ps]

    # 选两个代表性 context size 画
    ctx_to_plot = [context_sizes[0], context_sizes[len(context_sizes)//2], context_sizes[-1]]
    ctx_to_plot = sorted(set(c for c in ctx_to_plot if c in context_sizes))

    fig, axes = plt.subplots(1, len(ctx_to_plot), figsize=(6 * len(ctx_to_plot), 5))
    if len(ctx_to_plot) == 1:
        axes = [axes]

    colors = {"GP": "tab:blue", "TabPFN-base": "tab:orange", "TabPFN-tuned": "tab:green"}

    for ax_idx, n_ctx in enumerate(ctx_to_plot):
        ax = axes[ax_idx]
        # aggregate across variants
        agg_cov = {}
        for variant_result in group_results:
            ctx_res = variant_result.get(n_ctx, {})
            for sname, metrics in ctx_res.items():
                if "error" in metrics:
                    continue
                if sname not in agg_cov:
                    agg_cov[sname] = [[] for _ in expected_ps]
                for k_idx, ck in enumerate(cov_keys):
                    if ck in metrics:
                        agg_cov[sname][k_idx].append(metrics[ck])

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        for sname, cov_lists in agg_cov.items():
            actual = [np.mean(vals) if vals else np.nan for vals in cov_lists]
            ax.plot(expected_ps, actual, "o-", label=sname,
                   color=colors.get(sname, "gray"), linewidth=2, markersize=6)
        ax.set_xlabel("Expected Coverage")
        ax.set_ylabel("Actual Coverage")
        ax.set_title(f"n_ctx = {n_ctx}")
        ax.legend()
        ax.set_xlim(0.4, 1.02)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Calibration Curves: {group_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, f"calibration_{group_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_bin_diagnostics(diag, save_dir, group_name):
    """诊断 TabPFN 的 bin 离散化问题"""
    fig, axes = plt.subplots(1, len(diag), figsize=(7 * len(diag), 5))
    if len(diag) == 1:
        axes = [axes]

    for ax_idx, (name, d) in enumerate(diag.items()):
        ax = axes[ax_idx]
        borders = d["borders"]
        y_test = d["y_test"]
        y_ctx = d["y_ctx"]

        # Histogram of true y values vs bin boundaries
        ax.hist(y_test, bins=50, alpha=0.5, color="tab:blue", label="y_test distribution", density=True)
        ax.hist(y_ctx, bins=20, alpha=0.5, color="tab:green", label="y_ctx distribution", density=True)

        # Mark bin boundaries
        if borders is not None:
            bin_min = float(borders[0])
            bin_max = float(borders[-1])
            n_bins = len(borders) - 1
            ax.axvline(bin_min, color="red", linestyle="--", linewidth=2, label=f"bin range [{bin_min:.1f}, {bin_max:.1f}]")
            ax.axvline(bin_max, color="red", linestyle="--", linewidth=2)
            # show a few internal borders
            step = max(1, n_bins // 10)
            for b in borders[::step]:
                ax.axvline(float(b), color="red", linestyle=":", alpha=0.3, linewidth=0.5)
            ax.set_title(f"{name}\n{n_bins} bins, range=[{bin_min:.2f}, {bin_max:.2f}]")
        else:
            ax.set_title(name)

        ax.set_xlabel("y value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    fig.suptitle(f"TabPFN Bin Diagnostics: {group_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, f"bin_diagnostics_{group_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_prediction_scatter(
    task, variant_params, surrogates, n_ctx, save_dir, group_name, variant_idx, rng,
):
    """画一个 variant 上各 surrogate 的 pred_mean vs y_true 散点图"""
    lower, upper = task.bounds
    dim = task.dim

    X_ctx = rng.uniform(lower, upper, size=(n_ctx, dim)).astype(np.float32)
    y_ctx = task.evaluate_numpy(X_ctx, variant_params).astype(np.float64)

    sobol_seed = int(rng.integers(0, 100000))
    sobol = Sobol(d=dim, scramble=True, seed=sobol_seed)
    X_test = (sobol.random(300) * (upper - lower) + lower).astype(np.float32)
    y_test = task.evaluate_numpy(X_test, variant_params).astype(np.float64)

    n_surr = len(surrogates)
    fig, axes = plt.subplots(1, n_surr, figsize=(6 * n_surr, 5))
    if n_surr == 1:
        axes = [axes]

    colors = {"GP": "tab:blue", "TabPFN-base": "tab:orange", "TabPFN-tuned": "tab:green"}

    for ax_idx, (sname, predict_fn) in enumerate(surrogates.items()):
        ax = axes[ax_idx]
        try:
            pred_mean, pred_std = predict_fn(X_ctx, y_ctx, X_test)
            metrics = compute_metrics(y_test, pred_mean, pred_std)

            # scatter
            ax.errorbar(y_test, pred_mean, yerr=2 * pred_std,
                        fmt=".", alpha=0.15, color=colors.get(sname, "gray"),
                        elinewidth=0.5, markersize=3, label="±2σ")
            ax.scatter(y_test, pred_mean, s=8, alpha=0.6, color=colors.get(sname, "gray"), zorder=5)

            # diagonal
            ymin = min(y_test.min(), pred_mean.min())
            ymax = max(y_test.max(), pred_mean.max())
            ax.plot([ymin, ymax], [ymin, ymax], "k--", alpha=0.5, linewidth=1)

            ax.set_title(f"{sname}\nRMSE={metrics['rmse']:.3f}  R²={metrics['r2']:.3f}  NLL={metrics['nll']:.2f}")
        except Exception as e:
            ax.set_title(f"{sname}\nERROR: {e}")

        ax.set_xlabel("y_true")
        ax.set_ylabel("pred_mean")

    fig.suptitle(f"Prediction Scatter: {group_name} variant {variant_idx} (n_ctx={n_ctx})", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, f"scatter_{group_name}_v{variant_idx}_ctx{n_ctx}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_std_scatter(
    task, variant_params, surrogates, n_ctx, save_dir, group_name, variant_idx, rng,
):
    """画各 surrogate 的 pred_std vs |residual| 散点图，观察 std 校准"""
    lower, upper = task.bounds
    dim = task.dim

    X_ctx = rng.uniform(lower, upper, size=(n_ctx, dim)).astype(np.float32)
    y_ctx = task.evaluate_numpy(X_ctx, variant_params).astype(np.float64)

    sobol_seed = int(rng.integers(0, 100000))
    sobol = Sobol(d=dim, scramble=True, seed=sobol_seed)
    X_test = (sobol.random(300) * (upper - lower) + lower).astype(np.float32)
    y_test = task.evaluate_numpy(X_test, variant_params).astype(np.float64)

    n_surr = len(surrogates)
    fig, axes = plt.subplots(1, n_surr, figsize=(6 * n_surr, 5))
    if n_surr == 1:
        axes = [axes]

    colors = {"GP": "tab:blue", "TabPFN-base": "tab:orange", "TabPFN-tuned": "tab:green"}

    for ax_idx, (sname, predict_fn) in enumerate(surrogates.items()):
        ax = axes[ax_idx]
        try:
            pred_mean, pred_std = predict_fn(X_ctx, y_ctx, X_test)
            residual = np.abs(y_test - pred_mean)

            ax.scatter(pred_std, residual, s=8, alpha=0.5, color=colors.get(sname, "gray"))
            # ideal: |residual| ≈ pred_std on average
            max_val = max(pred_std.max(), residual.max())
            ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, linewidth=1, label="|res|=σ")
            ax.plot([0, max_val], [0, 2 * max_val], "k:", alpha=0.3, linewidth=1, label="|res|=2σ")

            # mean ratio
            valid = pred_std > 1e-8
            if valid.any():
                ratio = np.mean(residual[valid] / pred_std[valid])
                ax.set_title(f"{sname}\nmean |res|/σ = {ratio:.2f} (ideal ≈ 0.80)")
            else:
                ax.set_title(sname)
        except Exception as e:
            ax.set_title(f"{sname}\nERROR: {e}")

        ax.set_xlabel("pred_std (σ)")
        ax.set_ylabel("|residual|")
        ax.legend(fontsize=8)

    fig.suptitle(f"Std Calibration: {group_name} variant {variant_idx} (n_ctx={n_ctx})", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, f"std_scatter_{group_name}_v{variant_idx}_ctx{n_ctx}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ========================================================================
# Main
# ========================================================================
def main():
    parser = argparse.ArgumentParser(description="Surrogate prediction quality evaluation")
    parser.add_argument("--task", type=str, default="branin_family")
    parser.add_argument("--tabpfn_tuned_path", type=str,
                        default="./model/finetuned_tabpfn_branin_family.ckpt")
    parser.add_argument("--variants_path", type=str,
                        default="./data/branin_family_variants_k10_seed2026.npz",
                        help="Training variants path (to exclude from eval)")
    parser.add_argument("--n_variants_per_group", type=int, default=10)
    parser.add_argument("--n_test_points", type=int, default=500,
                        help="Number of test points per variant for evaluation")
    parser.add_argument("--context_sizes", type=str, default="2,5,8,12,16,20",
                        help="Comma-separated context sizes to evaluate")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save_dir", type=str, default="./results_fast/surrogate_quality")
    parser.add_argument("--groups", type=str, default="in_range,ood_level_1,ood_level_2,ood_level_3",
                        help="Comma-separated group names")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    context_sizes = [int(x) for x in args.context_sizes.split(",")]
    group_names = [g.strip() for g in args.groups.split(",")]

    print("=" * 70)
    print("Surrogate Prediction Quality Evaluation")
    print("=" * 70)
    print(f"Task:            {args.task}")
    print(f"TabPFN tuned:    {args.tabpfn_tuned_path}")
    print(f"Variants (excl): {args.variants_path}")
    print(f"Groups:          {group_names}")
    print(f"Variants/group:  {args.n_variants_per_group}")
    print(f"Test points:     {args.n_test_points}")
    print(f"Context sizes:   {context_sizes}")
    print(f"Device:          {args.device}")
    print(f"Save dir:        {args.save_dir}")
    print("=" * 70)

    task = get_task(args.task)
    suite_specs = task.default_variant_suite()

    # Load training variant keys (to exclude)
    forbidden_keys = load_training_variant_keys(args.variants_path)
    print(f"\nExcluding {len(forbidden_keys)} training variants")

    # Initialize surrogates
    print("\nInitializing surrogates...")
    gp_template_dim = task.dim

    tabpfn_base = None
    try:
        tabpfn_base = make_tabpfn(model_path=None, device=args.device)
        # Do a quick test fit to see if model weights are available
        tabpfn_base.fit(np.array([[0.0, 0.0]], dtype=np.float32), np.array([0.0], dtype=np.float32))
        tabpfn_base.predict(np.array([[1.0, 1.0]], dtype=np.float32))
        print("  TabPFN-base: initialized (pretrained weights)")
    except Exception as e:
        print(f"  TabPFN-base: UNAVAILABLE ({e})")
        print("  (Skipping TabPFN-base — may need HuggingFace login for gated model)")
        tabpfn_base = None

    tabpfn_tuned = make_tabpfn(model_path=args.tabpfn_tuned_path, device=args.device)
    print(f"  TabPFN-tuned: loaded from {args.tabpfn_tuned_path}")

    # Predict functions (closures)
    def pred_gp(X_ctx, y_ctx, X_test):
        gp = make_gp(gp_template_dim)
        return predict_gp(gp, X_ctx, y_ctx, X_test)

    def pred_tabpfn_base(X_ctx, y_ctx, X_test):
        return predict_tabpfn(tabpfn_base, X_ctx, y_ctx, X_test)

    def pred_tabpfn_tuned(X_ctx, y_ctx, X_test):
        return predict_tabpfn(tabpfn_tuned, X_ctx, y_ctx, X_test)

    surrogates = {"GP": pred_gp}
    if tabpfn_base is not None:
        surrogates["TabPFN-base"] = pred_tabpfn_base
    surrogates["TabPFN-tuned"] = pred_tabpfn_tuned

    rng = np.random.default_rng(args.seed)
    all_results = {}

    for group_name in group_names:
        if group_name not in suite_specs:
            print(f"\n[WARN] Group {group_name} not in task suite specs, skipping")
            continue

        spec = suite_specs[group_name]
        print(f"\n{'='*70}")
        print(f"Group: {group_name}")
        print(f"Spec: {spec}")
        print(f"{'='*70}")

        # Sample eval variants
        variants = sample_eval_variants(
            task, group_name, spec, args.n_variants_per_group, forbidden_keys,
            seed=int(rng.integers(0, 1_000_000)),
        )
        print(f"  Sampled {len(variants)} eval variants")

        group_dir = os.path.join(args.save_dir, group_name)
        os.makedirs(group_dir, exist_ok=True)

        group_variant_results = []

        for v_idx, vparams in enumerate(variants):
            param_str = ", ".join(f"{k}={v:.2f}" for k, v in sorted(vparams.items())
                                  if k not in ("alpha", "beta"))
            print(f"\n  Variant {v_idx+1}/{len(variants)}: {param_str}")

            variant_rng = np.random.default_rng(int(rng.integers(0, 2**31)))

            # Metrics evaluation
            results_by_ctx, X_test, y_test = evaluate_surrogates_on_variant(
                task, vparams, surrogates, context_sizes, args.n_test_points, variant_rng,
            )
            group_variant_results.append(results_by_ctx)

            # Print summary at largest context size
            ctx_max = context_sizes[-1]
            print(f"    n_ctx={ctx_max}:", end="")
            for sname in surrogates:
                m = results_by_ctx.get(ctx_max, {}).get(sname, {})
                if "error" in m:
                    print(f"  {sname}: ERROR", end="")
                else:
                    print(f"  {sname}: RMSE={m['rmse']:.3f} NLL={m['nll']:.2f} MACE={m['mace']:.3f}", end="")
            print()

            # Scatter plots for first 2 variants only (to save time)
            if v_idx < 2:
                scatter_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
                for ctx_plot in [context_sizes[0], context_sizes[-1]]:
                    plot_prediction_scatter(
                        task, vparams, surrogates, ctx_plot, group_dir,
                        group_name, v_idx, scatter_rng,
                    )
                    plot_std_scatter(
                        task, vparams, surrogates, ctx_plot, group_dir,
                        group_name, v_idx, np.random.default_rng(scatter_rng.integers(0, 2**31)),
                    )

        # Bin diagnostics (once per group)
        diag_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
        diag = collect_bin_diagnostics(
            tabpfn_tuned, tabpfn_base if tabpfn_base is not None else None,
            task, variants[0], diag_rng, n_ctx=5,
        )
        if diag:
            plot_bin_diagnostics(diag, group_dir, group_name)

        # Aggregated plots
        plot_metrics_by_context_size(group_variant_results, context_sizes, group_dir, group_name)
        plot_calibration_curves(group_variant_results, context_sizes, group_dir, group_name)

        # Store results
        all_results[group_name] = group_variant_results

    # ======== Overall summary ========
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY (mean across all variants, at largest context size)")
    print(f"{'='*70}")
    ctx_final = context_sizes[-1]
    summary_data = {}
    for group_name, group_variant_results in all_results.items():
        summary_data[group_name] = {}
        for sname in surrogates:
            metric_agg = {}
            for variant_result in group_variant_results:
                m = variant_result.get(ctx_final, {}).get(sname, {})
                if "error" in m:
                    continue
                for k, v in m.items():
                    if isinstance(v, (int, float)):
                        metric_agg.setdefault(k, []).append(v)
            summary_data[group_name][sname] = {
                k: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
                for k, vals in metric_agg.items()
            }

    # Print table
    header_metrics = ["rmse", "nll", "mace", "r2"]
    for group_name in all_results:
        print(f"\n  {group_name} (n_ctx={ctx_final}):")
        print(f"    {'Surrogate':<16} {'RMSE':>10} {'NLL':>10} {'MACE':>10} {'R²':>10}")
        print(f"    {'-'*56}")
        for sname in surrogates:
            sd = summary_data[group_name].get(sname, {})
            vals = []
            for mk in header_metrics:
                if mk in sd:
                    vals.append(f"{sd[mk]['mean']:>8.4f}±{sd[mk]['std']:.3f}"[:10])
                else:
                    vals.append(f"{'N/A':>10}")
            print(f"    {sname:<16} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10}")

    # Save JSON
    json_path = os.path.join(args.save_dir, "surrogate_quality_summary.json")
    os.makedirs(args.save_dir, exist_ok=True)

    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=2, default=_convert)
    print(f"\nSummary JSON saved to: {json_path}")
    print(f"All plots saved to: {args.save_dir}/")


if __name__ == "__main__":
    main()
