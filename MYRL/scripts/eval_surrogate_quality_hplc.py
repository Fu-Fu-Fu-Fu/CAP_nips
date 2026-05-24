"""
GP vs TabPFN-base surrogate quality on HPLC and Alkox emulators.
Compares ranking accuracy (Spearman), RMSE, calibration (MACE).
"""
from __future__ import annotations

import os
import sys
import argparse
import json
import time
import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore")

from _bootstrap import bootstrap_project_root
bootstrap_project_root()

from scipy.stats import spearmanr, norm
from scipy.stats.qmc import Sobol
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from tabpfn import TabPFNRegressor

from myrl.tasks import get_task
from myrl.bo.select_candidates import predict_tabpfn_with_normalization


def make_gp(dim: int) -> GaussianProcessRegressor:
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=[1.0] * dim,
        length_scale_bounds=(1e-5, 1e5),
        nu=2.5,
    )
    return GaussianProcessRegressor(
        kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=3,
    )


def compute_metrics(y_true, pred_mean, pred_std):
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    pred_mean = np.asarray(pred_mean, dtype=np.float64).ravel()
    pred_std = np.maximum(np.asarray(pred_std, dtype=np.float64).ravel(), 1e-12)

    residual = y_true - pred_mean
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
    sp, _ = spearmanr(pred_mean, y_true)
    sp = float(sp) if not np.isnan(sp) else 0.0

    n_top = max(1, len(y_true) // 10)
    true_top = set(np.argsort(y_true)[:n_top])
    pred_top = set(np.argsort(pred_mean)[:n_top])
    top10_overlap = len(true_top & pred_top) / n_top

    z_scores = np.abs(residual) / pred_std
    expected = np.array([0.5, 0.8, 0.9, 0.95])
    actual = np.array([float(np.mean(z_scores <= norm.ppf(0.5 + p / 2))) for p in expected])
    mace = float(np.mean(np.abs(expected - actual)))

    y_range = float(np.max(y_true) - np.min(y_true))
    nrmse = rmse / max(y_range, 1e-8)

    return {"spearman": sp, "top10_overlap": top10_overlap, "rmse": rmse,
            "nrmse": nrmse, "r2": r2, "mace": mace, "y_range": y_range}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_variants", type=int, default=3)
    parser.add_argument("--n_test_points", type=int, default=200)
    parser.add_argument("--context_sizes", type=str, default="5,10,20,40")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save_dir", type=str, default="./results_policies/surrogate_quality_hplc")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    context_sizes = [int(x) for x in args.context_sizes.split(",")]

    print("=" * 70)
    print("Surrogate Quality: GP vs TabPFN-base")
    print("=" * 70)
    print(f"Device:          {device}")
    print(f"Variants/task:   {args.n_variants}")
    print(f"Test points:     {args.n_test_points}")
    print(f"Context sizes:   {context_sizes}")
    print("=" * 70)

    tabpfn = TabPFNRegressor(
        device=device, n_estimators=1,
        random_state=42,
        inference_precision=torch.float32,
        ignore_pretraining_limits=True,
    )

    # Cache TabPFN model (same as training code) — avoids ~7s rebuild per fit()
    from tabpfn.base import RegressorModelSpecs
    dummy_X = np.random.rand(3, 6).astype(np.float32)
    dummy_y = np.random.randn(3).astype(np.float32)
    tabpfn.fit(dummy_X, dummy_y)
    tabpfn.model_path = RegressorModelSpecs(
        model=tabpfn.model_,
        config=tabpfn.config_,
        norm_criterion=tabpfn.znorm_space_bardist_,
    )
    print(f"TabPFN model cached — subsequent fit() calls skip model rebuild")

    tasks_to_test = [
        ("hplc_emulator", 6),
        ("alkox_emulator", 4),
    ]

    os.makedirs(args.save_dir, exist_ok=True)
    all_task_results = {}

    for task_name, dim in tasks_to_test:
        print(f"\n{'='*70}")
        print(f"Task: {task_name} (dim={dim})")
        print(f"{'='*70}")

        task = get_task(task_name)
        rng = np.random.default_rng(args.seed)

        suite = task.sample_eval_suite(n_per_group=args.n_variants, seed=args.seed)
        variants = suite.get("in_range", [{}] * args.n_variants)
        print(f"  Sampled {len(variants)} in-range eval variants")

        agg = {"GP": {c: [] for c in context_sizes},
               "TabPFN-base": {c: [] for c in context_sizes}}

        for vi, vparams in enumerate(variants):
            t_variant_start = time.time()
            lower, upper = task.bounds
            sobol = Sobol(d=dim, scramble=True, seed=int(rng.integers(0, 100000)))
            X_test = (sobol.random(args.n_test_points) * (upper - lower) + lower).astype(np.float32)
            y_test = task.evaluate_numpy(X_test, vparams).astype(np.float64)
            y_range = float(np.max(y_test) - np.min(y_test))

            for ci, n_ctx in enumerate(context_sizes):
                t_ctx = time.time()
                X_ctx = rng.uniform(lower, upper, size=(n_ctx, dim)).astype(np.float32)
                y_ctx = task.evaluate_numpy(X_ctx, vparams).astype(np.float64)

                gp_sp, tab_sp = "FAIL", "FAIL"

                # GP
                try:
                    gp = make_gp(dim)
                    gp.fit(X_ctx, y_ctx)
                    gp_mean, gp_std = gp.predict(X_test, return_std=True)
                    gp_m = compute_metrics(y_test, gp_mean, gp_std)
                    agg["GP"][n_ctx].append(gp_m)
                    gp_sp = f"{gp_m['spearman']:.3f}"
                except Exception as e:
                    print(f"    [WARN] GP failed v={vi} ctx={n_ctx}: {e}")

                # TabPFN (with y-normalization, same as training code)
                try:
                    tab_mean, tab_std, _ = predict_tabpfn_with_normalization(
                        tabpfn, X_ctx, y_ctx, X_test)
                    tab_m = compute_metrics(y_test, tab_mean, tab_std)
                    agg["TabPFN-base"][n_ctx].append(tab_m)
                    tab_sp = f"{tab_m['spearman']:.3f}"
                except Exception as e:
                    print(f"    [WARN] TabPFN failed v={vi} ctx={n_ctx}: {e}")

                dt_ctx = time.time() - t_ctx
                print(f"    v{vi} ctx={n_ctx:3d}: GP_sp={gp_sp}  Tab_sp={tab_sp}  ({dt_ctx:.1f}s)")

            dt = time.time() - t_variant_start
            print(f"  Variant {vi} done: y_range={y_range:.1f}  total={dt:.1f}s")

        # Summary table
        print(f"\n  {'ctx':>5} | {'Surrogate':>12} | {'Spearman':>10} | {'Top10%':>8} | {'NRMSE':>8} | {'R2':>8} | {'MACE':>8}")
        print("  " + "-" * 72)

        task_results = {}
        for n_ctx in context_sizes:
            for sname in ["GP", "TabPFN-base"]:
                vals = agg[sname][n_ctx]
                if not vals:
                    continue
                sp = np.mean([v["spearman"] for v in vals])
                t10 = np.mean([v["top10_overlap"] for v in vals])
                nrmse = np.mean([v["nrmse"] for v in vals])
                r2 = np.mean([v["r2"] for v in vals])
                mace = np.mean([v["mace"] for v in vals])
                print(f"  {n_ctx:5d} | {sname:>12} | {sp:10.4f} | {t10:8.3f} | {nrmse:8.4f} | {r2:8.4f} | {mace:8.4f}")
                task_results.setdefault(n_ctx, {})[sname] = {
                    "spearman": sp, "top10_overlap": t10,
                    "nrmse": nrmse, "r2": r2, "mace": mace,
                }

        print(f"\n  Overall averages:")
        for sname in ["GP", "TabPFN-base"]:
            all_sp = [v["spearman"] for c in context_sizes for v in agg[sname][c]]
            all_t10 = [v["top10_overlap"] for c in context_sizes for v in agg[sname][c]]
            all_nrmse = [v["nrmse"] for c in context_sizes for v in agg[sname][c]]
            if all_sp:
                print(f"    {sname:>12}: Spearman={np.mean(all_sp):.4f}  Top10%={np.mean(all_t10):.3f}  NRMSE={np.mean(all_nrmse):.4f}")

        all_task_results[task_name] = task_results

    # Cross-task comparison
    print(f"\n{'='*70}")
    print("CROSS-TASK COMPARISON: TabPFN-base ranking accuracy")
    print(f"{'='*70}")
    print(f"  {'Task':>20} | {'ctx':>5} | {'GP Spearman':>12} | {'TabPFN Spearman':>16} | {'Gap':>8}")
    print("  " + "-" * 70)

    for task_name, _ in tasks_to_test:
        tr = all_task_results.get(task_name, {})
        for n_ctx in context_sizes:
            cr = tr.get(n_ctx, {})
            gp_sp = cr.get("GP", {}).get("spearman", float("nan"))
            tab_sp = cr.get("TabPFN-base", {}).get("spearman", float("nan"))
            gap = tab_sp - gp_sp
            print(f"  {task_name:>20} | {n_ctx:5d} | {gp_sp:12.4f} | {tab_sp:16.4f} | {gap:+8.4f}")

    # Save
    results_path = os.path.join(args.save_dir, "surrogate_quality_results.json")
    def to_json(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, dict): return {str(k): to_json(v) for k, v in obj.items()}
        return obj
    with open(results_path, "w") as f:
        json.dump(to_json(all_task_results), f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
