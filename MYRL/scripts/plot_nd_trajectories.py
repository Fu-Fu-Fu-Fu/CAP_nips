"""
plot_nd_trajectories.py — 高维任务 2D 切面等高线 + 优化轨迹可视化

功能:
  对 dim>2 的任务（如 4D alkox、6D hplc），在指定的 2D 切面上画等高线图，
  并叠加各策略的 BO 优化轨迹。

  对最大化任务（alkox、hplc），自动将值取反为正值显示，标签标为
  "best"/"global best" 等，颜色绿色=好/高值。

数据来源:
  根据 eval_data pickle 中的 variant_params + run_seeds 重跑 BO 获取轨迹。

用法:
  python MYRL/scripts/plot_nd_trajectories.py \\
    --task alkox_emulator \\
    --eval_data results_policies/alkox_emulator_bnn_kl003/gp/optimal/eval_data_gp_optimal.pkl \\
    --rl_model_path runs/ppo_alkox_emulator_bnn_kl003/ppo_final.pt \\
    --taf_data_path data/taf_source_data_alkox_emulator_k10_transform.pkl \\
    --groups in_range ood_level_2 \\
    --variant_indices 0 4 10 \\
    --methods CAP-PPO EI \\
    --dim_pairs 0,1 2,3 \\
    --surrogate gp \\
    --save_dir results_policies/alkox_emulator_bnn_kl003/nd_trajectories
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
import pickle

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

# ── path bootstrap ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MYRL_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(MYRL_DIR)
for p in [MYRL_DIR, SCRIPT_DIR, os.path.join(REPO_ROOT, "olympus", "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from myrl.tasks.registry import get_task
from myrl.eval.eval_rl_new import (
    run_bo_with_policy,
    TaskVariantObjectiveFunction,
)
from myrl.policies.policies import EI, TAF, RLPolicy


# =====================================================================
# 最大化任务（框架内取反为 minimize -y）
# 绘图时将值取反回正值，让图上"高=好"
# =====================================================================
NEGATED_TASKS = {"alkox_emulator", "hplc_emulator"}

# 各任务显示的目标名称
OBJECTIVE_NAMES = {
    "alkox_emulator": "conversion",
    "hplc_emulator":  "peak_area",
    "benzylation_emulator": "yield (minimize)",
}

DIM_NAMES_MAP = {
    "alkox_emulator": ["catalase", "peroxidase", "alcohol_oxidase", "pH"],
    "hplc_emulator":  ["sample_loop", "additional_vol", "tubing_vol",
                       "sample_flow", "push_speed", "wait_time"],
    "benzylation_emulator": ["equiv", "res_time", "temp", "conc"],
}


# =====================================================================
# Helpers
# =====================================================================

def _load_eval_data(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _get_dim_names(task_name: str, dim: int):
    if task_name in DIM_NAMES_MAP:
        return DIM_NAMES_MAP[task_name]
    return [f"x{i}" for i in range(dim)]


def _build_policies(args, dim: int, max_steps: int):
    """构建需要的 policy 对象"""
    device = args.device
    x_features = [f"x{i}" for i in range(dim)]
    feature_order = ["posterior_mean", "posterior_std"] + x_features + ["incumbent", "timestep", "budget"]

    policies = {}
    configs = {}

    for m in args.methods:
        if m == "CAP-PPO":
            rl = RLPolicy(
                model_path=args.rl_model_path,
                coord_dim=dim,
                max_steps=max_steps,
                device=device,
                n_persistent_base=args.n_persistent_base,
                n_total_candidates=args.n_total_candidates,
                k_centers=args.k_centers,
                local_h=args.local_h,
                local_h_decay=args.local_h_decay,
            )
            policies[m] = rl
            configs[m] = {
                "n_candidates": args.n_total_candidates,
                "n_global": 0,
                "k_centers": args.k_centers,
                "n_persistent_base": args.n_persistent_base,
                "n_total_candidates": args.n_total_candidates,
            }
        elif m == "EI":
            policies[m] = EI(feature_order)
            configs[m] = {
                "n_candidates": 2048, "n_global": 2048, "k_centers": 0,
                "n_persistent_base": 0, "n_total_candidates": 0,
            }
        elif m == "TAF_ranking":
            policies[m] = TAF(args.taf_data_path, mode="ranking", rho=1.0)
            configs[m] = {
                "n_candidates": 2048, "n_global": 2048, "k_centers": 0,
                "n_persistent_base": 0, "n_total_candidates": 0,
            }
        elif m == "TAF_me":
            policies[m] = TAF(args.taf_data_path, mode="me")
            configs[m] = {
                "n_candidates": 2048, "n_global": 2048, "k_centers": 0,
                "n_persistent_base": 0, "n_total_candidates": 0,
            }
        else:
            print(f"[WARN] Unknown method '{m}', skipping")

    # taf_for_rl for CAP-PPO
    taf_for_rl = None
    if "CAP-PPO" in policies:
        rl_policy = policies["CAP-PPO"]
        if rl_policy.use_taf_feature and args.taf_data_path:
            if "TAF_ranking" in policies:
                taf_for_rl = policies["TAF_ranking"]
            else:
                taf_for_rl = TAF(args.taf_data_path, mode="ranking", rho=1.0)

    return policies, configs, taf_for_rl


def _run_trajectory(func, global_min, policy, policy_name, config,
                    X_init, y_init, max_steps, surrogate, device,
                    rng, taf_for_rl, local_h, local_h_decay):
    """调用 run_bo_with_policy 获取轨迹"""
    regrets, traj = run_bo_with_policy(
        func=func,
        global_min=global_min,
        policy=policy,
        policy_name=policy_name,
        X_init=X_init,
        y_init=y_init,
        max_steps=max_steps,
        surrogate_type=surrogate,
        rng=rng,
        n_candidates=config["n_candidates"],
        n_global=config.get("n_global", 0),
        k_centers=config["k_centers"],
        local_h=local_h,
        local_h_decay=local_h_decay,
        device=device,
        return_trajectory=True,
        n_persistent_base=config.get("n_persistent_base", 0),
        n_total_candidates=config.get("n_total_candidates", 0),
        taf_for_rl=taf_for_rl if isinstance(policy, RLPolicy) else None,
    )
    return regrets, traj


def eval_2d_slice(task, variant_params, fix_dims, fix_vals, vary_dims, n_grid=100):
    """在 2D 切面上评估目标函数，返回原始框架值（可能是负的）"""
    dim = task.dim
    g = np.linspace(0, 1, n_grid)
    G0, G1 = np.meshgrid(g, g)
    X = np.full((n_grid * n_grid, dim), 0.5, dtype=np.float32)
    for d, v in zip(fix_dims, fix_vals):
        X[:, d] = v
    X[:, vary_dims[0]] = G0.ravel()
    X[:, vary_dims[1]] = G1.ravel()
    y = task.evaluate_numpy(X, variant_params).reshape(n_grid, n_grid)
    return G0, G1, y


# =====================================================================
# Plotting
# =====================================================================

# method → (color, marker, colormap for progressive steps)
METHOD_STYLES = {
    "CAP-PPO":     ("dodgerblue", "D",  plt.cm.Blues),
    "EI":          ("orange",     "s",  plt.cm.Oranges),
    "TAF_ranking": ("limegreen",  "^",  plt.cm.Greens),
    "TAF_me":      ("violet",     "v",  plt.cm.Purples),
    "UCB":         ("red",        "p",  plt.cm.Reds),
    "PI":          ("brown",      "h",  plt.cm.copper),
    "Random":      ("gray",       "+",  plt.cm.Greys),
}


def plot_2d_contour_with_trajectories(
    ax, G0, G1, Z_raw, vary_dims, dim_names,
    trajectories: dict,
    n_init: int,
    is_negated: bool = False,
    obj_name: str = "f(x)",
    title: str = "",
    show_colorbar: bool = True,
):
    """
    在 ax 上画 2D 等高线 + 多策略轨迹。

    如果 is_negated=True（最大化任务），Z 值取反为正值显示，
    colormap 用 RdYlGn（绿=高=好），标签用 "Best" 而非 "min"。
    """
    d0, d1 = vary_dims

    if is_negated:
        Z = -Z_raw                       # 取反为正值
        cmap_name = "RdYlGn"             # 绿=高=好
        best_idx = np.unravel_index(Z.argmax(), Z.shape)
        best_label = "Slice best"
    else:
        Z = Z_raw
        cmap_name = "RdYlGn_r"           # 绿=低=好（minimization）
        best_idx = np.unravel_index(Z.argmin(), Z.shape)
        best_label = "Slice best"

    vmin, vmax = float(Z.min()), float(Z.max())
    levels = np.linspace(vmin, vmax, 30)

    cf = ax.contourf(G0, G1, Z, levels=levels, cmap=cmap_name, extend="both")
    ax.contour(G0, G1, Z, levels=levels, colors="k", linewidths=0.25, alpha=0.25)

    # 标记切面最优点
    ax.plot(G0[best_idx], G1[best_idx], "w*", markersize=14,
            markeredgecolor="k", markeredgewidth=1, zorder=20, label=best_label)

    # 画各策略的轨迹
    for method, traj in trajectories.items():
        color, marker, cmap = METHOD_STYLES.get(method, ("white", "o", plt.cm.Greys))

        # 初始点（所有策略共享同一初始随机种子）
        ax.scatter(traj[:n_init, d0], traj[:n_init, d1],
                   c="cyan", s=55, edgecolors="k", linewidths=0.8,
                   zorder=15, marker="o")

        # BO 步骤: 颜色递进（浅→深表示时间先→后）
        n_pts = len(traj)
        n_steps = n_pts - n_init
        if n_steps > 0:
            step_colors = cmap(np.linspace(0.3, 1.0, n_steps))
            for i in range(n_init, n_pts):
                ax.scatter(traj[i, d0], traj[i, d1],
                           c=[step_colors[i - n_init]], s=22,
                           edgecolors="k", linewidths=0.4, zorder=12, marker=marker)
                if i > n_init:
                    ax.plot([traj[i-1, d0], traj[i, d0]],
                            [traj[i-1, d1], traj[i, d1]],
                            color=color, linewidth=0.6, alpha=0.5, zorder=11)

        # 最终最优点（用星标）
        ax.scatter(traj[-1, d0], traj[-1, d1],
                   marker="*", c=color, s=100, edgecolors="white",
                   linewidths=1, zorder=16, label=f"{method} final")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(dim_names[d0], fontsize=9)
    ax.set_ylabel(dim_names[d1], fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    if show_colorbar:
        cb = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(obj_name, fontsize=8)
    return cf


def plot_3d_surface_with_trajectories(
    ax, G0, G1, Z_raw, vary_dims, dim_names,
    trajectories: dict,
    task, variant_params,
    n_init: int, fix_dims, fix_vals,
    is_negated: bool = False,
    obj_name: str = "f(x)",
    title: str = "",
):
    """在 3D ax 上画表面 + 轨迹"""
    d0, d1 = vary_dims

    if is_negated:
        Z = -Z_raw
        cmap_name = "RdYlGn"
    else:
        Z = Z_raw
        cmap_name = "RdYlGn_r"

    ax.plot_surface(G0, G1, Z, cmap=cmap_name, alpha=0.6,
                    edgecolor="none", rasterized=True)

    for method, traj in trajectories.items():
        color, marker, _ = METHOD_STYLES.get(method, ("white", "o", plt.cm.Greys))
        # 计算轨迹点在切面上投影后的函数值
        dim = task.dim
        X_eval = np.full((len(traj), dim), 0.5, dtype=np.float32)
        for fd, fv in zip(fix_dims, fix_vals):
            X_eval[:, fd] = fv
        X_eval[:, d0] = traj[:, d0]
        X_eval[:, d1] = traj[:, d1]
        z_traj = task.evaluate_numpy(X_eval, variant_params).reshape(-1)
        if is_negated:
            z_traj = -z_traj

        # 初始点
        ax.scatter(traj[:n_init, d0], traj[:n_init, d1], z_traj[:n_init],
                   c="cyan", s=60, edgecolors="k", linewidths=0.8,
                   zorder=15, marker="o", depthshade=False)
        # BO 点
        ax.scatter(traj[n_init:, d0], traj[n_init:, d1], z_traj[n_init:],
                   c=color, s=25, edgecolors="k", linewidths=0.3,
                   zorder=12, marker=marker, depthshade=False, label=method)
        # 连线
        ax.plot(traj[:, d0], traj[:, d1], z_traj,
                color=color, linewidth=0.8, alpha=0.6, zorder=11)

    ax.set_xlabel(dim_names[d0], fontsize=8)
    ax.set_ylabel(dim_names[d1], fontsize=8)
    ax.set_zlabel(obj_name, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="upper left")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="高维任务 2D 切面 + 轨迹可视化")
    parser.add_argument("--task", required=True, help="任务名，如 alkox_emulator")
    parser.add_argument("--eval_data", required=True, help="eval_data pickle 路径")
    parser.add_argument("--rl_model_path", default=None, help="CAP-PPO 模型路径")
    parser.add_argument("--taf_data_path", default=None, help="TAF 数据路径")
    parser.add_argument("--groups", nargs="+", default=["in_range"],
                        help="要可视化的 variant group")
    parser.add_argument("--variant_indices", nargs="+", type=int, default=[0],
                        help="每个 group 中要可视化的 variant 索引（0-based）")
    parser.add_argument("--methods", nargs="+", default=["CAP-PPO", "EI"],
                        help="要画轨迹的策略")
    parser.add_argument("--dim_pairs", nargs="+", default=None,
                        help="2D 切面的维度对，如 '0,1' '2,3'。默认自动选择")
    parser.add_argument("--fix_strategy", choices=["midpoint", "best"], default="best",
                        help="固定维度的值取法: midpoint=0.5, best=轨迹最优点")
    parser.add_argument("--surrogate", default="gp", help="代理模型类型")
    parser.add_argument("--max_steps", type=int, default=28)
    parser.add_argument("--n_init", type=int, default=2)
    parser.add_argument("--n_persistent_base", type=int, default=128)
    parser.add_argument("--n_total_candidates", type=int, default=192)
    parser.add_argument("--k_centers", type=int, default=2)
    parser.add_argument("--local_h", type=float, default=0.17)
    parser.add_argument("--local_h_decay", type=float, default=0.95)
    parser.add_argument("--n_grid", type=int, default=100, help="等高线网格分辨率")
    parser.add_argument("--run_index", type=int, default=0,
                        help="使用 eval 中第几次 run 的种子（默认 0）")
    parser.add_argument("--plot_3d", action="store_true", help="额外生成 3D 表面图")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", default=None, help="输出目录（默认 eval_data 同级 nd_trajectories/）")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    # ── 加载数据 ──
    print(f"Loading eval data from {args.eval_data} ...", flush=True)
    eval_data = _load_eval_data(args.eval_data)
    all_group_results = eval_data["all_group_results"]

    task = get_task(args.task)
    dim = int(task.dim)
    dim_names = _get_dim_names(args.task, dim)

    is_negated = args.task in NEGATED_TASKS
    obj_name = OBJECTIVE_NAMES.get(args.task, "f(x)")
    opt_word = "max" if is_negated else "min"

    print(f"Task: {args.task}, dim={dim}, dims={dim_names}")
    print(f"Negated (maximize): {is_negated}, objective: {obj_name}")

    # 确定切面维度对
    if args.dim_pairs is None:
        pairs = [(i, i+1) for i in range(0, dim, 2)]
        if dim % 2 == 1:
            pairs.append((dim-2, dim-1))
    else:
        pairs = []
        for p in args.dim_pairs:
            a, b = p.split(",")
            pairs.append((int(a), int(b)))
    print(f"Dimension pairs: {pairs}")

    # ── 构建 policies ──
    print("Building policies...", flush=True)
    policies, configs, taf_for_rl = _build_policies(args, dim, args.max_steps)
    print(f"Policies: {list(policies.keys())}")

    if args.save_dir is None:
        args.save_dir = os.path.join(os.path.dirname(args.eval_data), "nd_trajectories")
    os.makedirs(args.save_dir, exist_ok=True)

    # ── 逐 group 逐 variant 处理 ──
    for group_name in args.groups:
        if group_name not in all_group_results:
            print(f"[WARN] Group '{group_name}' not in eval data, skipping")
            continue

        group_data = all_group_results[group_name]
        meta = group_data["__meta__"]
        variants = meta["variants"]
        global_mins = meta["global_mins"]
        run_seeds = meta["run_seeds_by_variant"]

        for v_idx in args.variant_indices:
            if v_idx >= len(variants):
                print(f"[WARN] Variant index {v_idx} out of range (max {len(variants)-1}), skipping")
                continue

            variant_params = variants[v_idx]
            global_min = global_mins[v_idx]          # 框架内的 min 值（负的）
            global_best_display = -global_min if is_negated else global_min
            run_seed = run_seeds[v_idx][args.run_index]

            print(f"\n{'='*60}")
            print(f"Group={group_name}, Variant={v_idx}, "
                  f"global_{opt_word}={global_best_display:.2f} ({obj_name}), seed={run_seed}")
            print(f"{'='*60}")

            # 构建目标函数
            func = TaskVariantObjectiveFunction(task_name=args.task, variant_params=variant_params)
            lower, upper = func.bounds

            # 生成初始点（与 eval 相同的种子）
            init_rng = np.random.default_rng(run_seed)
            X_init = init_rng.uniform(lower, upper, size=(args.n_init, dim)).astype(np.float32)
            y_init = func(X_init)

            # 收集各策略轨迹 + 每个策略找到的最优值
            trajectories = {}
            method_best_display = {}
            for method_name, policy in policies.items():
                print(f"  Running {method_name}...", end="", flush=True)
                run_rng = np.random.default_rng(run_seed)
                config = configs[method_name]
                regrets, traj = _run_trajectory(
                    func=func, global_min=global_min,
                    policy=policy, policy_name=method_name,
                    config=config, X_init=X_init, y_init=y_init,
                    max_steps=args.max_steps, surrogate=args.surrogate,
                    device=args.device, rng=run_rng,
                    taf_for_rl=taf_for_rl,
                    local_h=args.local_h, local_h_decay=args.local_h_decay,
                )
                trajectories[method_name] = traj
                # 计算该策略找到的 best 值（原始尺度）
                y_all = func(traj).reshape(-1)
                best_found = float(y_all.min())
                best_display = -best_found if is_negated else best_found
                method_best_display[method_name] = best_display
                print(f" best_{obj_name}={best_display:.2f}, regret={regrets[-1]:.2f}, n_points={len(traj)}")

            # ── 确定固定维度值 ──
            ref_method = args.methods[0]
            ref_traj = trajectories[ref_method]
            ref_y = func(ref_traj)
            best_idx = int(np.argmin(ref_y))
            best_x = ref_traj[best_idx]

            # ── 构建 suptitle ──
            method_info = "  ".join(
                f"{m}={method_best_display[m]:.1f}" for m in trajectories
            )
            suptitle_single = (
                f"{args.task} — {group_name} Variant {v_idx}\n"
                f"global_{opt_word}={global_best_display:.1f} ({obj_name})  |  {method_info}"
            )
            suptitle_overlay = (
                f"{args.task} — {group_name} Variant {v_idx}  (all methods)\n"
                f"global_{opt_word}={global_best_display:.1f} ({obj_name})"
            )

            # ── 2D 等高线图: 每个策略独占一行 ──
            n_pairs = len(pairs)
            n_methods_label = len(trajectories)
            fig, axes = plt.subplots(n_methods_label, n_pairs,
                                     figsize=(6 * n_pairs, 5.5 * n_methods_label),
                                     squeeze=False)
            fig.suptitle(suptitle_single, fontsize=11, fontweight="bold", y=0.99)

            for col, (d0, d1) in enumerate(pairs):
                vary_dims = [d0, d1]
                fix_dims = [d for d in range(dim) if d not in vary_dims]
                if args.fix_strategy == "best":
                    fix_vals = [float(best_x[d]) for d in fix_dims]
                else:
                    fix_vals = [0.5] * len(fix_dims)
                fix_desc = ", ".join(f"{dim_names[d]}={v:.2f}" for d, v in zip(fix_dims, fix_vals))

                G0, G1, Z = eval_2d_slice(task, variant_params, fix_dims, fix_vals, vary_dims, args.n_grid)

                for row, method in enumerate(trajectories):
                    single_traj = {method: trajectories[method]}
                    title = (f"{method} (best {obj_name}={method_best_display[method]:.1f})\n"
                             f"{dim_names[d0]} vs {dim_names[d1]} | fix: {fix_desc}")
                    plot_2d_contour_with_trajectories(
                        axes[row, col], G0, G1, Z, vary_dims, dim_names,
                        single_traj, n_init=args.n_init,
                        is_negated=is_negated, obj_name=obj_name,
                        title=title,
                    )
                    if col == n_pairs - 1:
                        axes[row, col].legend(fontsize=6, loc="upper right")

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            fname = f"contour_{group_name}_v{v_idx:02d}.png"
            out_path = os.path.join(args.save_dir, fname)
            plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {out_path}")

            # ── 所有策略叠加在同一张图 ──
            fig2, axes2 = plt.subplots(1, n_pairs, figsize=(6.5 * n_pairs, 5.5), squeeze=False)
            fig2.suptitle(suptitle_overlay, fontsize=11, fontweight="bold", y=0.99)

            for col, (d0, d1) in enumerate(pairs):
                vary_dims = [d0, d1]
                fix_dims = [d for d in range(dim) if d not in vary_dims]
                if args.fix_strategy == "best":
                    fix_vals = [float(best_x[d]) for d in fix_dims]
                else:
                    fix_vals = [0.5] * len(fix_dims)
                fix_desc = ", ".join(f"{dim_names[d]}={v:.2f}" for d, v in zip(fix_dims, fix_vals))

                G0, G1, Z = eval_2d_slice(task, variant_params, fix_dims, fix_vals, vary_dims, args.n_grid)
                title = f"{dim_names[d0]} vs {dim_names[d1]}\nfix: {fix_desc}"
                plot_2d_contour_with_trajectories(
                    axes2[0, col], G0, G1, Z, vary_dims, dim_names,
                    trajectories, n_init=args.n_init,
                    is_negated=is_negated, obj_name=obj_name,
                    title=title,
                )
                axes2[0, col].legend(fontsize=7, loc="upper right")

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            fname2 = f"contour_overlay_{group_name}_v{v_idx:02d}.png"
            out_path2 = os.path.join(args.save_dir, fname2)
            plt.savefig(out_path2, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig2)
            print(f"  Saved: {out_path2}")

            # ── 3D 图（可选）──
            if args.plot_3d:
                for col, (d0, d1) in enumerate(pairs):
                    vary_dims = [d0, d1]
                    fix_dims = [d for d in range(dim) if d not in vary_dims]
                    if args.fix_strategy == "best":
                        fix_vals = [float(best_x[d]) for d in fix_dims]
                    else:
                        fix_vals = [0.5] * len(fix_dims)

                    G0, G1, Z = eval_2d_slice(task, variant_params, fix_dims, fix_vals, vary_dims, args.n_grid)

                    fig3 = plt.figure(figsize=(10, 7))
                    ax3 = fig3.add_subplot(111, projection="3d")
                    fix_desc = ", ".join(f"{dim_names[d]}={v:.2f}" for d, v in zip(fix_dims, fix_vals))
                    plot_3d_surface_with_trajectories(
                        ax3, G0, G1, Z, vary_dims, dim_names,
                        trajectories, task, variant_params,
                        n_init=args.n_init, fix_dims=fix_dims, fix_vals=fix_vals,
                        is_negated=is_negated, obj_name=obj_name,
                        title=(f"{group_name} V{v_idx} | {dim_names[d0]} vs {dim_names[d1]}\n"
                               f"fix: {fix_desc}"),
                    )
                    fname3 = f"surface3d_{group_name}_v{v_idx:02d}_d{d0}d{d1}.png"
                    out_path3 = os.path.join(args.save_dir, fname3)
                    plt.savefig(out_path3, dpi=args.dpi, bbox_inches="tight")
                    plt.close(fig3)
                    print(f"  Saved 3D: {out_path3}")

    # ── 跨 group 对比图（若有多个 group + 固定 variant_indices）──
    if len(args.groups) > 1 and len(args.variant_indices) == 1:
        v_idx = args.variant_indices[0]
        n_groups = len(args.groups)
        n_pairs = len(pairs)

        for method in args.methods:
            if method not in policies:
                continue
            fig_cmp, axes_cmp = plt.subplots(n_groups, n_pairs,
                                              figsize=(6 * n_pairs, 5.5 * n_groups),
                                              squeeze=False)
            fig_cmp.suptitle(f"{method} across groups — Variant {v_idx}", fontsize=13, y=0.99)

            for row, group_name in enumerate(args.groups):
                if group_name not in all_group_results:
                    continue
                meta = all_group_results[group_name]["__meta__"]
                if v_idx >= len(meta["variants"]):
                    continue

                variant_params = meta["variants"][v_idx]
                global_min = meta["global_mins"][v_idx]
                global_best_display = -global_min if is_negated else global_min

                run_seed = meta["run_seeds_by_variant"][v_idx][args.run_index]
                func = TaskVariantObjectiveFunction(task_name=args.task, variant_params=variant_params)
                lower, upper = func.bounds
                init_rng = np.random.default_rng(run_seed)
                X_init = init_rng.uniform(lower, upper, size=(args.n_init, dim)).astype(np.float32)
                y_init = func(X_init)

                run_rng = np.random.default_rng(run_seed)
                config = configs[method]
                _, traj = _run_trajectory(
                    func=func, global_min=global_min,
                    policy=policies[method], policy_name=method,
                    config=config, X_init=X_init, y_init=y_init,
                    max_steps=args.max_steps, surrogate=args.surrogate,
                    device=args.device, rng=run_rng,
                    taf_for_rl=taf_for_rl,
                    local_h=args.local_h, local_h_decay=args.local_h_decay,
                )

                ref_y = func(traj)
                best_idx = int(np.argmin(ref_y))
                best_x = traj[best_idx]
                best_display = float(-ref_y.min()) if is_negated else float(ref_y.min())

                for col, (d0, d1) in enumerate(pairs):
                    vary_dims = [d0, d1]
                    fix_dims = [d for d in range(dim) if d not in vary_dims]
                    fix_vals = [float(best_x[d]) for d in fix_dims]
                    fix_desc = ", ".join(f"{dim_names[d]}={v:.2f}" for d, v in zip(fix_dims, fix_vals))

                    G0, G1, Z = eval_2d_slice(task, variant_params, fix_dims, fix_vals, vary_dims, args.n_grid)
                    title = (f"{group_name} V{v_idx} | {opt_word}={global_best_display:.1f} "
                             f"found={best_display:.1f}\n"
                             f"{dim_names[d0]} vs {dim_names[d1]} | fix: {fix_desc}")
                    plot_2d_contour_with_trajectories(
                        axes_cmp[row, col], G0, G1, Z, vary_dims, dim_names,
                        {method: traj}, n_init=args.n_init,
                        is_negated=is_negated, obj_name=obj_name,
                        title=title,
                    )

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            fname_cmp = f"cross_group_{method}_v{v_idx:02d}.png"
            out_cmp = os.path.join(args.save_dir, fname_cmp)
            plt.savefig(out_cmp, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig_cmp)
            print(f"\nSaved cross-group comparison: {out_cmp}")

    print("\nDone!")


if __name__ == "__main__":
    main()
