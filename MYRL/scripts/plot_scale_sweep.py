"""
plot_scale_sweep.py — Re-plot scale sweep results from saved pickle data.

Loads the pickle saved by eval_scale_sweep.py and regenerates all plots
without re-running evaluation.  Supports customizable options.

Usage:
    python MYRL/scripts/plot_scale_sweep.py \
        --data ./results_policies/alkox_emulator_scale_sweep/scale_sweep_data.pkl \
        [--save_dir ./results_policies/alkox_emulator_scale_sweep/replot] \
        [--yscale linear] \
        [--shared_y] \
        [--no_random] \
        [--formats png pdf]
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.normpath(os.path.join(_this_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

CAP_PPO_NAME = "CAP-PPO"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BASE_DX = 0.08
_BASE_ROT = 28.0
_BASE_SX_DELTA = 0.16

COLORS = {
    CAP_PPO_NAME: "#E24A33",
    "EI": "#348ABD",
    "UCB": "#467821",
    "PI": "#988ED5",
    "TAF_ranking": "#FBC15E",
    "TAF_me": "#8C6D31",
    "PFNs4BO": "#00CED1",
    "TuRBO": "#E755BA",
    "FunBO": "#777777",
    "Random": "#000000",
}
MARKERS = {
    CAP_PPO_NAME: "o",
    "Random": "s",
    "EI": "^", "UCB": "v", "PI": "<",
    "TAF_me": "D", "TAF_ranking": "d",
    "PFNs4BO": "P", "TuRBO": "X", "FunBO": "h",
}
LINESTYLES = {
    CAP_PPO_NAME: "-",
    "EI": "--", "UCB": "--", "PI": "--",
    "TAF_ranking": "-.", "TAF_me": "-.",
    "PFNs4BO": ":", "TuRBO": ":", "FunBO": ":",
    "Random": "--",
}
TASK_DISPLAY_NAMES = {
    "branin_family": "Branin 2D",
    "hartmann_3d_family": "Hartmann 3D",
    "hartmann_6d_family": "Hartmann 6D",
    "alkox_emulator": "Alkox 4D",
    "hplc_emulator": "HPLC 6D",
    "benzylation_emulator": "Benzylation 4D",
}


def make_spec_for_scale(scale: float) -> Dict:
    dx_max = _BASE_DX * scale
    rot_max = _BASE_ROT * scale
    sx_min = max(0.60, 1.0 - _BASE_SX_DELTA * scale)
    return {"dx": [(-dx_max, dx_max)], "rot": [(-rot_max, rot_max)], "sx": [(sx_min, 1.0)]}


def style_axes(ax):
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.10, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.35)
    ax.spines["bottom"].set_alpha(0.35)
    ax.tick_params(labelsize=11)


def paper_task_title(task_name: str, surrogate: str) -> str:
    display = TASK_DISPLAY_NAMES.get(task_name, task_name)
    surr = "GP" if surrogate == "gp" else "TabPFN"
    return f"{display} ({surr} surrogate)"


# ---------------------------------------------------------------------------
# Plot 1: Regret vs. Scale (sweep summary)
# ---------------------------------------------------------------------------
def _compute_normalized_stats(data: Dict, scales, method_names):
    """Compute per-scale normalized regret stats (mean and SEM).

    For each variant, divides final regret by step-0 regret (initial gap).
    Returns dict: {scale_str: {method: {"mean": float, "sem": float}}}.
    """
    all_trajectories = data["trajectories"]
    stats = {}
    for s in scales:
        sk = str(s)
        # Step-0 regret per variant: use Random (same init for all methods)
        # Shape: (n_variants, n_runs, n_steps+1)
        ref_trajs = all_trajectories[sk]["Random"]
        step0_per_variant = ref_trajs[:, :, 0].mean(axis=1)  # (n_variants,)
        # Avoid division by zero (shouldn't happen)
        step0_per_variant = np.maximum(step0_per_variant, 1e-8)

        stats[sk] = {}
        for m in method_names:
            trajs = all_trajectories[sk][m]
            # Final regret per variant: mean over runs, take last step
            final_per_variant = trajs[:, :, -1].mean(axis=1)  # (n_variants,)
            normalized = final_per_variant / step0_per_variant
            stats[sk][m] = {
                "mean": float(normalized.mean()),
                "sem": float(normalized.std() / np.sqrt(len(normalized))),
            }
    return stats


def plot_sweep(
    data: Dict,
    save_dir: str,
    *,
    exclude_methods=(),
    formats=("png",),
    show_transform_params: bool = False,
    normalize: bool = True,
):
    scales = data["scales"]
    method_names = [m for m in data["method_names"] if m not in exclude_methods]

    if normalize and "trajectories" in data:
        norm_stats = _compute_normalized_stats(data, scales, method_names)

    fig, ax = plt.subplots(1, 1, figsize=(10.8, 5.8))

    for m in method_names:
        if normalize and "trajectories" in data:
            means = np.array([norm_stats[str(s)][m]["mean"] for s in scales])
            sems = np.array([norm_stats[str(s)][m]["sem"] for s in scales])
        else:
            all_results = data["results"]
            means = np.array([all_results[str(s)][m]["mean"] for s in scales])
            stds = np.array([all_results[str(s)][m]["std"] for s in scales])
            n = data["n_variants"]
            sems = stds / np.sqrt(n)

        color = COLORS.get(m, "tab:gray")
        marker = MARKERS.get(m, "o")
        ls = LINESTYLES.get(m, "-")
        lw = 2.5 if m == CAP_PPO_NAME else 1.5
        zorder = 10 if m == CAP_PPO_NAME else 5
        alpha_fill = 0.06 if m == "Random" else 0.12
        alpha_line = 0.5 if m == "Random" else 1.0

        ax.plot(scales, means, marker=marker, label=m, color=color,
                linewidth=lw, linestyle=ls, markersize=4.8, zorder=zorder,
                alpha=alpha_line, markeredgewidth=0.5, markeredgecolor="white")
        ax.fill_between(scales, means - sems, means + sems, alpha=alpha_fill, color=color)

    ax.set_xlabel("Variant Scale", fontsize=12)
    ylabel = "Normalized Regret (mean ± SEM)" if normalize else "Simple Regret (mean ± SEM)"
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(paper_task_title(data["task"], data["surrogate"]), fontsize=13, pad=10)
    ax.legend(fontsize=9, loc="upper left", ncol=2,
              frameon=True, framealpha=0.92, facecolor="white",
              edgecolor="#d7d7d7", handlelength=2.0, columnspacing=1.0)
    style_axes(ax)

    if show_transform_params:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(scales)
        tick_labels = []
        for s in scales:
            spec = make_spec_for_scale(s)
            dx_hi = spec["dx"][0][1]
            rot_hi = spec["rot"][0][1]
            sx_lo = spec["sx"][0][0]
            tick_labels.append(f"dx±{dx_hi:.2f}\nrot±{rot_hi:.0f}°\nsx≥{sx_lo:.2f}")
        ax2.set_xticklabels(tick_labels, fontsize=7)
        ax2.set_xlabel("Variant transformation parameters", fontsize=10)

    plt.tight_layout()
    for fmt in formats:
        fig.savefig(os.path.join(save_dir, f"scale_sweep_regret.{fmt}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Sweep plot saved")


# ---------------------------------------------------------------------------
# Plot 2: CAP-PPO vs key baselines with crossover
# ---------------------------------------------------------------------------
def plot_highlight(data: Dict, save_dir: str, *, formats=("png",), normalize: bool = True):
    scales = data["scales"]
    method_names = data["method_names"]
    highlight = [CAP_PPO_NAME, "EI", "UCB", "TAF_ranking"]

    if normalize and "trajectories" in data:
        norm_stats = _compute_normalized_stats(data, scales, method_names)

    fig, ax = plt.subplots(1, 1, figsize=(10.2, 5.6))

    for m in highlight:
        if m not in method_names:
            continue
        if normalize and "trajectories" in data:
            means = np.array([norm_stats[str(s)][m]["mean"] for s in scales])
            sems = np.array([norm_stats[str(s)][m]["sem"] for s in scales])
        else:
            all_results = data["results"]
            means = np.array([all_results[str(s)][m]["mean"] for s in scales])
            stds = np.array([all_results[str(s)][m]["std"] for s in scales])
            sems = stds / np.sqrt(data["n_variants"])

        color = COLORS.get(m, "tab:gray")
        marker = MARKERS.get(m, "o")
        ls = LINESTYLES.get(m, "-")
        lw = 2.5 if m == CAP_PPO_NAME else 1.5

        ax.plot(scales, means, marker=marker, label=m, color=color,
                linewidth=lw, linestyle=ls, markersize=5.2,
                markeredgewidth=0.6, markeredgecolor="white")
        ax.fill_between(scales, means - sems, means + sems, alpha=0.15, color=color)

    ax.set_xlabel("Variant Scale", fontsize=12)
    ylabel = "Normalized Regret (mean ± SEM)" if normalize else "Simple Regret (mean ± SEM)"
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(
        f"CAP-PPO vs Key Baselines — {paper_task_title(data['task'], data['surrogate'])}",
        fontsize=13,
        pad=10,
    )
    ax.legend(fontsize=10, frameon=True, framealpha=0.92,
              facecolor="white", edgecolor="#d7d7d7")
    style_axes(ax)

    # Crossover annotations
    if normalize and "trajectories" in data:
        cap_means = np.array([norm_stats[str(s)][CAP_PPO_NAME]["mean"] for s in scales])
        for baseline in ["EI", "UCB", "TAF_ranking"]:
            if baseline not in method_names:
                continue
            bl_means = np.array([norm_stats[str(s)][baseline]["mean"] for s in scales])
            diff = cap_means - bl_means
            for i in range(len(diff) - 1):
                if diff[i] <= 0 < diff[i + 1]:
                    frac = -diff[i] / (diff[i + 1] - diff[i])
                    cross_scale = scales[i] + frac * (scales[i + 1] - scales[i])
                    cross_regret = cap_means[i] + frac * (cap_means[i + 1] - cap_means[i])
                    ax.axvline(x=cross_scale, color=COLORS.get(baseline, "gray"),
                              linestyle="--", alpha=0.5, linewidth=1)
                    ax.annotate(
                        f"x{baseline}\n@{cross_scale:.2f}",
                        xy=(cross_scale, cross_regret),
                        xytext=(cross_scale + 0.05, cross_regret + 0.05),
                        fontsize=8, color=COLORS.get(baseline, "gray"),
                        arrowprops=dict(arrowstyle="->", color=COLORS.get(baseline, "gray"), alpha=0.7),
                    )
                    break
    else:
        all_results = data["results"]
        cap_means = np.array([all_results[str(s)][CAP_PPO_NAME]["mean"] for s in scales])
        for baseline in ["EI", "UCB", "TAF_ranking"]:
            if baseline not in method_names:
                continue
            bl_means = np.array([all_results[str(s)][baseline]["mean"] for s in scales])
            diff = cap_means - bl_means
            for i in range(len(diff) - 1):
                if diff[i] <= 0 < diff[i + 1]:
                    frac = -diff[i] / (diff[i + 1] - diff[i])
                    cross_scale = scales[i] + frac * (scales[i + 1] - scales[i])
                    cross_regret = cap_means[i] + frac * (cap_means[i + 1] - cap_means[i])
                    ax.axvline(x=cross_scale, color=COLORS.get(baseline, "gray"),
                              linestyle="--", alpha=0.5, linewidth=1)
                    ax.annotate(
                        f"x{baseline}\n@{cross_scale:.2f}",
                        xy=(cross_scale, cross_regret),
                        xytext=(cross_scale + 0.05, cross_regret + 2),
                        fontsize=8, color=COLORS.get(baseline, "gray"),
                        arrowprops=dict(arrowstyle="->", color=COLORS.get(baseline, "gray"), alpha=0.7),
                    )
                    break

    plt.tight_layout()
    for fmt in formats:
        fig.savefig(os.path.join(save_dir, f"scale_sweep_highlight.{fmt}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Highlight plot saved")


# ---------------------------------------------------------------------------
# Plot 3: Per-scale optimization trajectory grid
# ---------------------------------------------------------------------------
def plot_trajectory_grid(
    data: Dict,
    save_dir: str,
    *,
    shared_y: bool = True,
    yscale: str = "linear",
    exclude_methods: tuple = (),
    formats: tuple = ("png",),
    normalize: bool = True,
):
    scales = data["scales"]
    all_trajectories = data["trajectories"]
    method_names = [m for m in data["method_names"] if m not in exclude_methods]
    n_init = data["n_init"]
    max_steps = data["max_steps"]

    n_scales = len(scales)
    n_cols = min(4, n_scales)
    n_rows = (n_scales + n_cols - 1) // n_cols

    # Pre-compute normalized trajectories if needed
    norm_trajectories = {}
    if normalize:
        for scale in scales:
            sk = str(scale)
            ref_trajs = all_trajectories[sk]["Random"]
            # step-0 regret per variant, mean over runs: (n_variants,)
            step0 = np.maximum(ref_trajs[:, :, 0].mean(axis=1), 1e-8)
            norm_trajectories[sk] = {}
            for m in method_names:
                trajs = all_trajectories[sk][m]  # (n_variants, n_runs, n_steps+1)
                # Normalize: divide each variant by its step-0 regret
                norm_trajectories[sk][m] = trajs / step0[:, None, None]

    src = norm_trajectories if normalize else all_trajectories

    # Compute global y-max (excluding Random) for shared y-axis
    global_y_max = 0
    for scale in scales:
        sk = str(scale)
        for m in method_names:
            trajs = src[sk][m]
            unit_curves = trajs.mean(axis=1)
            q70 = np.quantile(unit_curves, 0.70, axis=0)
            candidate = float(q70.max())
            if m == "Random":
                candidate = min(candidate, float(np.median(unit_curves, axis=0)[0]))
            global_y_max = max(global_y_max, candidate)
    global_y_max *= 1.1

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False, sharey=shared_y,
    )

    n_total = n_init + max_steps
    x = np.arange(n_total + 1)

    for idx, scale in enumerate(scales):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row][col]
        sk = str(scale)
        spec = make_spec_for_scale(scale)

        for m in method_names:
            trajs = src[sk][m]
            unit_curves = trajs.mean(axis=1)
            median = np.median(unit_curves, axis=0)
            q30 = np.quantile(unit_curves, 0.30, axis=0)
            q70 = np.quantile(unit_curves, 0.70, axis=0)

            color = COLORS.get(m, "tab:gray")
            lw = 2.0 if m == CAP_PPO_NAME else 1.2
            zorder = 10 if m == CAP_PPO_NAME else 5
            alpha_line = 1.0 if m == CAP_PPO_NAME else 0.8

            ax.plot(x[:len(median)], median, label=m, color=color,
                    linewidth=lw, zorder=zorder, alpha=alpha_line)
            ax.fill_between(x[:len(median)], q30, q70, alpha=0.12, color=color)

        dx_hi = spec["dx"][0][1]
        rot_hi = spec["rot"][0][1]
        sx_lo = spec["sx"][0][0]
        ax.set_title(f"scale={scale:.2f}\ndx±{dx_hi:.2f}  rot±{rot_hi:.0f}°  sx≥{sx_lo:.2f}",
                      fontsize=9)
        ax.grid(True, alpha=0.3)
        if yscale == "linear":
            ax.set_ylim(bottom=0, top=global_y_max if shared_y else None)
        else:
            ax.set_yscale(yscale)
        if row == n_rows - 1:
            ax.set_xlabel("Evaluations", fontsize=10)
        if col == 0:
            ylabel = "Normalized Regret" if normalize else "Simple Regret"
            ax.set_ylabel(ylabel, fontsize=10)

    for idx in range(n_scales, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row][col].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(7, len(method_names)), fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Optimization Curves per Scale: {data['task']} ({data['surrogate'].upper()})\n"
        f"{data['n_variants']} variants × {data['n_runs']} runs  |  median + 30-70% band",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])

    for fmt in formats:
        fig.savefig(os.path.join(save_dir, f"scale_sweep_trajectories.{fmt}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Trajectory grid plot saved")


# ---------------------------------------------------------------------------
# Plot 4: Per-scale individual trajectory plot (one figure each)
# ---------------------------------------------------------------------------
def plot_trajectory_individual(
    data: Dict,
    save_dir: str,
    *,
    yscale: str = "linear",
    exclude_methods: tuple = (),
    formats: tuple = ("png",),
    normalize: bool = True,
):
    scales = data["scales"]
    all_trajectories = data["trajectories"]
    method_names = [m for m in data["method_names"] if m not in exclude_methods]
    n_init = data["n_init"]
    max_steps = data["max_steps"]

    ind_dir = os.path.join(save_dir, "per_scale")
    os.makedirs(ind_dir, exist_ok=True)

    n_total = n_init + max_steps
    x = np.arange(n_total + 1)

    for scale in scales:
        sk = str(scale)
        spec = make_spec_for_scale(scale)

        # Normalize if requested
        if normalize:
            ref_trajs = all_trajectories[sk]["Random"]
            step0 = np.maximum(ref_trajs[:, :, 0].mean(axis=1), 1e-8)

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        for m in method_names:
            trajs = all_trajectories[sk][m]
            if normalize:
                trajs = trajs / step0[:, None, None]
            unit_curves = trajs.mean(axis=1)
            median = np.median(unit_curves, axis=0)
            q30 = np.quantile(unit_curves, 0.30, axis=0)
            q70 = np.quantile(unit_curves, 0.70, axis=0)

            color = COLORS.get(m, "tab:gray")
            lw = 2.5 if m == CAP_PPO_NAME else 1.5
            zorder = 10 if m == CAP_PPO_NAME else 5

            ax.plot(x[:len(median)], median, label=m, color=color,
                    linewidth=lw, zorder=zorder)
            ax.fill_between(x[:len(median)], q30, q70, alpha=0.15, color=color)

        dx_hi = spec["dx"][0][1]
        rot_hi = spec["rot"][0][1]
        sx_lo = spec["sx"][0][0]
        ax.set_xlabel("Number of Evaluations", fontsize=12)
        ylabel = "Normalized Regret" if normalize else "Simple Regret"
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(
            f"scale={scale:.2f}: dx±{dx_hi:.2f}, rot±{rot_hi:.0f}°, sx≥{sx_lo:.2f}\n"
            f"{data['n_variants']} variants × {data['n_runs']} runs  ({data['surrogate'].upper()})",
            fontsize=12,
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if yscale == "linear":
            ax.set_ylim(bottom=0)
        else:
            ax.set_yscale(yscale)

        plt.tight_layout()
        for fmt in formats:
            fig.savefig(os.path.join(ind_dir, f"trajectory_scale_{scale:.2f}.{fmt}"),
                        dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Individual trajectory plots saved to {ind_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Re-plot scale sweep results from saved pickle")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to scale_sweep_data.pkl")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Output dir (defaults to same dir as pkl)")
    parser.add_argument("--yscale", type=str, default="linear",
                        choices=["linear", "log", "symlog"])
    parser.add_argument("--shared_y", action="store_true", default=True,
                        help="Share y-axis across trajectory subplots (default)")
    parser.add_argument("--no_shared_y", action="store_true",
                        help="Independent y-axis per subplot")
    parser.add_argument("--no_random", action="store_true",
                        help="Exclude Random from plots")
    parser.add_argument("--individual", action="store_true",
                        help="Also generate individual per-scale plots")
    parser.add_argument("--show_transform_params", action="store_true",
                        help="Show the top x-axis with dx/rot/sx labels.")
    parser.add_argument("--no_normalize", action="store_true",
                        help="Disable gap normalization (plot raw simple regret)")
    parser.add_argument("--formats", nargs="+", default=["png"],
                        help="Output formats (png, pdf, svg)")
    args = parser.parse_args()

    with open(args.data, "rb") as f:
        data = pickle.load(f)

    save_dir = args.save_dir or os.path.dirname(args.data)
    os.makedirs(save_dir, exist_ok=True)

    exclude = ("Random",) if args.no_random else ()
    shared_y = not args.no_shared_y
    formats = tuple(args.formats)
    normalize = not args.no_normalize

    print(f"Loaded: {args.data}")
    print(f"  Task: {data['task']}, Surrogate: {data['surrogate']}")
    print(f"  Scales: {data['scales']}")
    print(f"  {data['n_variants']} variants × {data['n_runs']} runs")
    print(f"  Methods: {data['method_names']}")
    print(f"  Normalize: {normalize}")
    print(f"  Saving to: {save_dir}")
    print()

    plot_sweep(
        data,
        save_dir,
        exclude_methods=exclude,
        formats=formats,
        show_transform_params=args.show_transform_params,
        normalize=normalize,
    )
    plot_highlight(data, save_dir, formats=formats, normalize=normalize)
    plot_trajectory_grid(data, save_dir, shared_y=shared_y,
                         yscale=args.yscale, exclude_methods=exclude,
                         formats=formats, normalize=normalize)
    if args.individual:
        plot_trajectory_individual(data, save_dir, yscale=args.yscale,
                                   exclude_methods=exclude, formats=formats,
                                   normalize=normalize)

    print(f"\nAll plots saved to {save_dir}/")


if __name__ == "__main__":
    main()
