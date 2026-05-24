"""
Plot normalized scale-sweep figures from compact plot-only JSON.

The expected input is the compact JSON produced from scale_sweep_data.pkl,
with:
  - normalized_final_stats
  - normalized_trajectory_summary

Example:
    python MYRL/scripts/plot_scale_sweep_plot_json.py \
        --data paper_experiments/alkox/results/tabpfn/scale_sweep_plot_data.json \
        --save_dir paper_experiments/alkox/results/tabpfn/plot_json_replot \
        --formats png pdf
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CAP_PPO_NAME = "CAP-PPO"

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
    "EI": "^",
    "UCB": "v",
    "PI": "<",
    "TAF_me": "D",
    "TAF_ranking": "d",
    "PFNs4BO": "P",
    "TuRBO": "X",
    "FunBO": "h",
}
LINESTYLES = {
    CAP_PPO_NAME: "-",
    "EI": "--",
    "UCB": "--",
    "PI": "--",
    "TAF_ranking": "-.",
    "TAF_me": "-.",
    "PFNs4BO": ":",
    "TuRBO": ":",
    "FunBO": ":",
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


def style_axes(ax):
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.10, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.35)
    ax.spines["bottom"].set_alpha(0.35)
    ax.tick_params(labelsize=11)


def task_title(data: Dict) -> str:
    task = TASK_DISPLAY_NAMES.get(data.get("task", ""), data.get("task", "Unknown task"))
    surrogate = str(data.get("surrogate", "")).upper()
    return f"{task} ({surrogate} surrogate)"


def save_all(fig, save_dir: str, basename: str, formats: Iterable[str]):
    for fmt in formats:
        out = os.path.join(save_dir, f"{basename}.{fmt}")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")


def plot_scale_sweep(data: Dict, save_dir: str, formats: List[str], exclude_random: bool):
    scales = [float(s) for s in data["scales"]]
    stats = data["normalized_final_stats"]
    methods = list(data["method_names"])
    if exclude_random:
        methods = [m for m in methods if m != "Random"]

    fig, ax = plt.subplots(1, 1, figsize=(10.8, 5.8))

    for method in methods:
        means = np.array([stats[str(s)][method]["mean"] for s in data["scales"]], dtype=float)
        sems = np.array([stats[str(s)][method]["sem"] for s in data["scales"]], dtype=float)

        color = COLORS.get(method, "tab:gray")
        marker = MARKERS.get(method, "o")
        ls = LINESTYLES.get(method, "-")
        lw = 2.5 if method == CAP_PPO_NAME else 1.5
        alpha_line = 0.5 if method == "Random" else 1.0
        alpha_fill = 0.06 if method == "Random" else 0.12

        ax.plot(
            scales,
            means,
            marker=marker,
            label=method,
            color=color,
            linewidth=lw,
            linestyle=ls,
            markersize=4.8,
            alpha=alpha_line,
            markeredgewidth=0.5,
            markeredgecolor="white",
        )
        ax.fill_between(scales, means - sems, means + sems, alpha=alpha_fill, color=color)

    ax.set_xlabel("Variant Scale", fontsize=12)
    ax.set_ylabel("Normalized Regret (mean ± SEM)", fontsize=12)
    ax.set_title(task_title(data), fontsize=13, pad=10)
    ax.legend(
        fontsize=9,
        loc="upper left",
        ncol=2,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#d7d7d7",
        handlelength=2.0,
        columnspacing=1.0,
    )
    style_axes(ax)
    fig.tight_layout()
    save_all(fig, save_dir, "scale_sweep_regret", formats)
    plt.close(fig)


def plot_highlight(data: Dict, save_dir: str, formats: List[str]):
    scales = [float(s) for s in data["scales"]]
    stats = data["normalized_final_stats"]
    methods = set(data["method_names"])
    highlight = [m for m in [CAP_PPO_NAME, "EI", "UCB", "TAF_ranking"] if m in methods]

    fig, ax = plt.subplots(1, 1, figsize=(10.2, 5.6))

    for method in highlight:
        means = np.array([stats[str(s)][method]["mean"] for s in data["scales"]], dtype=float)
        sems = np.array([stats[str(s)][method]["sem"] for s in data["scales"]], dtype=float)
        color = COLORS.get(method, "tab:gray")
        lw = 2.5 if method == CAP_PPO_NAME else 1.5

        ax.plot(
            scales,
            means,
            marker=MARKERS.get(method, "o"),
            label=method,
            color=color,
            linewidth=lw,
            linestyle=LINESTYLES.get(method, "-"),
            markersize=5.2,
            markeredgewidth=0.6,
            markeredgecolor="white",
        )
        ax.fill_between(scales, means - sems, means + sems, alpha=0.15, color=color)

    ax.set_xlabel("Variant Scale", fontsize=12)
    ax.set_ylabel("Normalized Regret (mean ± SEM)", fontsize=12)
    ax.set_title(f"CAP-PPO vs Key Baselines - {task_title(data)}", fontsize=13, pad=10)
    ax.legend(fontsize=10, frameon=True, framealpha=0.92, facecolor="white", edgecolor="#d7d7d7")
    style_axes(ax)
    fig.tight_layout()
    save_all(fig, save_dir, "scale_sweep_highlight", formats)
    plt.close(fig)


def plot_trajectory_grid(data: Dict, save_dir: str, formats: List[str], exclude_random: bool):
    scales = data["scales"]
    methods = list(data["method_names"])
    if exclude_random:
        methods = [m for m in methods if m != "Random"]
    summaries = data["normalized_trajectory_summary"]
    x = np.asarray(data.get("x") or list(range(data["n_init"] + data["max_steps"] + 1)), dtype=float)

    n_scales = len(scales)
    n_cols = min(4, n_scales)
    n_rows = (n_scales + n_cols - 1) // n_cols

    global_y_max = 0.0
    for sk in scales:
        for method in methods:
            q70 = np.asarray(summaries[str(sk)][method]["q70"], dtype=float)
            candidate = float(np.nanmax(q70))
            if method == "Random":
                median0 = float(np.asarray(summaries[str(sk)][method]["median"], dtype=float)[0])
                candidate = min(candidate, median0)
            global_y_max = max(global_y_max, candidate)
    global_y_max = max(global_y_max * 1.1, 1e-8)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,
        sharey=True,
    )

    for idx, sk in enumerate(scales):
        ax = axes[idx // n_cols][idx % n_cols]
        for method in methods:
            item = summaries[str(sk)][method]
            median = np.asarray(item["median"], dtype=float)
            q30 = np.asarray(item["q30"], dtype=float)
            q70 = np.asarray(item["q70"], dtype=float)
            xs = x[: len(median)]

            color = COLORS.get(method, "tab:gray")
            lw = 2.0 if method == CAP_PPO_NAME else 1.2
            alpha_line = 1.0 if method == CAP_PPO_NAME else 0.8

            ax.plot(xs, median, label=method, color=color, linewidth=lw, alpha=alpha_line)
            ax.fill_between(xs, q30, q70, alpha=0.12, color=color)

        ax.set_title(f"scale={float(sk):.2f}", fontsize=10)
        ax.set_ylim(bottom=0, top=global_y_max)
        ax.grid(True, alpha=0.3)
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel("Evaluations", fontsize=10)
        if idx % n_cols == 0:
            ax.set_ylabel("Normalized Regret", fontsize=10)

    for idx in range(n_scales, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(7, len(methods)),
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        f"Optimization Curves per Scale: {task_title(data)}\n"
        f"{data['n_variants']} variants x {data['n_runs']} runs | median + 30-70% band",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    save_all(fig, save_dir, "scale_sweep_trajectories", formats)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot normalized scale-sweep figures from compact JSON")
    parser.add_argument("--data", required=True, help="Path to scale_sweep_plot_data.json")
    parser.add_argument("--save_dir", default=None, help="Output directory. Defaults to JSON directory.")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], help="Output formats")
    parser.add_argument("--no_random", action="store_true", help="Exclude Random from plots")
    parser.add_argument("--skip_highlight", action="store_true", help="Do not create highlight plot")
    args = parser.parse_args()

    with open(args.data, "r") as f:
        data = json.load(f)

    save_dir = args.save_dir or os.path.dirname(args.data)
    os.makedirs(save_dir, exist_ok=True)

    plot_scale_sweep(data, save_dir, args.formats, args.no_random)
    if not args.skip_highlight:
        plot_highlight(data, save_dir, args.formats)
    plot_trajectory_grid(data, save_dir, args.formats, args.no_random)


if __name__ == "__main__":
    main()
