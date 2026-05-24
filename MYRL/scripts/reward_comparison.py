"""
reward_comparison.py — Compare reward functions across 5 tasks.

Runs 100-episode micro-trainings for each (task, reward_mode, preset) combo.
Produces CSV + comparison plots (reward curves, regret curves, correlation heatmap).

Usage:
    python MYRL/scripts/reward_comparison.py \
        --episodes 100 --save_dir paper_experiments/reward_comparison --seed 2026

    # Smoke test (single task, single config, 5 episodes):
    python MYRL/scripts/reward_comparison.py \
        --tasks branin_family --reward_configs auc_A --episodes 5 \
        --save_dir /tmp/reward_cmp_test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Bootstrap project root
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.normpath(os.path.join(_this_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from myrl.rl.train_rl import (
    ImprovedBraninBOEnv,
    ImprovedPPO,
    ImprovedRolloutBuffer,
)

# ---------------------------------------------------------------------------
# Repo root (for resolving relative data paths)
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.normpath(os.path.join(_this_dir, "..", ".."))

def _p(rel: str) -> str:
    """Resolve a repo-relative path."""
    return os.path.join(REPO_ROOT, rel)


# ===================================================================
# Task configurations — mirrors production training configs
# ===================================================================
TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "branin_family": dict(
        task_name="branin_family",
        objective_source="oracle_gp",
        normalize_oracle_gp=True,
        normalize_bnn=False,
        bnn_params_path=None,
        bnn_rff_alpha=5.0,
        bnn_rff_length_scale=0.3,
        variants_path=_p("data/branin_family_variants_k10_seed2026.npz"),
        trajectories_path=_p("data/branin_family_bo_trajs_k10_boSeed2026.npz"),
        taf_data_path=_p("data/taf_source_data_branin_family_k10.pkl"),
        max_steps=18,
        n_init_context=2,
        n_persistent_base=128,
        n_total_candidates=192,
        k_centers=2,
        local_h=1.5,
        local_h_decay=0.9,
        oracle_gp_min_grid_size=80,
        oracle_gp_min_n_lbfgs_starts=25,
    ),
    "hartmann_3d_family": dict(
        task_name="hartmann_3d_family",
        objective_source="oracle_gp",
        normalize_oracle_gp=True,
        normalize_bnn=False,
        bnn_params_path=None,
        bnn_rff_alpha=5.0,
        bnn_rff_length_scale=0.3,
        variants_path=_p("data/hartmann_3d_family_variants_k10_seed2026.npz"),
        trajectories_path=_p("data/hartmann_3d_family_bo_trajs_k10_boSeed2026.npz"),
        taf_data_path=_p("data/taf_source_data_hartmann_3d_family_k10.pkl"),
        max_steps=18,
        n_init_context=2,
        n_persistent_base=128,
        n_total_candidates=192,
        k_centers=2,
        local_h=1.5,
        local_h_decay=0.9,
        oracle_gp_min_grid_size=80,
        oracle_gp_min_n_lbfgs_starts=25,
    ),
    "hartmann_6d_family": dict(
        task_name="hartmann_6d_family",
        objective_source="oracle_gp",
        normalize_oracle_gp=True,
        normalize_bnn=False,
        bnn_params_path=None,
        bnn_rff_alpha=5.0,
        bnn_rff_length_scale=0.3,
        variants_path=_p("data/hartmann_6d_family_variants_k10_seed2026.npz"),
        trajectories_path=_p("data/hartmann_6d_family_bo_trajs_k10_boSeed2026.npz"),
        taf_data_path=_p("data/taf_source_data_hartmann_6d_family_k10.pkl"),
        max_steps=48,
        n_init_context=2,
        n_persistent_base=128,
        n_total_candidates=256,
        k_centers=3,
        local_h=0.15,
        local_h_decay=0.95,
        oracle_gp_min_grid_size=100,
        oracle_gp_min_n_lbfgs_starts=10,
    ),
    "hplc_emulator": dict(
        task_name="hplc_emulator",
        objective_source="bnn",
        normalize_oracle_gp=False,
        normalize_bnn=True,
        bnn_params_path=_p("data/bnn_surrogates_hplc_emulator_k10_kl0.001_single_inrange_dx004_rot14_sx088.npz"),
        bnn_rff_alpha=1.5,
        bnn_rff_length_scale=0.3,
        variants_path=_p("data/hplc_emulator_variants_k10_seed2026_single_inrange_dx004_rot14_sx088.npz"),
        trajectories_path=_p("data/hplc_emulator_bo_trajs_k10_boSeed2026_single_inrange_dx004_rot14_sx088.npz"),
        taf_data_path=_p("data/taf_source_data_hplc_emulator_k10_single_inrange_dx004_rot14_sx088.pkl"),
        max_steps=48,
        n_init_context=2,
        n_persistent_base=128,
        n_total_candidates=256,
        k_centers=3,
        local_h=0.15,
        local_h_decay=0.95,
        oracle_gp_min_grid_size=100,
        oracle_gp_min_n_lbfgs_starts=10,
    ),
    # Alkox without normalize_bnn (current production setup)
    "alkox_emulator": dict(
        task_name="alkox_emulator",
        objective_source="bnn",
        normalize_oracle_gp=False,
        normalize_bnn=False,
        bnn_params_path=_p("data/bnn_surrogates_alkox_emulator_k10_kl0.001_transform.npz"),
        bnn_rff_alpha=5.0,
        bnn_rff_length_scale=0.3,
        variants_path=_p("data/alkox_emulator_variants_k10_seed2026_transform.npz"),
        trajectories_path=_p("data/alkox_emulator_bo_trajs_k10_boSeed2026_transform.npz"),
        taf_data_path=_p("data/taf_source_data_alkox_emulator_k10_transform.pkl"),
        max_steps=28,
        n_init_context=2,
        n_persistent_base=128,
        n_total_candidates=192,
        k_centers=2,
        local_h=0.17,
        local_h_decay=0.95,
        oracle_gp_min_grid_size=100,
        oracle_gp_min_n_lbfgs_starts=10,
    ),
    # Alkox WITH normalize_bnn (test variant)
    "alkox_emulator_znorm": dict(
        task_name="alkox_emulator",
        objective_source="bnn",
        normalize_oracle_gp=False,
        normalize_bnn=True,
        bnn_params_path=_p("data/bnn_surrogates_alkox_emulator_k10_kl0.001_transform.npz"),
        bnn_rff_alpha=5.0,
        bnn_rff_length_scale=0.3,
        variants_path=_p("data/alkox_emulator_variants_k10_seed2026_transform.npz"),
        trajectories_path=_p("data/alkox_emulator_bo_trajs_k10_boSeed2026_transform.npz"),
        taf_data_path=_p("data/taf_source_data_alkox_emulator_k10_transform.pkl"),
        max_steps=28,
        n_init_context=2,
        n_persistent_base=128,
        n_total_candidates=192,
        k_centers=2,
        local_h=0.17,
        local_h_decay=0.95,
        oracle_gp_min_grid_size=100,
        oracle_gp_min_n_lbfgs_starts=10,
    ),
}


# ===================================================================
# Reward configurations — 4 modes × 2 presets
# ===================================================================
_PRESET_A = dict(  # Conservative / default
    reward_mixed_lambda=0.3,
    reward_terminal_weight=1.0,
    reward_frontload_power=1.0,
    reward_stage_midpoint=0.4,
    reward_regret_auc_weight=0.2,
    reward_regret_delta_weight=1.0,
    reward_regret_early_power=0.5,
    reward_regret_terminal_power=3.0,
    reward_regret_scale_floor_ratio=0.02,
)

_PRESET_B = dict(  # Aggressive exploitation
    reward_mixed_lambda=0.5,
    reward_terminal_weight=2.0,
    reward_frontload_power=1.5,
    reward_stage_midpoint=0.3,
    reward_regret_auc_weight=0.3,
    reward_regret_delta_weight=1.5,
    reward_regret_early_power=1.0,
    reward_regret_terminal_power=5.0,
    reward_regret_scale_floor_ratio=0.05,
)

REWARD_CONFIGS: Dict[str, Dict[str, Any]] = {}

for mode in ["auc", "delta_terminal", "regret_balanced", "staged_mixed"]:
    for preset_name, preset in [("A", _PRESET_A), ("B", _PRESET_B)]:
        name = f"{mode}_{preset_name}"
        REWARD_CONFIGS[name] = dict(reward_mode=mode, **preset)


# ===================================================================
# Micro-training
# ===================================================================

def run_micro_training(
    task_cfg: Dict[str, Any],
    reward_cfg: Dict[str, Any],
    n_episodes: int,
    update_every: int,
    seed: int,
    print_every: int = 10,
) -> List[Dict[str, float]]:
    """Run a short PPO training and collect per-episode metrics."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Merge task config with reward config for env creation
    env_kwargs = dict(task_cfg)
    env_kwargs.update(reward_cfg)
    env_kwargs["variant_sampling"] = "shuffled_cycle"
    env_kwargs["inference_precision"] = "float32"
    env_kwargs["device"] = device
    env_kwargs["seed"] = seed

    env = ImprovedBraninBOEnv(**env_kwargs)

    coord_dim = int(env.task.dim)
    max_steps = int(env.max_steps)
    use_taf_feature = (env_kwargs.get("taf_data_path") is not None)

    ppo = ImprovedPPO(
        coord_dim=coord_dim,
        hidden_dim=128,
        n_self_attn_layers=3,
        n_cross_attn_layers=3,
        n_heads=8,
        max_steps=max_steps,
        device=device,
        use_taf_feature=use_taf_feature,
        ent_coef=0.02,
        ent_coef_end=0.01,
    )

    total_updates = max(1, n_episodes // update_every)
    ppo.setup_scheduler(total_updates, warmup_fraction=0.05)

    buffer = ImprovedRolloutBuffer()
    metrics: List[Dict[str, float]] = []

    for episode in range(1, n_episodes + 1):
        obs = env.reset()
        ep_reward = 0.0
        bounds = env.current_func.bounds
        step_rewards = []

        for step in range(max_steps):
            action, log_prob, value = ppo.select_candidate(
                obs["X_context"], obs["y_context"],
                obs["X_candidates"], obs["pred_mean"], obs["pred_std"],
                bounds, obs["step"],
                is_persistent=obs["is_persistent"],
                taf_rank_norm=obs.get("taf_rank_norm"),
            )

            next_obs, reward, done, info = env.step(action)
            ep_reward += reward
            step_rewards.append(reward)

            context_feat, candidate_feat = ppo._build_features(
                obs["X_context"], obs["y_context"],
                obs["X_candidates"], obs["pred_mean"], obs["pred_std"],
                bounds, obs["step"],
                is_persistent=obs["is_persistent"],
                taf_rank_norm=obs.get("taf_rank_norm"),
            )
            buffer.add(context_feat, candidate_feat, obs["step"],
                       action, reward, value, log_prob, done)

            if done:
                break
            obs = next_obs

        metrics.append({
            "episode_reward": ep_reward,
            "final_regret": info["regret"],
            "initial_regret": float(env.initial_regret),
            "reward_scale": float(env.reward_scale),
            "best_y": info["best_y"],
            "global_min": info["global_min"],
        })

        # PPO update
        if episode % update_every == 0 and len(buffer) > 0:
            rollout = buffer.get(last_value=0.0, gamma=ppo.gamma, lam=ppo.lam)
            ppo.update(rollout, n_epochs=4, batch_size=64)
            buffer.reset()

        # Per-episode logging
        if episode % print_every == 0 or episode == 1:
            recent = metrics[-min(print_every, len(metrics)):]
            avg_rwd = np.mean([m["episode_reward"] for m in recent])
            avg_reg = np.mean([m["final_regret"] for m in recent])
            print(f"  Ep {episode:4d} | Reward: {avg_rwd:7.3f} | Regret: {avg_reg:.4f}")

    return metrics


# ===================================================================
# Plotting
# ===================================================================

def _smooth(arr, window=10):
    """Simple rolling mean."""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def plot_reward_curves(df: pd.DataFrame, save_dir: str):
    tasks = df["task"].unique()
    configs = sorted(df["reward_config"].unique())
    n_tasks = len(tasks)

    fig, axes = plt.subplots(n_tasks, 1, figsize=(12, 4 * n_tasks), squeeze=False)
    for i, task in enumerate(tasks):
        ax = axes[i, 0]
        sub = df[df["task"] == task]
        for cfg in configs:
            vals = sub[sub["reward_config"] == cfg]["episode_reward"].values
            smoothed = _smooth(vals)
            ax.plot(smoothed, label=cfg, alpha=0.8)
        ax.set_title(f"{task} — Episode Reward")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "reward_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_regret_curves(df: pd.DataFrame, save_dir: str):
    tasks = df["task"].unique()
    configs = sorted(df["reward_config"].unique())
    n_tasks = len(tasks)

    fig, axes = plt.subplots(n_tasks, 1, figsize=(12, 4 * n_tasks), squeeze=False)
    for i, task in enumerate(tasks):
        ax = axes[i, 0]
        sub = df[df["task"] == task]
        for cfg in configs:
            vals = sub[sub["reward_config"] == cfg]["final_regret"].values
            smoothed = _smooth(vals)
            ax.plot(smoothed, label=cfg, alpha=0.8)
        ax.set_title(f"{task} — Final Regret")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Regret")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "regret_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_correlation_heatmap(df: pd.DataFrame, save_dir: str):
    """Pearson correlation between cumulative reward and negative final regret."""
    tasks = df["task"].unique()
    configs = sorted(df["reward_config"].unique())

    corr_matrix = np.full((len(tasks), len(configs)), np.nan)
    for i, task in enumerate(tasks):
        for j, cfg in enumerate(configs):
            sub = df[(df["task"] == task) & (df["reward_config"] == cfg)]
            if len(sub) > 5:
                r = sub["episode_reward"].values
                reg = sub["final_regret"].values
                corr = np.corrcoef(r, -reg)[0, 1]
                corr_matrix[i, j] = corr

    fig, ax = plt.subplots(figsize=(max(10, len(configs) * 1.2), max(4, len(tasks) * 0.8)))
    im = ax.imshow(corr_matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks, fontsize=9)
    for i in range(len(tasks)):
        for j in range(len(configs)):
            v = corr_matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.5 else "black")
    plt.colorbar(im, ax=ax, label="Pearson(reward, -regret)")
    ax.set_title("Reward–Regret Correlation")
    plt.tight_layout()
    path = os.path.join(save_dir, "correlation_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_summary_table(df: pd.DataFrame, save_dir: str):
    """Mean final regret over last 20 episodes."""
    tasks = df["task"].unique()
    configs = sorted(df["reward_config"].unique())

    regret_matrix = np.full((len(tasks), len(configs)), np.nan)
    for i, task in enumerate(tasks):
        for j, cfg in enumerate(configs):
            sub = df[(df["task"] == task) & (df["reward_config"] == cfg)]
            if len(sub) >= 20:
                regret_matrix[i, j] = sub["final_regret"].values[-20:].mean()

    fig, ax = plt.subplots(figsize=(max(10, len(configs) * 1.2), max(4, len(tasks) * 0.8)))
    im = ax.imshow(regret_matrix, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks, fontsize=9)
    for i in range(len(tasks)):
        for j in range(len(configs)):
            v = regret_matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, label="Mean Regret (last 20 ep)")
    ax.set_title("Final Regret Summary (lower = better)")
    plt.tight_layout()
    path = os.path.join(save_dir, "summary_table.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Reward function comparison across tasks")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Task names to test (default: all). E.g. branin_family hplc_emulator")
    parser.add_argument("--reward_configs", nargs="*", default=None,
                        help="Reward config names to test (default: all). E.g. auc_A regret_balanced_B")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of training episodes per micro-run")
    parser.add_argument("--update_every", type=int, default=10,
                        help="PPO update frequency (episodes)")
    parser.add_argument("--save_dir", type=str, default="paper_experiments/reward_comparison")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Filter tasks/configs
    task_names = args.tasks if args.tasks else list(TASK_CONFIGS.keys())
    config_names = args.reward_configs if args.reward_configs else list(REWARD_CONFIGS.keys())

    total_combos = len(task_names) * len(config_names)
    print(f"=== Reward Comparison: {len(task_names)} tasks × {len(config_names)} configs = {total_combos} runs ===")
    print(f"Episodes per run: {args.episodes}, update_every: {args.update_every}")
    print(f"Tasks: {task_names}")
    print(f"Configs: {config_names}")
    print(f"Save dir: {args.save_dir}")
    print()

    all_results = []
    run_idx = 0

    for task_name in task_names:
        if task_name not in TASK_CONFIGS:
            print(f"WARNING: Unknown task '{task_name}', skipping")
            continue
        task_cfg = TASK_CONFIGS[task_name]

        for config_name in config_names:
            if config_name not in REWARD_CONFIGS:
                print(f"WARNING: Unknown config '{config_name}', skipping")
                continue
            reward_cfg = REWARD_CONFIGS[config_name]

            run_idx += 1
            print(f"\n{'='*60}")
            print(f"[{run_idx}/{total_combos}] {task_name} / {config_name}")
            print(f"  reward_mode={reward_cfg['reward_mode']}, "
                  f"terminal_weight={reward_cfg['reward_terminal_weight']}")
            print(f"{'='*60}")

            t0 = time.time()
            metrics = run_micro_training(
                task_cfg=task_cfg,
                reward_cfg=reward_cfg,
                n_episodes=args.episodes,
                update_every=args.update_every,
                seed=args.seed,
            )
            elapsed = time.time() - t0

            # Append to results
            for ep_idx, m in enumerate(metrics):
                all_results.append({
                    "task": task_name,
                    "reward_config": config_name,
                    "reward_mode": reward_cfg["reward_mode"],
                    "preset": config_name.split("_")[-1],
                    "episode": ep_idx + 1,
                    **m,
                })

            # Quick summary
            last20_regret = np.mean([m["final_regret"] for m in metrics[-20:]])
            last20_reward = np.mean([m["episode_reward"] for m in metrics[-20:]])
            print(f"  Done in {elapsed:.1f}s | "
                  f"Last-20 avg: reward={last20_reward:.3f}, regret={last20_regret:.4f}")

    # Save CSV
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(args.save_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path} ({len(df)} rows)")

    # Save config info
    config_info = {
        "tasks": task_names,
        "reward_configs": {k: v for k, v in REWARD_CONFIGS.items() if k in config_names},
        "episodes": args.episodes,
        "update_every": args.update_every,
        "seed": args.seed,
    }
    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(config_info, f, indent=2)

    # Generate plots
    if len(df) > 0:
        print("\nGenerating plots...")
        plot_reward_curves(df, args.save_dir)
        plot_regret_curves(df, args.save_dir)
        plot_correlation_heatmap(df, args.save_dir)
        plot_summary_table(df, args.save_dir)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
