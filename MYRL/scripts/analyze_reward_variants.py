#!/usr/bin/env python
"""
Offline reward-variant diagnostics on saved scale-sweep trajectories.

This script does not retrain PPO. It replays saved trajectory traces and
computes discounted episode returns under several candidate reward designs,
then measures how well those returns align with final regret.

Current use case:
    python MYRL/scripts/analyze_reward_variants.py \
      --data results_policies/hplc_emulator_single_inrange_dx004_rot14_sx088_scale_sweep/scale_sweep_data.pkl \
      --output_json results_policies/hplc_emulator_single_inrange_dx004_rot14_sx088_scale_sweep/reward_variant_analysis.json
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.stats import spearmanr


DEFAULT_METHODS = ("EI", "TAF_ranking", "CAP-PPO")
DEFAULT_MODES = ("auc", "delta", "delta_terminal", "mixed", "frontload_mixed", "staged_mixed")


def _discounted_return(rewards: np.ndarray, gamma: float) -> float:
    ret = 0.0
    coeff = 1.0
    for r in rewards:
        ret += coeff * float(r)
        coeff *= float(gamma)
    return ret


def _build_rewards(
    best_y_trace: np.ndarray,
    global_min: float,
    mode: str,
    *,
    mixed_lambda: float,
    terminal_weight: float,
    frontload_power: float,
    stage_midpoint: float,
) -> np.ndarray:
    best = np.asarray(best_y_trace, dtype=np.float64).reshape(-1)
    if best.size < 2:
        return np.zeros(0, dtype=np.float64)

    best0 = float(best[0])
    scale = max(best0 - float(global_min), 1e-8)

    rewards: List[float] = []
    for t in range(1, best.size):
        cumulative_improvement = max(0.0, best0 - float(best[t])) / scale
        delta_improvement = max(0.0, float(best[t - 1]) - float(best[t])) / scale

        if mode == "auc":
            rewards.append(cumulative_improvement)
        elif mode == "delta":
            rewards.append(delta_improvement)
        elif mode == "delta_terminal":
            rewards.append(delta_improvement)
        elif mode == "mixed":
            rewards.append(
                float(mixed_lambda) * cumulative_improvement
                + (1.0 - float(mixed_lambda)) * delta_improvement
            )
        elif mode == "frontload_mixed":
            n_steps = max(best.size - 1, 1)
            step_idx = t
            frontload_weight = max(0.0, 1.0 - (float(step_idx) - 1.0) / float(n_steps))
            frontload_weight = frontload_weight ** float(frontload_power)
            rewards.append(
                frontload_weight * (
                    float(mixed_lambda) * cumulative_improvement
                    + (1.0 - float(mixed_lambda)) * delta_improvement
                )
            )
        elif mode == "staged_mixed":
            n_steps = max(best.size - 1, 1)
            step_progress = float(t) / float(n_steps)
            stage = min(max(float(stage_midpoint), 1e-6), 1.0)
            early_weight = max(0.0, 1.0 - step_progress / stage)
            early_weight = early_weight ** float(frontload_power)
            dense_reward = (
                float(mixed_lambda) * cumulative_improvement
                + (1.0 - float(mixed_lambda)) * delta_improvement
            )
            rewards.append(early_weight * dense_reward + (1.0 - early_weight) * delta_improvement)
        else:
            raise ValueError(f"Unsupported reward mode: {mode}")

    if mode == "delta_terminal" and rewards:
        total_improvement = max(0.0, best0 - float(best[-1])) / scale
        rewards[-1] += float(terminal_weight) * total_improvement
    elif mode == "frontload_mixed" and rewards:
        total_improvement = max(0.0, best0 - float(best[-1])) / scale
        rewards[-1] += float(terminal_weight) * total_improvement
    elif mode == "staged_mixed" and rewards:
        total_improvement = max(0.0, best0 - float(best[-1])) / scale
        rewards[-1] += float(terminal_weight) * total_improvement

    return np.asarray(rewards, dtype=np.float64)


def _safe_spearman(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    rho = spearmanr(xs, ys).correlation
    if rho is None:
        return float("nan")
    return float(rho)


def _pairwise_accuracy(items: List[Dict[str, float]]) -> float:
    total = 0
    correct = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            reg_i = items[i]["final_regret"]
            reg_j = items[j]["final_regret"]
            ret_i = items[i]["episode_return"]
            ret_j = items[j]["episode_return"]
            if math.isclose(reg_i, reg_j, rel_tol=0.0, abs_tol=1e-12):
                continue
            total += 1
            better_i = reg_i < reg_j
            higher_i = ret_i > ret_j
            if better_i == higher_i:
                correct += 1
    if total == 0:
        return float("nan")
    return float(correct / total)


def _prefix_regret_auc(regret_trace: np.ndarray, budget: int) -> float:
    regrets = np.asarray(regret_trace, dtype=np.float64).reshape(-1)
    idx = min(max(int(budget), 0), regrets.size - 1)
    return float(np.mean(regrets[: idx + 1]))


def _budget_metrics(items: List[Dict[str, float]], budget: int) -> Dict[str, float]:
    returns = [x["episode_return"] for x in items]
    neg_budget_regrets = [-x["budget_regret"] for x in items]
    neg_prefix_auc = [-x["prefix_regret_auc"] for x in items]

    chosen_idx = max(range(len(items)), key=lambda i: items[i]["episode_return"])
    regret_order = sorted(range(len(items)), key=lambda i: items[i]["budget_regret"])
    auc_order = sorted(range(len(items)), key=lambda i: items[i]["prefix_regret_auc"])

    return {
        "budget_actions": int(budget),
        "budget_total_evals": int(budget + 2),
        "spearman_return_vs_neg_budget_regret": _safe_spearman(returns, neg_budget_regrets),
        "spearman_return_vs_neg_prefix_regret_auc": _safe_spearman(returns, neg_prefix_auc),
        "mean_variant_pairwise_accuracy_budget_regret": _pairwise_accuracy(
            [
                {
                    "final_regret": x["budget_regret"],
                    "episode_return": x["episode_return"],
                }
                for x in items
            ]
        ),
        "mean_variant_pairwise_accuracy_prefix_auc": _pairwise_accuracy(
            [
                {
                    "final_regret": x["prefix_regret_auc"],
                    "episode_return": x["episode_return"],
                }
                for x in items
            ]
        ),
        "chosen_rank_by_budget_regret": int(regret_order.index(chosen_idx)),
        "chosen_rank_by_prefix_regret_auc": int(auc_order.index(chosen_idx)),
    }


def _analyze_scale(
    data: Dict,
    scale: str,
    methods: Iterable[str],
    mode: str,
    *,
    gamma: float,
    mixed_lambda: float,
    terminal_weight: float,
    frontload_power: float,
    stage_midpoint: float,
    budgets: List[int],
) -> Dict:
    run_details = data["run_details"][scale]

    all_items: List[Dict[str, float]] = []
    chosen_method_counts = {m: 0 for m in methods}
    chosen_final_ranks: List[int] = []
    variant_pairwise_accs: List[float] = []
    budget_summaries: Dict[int, Dict[str, List[float]]] = {
        int(b): {
            "budget_regret_rho": [],
            "prefix_auc_rho": [],
            "budget_regret_pair_acc": [],
            "prefix_auc_pair_acc": [],
            "chosen_rank_budget_regret": [],
            "chosen_rank_prefix_auc": [],
        }
        for b in budgets
    }

    n_variants = len(run_details[next(iter(methods))])
    for v_idx in range(n_variants):
        per_variant: List[Dict[str, float]] = []
        for method in methods:
            for run_idx, run in enumerate(run_details[method][v_idx]):
                rewards = _build_rewards(
                    run["best_y_trace"],
                    float(run["global_min"]),
                    mode,
                    mixed_lambda=mixed_lambda,
                    terminal_weight=terminal_weight,
                    frontload_power=frontload_power,
                    stage_midpoint=stage_midpoint,
                )
                item = {
                    "variant_idx": int(v_idx),
                    "run_idx": int(run_idx),
                    "method": method,
                    "episode_return": _discounted_return(rewards, gamma),
                    "final_regret": float(np.asarray(run["regret_trace"], dtype=np.float64)[-1]),
                    "regret_trace": np.asarray(run["regret_trace"], dtype=np.float64).reshape(-1),
                }
                per_variant.append(item)
                all_items.append(item)

        variant_pairwise_accs.append(_pairwise_accuracy(per_variant))
        chosen_idx = max(range(len(per_variant)), key=lambda i: per_variant[i]["episode_return"])
        chosen = per_variant[chosen_idx]
        chosen_method_counts[chosen["method"]] += 1
        regret_order = sorted(range(len(per_variant)), key=lambda i: per_variant[i]["final_regret"])
        chosen_final_ranks.append(int(regret_order.index(chosen_idx)))

        for budget in budgets:
            budget_items = []
            for item in per_variant:
                regret_trace = item["regret_trace"]
                budget_idx = min(max(int(budget), 0), regret_trace.size - 1)
                budget_items.append(
                    {
                        "episode_return": item["episode_return"],
                        "budget_regret": float(regret_trace[budget_idx]),
                        "prefix_regret_auc": _prefix_regret_auc(regret_trace, budget_idx),
                    }
                )
            metrics = _budget_metrics(budget_items, budget)
            summary = budget_summaries[int(budget)]
            summary["budget_regret_rho"].append(metrics["spearman_return_vs_neg_budget_regret"])
            summary["prefix_auc_rho"].append(metrics["spearman_return_vs_neg_prefix_regret_auc"])
            summary["budget_regret_pair_acc"].append(metrics["mean_variant_pairwise_accuracy_budget_regret"])
            summary["prefix_auc_pair_acc"].append(metrics["mean_variant_pairwise_accuracy_prefix_auc"])
            summary["chosen_rank_budget_regret"].append(metrics["chosen_rank_by_budget_regret"])
            summary["chosen_rank_prefix_auc"].append(metrics["chosen_rank_by_prefix_regret_auc"])

    returns = [x["episode_return"] for x in all_items]
    neg_final_regrets = [-x["final_regret"] for x in all_items]

    mean_return_by_method = {}
    mean_final_regret_by_method = {}
    for method in methods:
        method_items = [x for x in all_items if x["method"] == method]
        mean_return_by_method[method] = float(np.mean([x["episode_return"] for x in method_items]))
        mean_final_regret_by_method[method] = float(np.mean([x["final_regret"] for x in method_items]))

    budget_metrics = {}
    for budget, summary in budget_summaries.items():
        budget_metrics[str(budget)] = {
            "budget_actions": int(budget),
            "budget_total_evals": int(budget + 2),
            "spearman_return_vs_neg_budget_regret": float(np.nanmean(summary["budget_regret_rho"])),
            "spearman_return_vs_neg_prefix_regret_auc": float(np.nanmean(summary["prefix_auc_rho"])),
            "mean_variant_pairwise_accuracy_budget_regret": float(np.nanmean(summary["budget_regret_pair_acc"])),
            "mean_variant_pairwise_accuracy_prefix_auc": float(np.nanmean(summary["prefix_auc_pair_acc"])),
            "avg_chosen_rank_by_budget_regret_among_9": float(np.mean(summary["chosen_rank_budget_regret"])),
            "avg_chosen_rank_by_prefix_regret_auc_among_9": float(np.mean(summary["chosen_rank_prefix_auc"])),
        }

    return {
        "n_variants": int(n_variants),
        "n_trajectories": int(len(all_items)),
        "spearman_return_vs_neg_final_regret": _safe_spearman(returns, neg_final_regrets),
        "mean_variant_pairwise_accuracy": float(np.nanmean(variant_pairwise_accs)),
        "chosen_method_counts": chosen_method_counts,
        "avg_chosen_final_rank_among_9": float(np.mean(chosen_final_ranks)),
        "mean_return_by_method": mean_return_by_method,
        "mean_final_regret_by_method": mean_final_regret_by_method,
        "budget_metrics": budget_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline reward-variant diagnostics")
    parser.add_argument("--data", required=True, type=str, help="Path to scale_sweep_data.pkl")
    parser.add_argument("--output_json", required=True, type=str, help="Where to save analysis JSON")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--mixed_lambda", type=float, default=0.30)
    parser.add_argument("--terminal_weight", type=float, default=1.0)
    parser.add_argument("--frontload_power", type=float, default=1.0)
    parser.add_argument("--stage_midpoint", type=float, default=0.4)
    parser.add_argument("--budgets", nargs="+", type=int, default=[6, 10, 14])
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    args = parser.parse_args()

    data_path = Path(args.data)
    with data_path.open("rb") as f:
        data = pickle.load(f)

    methods = list(args.methods)
    scales = [str(s) for s in data["scales"]]
    modes = list(args.modes)

    analysis = {
        "data": str(data_path),
        "task": data.get("task"),
        "gamma": float(args.gamma),
        "mixed_lambda": float(args.mixed_lambda),
        "terminal_weight": float(args.terminal_weight),
        "frontload_power": float(args.frontload_power),
        "stage_midpoint": float(args.stage_midpoint),
        "budgets": [int(b) for b in args.budgets],
        "methods": methods,
        "scales": scales,
        "modes": {},
    }

    for mode in modes:
        per_scale = {}
        for scale in scales:
            per_scale[scale] = _analyze_scale(
                data,
                scale,
                methods,
                mode,
                gamma=float(args.gamma),
                mixed_lambda=float(args.mixed_lambda),
                terminal_weight=float(args.terminal_weight),
                frontload_power=float(args.frontload_power),
                stage_midpoint=float(args.stage_midpoint),
                budgets=[int(b) for b in args.budgets],
            )

        aggregate = _analyze_scale(
            {"run_details": {"all": _merge_scales(data["run_details"], scales, methods)}},
            "all",
            methods,
            mode,
            gamma=float(args.gamma),
            mixed_lambda=float(args.mixed_lambda),
            terminal_weight=float(args.terminal_weight),
            frontload_power=float(args.frontload_power),
            stage_midpoint=float(args.stage_midpoint),
            budgets=[int(b) for b in args.budgets],
        )
        analysis["modes"][mode] = {
            "per_scale": per_scale,
            "aggregate": aggregate,
        }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False))

    print(f"Saved analysis to {output_path}")
    for mode in modes:
        agg = analysis["modes"][mode]["aggregate"]
        rho = agg["spearman_return_vs_neg_final_regret"]
        acc = agg["mean_variant_pairwise_accuracy"]
        rank = agg["avg_chosen_final_rank_among_9"]
        budget_summary = []
        for budget in args.budgets:
            b = agg["budget_metrics"][str(int(budget))]
            budget_summary.append(
                f"K={int(budget) + 2}eval rho={b['spearman_return_vs_neg_budget_regret']:.4f}"
            )
        print(
            f"{mode:15s} rho={rho:.4f}  pair_acc={acc:.4f}  "
            f"chosen_rank={rank:.3f}  chosen={agg['chosen_method_counts']}  "
            + "  ".join(budget_summary)
        )


def _merge_scales(run_details: Dict, scales: List[str], methods: List[str]) -> Dict[str, List[List[Dict]]]:
    merged = {m: [] for m in methods}
    for scale in scales:
        for m in methods:
            merged[m].extend(run_details[scale][m])
    return merged


if __name__ == "__main__":
    main()
