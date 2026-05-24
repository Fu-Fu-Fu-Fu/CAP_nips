#!/usr/bin/env python
"""
Analyze reward designs using synthetic best-regret episodes and offline HPLC traces.

This script does not retrain PPO. It compares existing reward modes with a
new best-regret-based candidate reward under:
1. hand-designed scenario preferences
2. scale-invariance checks
3. offline replay on saved HPLC scale-sweep trajectories
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.stats import spearmanr


DEFAULT_METHODS = ("EI", "TAF_ranking", "CAP-PPO")
EXISTING_MODES = ("auc", "delta", "delta_terminal", "mixed", "frontload_mixed", "staged_mixed")


@dataclass(frozen=True)
class CandidateParams:
    auc_weight: float
    delta_weight: float
    terminal_weight: float
    early_power: float
    terminal_power: float
    scale_floor_ratio: float


def discounted_return(rewards: np.ndarray, gamma: float) -> float:
    ret = 0.0
    coeff = 1.0
    for r in rewards:
        ret += coeff * float(r)
        coeff *= float(gamma)
    return ret


def build_existing_rewards(
    best_y_trace: np.ndarray,
    global_min: float,
    mode: str,
    *,
    mixed_lambda: float = 0.7,
    terminal_weight: float = 1.0,
    frontload_power: float = 1.0,
    stage_midpoint: float = 0.4,
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

    if mode in {"delta_terminal", "frontload_mixed", "staged_mixed"} and rewards:
        total_improvement = max(0.0, best0 - float(best[-1])) / scale
        rewards[-1] += float(terminal_weight) * total_improvement

    return np.asarray(rewards, dtype=np.float64)


def build_regret_balanced_rewards(
    best_y_trace: np.ndarray,
    global_min: float,
    reward_scale: float,
    params: CandidateParams,
) -> np.ndarray:
    best = np.asarray(best_y_trace, dtype=np.float64).reshape(-1)
    if best.size < 2:
        return np.zeros(0, dtype=np.float64)

    g = np.maximum(best - float(global_min), 0.0)
    g0 = float(g[0])
    n_steps = max(best.size - 1, 1)
    norm = max(g0, float(params.scale_floor_ratio) * float(reward_scale), 1e-8)

    rewards: List[float] = []
    prev_progress = 0.0
    for t in range(1, best.size):
        progress = np.clip((g0 - float(g[t])) / norm, 0.0, 1.0)
        delta_progress = max(0.0, progress - prev_progress)
        early_weight = ((float(n_steps) - float(t) + 1.0) / float(n_steps)) ** float(params.early_power)

        reward = (
            float(params.auc_weight) * (progress / float(n_steps))
            + float(params.delta_weight) * early_weight * delta_progress
        )

        if t == n_steps:
            reward += float(params.terminal_weight) * (progress ** float(params.terminal_power))

        rewards.append(float(reward))
        prev_progress = progress

    return np.asarray(rewards, dtype=np.float64)


def build_best_y_trace(best_regret_trace: Iterable[float], global_min: float = 0.0) -> np.ndarray:
    g = np.asarray(list(best_regret_trace), dtype=np.float64).reshape(-1)
    if g.size < 2:
        raise ValueError("best_regret_trace must contain at least 2 points")
    if np.any(np.diff(g) > 1e-12):
        raise ValueError("best_regret_trace must be best-so-far regret, so it must be non-increasing")
    return g + float(global_min)


def scenario_definitions() -> Tuple[Dict[str, Dict], List[Tuple[str, str, str]]]:
    scenarios = {
        "fast_good": {
            "best_regret_trace": [1.00, 0.55, 0.30, 0.18, 0.12, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
            "reward_scale": 2.0,
        },
        "slow_good": {
            "best_regret_trace": [1.00, 0.97, 0.93, 0.84, 0.70, 0.48, 0.28, 0.17, 0.12, 0.10, 0.10],
            "reward_scale": 2.0,
        },
        "fast_mid": {
            "best_regret_trace": [1.00, 0.62, 0.48, 0.41, 0.38, 0.36, 0.35, 0.35, 0.35, 0.35, 0.35],
            "reward_scale": 2.0,
        },
        "slow_best": {
            "best_regret_trace": [1.00, 0.98, 0.95, 0.90, 0.78, 0.58, 0.38, 0.22, 0.12, 0.07, 0.05],
            "reward_scale": 2.0,
        },
        "early_lucky_flat": {
            "best_regret_trace": [1.00, 0.50, 0.43, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40],
            "reward_scale": 2.0,
        },
        "steady_best": {
            "best_regret_trace": [1.00, 0.86, 0.73, 0.60, 0.48, 0.36, 0.26, 0.18, 0.11, 0.07, 0.04],
            "reward_scale": 2.0,
        },
        "no_improve": {
            "best_regret_trace": [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
            "reward_scale": 2.0,
        },
        # Same normalized path as fast_good, but different absolute scale.
        "fast_good_scaled_up": {
            "best_regret_trace": [10.0, 5.5, 3.0, 1.8, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "reward_scale": 20.0,
        },
        "fast_good_scaled_down": {
            "best_regret_trace": [0.10, 0.055, 0.030, 0.018, 0.012, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010],
            "reward_scale": 0.20,
        },
    }

    preferences = [
        ("fast_good", "slow_good", "same final regret, earlier convergence should score higher"),
        ("slow_best", "fast_mid", "much lower final regret should beat faster but mediocre endpoint"),
        ("steady_best", "early_lucky_flat", "low final regret should beat early lucky jump then plateau"),
        ("fast_mid", "no_improve", "real improvement should beat no improvement"),
        ("fast_good", "fast_mid", "same speed class, lower final regret should score higher"),
        ("slow_good", "early_lucky_flat", "better final regret should outweigh early but shallow jump"),
        ("early_lucky_flat", "no_improve", "some improvement should beat none"),
    ]
    return scenarios, preferences


def evaluate_scenarios(
    mode: str,
    gamma: float,
    candidate_params: CandidateParams | None = None,
) -> Dict:
    scenarios, preferences = scenario_definitions()
    scores: Dict[str, float] = {}
    per_step: Dict[str, List[float]] = {}

    for name, spec in scenarios.items():
        best_y = build_best_y_trace(spec["best_regret_trace"], global_min=0.0)
        reward_scale = float(spec["reward_scale"])
        if mode == "regret_balanced":
            assert candidate_params is not None
            rewards = build_regret_balanced_rewards(best_y, 0.0, reward_scale, candidate_params)
        else:
            rewards = build_existing_rewards(best_y, 0.0, mode)
        per_step[name] = [float(x) for x in rewards]
        scores[name] = discounted_return(rewards, gamma)

    checks = []
    n_pass = 0
    for better, worse, reason in preferences:
        passed = bool(scores[better] > scores[worse])
        n_pass += int(passed)
        checks.append(
            {
                "better": better,
                "worse": worse,
                "passed": passed,
                "margin": float(scores[better] - scores[worse]),
                "reason": reason,
            }
        )

    scale_gap = abs(scores["fast_good_scaled_up"] - scores["fast_good"])
    scale_gap += abs(scores["fast_good_scaled_down"] - scores["fast_good"])

    return {
        "scores": scores,
        "per_step_rewards": per_step,
        "preference_checks": checks,
        "n_preferences_passed": int(n_pass),
        "n_preferences_total": int(len(preferences)),
        "scale_invariance_gap": float(scale_gap),
    }


def select_candidate_params(gamma: float) -> Tuple[CandidateParams, Dict]:
    best_params = None
    best_metrics = None
    best_key = None

    for vals in itertools.product(
        [0.2, 0.3, 0.4, 0.5],      # auc_weight
        [0.4, 0.6, 0.8, 1.0],      # delta_weight
        [0.8, 1.0, 1.2, 1.5],      # terminal_weight
        [0.5, 1.0, 1.5],           # early_power
        [1.0, 2.0, 3.0],           # terminal_power
        [0.02, 0.05, 0.10],        # scale_floor_ratio
    ):
        params = CandidateParams(*[float(v) for v in vals])
        metrics = evaluate_scenarios("regret_balanced", gamma=gamma, candidate_params=params)
        key = (
            metrics["n_preferences_passed"],
            -metrics["scale_invariance_gap"],
            metrics["scores"]["slow_best"] - metrics["scores"]["fast_mid"],
            metrics["scores"]["fast_good"] - metrics["scores"]["slow_good"],
            metrics["scores"]["steady_best"] - metrics["scores"]["early_lucky_flat"],
        )
        if best_key is None or key > best_key:
            best_key = key
            best_params = params
            best_metrics = metrics

    assert best_params is not None and best_metrics is not None
    return best_params, best_metrics


def safe_spearman(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    rho = spearmanr(xs, ys).correlation
    return float("nan") if rho is None else float(rho)


def pairwise_accuracy(items: List[Dict[str, float]]) -> float:
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
            if (reg_i < reg_j) == (ret_i > ret_j):
                correct += 1
    return float(correct / total) if total else float("nan")


def replay_mode_on_scale_sweep(
    data_path: Path,
    mode: str,
    gamma: float,
    candidate_params: CandidateParams | None = None,
    methods: Iterable[str] = DEFAULT_METHODS,
) -> Dict:
    with data_path.open("rb") as f:
        data = pickle.load(f)

    methods = list(methods)
    scales = [str(x) for x in data["scales"]]
    all_items = []

    for scale in scales:
        run_details = data["run_details"][scale]
        n_variants = len(run_details[methods[0]])
        for v_idx in range(n_variants):
            for method in methods:
                for run in run_details[method][v_idx]:
                    best_y_trace = np.asarray(run["best_y_trace"], dtype=np.float64).reshape(-1)
                    regret_trace = np.asarray(run["regret_trace"], dtype=np.float64).reshape(-1)
                    global_min = float(run["global_min"])
                    reward_scale = max(float(best_y_trace[0] - global_min), 1e-8)

                    if mode == "regret_balanced":
                        assert candidate_params is not None
                        rewards = build_regret_balanced_rewards(
                            best_y_trace, global_min, reward_scale, candidate_params
                        )
                    else:
                        rewards = build_existing_rewards(best_y_trace, global_min, mode)

                    all_items.append(
                        {
                            "scale": scale,
                            "method": method,
                            "episode_return": discounted_return(rewards, gamma),
                            "final_regret": float(regret_trace[-1]),
                            "prefix_regret_auc": float(np.mean(regret_trace)),
                            "regret_at_12eval": float(regret_trace[min(10, len(regret_trace) - 1)]),
                        }
                    )

    returns = [x["episode_return"] for x in all_items]
    neg_final_regret = [-x["final_regret"] for x in all_items]
    neg_prefix_auc = [-x["prefix_regret_auc"] for x in all_items]
    neg_regret_12 = [-x["regret_at_12eval"] for x in all_items]

    return {
        "n_trajectories": int(len(all_items)),
        "spearman_return_vs_neg_final_regret": safe_spearman(returns, neg_final_regret),
        "spearman_return_vs_neg_prefix_regret_auc": safe_spearman(returns, neg_prefix_auc),
        "spearman_return_vs_neg_regret_at_12eval": safe_spearman(returns, neg_regret_12),
        "pairwise_accuracy_final_regret": pairwise_accuracy(all_items),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze best-regret-based reward design")
    parser.add_argument(
        "--data",
        type=str,
        default="results_policies/hplc_emulator_single_inrange_dx004_rot14_sx088_scale_sweep/scale_sweep_data.pkl",
        help="Optional offline HPLC scale-sweep data for replay analysis.",
    )
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()

    candidate_params, candidate_synth = select_candidate_params(gamma=float(args.gamma))

    results = {
        "gamma": float(args.gamma),
        "candidate_params": candidate_params.__dict__,
        "synthetic": {},
        "offline_replay": {},
    }

    for mode in EXISTING_MODES:
        results["synthetic"][mode] = evaluate_scenarios(mode, gamma=float(args.gamma))
    results["synthetic"]["regret_balanced"] = candidate_synth

    data_path = Path(args.data)
    if data_path.exists():
        for mode in EXISTING_MODES:
            results["offline_replay"][mode] = replay_mode_on_scale_sweep(
                data_path, mode, gamma=float(args.gamma)
            )
        results["offline_replay"]["regret_balanced"] = replay_mode_on_scale_sweep(
            data_path,
            "regret_balanced",
            gamma=float(args.gamma),
            candidate_params=candidate_params,
        )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"Saved analysis to {output_path}")
    print("\nBest candidate params:")
    for k, v in candidate_params.__dict__.items():
        print(f"  {k}: {v}")

    print("\nSynthetic preference pass counts:")
    for mode in list(EXISTING_MODES) + ["regret_balanced"]:
        info = results["synthetic"][mode]
        print(
            f"  {mode:16s} "
            f"{info['n_preferences_passed']}/{info['n_preferences_total']} "
            f"scale_gap={info['scale_invariance_gap']:.6f}"
        )

    if results["offline_replay"]:
        print("\nOffline replay alignment:")
        for mode in list(EXISTING_MODES) + ["regret_balanced"]:
            info = results["offline_replay"][mode]
            print(
                f"  {mode:16s} "
                f"rho_final={info['spearman_return_vs_neg_final_regret']:.4f} "
                f"rho_prefix_auc={info['spearman_return_vs_neg_prefix_regret_auc']:.4f} "
                f"rho_K12={info['spearman_return_vs_neg_regret_at_12eval']:.4f} "
                f"pair_acc={info['pairwise_accuracy_final_regret']:.4f}"
            )


if __name__ == "__main__":
    main()
