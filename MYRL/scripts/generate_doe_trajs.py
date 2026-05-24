#!/usr/bin/env python
from __future__ import annotations

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

"""Generate DOE trajectories for prior-data collection ablations.

Two collection modes are supported:

* static_sobol: one-shot Sobol design over the full budget.
* seq_local_sobol: initial global Sobol screening, then batched local Sobol
  refinement around the current best point with a fixed fraction of global
  exploration in every batch.

The output format matches the BO trajectory caches consumed by train_rl.py:
X_trajs, y_trajs, variant_indices, variants, variant_infos, metadata.
"""

import argparse
import json
import os
from typing import Any, Dict, Tuple

import numpy as np
from scipy.stats.qmc import Sobol
from tqdm import tqdm

from myrl.tasks.registry import get_task


def _sobol_box(
    *,
    n: int,
    dim: int,
    lower: np.ndarray,
    upper: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Draw n scrambled Sobol points in [lower, upper].

    Sobol balance properties are best at powers of two; for arbitrary budgets
    we draw the next power of two and keep the prefix. This avoids scipy's
    non-power-of-two warning and gives stable deterministic designs.
    """
    n = int(n)
    if n <= 0:
        return np.zeros((0, int(dim)), dtype=np.float32)
    m = int(np.ceil(np.log2(max(2, n))))
    sampler = Sobol(d=int(dim), scramble=True, seed=int(seed))
    u = sampler.random_base2(m=m)[:n]
    return (lower[None, :] + u * (upper[None, :] - lower[None, :])).astype(np.float32)


def _evaluate(task: Any, X: np.ndarray, variant_params: Dict[str, float]) -> np.ndarray:
    return task.evaluate_numpy(X, variant_params).astype(np.float32).reshape(-1)


def _static_sobol(
    *,
    task: Any,
    variant_params: Dict[str, float],
    total_evals: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    lower, upper = [np.asarray(a, dtype=np.float32) for a in task.bounds]
    X = _sobol_box(n=total_evals, dim=task.dim, lower=lower, upper=upper, seed=seed)
    y = _evaluate(task, X, variant_params)
    return X, y


def _seq_local_sobol(
    *,
    task: Any,
    variant_params: Dict[str, float],
    total_evals: int,
    seed: int,
    init_global: int,
    batch_size: int,
    radius0: float,
    radius_min: float,
    decay: float,
    global_frac: float,
) -> Tuple[np.ndarray, np.ndarray]:
    lower, upper = [np.asarray(a, dtype=np.float32) for a in task.bounds]
    init_global = int(min(max(1, init_global), total_evals))

    X_parts = [
        _sobol_box(
            n=init_global,
            dim=task.dim,
            lower=lower,
            upper=upper,
            seed=seed,
        )
    ]
    y_parts = [_evaluate(task, X_parts[0], variant_params)]

    n_done = init_global
    round_idx = 0
    while n_done < total_evals:
        cur_batch = int(min(max(1, batch_size), total_evals - n_done))
        X_seen = np.vstack(X_parts)
        y_seen = np.concatenate(y_parts)
        x_best = X_seen[int(np.argmin(y_seen))]

        n_global = int(round(cur_batch * float(global_frac)))
        if cur_batch >= 3:
            n_global = max(1, min(cur_batch - 1, n_global))
        else:
            n_global = 0
        n_local = cur_batch - n_global

        radius = max(float(radius_min), float(radius0) * (float(decay) ** round_idx))
        local_lower = np.maximum(lower, x_best - radius)
        local_upper = np.minimum(upper, x_best + radius)

        batch_parts = []
        if n_local > 0:
            batch_parts.append(
                _sobol_box(
                    n=n_local,
                    dim=task.dim,
                    lower=local_lower,
                    upper=local_upper,
                    seed=seed + 1009 + round_idx * 17,
                )
            )
        if n_global > 0:
            batch_parts.append(
                _sobol_box(
                    n=n_global,
                    dim=task.dim,
                    lower=lower,
                    upper=upper,
                    seed=seed + 2003 + round_idx * 19,
                )
            )

        X_new = np.vstack(batch_parts).astype(np.float32)
        y_new = _evaluate(task, X_new, variant_params)
        X_parts.append(X_new)
        y_parts.append(y_new)
        n_done += cur_batch
        round_idx += 1

    return np.vstack(X_parts)[:total_evals], np.concatenate(y_parts)[:total_evals]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DOE trajectory caches")
    parser.add_argument("--task", required=True)
    parser.add_argument("--variants_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--mode",
        choices=["static_sobol", "seq_local_sobol"],
        required=True,
    )
    parser.add_argument("--n_trajs_per_variant", type=int, default=1)
    parser.add_argument("--total_evals", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--init_global", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--radius0", type=float, default=0.20)
    parser.add_argument("--radius_min", type=float, default=0.05)
    parser.add_argument("--decay", type=float, default=0.90)
    parser.add_argument("--global_frac", type=float, default=0.40)
    args = parser.parse_args()

    variants_data = np.load(args.variants_path, allow_pickle=True)
    variants = variants_data["variants"]
    task = get_task(args.task)

    lower, _ = task.bounds
    dim = int(np.asarray(lower).shape[0])
    n_variants = int(len(variants))
    n_total = n_variants * int(args.n_trajs_per_variant)

    X_trajs = np.zeros((n_total, int(args.total_evals), dim), dtype=np.float32)
    y_trajs = np.zeros((n_total, int(args.total_evals)), dtype=np.float32)
    variant_indices = np.zeros((n_total,), dtype=np.int64)
    variant_infos = []

    idx = 0
    pbar = tqdm(total=n_total, desc=f"Generate {args.mode} trajectories")
    for v_idx in range(n_variants):
        variant = variants[v_idx]
        variant_params = dict(variant) if not isinstance(variant, dict) else variant
        for traj_idx in range(int(args.n_trajs_per_variant)):
            cur_seed = int(args.seed) + 10000 * v_idx + traj_idx
            if args.mode == "static_sobol":
                X, y = _static_sobol(
                    task=task,
                    variant_params=variant_params,
                    total_evals=int(args.total_evals),
                    seed=cur_seed,
                )
            else:
                X, y = _seq_local_sobol(
                    task=task,
                    variant_params=variant_params,
                    total_evals=int(args.total_evals),
                    seed=cur_seed,
                    init_global=int(args.init_global),
                    batch_size=int(args.batch_size),
                    radius0=float(args.radius0),
                    radius_min=float(args.radius_min),
                    decay=float(args.decay),
                    global_frac=float(args.global_frac),
                )

            X_trajs[idx] = X
            y_trajs[idx] = y
            variant_indices[idx] = v_idx
            variant_infos.append(
                {
                    "variant_index": int(v_idx),
                    "trajectory_index": int(traj_idx),
                    "seed": int(cur_seed),
                    "trajectory_type": str(args.mode),
                    "best_y": float(np.min(y)),
                }
            )
            idx += 1
            pbar.update(1)
    pbar.close()

    metadata = {
        "task": args.task,
        "k": n_variants,
        "n_trajs_per_variant": int(args.n_trajs_per_variant),
        "total_evals": int(args.total_evals),
        "seed": int(args.seed),
        "trajectory_type": str(args.mode),
        "mode": str(args.mode),
        "init_global": int(args.init_global),
        "batch_size": int(args.batch_size),
        "radius0": float(args.radius0),
        "radius_min": float(args.radius_min),
        "decay": float(args.decay),
        "global_frac": float(args.global_frac),
    }

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    np.savez(
        args.output_path,
        variants=variants,
        X_trajs=X_trajs,
        y_trajs=y_trajs,
        variant_indices=variant_indices,
        variant_infos=np.asarray(variant_infos, dtype=object),
        metadata=np.asarray([metadata], dtype=object),
    )

    print(
        f"Saved {n_total} {args.mode} trajectories "
        f"({n_variants} variants x {args.n_trajs_per_variant}) to {args.output_path}"
    )
    print("metadata:", json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
