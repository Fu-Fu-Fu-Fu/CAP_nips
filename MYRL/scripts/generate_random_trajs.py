#!/usr/bin/env python
"""Generate random (uniform) trajectories as an ablation against BO expert trajectories.

For each variant, samples X uniformly in [0,1]^d and evaluates y via task emulator.
Output format matches finetune.py BO trajectory output (X_trajs, y_trajs, variant_indices).

Usage:
    python MYRL/scripts/generate_random_trajs.py \
        --task hartmann_6d_family \
        --variants_path data/hartmann_6d_family_variants_k10_seed2026.npz \
        --output_path data/hartmann_6d_family_random_trajs_k10.npz \
        --n_trajs_per_variant 101 \
        --total_evals 50 \
        --seed 2026
"""

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from myrl.tasks.registry import get_task


def main():
    parser = argparse.ArgumentParser(description="Generate random trajectories for ablation")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--variants_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--n_trajs_per_variant", type=int, default=1)
    parser.add_argument("--total_evals", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    # Load variants
    variants_data = np.load(args.variants_path, allow_pickle=True)
    variants = variants_data["variants"]  # array of dicts

    task = get_task(args.task)
    lower, upper = task.bounds
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    dim = lower.shape[0]

    rng = np.random.default_rng(args.seed)

    k = len(variants)
    n_traj = k * args.n_trajs_per_variant

    X_trajs = np.zeros((n_traj, args.total_evals, dim), dtype=np.float32)
    y_trajs = np.zeros((n_traj, args.total_evals), dtype=np.float32)
    variant_indices = np.zeros((n_traj,), dtype=np.int64)

    t_idx = 0
    pbar = tqdm(total=n_traj, desc="Generate random trajectories")
    for v_idx in range(k):
        params = variants[v_idx]
        params_dict = dict(params) if not isinstance(params, dict) else params
        for _ in range(args.n_trajs_per_variant):
            X = rng.uniform(lower, upper, size=(args.total_evals, dim)).astype(np.float32)
            y = task.evaluate_numpy(X, params_dict).astype(np.float32).ravel()
            X_trajs[t_idx] = X
            y_trajs[t_idx] = y
            variant_indices[t_idx] = v_idx
            t_idx += 1
            pbar.update(1)
    pbar.close()

    # Save
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    metadata = {
        "k": k,
        "n_trajs_per_variant": args.n_trajs_per_variant,
        "total_evals": args.total_evals,
        "seed": args.seed,
        "trajectory_type": "random_uniform",
    }
    variant_infos = np.array(
        [{"variant_index": i, "type": "random"} for i in range(k)],
        dtype=object,
    )
    np.savez(
        args.output_path,
        variants=variants,
        X_trajs=X_trajs,
        y_trajs=y_trajs,
        variant_indices=variant_indices,
        variant_infos=variant_infos,
        metadata=np.array([metadata], dtype=object),
    )
    print(f"Saved {n_traj} random trajectories ({k} variants x {args.n_trajs_per_variant}) to {args.output_path}")


if __name__ == "__main__":
    main()
