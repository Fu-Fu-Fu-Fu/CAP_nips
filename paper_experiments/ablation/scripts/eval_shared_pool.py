"""
eval_shared_pool.py — Ablation 1: w/o RL (shared candidate pool)

All methods (EI, UCB, PI, Random, TAF, CAP-PPO) see the **exact same** candidate
pool at each BO step. The candidate pool is generated using CAP-PPO's persistent +
local strategy.  This isolates the contribution of the *learned selection strategy*
from the candidate pool design.

TuRBO and PFNs4BO are excluded because they use their own internal surrogates /
candidate generation (TuRBO has its own trust region, PFNs4BO has its own model).

Usage:
    python paper_experiments/ablation_study/scripts/eval_shared_pool.py \
        --task hartmann_6d_family \
        --rl_model_path paper_experiments/hartmann6d/model/ppo_best.pt \
        --taf_data_path ./data/taf_source_data_hartmann_6d_family_k10.pkl \
        --surrogate gp \
        --scales 0.5 0.75 1.0 1.25 1.5 1.75 2.0 \
        --n_variants 20 --n_runs 3 \
        --max_steps 48 --n_init 2 \
        --n_persistent_base 128 --n_total_candidates 256 \
        --k_centers 3 --local_h 0.15 --local_h_decay 0.95 \
        --n_workers 4 \
        --save_dir paper_experiments/ablation_study/wo_rl/hartmann6d
"""

from __future__ import annotations

import os
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
import multiprocessing as mp
import pickle
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

# Bootstrap project root
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.normpath(os.path.join(_this_dir, "..", "..", "..", "MYRL"))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_sweep_dir = os.path.normpath(os.path.join(_this_dir, "..", "..", "..", "MYRL", "scripts"))
if _sweep_dir not in sys.path:
    sys.path.insert(0, _sweep_dir)

import warnings
from sklearn.exceptions import ConvergenceWarning
from linear_operator.utils.warnings import NumericalWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")
warnings.filterwarnings("ignore", message=".*ABNORMAL_TERMINATION_IN_LNSRCH.*")
warnings.filterwarnings("ignore", message=".*scale the data.*")
warnings.filterwarnings("ignore", message=".*optimal value found.*close to the specified.*bound.*")
warnings.filterwarnings("ignore", category=NumericalWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module=r"pfns4bo(\..*)?")


# =========================================================================
# Variant generation — mirrors eval_scale_sweep.py exactly
# =========================================================================
def _get_sample_variant_fn(task_name: str):
    if task_name == "alkox_emulator":
        from myrl.tasks.alkox_emulator import sample_variant_from_spec
        return sample_variant_from_spec
    elif task_name == "benzylation_emulator":
        from myrl.tasks.benzylation_emulator import sample_variant_from_spec
        return sample_variant_from_spec
    elif task_name == "hplc_emulator":
        from myrl.tasks.hplc_emulator import sample_variant_from_spec
        return sample_variant_from_spec
    elif task_name == "hartmann_6d_family":
        from myrl.tasks.hartmann_6d_family import sample_variant_from_spec
        return sample_variant_from_spec
    elif task_name == "branin_family":
        from myrl.tasks.branin_family import sample_variant_from_spec as _branin_svfs
        def _fn(rng, spec):
            branin_spec = {"dx": spec["dx"], "rotation": spec["rot"], "sx": spec["sx"]}
            return _branin_svfs(rng, branin_spec)
        return _fn
    elif task_name == "hartmann_3d_family":
        from myrl.tasks.hartmann_3d_family import sample_variant_from_spec
        return sample_variant_from_spec
    else:
        raise ValueError(f"Unsupported task: {task_name}")


def build_work_plan(task_name, scales, n_variants, n_runs, seed):
    from eval_scale_sweep import make_spec_for_scale
    sample_variant_from_spec = _get_sample_variant_fn(task_name)
    master_rng = np.random.default_rng(seed)
    work_plan = {}
    for scale in scales:
        scale_rng = np.random.default_rng(master_rng.integers(0, 10**9))
        variants_and_seeds = []
        for _ in range(n_variants):
            spec = make_spec_for_scale(task_name, scale)
            vp = sample_variant_from_spec(scale_rng, spec)
            seeds = [int(scale_rng.integers(0, 100000)) for _ in range(n_runs)]
            variants_and_seeds.append((vp, seeds))
        work_plan[scale] = variants_and_seeds
    return work_plan


# =========================================================================
# Shared-pool BO run: all methods see the same candidate pool per step
# =========================================================================
def run_shared_pool_bo(
    func, global_min, policies, X_init, y_init, max_steps,
    surrogate_type, tabpfn_regressor, rng, rl_policy,
    n_total_candidates, k_centers, local_h, local_h_decay,
    explore_fraction=0.0, taf_for_rl=None,
):
    from myrl.eval.eval_rl_new import (
        generate_persistent_adaptive_candidates,
        get_gp_predictions, get_tabpfn_predictions,
        build_state_for_policies, CAP_PPO_NAME,
    )
    from myrl.policies.policies import RLPolicy, TAF

    lower, upper = func.bounds
    bounds = (lower, upper)
    cap_name = CAP_PPO_NAME

    method_states = {}
    for name in policies:
        method_states[name] = {
            "X_context": X_init.copy(),
            "y_context": y_init.copy(),
            "regret_trace": [float(y_init.min() - global_min)],
        }

    rl_policy.reset_persistent_pool(lower, upper, rng)

    for step in range(max_steps):
        cap_X = method_states[cap_name]["X_context"]
        cap_y = method_states[cap_name]["y_context"]

        X_candidates, is_persistent, n_pers_in_cand = generate_persistent_adaptive_candidates(
            lower=lower, upper=upper, X_context=cap_X, y_context=cap_y,
            step=step, rng=rng,
            persistent_pool=rl_policy.persistent_pool,
            persistent_available=rl_policy.persistent_available,
            n_total_candidates=n_total_candidates,
            k_centers=k_centers, local_h=local_h, local_h_decay=local_h_decay,
            explore_fraction=explore_fraction,
        )

        for name, policy in policies.items():
            ms = method_states[name]
            X_ctx, y_ctx = ms["X_context"], ms["y_context"]

            if surrogate_type == 'gp':
                pred_mean, pred_std, model_target = get_gp_predictions(X_ctx, y_ctx, X_candidates)
            elif surrogate_type in {'tabpfn', 'tabpfn_base', 'tabpfn_tuned'}:
                pred_mean, pred_std, model_target = get_tabpfn_predictions(
                    tabpfn_regressor, X_ctx, y_ctx, X_candidates)
            else:
                raise ValueError(f"Unknown surrogate_type: {surrogate_type}")

            state = build_state_for_policies(X_candidates, pred_mean, pred_std, y_ctx, step, max_steps)

            if isinstance(policy, RLPolicy):
                taf_rank_norm = None
                if policy.use_taf_feature and taf_for_rl is not None:
                    taf_scores = taf_for_rl.af(state.numpy(), X_ctx, model_target)
                    ranks = np.argsort(np.argsort(-taf_scores)).astype(np.float32)
                    taf_rank_norm = ranks / max(len(taf_scores) - 1, 1)
                policy.set_context(X_ctx, y_ctx, X_candidates, pred_mean, pred_std,
                                   bounds, step, is_persistent=is_persistent,
                                   taf_rank_norm=taf_rank_norm)
                action, _ = policy.act(state)
            elif isinstance(policy, TAF):
                action, _ = policy.act(state, X_ctx, model_target)
            else:
                action, _ = policy.act(state, rng=np.random.default_rng(rng.integers(2**31)))

            action = int(action.item())
            x_new = X_candidates[action:action+1]
            y_new = func(x_new)[0]
            ms["X_context"] = np.vstack([ms["X_context"], x_new])
            ms["y_context"] = np.concatenate([ms["y_context"], [y_new]])
            ms["regret_trace"].append(float(ms["y_context"].min() - global_min))

        # Update persistent pool based on CAP-PPO's selection
        cap_x_new = method_states[cap_name]["X_context"][-1]
        for idx in range(len(X_candidates)):
            if np.allclose(X_candidates[idx], cap_x_new):
                rl_policy.consume_persistent_point(idx, n_pers_in_cand)
                break

    results = {}
    for name in policies:
        results[name] = {
            "regret_trace": np.array(method_states[name]["regret_trace"], dtype=np.float64),
        }
    return results


# =========================================================================
# Checkpoint helpers
# =========================================================================
def _now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _checkpoint_path(save_dir, scale, v_idx, run_idx):
    tag = f"{scale:.4f}".replace(".", "p", 1)
    return os.path.join(save_dir, "_resume", f"scale_{tag}_v{v_idx:03d}_r{run_idx:02d}.pkl")

def _atomic_pickle_dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)

def _load_checkpoint(path):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


# =========================================================================
# Worker process state & init (for multiprocessing)
# =========================================================================
_wstate: Dict[str, Any] = {}

def _init_worker(args_dict: Dict):
    """Initialize worker process: rebuild policies & task (not picklable)."""
    from myrl.tasks import get_task
    from myrl.eval.eval_rl_new import CAP_PPO_NAME
    from myrl.policies.policies import (
        EI, UCB, PI, TAF, RandomPolicy, RLPolicy, FunBO,
    )

    a = argparse.Namespace(**args_dict)
    task = get_task(a.task)
    dim = task.dim
    device = a.device

    x_feature_names = [f"x{i}" for i in range(dim)]
    feature_order = ["posterior_mean", "posterior_std"] + x_feature_names + ["incumbent", "timestep", "budget"]

    all_policies = {
        "Random": lambda: RandomPolicy(),
        "EI": lambda: EI(feature_order),
        "UCB": lambda: UCB(feature_order, kappa=2.0, D=dim, delta=0.1),
        "PI": lambda: PI(feature_order, xi=0.01),
        "FunBO": lambda: FunBO(feature_order, beta=2.0),
        "TAF_me": lambda: TAF(a.taf_data_path, mode="me") if a.taf_data_path else None,
        "TAF_ranking": lambda: TAF(a.taf_data_path, mode="ranking", rho=1.0) if a.taf_data_path else None,
        CAP_PPO_NAME: lambda: RLPolicy(
            model_path=a.rl_model_path, coord_dim=dim, max_steps=a.max_steps, device=device,
            n_persistent_base=a.n_persistent_base, n_total_candidates=a.n_total_candidates,
            k_centers=a.k_centers, local_h=a.local_h, local_h_decay=a.local_h_decay,
        ),
    }
    methods = a.methods if a.methods else list(all_policies.keys())
    policies = {}
    for m in methods:
        if m in all_policies:
            p = all_policies[m]()
            if p is not None:
                policies[m] = p

    tabpfn = None
    if a.surrogate in ("tabpfn", "tabpfn_base", "tabpfn_tuned"):
        from tabpfn import TabPFNRegressor
        tabpfn = TabPFNRegressor(device=device, n_estimators=8)

    rl_policy = policies[CAP_PPO_NAME]
    taf_for_rl = policies.get("TAF_ranking", None) if rl_policy.use_taf_feature else None

    _wstate.update({
        "task": task, "dim": dim, "policies": policies,
        "method_names": list(policies.keys()),
        "tabpfn": tabpfn, "rl_policy": rl_policy,
        "taf_for_rl": taf_for_rl, "args": a, "pid": os.getpid(),
    })


def _eval_single_work_item(work_item: Dict) -> Dict:
    """Evaluate one (scale, variant, run) — all methods share candidate pool."""
    from myrl.rl.train_rl import TaskVariantObjectiveFunction
    from myrl.eval.eval_rl_new import CAP_PPO_NAME

    ws = _wstate
    a = ws["args"]
    task = ws["task"]
    dim = ws["dim"]
    policies = ws["policies"]
    tabpfn = ws["tabpfn"]
    rl_policy = ws["rl_policy"]
    taf_for_rl = ws["taf_for_rl"]

    scale = work_item["scale"]
    v_idx = work_item["v_idx"]
    run_idx = work_item["run_idx"]
    run_seed = work_item["run_seed"]
    vp = work_item["variant_params"]

    # Check checkpoint
    cp_path = _checkpoint_path(a.save_dir, scale, v_idx, run_idx)
    cp = _load_checkpoint(cp_path)
    if cp is not None and cp.get("complete", False):
        return {
            "scale": scale, "v_idx": v_idx, "run_idx": run_idx,
            "trajectories": {m: np.array(t, dtype=np.float64) for m, t in cp["trajectories"].items()},
            "cached": True,
        }

    func = TaskVariantObjectiveFunction(task_name=a.task, variant_params=vp)
    global_min = float(task.estimate_global_min(vp))
    lower, upper = func.bounds

    init_rng = np.random.default_rng(run_seed)
    X_init = init_rng.uniform(lower, upper, size=(a.n_init, dim)).astype(np.float32)
    y_init = func(X_init)
    run_rng = np.random.default_rng(run_seed)

    t0 = time.time()
    results = run_shared_pool_bo(
        func=func, global_min=global_min, policies=policies,
        X_init=X_init, y_init=y_init, max_steps=a.max_steps,
        surrogate_type=a.surrogate, tabpfn_regressor=tabpfn,
        rng=run_rng, rl_policy=rl_policy,
        n_total_candidates=a.n_total_candidates,
        k_centers=a.k_centers, local_h=a.local_h, local_h_decay=a.local_h_decay,
        explore_fraction=getattr(a, 'explore_fraction', 0.0),
        taf_for_rl=taf_for_rl,
    )
    elapsed = time.time() - t0

    # Save checkpoint
    cp_data = {
        "scale": scale, "v_idx": v_idx, "run_idx": run_idx,
        "trajectories": {m: results[m]["regret_trace"] for m in results},
        "complete": True, "updated_at": _now_str(),
    }
    _atomic_pickle_dump(cp_data, cp_path)

    cap_final = float(results[CAP_PPO_NAME]["regret_trace"][-1])
    print(f"  [pid={ws['pid']}] scale={scale:.2f} v={v_idx:02d} run={run_idx} "
          f"CAP={cap_final:.4f} elapsed={elapsed:.1f}s")

    return {
        "scale": scale, "v_idx": v_idx, "run_idx": run_idx,
        "trajectories": {m: results[m]["regret_trace"] for m in results},
        "cached": False,
    }


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Ablation 1: w/o RL — shared candidate pool eval")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--rl_model_path", type=str, required=True)
    parser.add_argument("--taf_data_path", type=str, default=None)
    parser.add_argument("--surrogate", type=str, default="gp", choices=["gp", "tabpfn_base"])
    parser.add_argument("--scales", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    parser.add_argument("--n_variants", type=int, default=20)
    parser.add_argument("--n_runs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=48)
    parser.add_argument("--n_init", type=int, default=2)
    parser.add_argument("--n_persistent_base", type=int, default=128)
    parser.add_argument("--n_total_candidates", type=int, default=256)
    parser.add_argument("--k_centers", type=int, default=3)
    parser.add_argument("--local_h", type=float, default=0.15)
    parser.add_argument("--local_h_decay", type=float, default=0.95)
    parser.add_argument("--explore_fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["Random", "EI", "UCB", "PI", "TAF_ranking", "CAP-PPO"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_workers", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    from eval_scale_sweep import make_spec_for_scale

    print(f"=== Ablation 1: w/o RL (Shared Candidate Pool) ===")
    print(f"Task:      {args.task}")
    print(f"Model:     {args.rl_model_path}")
    print(f"Surrogate: {args.surrogate}")
    print(f"Methods:   {args.methods}")
    print(f"Scales:    {args.scales}")
    print(f"Variants:  {args.n_variants} x {args.n_runs} runs")
    print(f"Workers:   {args.n_workers}")
    print(f"Save:      {args.save_dir}")
    print()

    # Pre-sample work plan
    work_plan = build_work_plan(args.task, args.scales, args.n_variants,
                                args.n_runs, args.seed)

    # Build flat work items: one per (scale, variant, run)
    all_work_items = []
    for scale in args.scales:
        for v_idx, (vp, run_seeds) in enumerate(work_plan[scale]):
            for run_idx, run_seed in enumerate(run_seeds):
                all_work_items.append({
                    "scale": scale, "v_idx": v_idx, "run_idx": run_idx,
                    "run_seed": run_seed, "variant_params": vp,
                })

    total_items = len(all_work_items)
    print(f"Total work items: {total_items} "
          f"({len(args.scales)} scales x {args.n_variants} variants x {args.n_runs} runs)\n")

    # Execute
    args_dict = vars(args)
    sweep_start = time.time()

    if args.n_workers > 1:
        n_cpus = os.cpu_count() or 1
        threads_per_worker = max(1, n_cpus // args.n_workers)
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                     "NUMEXPR_MAX_THREADS"):
            os.environ.setdefault(var, str(threads_per_worker))
        print(f"Starting {args.n_workers} workers (spawn), "
              f"{threads_per_worker} threads/worker ({n_cpus} CPUs)\n")
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.n_workers,
                      initializer=_init_worker, initargs=(args_dict,)) as pool:
            raw_results = []
            completed = 0
            for result in pool.imap_unordered(_eval_single_work_item, all_work_items):
                completed += 1
                tag = "cached" if result.get("cached") else "done"
                cap_r = float(result["trajectories"].get("CAP-PPO", np.array([0]))[-1])
                print(f"  [{completed}/{total_items}] "
                      f"scale={result['scale']:.2f} v={result['v_idx']:02d} "
                      f"run={result['run_idx']} CAP={cap_r:.4f} [{tag}]")
                raw_results.append(result)
    else:
        print("Sequential mode (n_workers=1)\n")
        _init_worker(args_dict)
        raw_results = []
        completed = 0
        for item in all_work_items:
            result = _eval_single_work_item(item)
            completed += 1
            raw_results.append(result)

    elapsed = time.time() - sweep_start
    print(f"\nAll evaluations done in {elapsed / 60:.1f}m")

    # ------------------------------------------------------------------
    # Aggregate results
    # ------------------------------------------------------------------
    # Figure out method_names from first result
    method_names = list(raw_results[0]["trajectories"].keys())

    all_results = {}
    all_trajectories = {}

    for scale in args.scales:
        sk = str(scale)
        # Collect trajectories: [n_variants][n_runs] = array
        scale_trajs = {m: [[None] * args.n_runs for _ in range(args.n_variants)]
                       for m in method_names}

        for r in raw_results:
            if r["scale"] != scale:
                continue
            v_idx, run_idx = r["v_idx"], r["run_idx"]
            for m in method_names:
                scale_trajs[m][v_idx][run_idx] = np.array(r["trajectories"][m], dtype=np.float64)

        # Convert to ndarray (n_variants, n_runs, n_steps+1)
        scale_trajs_np = {}
        scale_result = {}
        for m in method_names:
            arr = np.array(scale_trajs[m])
            scale_trajs_np[m] = arr
            final_per_variant = arr[:, :, -1].mean(axis=1)
            scale_result[m] = {
                "mean": float(final_per_variant.mean()),
                "std": float(final_per_variant.std()),
                "per_variant": final_per_variant.tolist(),
            }

        all_results[sk] = scale_result
        all_trajectories[sk] = scale_trajs_np

        print(f"\n  --- Scale {scale:.2f} ---")
        for m in method_names:
            r = scale_result[m]
            print(f"    {m:<15} mean={r['mean']:.4f} std={r['std']:.4f}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    pkl_path = os.path.join(args.save_dir, "scale_sweep_data.pkl")
    pkl_data = {
        "task": args.task, "surrogate": args.surrogate,
        "model": args.rl_model_path, "scales": args.scales,
        "n_variants": args.n_variants, "n_runs": args.n_runs,
        "max_steps": args.max_steps, "n_init": args.n_init,
        "seed": args.seed, "method_names": method_names,
        "specs": {str(s): make_spec_for_scale(args.task, s) for s in args.scales},
        "results": all_results,
        "trajectories": all_trajectories,
        "ablation": "wo_rl_shared_pool",
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(pkl_data, f)
    print(f"\nSaved: {pkl_path}")

    def _make_json_safe(obj):
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, dict):
            return {k: _make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_make_json_safe(v) for v in obj]
        return obj

    json_path = os.path.join(args.save_dir, "scale_sweep_results.json")
    json_out = {
        "task": args.task, "surrogate": args.surrogate,
        "scales": args.scales, "n_variants": args.n_variants,
        "n_runs": args.n_runs, "ablation": "wo_rl_shared_pool",
        "results": all_results,
    }
    with open(json_path, "w") as f:
        json.dump(_make_json_safe(json_out), f, indent=2)
    print(f"Saved: {json_path}")

    # Auto replot
    replot_dir = os.path.join(args.save_dir, "replot")
    os.makedirs(replot_dir, exist_ok=True)
    try:
        from plot_scale_sweep import plot_sweep, plot_sweep_highlight
        plot_sweep(pkl_data, replot_dir, normalize=True)
        plot_sweep_highlight(pkl_data, replot_dir, normalize=True)
        print(f"Plots saved to {replot_dir}/")
    except Exception as e:
        print(f"  [WARN] Plot generation failed: {e}")
        print(f"  Manual: python MYRL/scripts/plot_scale_sweep.py --pkl {pkl_path} --save_dir {replot_dir}")


if __name__ == "__main__":
    main()
