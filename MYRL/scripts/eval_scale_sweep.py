"""
eval_scale_sweep.py — Scale sweep evaluation for variant-family tasks.

Evaluates a trained CAP-PPO model across different variant "scale" levels.
For each scale, variant transformation parameters (dx, rot, sx) are scaled
proportionally from the base in_range spec.  This produces a
**regret vs. scale** curve showing at what similarity level CAP-PPO
maintains its advantage over classical baselines.

Supports parallel evaluation via --n_workers.

Usage:
    python MYRL/scripts/eval_scale_sweep.py \
        --task alkox_emulator \
        --rl_model_path ./runs/ppo_alkox_emulator_bnn_kl001/ppo_final.pt \
        --taf_data_path ./data/taf_source_data_alkox_emulator_k10_transform.pkl \
        --scales 0.5 0.75 1.0 1.25 1.5 1.75 2.0 \
        --n_variants 20 --n_runs 3 \
        --n_workers 4 \
        --save_dir ./results_policies/alkox_emulator_scale_sweep
"""

from __future__ import annotations

# MUST be set before TF is imported (including in spawned workers)
import os
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import contextlib
import json
import multiprocessing as mp
import pickle
import shutil
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Bootstrap project root
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.normpath(os.path.join(_this_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from myrl.tasks import get_task
from myrl.rl.train_rl import TaskVariantObjectiveFunction
from myrl.eval.eval_rl_new import (
    run_bo_with_policy,
    generate_mixed_candidates,
    generate_persistent_adaptive_candidates,
    get_gp_predictions,
    get_tabpfn_predictions,
    build_state_for_policies,
    CAP_PPO_NAME,
)
from myrl.policies.policies import EI, UCB, PI, TAF, RandomPolicy, RLPolicy, FunBO, PFNs4BOPolicy, TuRBOPolicy

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


# ---------------------------------------------------------------------------
# Scale → variant spec
# ---------------------------------------------------------------------------
def _get_base_in_range_spec(task_name: str) -> Dict[str, List[Tuple[float, float]]]:
    """Return the task-specific base in-range spec used as scale=1.0.

    The returned dict always uses normalised keys: "dx", "rot", "sx".
    (branin_family internally uses "rotation"; we remap it here.)
    """
    if task_name == "alkox_emulator":
        from myrl.tasks.alkox_emulator import _VARIANT_SUITE_SPECS as suite_specs
    elif task_name == "benzylation_emulator":
        from myrl.tasks.benzylation_emulator import _VARIANT_SUITE_SPECS as suite_specs
    elif task_name == "hplc_emulator":
        from myrl.tasks.hplc_emulator import _VARIANT_SUITE_SPECS as suite_specs
    elif task_name == "hartmann_6d_family":
        from myrl.tasks.hartmann_6d_family import _VARIANT_SUITE_SPECS as suite_specs
    elif task_name == "ackley_5d_family":
        from myrl.tasks.ackley_family import _VARIANT_SUITE_SPECS_5D as suite_specs
    elif task_name == "ackley_10d_family":
        from myrl.tasks.ackley_family import _VARIANT_SUITE_SPECS_10D as suite_specs
    elif task_name == "branin_family":
        from myrl.tasks.branin_family import _VARIANT_SUITE_SPECS as suite_specs
        base = suite_specs["in_range"]
        # branin uses "rotation" key; normalise to "rot"
        return {"dx": base["dx"], "rot": base["rotation"], "sx": base["sx"]}
    elif task_name == "hartmann_3d_family":
        from myrl.tasks.hartmann_3d_family import _VARIANT_SUITE_SPECS as suite_specs
    else:
        raise ValueError(f"Unsupported task for scale sweep: {task_name}")
    return suite_specs["in_range"]


def make_spec_for_scale(
    task_name: str, scale: float
) -> Dict[str, List[Tuple[float, float]]]:
    """Generate a variant spec by scaling the task's base in-range parameters."""
    base_spec = _get_base_in_range_spec(task_name)
    dx_max = max(max(abs(float(lo)), abs(float(hi))) for lo, hi in base_spec["dx"])
    rot_max = max(max(abs(float(lo)), abs(float(hi))) for lo, hi in base_spec["rot"])
    sx_min_base = min(float(lo) for lo, _ in base_spec["sx"])
    sx_max_base = max(float(hi) for _, hi in base_spec["sx"])
    sx_low_delta = max(0.0, 1.0 - sx_min_base)
    sx_high_delta = max(0.0, sx_max_base - 1.0)

    dx_max *= scale
    rot_max *= scale
    sx_min = max(0.60, 1.0 - sx_low_delta * scale)
    sx_max = 1.0 + sx_high_delta * scale
    return {
        "dx": [(-dx_max, dx_max)],
        "rot": [(-rot_max, rot_max)],
        "sx": [(sx_min, sx_max)],
    }


def spec_summary(spec: Dict) -> str:
    dx_lo, dx_hi = spec["dx"][0]
    rot_lo, rot_hi = spec["rot"][0]
    sx_lo, sx_hi = spec["sx"][0]
    return f"dx ±{dx_hi:.3f}, rot ±{rot_hi:.1f}°, sx [{sx_lo:.2f}, {sx_hi:.2f}]"


# ---------------------------------------------------------------------------
# Policy config builder (shared between main and worker)
# ---------------------------------------------------------------------------
def _build_policy_configs(args_ns) -> Dict[str, Dict]:
    _bl = {"n_candidates": args_ns.n_candidates_baseline,
           "n_global": args_ns.n_candidates_baseline, "k_centers": 0}
    return {
        "Random": {"n_candidates": args_ns.n_total_candidates,
                    "n_global": args_ns.n_total_candidates, "k_centers": 0},
        "EI": dict(_bl),
        "UCB": dict(_bl),
        "PI": dict(_bl),
        "FunBO": dict(_bl),
        "PFNs4BO": dict(_bl),
        "TuRBO": {"n_candidates": 0, "n_global": 0, "k_centers": 0},
        "TAF_me": dict(_bl),
        "TAF_ranking": dict(_bl),
        CAP_PPO_NAME: {
            "n_candidates": args_ns.n_total_candidates,
            "n_global": 0,
            "k_centers": args_ns.k_centers,
            "n_persistent_base": args_ns.n_persistent_base,
            "n_total_candidates": args_ns.n_total_candidates,
        },
    }

# ---------------------------------------------------------------------------
# Worker process state & functions  (for parallel execution)
# ---------------------------------------------------------------------------
_wstate: Dict[str, Any] = {}


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _fmt_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60.0)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(minutes)}m {sec:.1f}s"


def _should_log_progress(detail: str, needed: str) -> bool:
    order = {"summary": 0, "variant": 1, "run": 2, "method": 3}
    return order.get(detail, 0) >= order.get(needed, 0)


def _progress_log(detail: str, needed: str, message: str):
    if _should_log_progress(detail, needed):
        print(f"[{_now_str()}] {message}", flush=True)


def _scale_tag(scale: float) -> str:
    return f"{scale:.4f}".replace("-", "m").replace(".", "p")


def _checkpoint_dir(save_dir: str) -> str:
    return os.path.join(save_dir, "_resume")


def _checkpoint_path(save_dir: str, scale: float, v_idx: int) -> str:
    return os.path.join(_checkpoint_dir(save_dir), f"scale_{_scale_tag(scale)}_v{v_idx:03d}.pkl")


def _atomic_pickle_dump(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp_path, path)


def _safe_remove(path: str):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _variant_signature(variant_params: Dict, run_seeds: List[int]) -> str:
    payload = {
        "variant_params": variant_params,
        "run_seeds": list(run_seeds),
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _init_variant_checkpoint(
    *,
    scale: float,
    v_idx: int,
    variant_params: Dict,
    run_seeds: List[int],
    method_names: List[str],
) -> Dict[str, Any]:
    n_runs = len(run_seeds)
    return {
        "version": 1,
        "complete": False,
        "scale": scale,
        "v_idx": v_idx,
        "variant_signature": _variant_signature(variant_params, run_seeds),
        "variant_params": variant_params,
        "run_seeds": list(run_seeds),
        "global_min": None,
        "trajectories": {m: [None] * n_runs for m in method_names},
        "run_details": {m: [None] * n_runs for m in method_names},
        "updated_at": _now_str(),
    }


def _normalize_checkpoint_methods(state: Dict[str, Any], method_names: List[str], n_runs: int):
    trajs = state.setdefault("trajectories", {})
    details = state.setdefault("run_details", {})
    for m in method_names:
        if m not in trajs or len(trajs[m]) != n_runs:
            trajs[m] = [None] * n_runs
        if m not in details or len(details[m]) != n_runs:
            details[m] = [None] * n_runs


def _is_method_run_complete(state: Dict[str, Any], method_name: str, run_idx: int) -> bool:
    return (
        method_name in state["trajectories"]
        and run_idx < len(state["trajectories"][method_name])
        and state["trajectories"][method_name][run_idx] is not None
        and state["run_details"][method_name][run_idx] is not None
    )


def _count_completed_method_runs(state: Dict[str, Any], method_names: List[str], n_runs: int) -> int:
    total = 0
    for m in method_names:
        for run_idx in range(n_runs):
            if _is_method_run_complete(state, m, run_idx):
                total += 1
    return total


def _checkpoint_complete_for_selection(state: Dict[str, Any], method_names: List[str], n_runs: int) -> bool:
    return _count_completed_method_runs(state, method_names, n_runs) == len(method_names) * n_runs


def _checkpoint_to_result(state: Dict[str, Any], method_names: List[str]) -> Dict[str, Any]:
    trajectories = {
        m: [np.asarray(t, dtype=np.float64) for t in state["trajectories"][m]]
        for m in method_names
    }
    run_details = {
        m: list(state["run_details"][m])
        for m in method_names
    }
    final_regrets = {
        m: float(np.mean([traj[-1] for traj in trajectories[m]]))
        for m in method_names
    }
    return {
        "scale": state["scale"],
        "v_idx": state["v_idx"],
        "global_min": float(state["global_min"]),
        "trajectories": trajectories,
        "run_details": run_details,
        "final_regrets": final_regrets,
    }


def _load_variant_checkpoint(
    *,
    checkpoint_path: str,
    scale: float,
    v_idx: int,
    variant_params: Dict,
    run_seeds: List[int],
    method_names: List[str],
) -> Dict[str, Any] | None:
    if not os.path.exists(checkpoint_path):
        return None
    with open(checkpoint_path, "rb") as f:
        state = pickle.load(f)
    expected_signature = _variant_signature(variant_params, run_seeds)
    if (
        state.get("version") != 1
        or state.get("scale") != scale
        or state.get("v_idx") != v_idx
        or state.get("variant_signature") != expected_signature
    ):
        return None
    _normalize_checkpoint_methods(state, method_names, len(run_seeds))
    return state


@contextlib.contextmanager
def _maybe_silence_third_party_output(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def _init_worker(args_dict: Dict):
    """Called once per worker process.  Initializes task, models, policies."""
    # Ensure TF doesn't preallocate all GPU memory in this worker
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import torch as _torch
    from tabpfn import TabPFNRegressor as _TabPFNRegressor

    # Respect thread limits inherited from parent (set before spawn)
    _tpw = os.environ.get("OMP_NUM_THREADS")
    if _tpw is not None:
        _torch.set_num_threads(int(_tpw))

    # Suppress warnings in workers too
    import warnings as _w
    from sklearn.exceptions import ConvergenceWarning as _CW
    from linear_operator.utils.warnings import NumericalWarning as _NW
    _w.filterwarnings("ignore", category=_CW)
    _w.filterwarnings("ignore", message=".*lbfgs failed to converge.*")
    _w.filterwarnings("ignore", message=".*ABNORMAL_TERMINATION_IN_LNSRCH.*")
    _w.filterwarnings("ignore", message=".*scale the data.*")
    _w.filterwarnings("ignore", message=".*optimal value found.*close to the specified.*bound.*")
    _w.filterwarnings("ignore", category=_NW)
    _w.filterwarnings("ignore", category=FutureWarning, module=r"pfns4bo(\..*)?")

    a = SimpleNamespace(**args_dict)
    device = "cuda" if _torch.cuda.is_available() else "cpu"

    task = get_task(a.task)
    dim = int(task.dim)

    x_feature_names = [f"x{i}" for i in range(dim)]
    feature_order = (["posterior_mean", "posterior_std"]
                     + x_feature_names
                     + ["incumbent", "timestep", "budget"])

    all_policies = {
        "Random": lambda: RandomPolicy(),
        "EI": lambda: EI(feature_order),
        "UCB": lambda: UCB(feature_order, kappa=2.0, D=dim, delta=0.1),
        "PI": lambda: PI(feature_order, xi=0.01),
        "FunBO": lambda: FunBO(feature_order, beta=1.0),
        "PFNs4BO": lambda: PFNs4BOPolicy(device=device),
        "TuRBO": lambda: TuRBOPolicy(device=device),
        "TAF_me": lambda: TAF(a.taf_data_path, mode="me"),
        "TAF_ranking": lambda: TAF(a.taf_data_path, mode="ranking", rho=1.0),
        CAP_PPO_NAME: lambda: RLPolicy(
            model_path=a.rl_model_path,
            coord_dim=dim,
            hidden_dim=a.hidden_dim,
            n_self_attn_layers=a.n_self_attn_layers,
            n_cross_attn_layers=a.n_cross_attn_layers,
            n_heads=a.n_heads,
            max_steps=a.max_steps,
            device=device,
            n_persistent_base=a.n_persistent_base,
            n_total_candidates=a.n_total_candidates,
            k_centers=a.k_centers,
            local_h=a.local_h,
            local_h_decay=a.local_h_decay,
        ),
    }
    selected = a.methods if a.methods else list(all_policies.keys())
    # Always include TAF_ranking if CAP-PPO uses taf_feature and it's not selected
    if CAP_PPO_NAME in selected and "TAF_ranking" not in selected:
        selected.append("TAF_ranking")
    policies = {name: all_policies[name]() for name in selected if name in all_policies}

    tabpfn_regressor = None
    if a.surrogate == "tabpfn_base":
        tabpfn_regressor = _TabPFNRegressor(
            device=device,
            n_estimators=1,
            random_state=42,
            inference_precision=_torch.float32,
            ignore_pretraining_limits=True,
        )

    _wstate.update({
        "task": task,
        "dim": dim,
        "policies": policies,
        "method_names": list(policies.keys()),
        "policy_configs": _build_policy_configs(a),
        "tabpfn": tabpfn_regressor,
        "device": device,
        "args": a,
        "pid": os.getpid(),
    })


def _eval_single_variant(work_item: Dict) -> Dict:
    """Evaluate one variant — all methods × all runs.

    Called by pool.map (parallel) or directly (sequential).
    """
    variant_params = work_item["variant_params"]
    run_seeds = work_item["run_seeds"]
    scale = work_item["scale"]
    v_idx = work_item["v_idx"]

    ws = _wstate
    a = ws["args"]
    task = ws["task"]
    dim = ws["dim"]
    policies = ws["policies"]
    method_names = ws["method_names"]
    policy_configs = ws["policy_configs"]
    tabpfn = ws["tabpfn"]
    device = ws["device"]
    progress_detail = getattr(a, "progress_detail", "summary")
    suppress_third_party_output = getattr(a, "suppress_third_party_output", True)
    pid = ws["pid"]
    checkpoint_path = _checkpoint_path(a.save_dir, scale, v_idx)

    func = TaskVariantObjectiveFunction(task_name=a.task, variant_params=variant_params)
    global_min = float(task.estimate_global_min(variant_params))
    lower, upper = func.bounds
    variant_start = time.time()
    n_methods = len(method_names)
    n_runs = len(run_seeds)

    state = _load_variant_checkpoint(
        checkpoint_path=checkpoint_path,
        scale=scale,
        v_idx=v_idx,
        variant_params=variant_params,
        run_seeds=run_seeds,
        method_names=method_names,
    )
    if state is None:
        state = _init_variant_checkpoint(
            scale=scale,
            v_idx=v_idx,
            variant_params=variant_params,
            run_seeds=run_seeds,
            method_names=method_names,
        )
        state["global_min"] = global_min
    else:
        state["global_min"] = global_min
        completed_method_runs = _count_completed_method_runs(state, method_names, n_runs)
        _progress_log(
            progress_detail,
            "variant",
            f"[pid={pid}] resume found   scale={scale:.2f} v={v_idx:02d} "
            f"completed={completed_method_runs}/{n_runs * n_methods} method-runs",
        )
        if _checkpoint_complete_for_selection(state, method_names, n_runs):
            _progress_log(
                progress_detail,
                "variant",
                f"[pid={pid}] variant cached scale={scale:.2f} v={v_idx:02d} "
                f"loaded from {_checkpoint_path(a.save_dir, scale, v_idx)}",
            )
            state["complete"] = True
            return _checkpoint_to_result(state, method_names)

    _progress_log(
        progress_detail,
        "variant",
        f"[pid={pid}] variant start  scale={scale:.2f} v={v_idx:02d} "
        f"runs={len(run_seeds)} methods={n_methods} gmin={global_min:.4f}",
    )

    for run_idx, run_seed in enumerate(run_seeds):
        run_start = time.time()
        init_rng = np.random.default_rng(run_seed)
        X_init = init_rng.uniform(lower, upper, size=(a.n_init, dim)).astype(np.float32)
        y_init = func(X_init)

        _progress_log(
            progress_detail,
            "run",
            f"[pid={pid}] run start      scale={scale:.2f} v={v_idx:02d} "
            f"run={run_idx + 1}/{len(run_seeds)} seed={run_seed}",
        )

        for policy_name, policy in policies.items():
            if _is_method_run_complete(state, policy_name, run_idx):
                cached_regret = float(np.asarray(state["trajectories"][policy_name][run_idx])[-1])
                _progress_log(
                    progress_detail,
                    "method",
                    f"[pid={pid}] method cached   scale={scale:.2f} v={v_idx:02d} "
                    f"run={run_idx + 1}/{len(run_seeds)} method={policy_name} "
                    f"final_regret={cached_regret:.4f}",
                )
                continue

            policy_start = time.time()
            run_rng = np.random.default_rng(run_seed)
            pc = policy_configs[policy_name]

            _taf_for_rl = None
            if isinstance(policy, RLPolicy) and policy.use_taf_feature:
                _taf_for_rl = policies.get("TAF_ranking", None)

            _progress_log(
                progress_detail,
                "method",
                f"[pid={pid}] method start   scale={scale:.2f} v={v_idx:02d} "
                f"run={run_idx + 1}/{len(run_seeds)} method={policy_name}",
            )

            with _maybe_silence_third_party_output(suppress_third_party_output):
                trace = run_bo_with_policy(
                    func=func,
                    global_min=global_min,
                    policy=policy,
                    policy_name=policy_name,
                    X_init=X_init,
                    y_init=y_init,
                    max_steps=a.max_steps,
                    surrogate_type=a.surrogate,
                    tabpfn_regressor=tabpfn,
                    rng=run_rng,
                    n_candidates=pc["n_candidates"],
                    n_global=pc.get("n_global", 0),
                    k_centers=pc["k_centers"],
                    local_h=a.local_h,
                    local_h_decay=a.local_h_decay,
                    device=device,
                    return_trace=True,
                    n_persistent_base=pc.get("n_persistent_base", 0),
                    n_total_candidates=pc.get("n_total_candidates", 0),
                    taf_for_rl=_taf_for_rl,
                    explore_fraction=getattr(a, 'explore_fraction', 0.0),
                )

            regrets = np.asarray(trace["regret_trace"], dtype=np.float64)
            state["trajectories"][policy_name][run_idx] = regrets
            state["run_details"][policy_name][run_idx] = trace
            state["updated_at"] = _now_str()
            state["complete"] = _checkpoint_complete_for_selection(state, method_names, n_runs)
            _atomic_pickle_dump(state, checkpoint_path)

            _progress_log(
                progress_detail,
                "method",
                f"[pid={pid}] method done    scale={scale:.2f} v={v_idx:02d} "
                f"run={run_idx + 1}/{len(run_seeds)} method={policy_name} "
                f"final_regret={regrets[-1]:.4f} elapsed={_fmt_elapsed(time.time() - policy_start)} "
                f"saved={checkpoint_path}",
            )

        _progress_log(
            progress_detail,
            "run",
            f"[pid={pid}] run done       scale={scale:.2f} v={v_idx:02d} "
            f"run={run_idx + 1}/{len(run_seeds)} elapsed={_fmt_elapsed(time.time() - run_start)}",
        )

    state["complete"] = _checkpoint_complete_for_selection(state, method_names, n_runs)
    state["updated_at"] = _now_str()
    _atomic_pickle_dump(state, checkpoint_path)

    _progress_log(
        progress_detail,
        "variant",
        f"[pid={pid}] variant done   scale={scale:.2f} v={v_idx:02d} "
        f"elapsed={_fmt_elapsed(time.time() - variant_start)} "
        f"checkpoint={checkpoint_path}",
    )

    return _checkpoint_to_result(state, method_names)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Scale sweep evaluation for variant-family tasks")

    # Task & model
    parser.add_argument("--task", type=str, default="alkox_emulator")
    parser.add_argument("--rl_model_path", type=str, required=True)
    parser.add_argument("--taf_data_path", type=str, required=True)

    # Scale sweep
    parser.add_argument("--scales", type=float, nargs="+",
                        default=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    parser.add_argument("--n_variants", type=int, default=20)
    parser.add_argument("--n_runs", type=int, default=3)

    # BO settings
    parser.add_argument("--max_steps", type=int, default=28)
    parser.add_argument("--n_init", type=int, default=2)
    parser.add_argument("--surrogate", type=str, default="tabpfn_base",
                        choices=["gp", "tabpfn_base"])

    # Candidate generation (match training config)
    parser.add_argument("--n_persistent_base", type=int, default=128)
    parser.add_argument("--n_total_candidates", type=int, default=192)
    parser.add_argument("--k_centers", type=int, default=2)
    parser.add_argument("--local_h", type=float, default=0.17)
    parser.add_argument("--local_h_decay", type=float, default=0.95)
    parser.add_argument("--explore_fraction", type=float, default=0.0,
                        help="Fraction of fresh candidates for exploration (0=off)")
    parser.add_argument("--n_candidates_baseline", type=int, default=2048)

    # Network arch (must match training)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_self_attn_layers", type=int, default=3)
    parser.add_argument("--n_cross_attn_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=8)

    # Parallelism
    parser.add_argument("--n_workers", type=int, default=1,
                        help="Number of parallel worker processes. "
                             "Each worker loads its own TabPFN + RL model on GPU. "
                             "Set based on GPU memory (each worker ~2-3 GB).")

    # Method selection
    parser.add_argument("--methods", type=str, nargs="+", default=None,
                        help="Methods to evaluate. Default: all 7. "
                             "Example: --methods EI TAF_ranking CAP-PPO")

    # Other
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save_dir", type=str, default="./results_policies/scale_sweep")
    parser.add_argument("--show_transform_params", action="store_true",
                        help="Show the top x-axis with dx/rot/sx labels.")
    parser.add_argument(
        "--progress_detail",
        type=str,
        default="summary",
        choices=["summary", "variant", "run", "method"],
        help="How much inner-loop progress to print. "
             "'summary' prints only per-variant completion; "
             "'variant' adds variant start/end; "
             "'run' adds per-run logs; "
             "'method' adds per-method start/end logs.",
    )
    parser.add_argument(
        "--show_third_party_output",
        action="store_true",
        help="Do not silence stdout/stderr noise from third-party libraries.",
    )
    parser.add_argument(
        "--fresh_run",
        action="store_true",
        help="Ignore and delete any existing per-variant resume checkpoints before evaluation starts.",
    )

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 80)
    print("Scale Sweep Evaluation")
    print("=" * 80)
    print(f"Task: {args.task}")
    print(f"Model: {args.rl_model_path}")
    print(f"Surrogate: {args.surrogate}")
    print(f"Scales: {args.scales}")
    print(f"Variants/scale: {args.n_variants}, Runs/variant: {args.n_runs}")
    print(f"BO: {args.n_init} init + {args.max_steps} steps = {args.n_init + args.max_steps} evals")
    print(f"Methods: {args.methods or 'all'}")
    print(f"Device: {device}")
    print(f"Workers: {args.n_workers}")
    print(f"Save: {args.save_dir}")
    print(f"Progress detail: {args.progress_detail}")
    print(f"Third-party output: {'shown' if args.show_third_party_output else 'suppressed'}")
    print(f"Resume: {'disabled (fresh run)' if args.fresh_run else 'enabled'}")
    print()

    args.suppress_third_party_output = not args.show_third_party_output
    resume_dir = _checkpoint_dir(args.save_dir)
    if args.fresh_run and os.path.isdir(resume_dir):
        shutil.rmtree(resume_dir)
        print(f"Removed existing resume checkpoints: {resume_dir}")
    os.makedirs(resume_dir, exist_ok=True)
    print(f"Resume checkpoint dir: {resume_dir}")
    print()

    for s in args.scales:
        spec = make_spec_for_scale(args.task, s)
        print(f"  scale={s:.2f}: {spec_summary(spec)}")
    print()

    # ------------------------------------------------------------------
    # 1. Import task-specific sample function (main process)
    # ------------------------------------------------------------------
    if args.task == "alkox_emulator":
        from myrl.tasks.alkox_emulator import sample_variant_from_spec
    elif args.task == "benzylation_emulator":
        from myrl.tasks.benzylation_emulator import sample_variant_from_spec
    elif args.task == "hplc_emulator":
        from myrl.tasks.hplc_emulator import sample_variant_from_spec
    elif args.task == "hartmann_6d_family":
        from myrl.tasks.hartmann_6d_family import sample_variant_from_spec
    elif args.task == "ackley_5d_family":
        from myrl.tasks.ackley_family import sample_variant_from_spec as _ackley_svfs
        def sample_variant_from_spec(rng, spec):
            return _ackley_svfs(rng, spec, dim=5)
    elif args.task == "ackley_10d_family":
        from myrl.tasks.ackley_family import sample_variant_from_spec as _ackley_svfs
        def sample_variant_from_spec(rng, spec):
            return _ackley_svfs(rng, spec, dim=10)
    elif args.task == "branin_family":
        from myrl.tasks.branin_family import sample_variant_from_spec as _branin_svfs
        def sample_variant_from_spec(rng, spec):
            # make_spec_for_scale outputs "rot"; branin expects "rotation"
            branin_spec = {"dx": spec["dx"], "rotation": spec["rot"], "sx": spec["sx"]}
            return _branin_svfs(rng, branin_spec)
    elif args.task == "hartmann_3d_family":
        from myrl.tasks.hartmann_3d_family import sample_variant_from_spec
    else:
        raise ValueError(f"Unsupported task for scale sweep: {args.task}")

    # ------------------------------------------------------------------
    # 2. Pre-sample ALL variants and run seeds (deterministic, main proc)
    # ------------------------------------------------------------------
    master_rng = np.random.default_rng(args.seed)

    # work_plan[scale] = [(variant_params, [run_seeds]), ...]
    work_plan: Dict[float, List[Tuple[Dict, List[int]]]] = {}
    for scale in args.scales:
        scale_rng = np.random.default_rng(master_rng.integers(0, 10**9))
        variants_and_seeds = []
        for _ in range(args.n_variants):
            spec = make_spec_for_scale(args.task, scale)
            vp = sample_variant_from_spec(scale_rng, spec)
            seeds = [int(scale_rng.integers(0, 100000)) for _ in range(args.n_runs)]
            variants_and_seeds.append((vp, seeds))
        work_plan[scale] = variants_and_seeds

    total_variants = sum(len(v) for v in work_plan.values())
    print(f"Pre-sampled {total_variants} variant evaluations across {len(args.scales)} scales\n")

    # ------------------------------------------------------------------
    # 3. Build work items
    # ------------------------------------------------------------------
    # Flatten into per-variant work items, grouped by scale for output
    all_work_items: List[Dict] = []
    for scale in args.scales:
        for v_idx, (vp, seeds) in enumerate(work_plan[scale]):
            all_work_items.append({
                "scale": scale,
                "v_idx": v_idx,
                "variant_params": vp,
                "run_seeds": seeds,
            })

    # ------------------------------------------------------------------
    # 4. Execute (parallel or sequential)
    # ------------------------------------------------------------------
    args_dict = vars(args)
    sweep_start = time.time()

    if args.n_workers > 1:
        # Auto-limit threads per worker to avoid CPU over-subscription.
        # Each worker inherits env vars set here (spawn mode).
        n_cpus = os.cpu_count() or 1
        threads_per_worker = max(1, n_cpus // args.n_workers)
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                     "NUMEXPR_MAX_THREADS"):
            os.environ.setdefault(var, str(threads_per_worker))
        print(f"Starting {args.n_workers} worker processes (spawn), "
              f"{threads_per_worker} threads/worker ({n_cpus} CPUs) ...\n")
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=args.n_workers,
            initializer=_init_worker,
            initargs=(args_dict,),
        ) as pool:
            # Use imap_unordered for streaming progress
            completed = 0
            raw_results: List[Dict] = []
            for result in pool.imap_unordered(_eval_single_variant, all_work_items):
                completed += 1
                s, vi = result["scale"], result["v_idx"]
                cap_r = result["final_regrets"].get(CAP_PPO_NAME, -1)
                ei_r = result["final_regrets"].get("EI", -1)
                print(f"  [{completed}/{total_variants}] "
                      f"scale={s:.2f} v={vi:02d}  "
                      f"{CAP_PPO_NAME}={cap_r:.3f}  EI={ei_r:.3f}  "
                      f"gmin={result['global_min']:.4f}")
                raw_results.append(result)
    else:
        # Sequential — initialize in main process
        print("Sequential mode (n_workers=1)\n")
        _init_worker(args_dict)
        raw_results = []
        completed = 0
        for item in all_work_items:
            result = _eval_single_variant(item)
            completed += 1
            s, vi = result["scale"], result["v_idx"]
            cap_r = result["final_regrets"].get(CAP_PPO_NAME, -1)
            ei_r = result["final_regrets"].get("EI", -1)
            print(f"  [{completed}/{total_variants}] "
                  f"scale={s:.2f} v={vi:02d}  "
                  f"{CAP_PPO_NAME}={cap_r:.3f}  EI={ei_r:.3f}  "
                  f"gmin={result['global_min']:.4f}")
            raw_results.append(result)

    elapsed = time.time() - sweep_start

    # ------------------------------------------------------------------
    # 5. Aggregate results by scale
    # ------------------------------------------------------------------
    # Get method_names from first result (consistent across all)
    method_names = list(raw_results[0]["final_regrets"].keys())

    all_results: Dict[str, Dict] = {}
    all_trajectories: Dict[str, Dict] = {}
    all_run_details: Dict[str, Dict] = {}

    for scale in args.scales:
        sk = str(scale)
        # Gather results for this scale, sorted by v_idx for determinism
        scale_items = sorted(
            [r for r in raw_results if r["scale"] == scale],
            key=lambda r: r["v_idx"],
        )

        # Per-method: collect trajectories and final regrets
        scale_regrets: Dict[str, List[float]] = {m: [] for m in method_names}
        scale_trajectories: Dict[str, List[List[np.ndarray]]] = {m: [] for m in method_names}
        scale_run_details: Dict[str, List[List[Dict[str, Any]]]] = {m: [] for m in method_names}

        for item in scale_items:
            for m in method_names:
                scale_regrets[m].append(item["final_regrets"][m])
                scale_trajectories[m].append(item["trajectories"][m])
                scale_run_details[m].append(item["run_details"][m])

        # Summary stats
        scale_result = {}
        for m in method_names:
            arr = np.array(scale_regrets[m])
            scale_result[m] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "per_variant": arr.tolist(),
            }
        all_results[sk] = scale_result

        # Trajectories: [n_variants, n_runs, n_steps+1]
        scale_trajs = {}
        for m in method_names:
            scale_trajs[m] = np.array(scale_trajectories[m])
        all_trajectories[sk] = scale_trajs
        all_run_details[sk] = scale_run_details

        # Print scale summary
        print(f"\n  --- Scale {scale:.2f} Summary ---")
        for m in method_names:
            r = scale_result[m]
            print(f"    {m}: {r['mean']:.4f} ± {r['std']:.4f}")

    print(f"\n{'=' * 80}")
    print(f"Scale sweep completed in {elapsed / 3600:.1f}h {(elapsed % 3600) / 60:.0f}m")
    if args.n_workers > 1:
        print(f"  ({args.n_workers} workers, "
              f"~{elapsed / total_variants:.1f}s/variant effective, "
              f"~{elapsed * args.n_workers / total_variants:.1f}s/variant per-worker)")
    print(f"{'=' * 80}")

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    # JSON (summary only — no trajectories)
    json_path = os.path.join(args.save_dir, "scale_sweep_results.json")
    save_data = {
        "task": args.task,
        "surrogate": args.surrogate,
        "model": args.rl_model_path,
        "scales": args.scales,
        "n_variants": args.n_variants,
        "n_runs": args.n_runs,
        "max_steps": args.max_steps,
        "n_init": args.n_init,
        "seed": args.seed,
        "n_workers": args.n_workers,
        "elapsed_seconds": elapsed,
        "resume_dir": resume_dir,
        "specs": {str(s): make_spec_for_scale(args.task, s) for s in args.scales},
        "results": all_results,
    }

    def _make_json_safe(obj):
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, dict):
            return {k: _make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_make_json_safe(v) for v in obj]
        return obj

    with open(json_path, "w") as f:
        json.dump(_make_json_safe(save_data), f, indent=2)
    print(f"Results saved to {json_path}")

    # Pickle (full data including trajectories — for re-plotting)
    pkl_path = os.path.join(args.save_dir, "scale_sweep_data.pkl")
    pkl_data = {
        "task": args.task,
        "surrogate": args.surrogate,
        "model": args.rl_model_path,
        "scales": args.scales,
        "n_variants": args.n_variants,
        "n_runs": args.n_runs,
        "max_steps": args.max_steps,
        "n_init": args.n_init,
        "seed": args.seed,
        "method_names": method_names,
        "resume_dir": resume_dir,
        "specs": {str(s): make_spec_for_scale(args.task, s) for s in args.scales},
        "results": all_results,
        "trajectories": all_trajectories,
        "run_details": all_run_details,
        "raw_results": raw_results,
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(pkl_data, f)
    print(f"Full data (incl. trajectories) saved to {pkl_path}")

    # ------------------------------------------------------------------
    # 7. Plots
    # ------------------------------------------------------------------
    _plot_sweep(args.scales, all_results, method_names, args, args.save_dir)
    _plot_trajectory_grid(args.scales, all_trajectories, method_names, args, args.save_dir)


# ---------------------------------------------------------------------------
# Paper-quality plot constants
# ---------------------------------------------------------------------------
_PAPER_COLORS = {
    CAP_PPO_NAME: "#E24A33",   # red (ours)
    "EI":          "#348ABD",   # blue
    "UCB":         "#467821",   # green
    "PI":          "#988ED5",   # purple
    "TAF_ranking": "#FBC15E",   # orange
    "TAF_me":      "#8C6D31",   # brown
    "PFNs4BO":     "#00CED1",   # cyan
    "TuRBO":       "#E755BA",   # magenta
    "FunBO":       "#777777",   # gray
    "Random":      "#000000",   # black
}
_PAPER_MARKERS = {
    CAP_PPO_NAME: "o",
    "EI": "^", "UCB": "v", "PI": "<",
    "TAF_ranking": "d", "TAF_me": "D",
    "PFNs4BO": "P", "TuRBO": "X", "FunBO": "h",
    "Random": "s",
}
_PAPER_LINESTYLES = {
    CAP_PPO_NAME: "-",
    "EI": "--", "UCB": "--", "PI": "--",
    "TAF_ranking": "-.", "TAF_me": "-.",
    "PFNs4BO": ":", "TuRBO": ":", "FunBO": ":",
    "Random": "--",
}

_TASK_DISPLAY_NAMES = {
    "branin_family":      "Branin 2D",
    "ackley_5d_family":   "Ackley 5D",
    "ackley_10d_family":  "Ackley 10D",
    "hartmann_3d_family": "Hartmann 3D",
    "hartmann_6d_family": "Hartmann 6D",
    "alkox_emulator":     "Alkox 4D",
    "hplc_emulator":      "HPLC 6D",
    "benzylation_emulator": "Benzylation 4D",
}


def _paper_task_title(task_name: str, surrogate: str) -> str:
    """Short, paper-quality title for a task + surrogate combination."""
    display = _TASK_DISPLAY_NAMES.get(task_name, task_name)
    surr = "GP" if surrogate == "gp" else "TabPFN"
    return f"{display} ({surr} surrogate)"


def _savefig(fig, save_dir: str, stem: str):
    """Save figure as both PNG (300 dpi) and PDF (vector)."""
    png_path = os.path.join(save_dir, f"{stem}.png")
    pdf_path = os.path.join(save_dir, f"{stem}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")


def _style_paper_axes(ax):
    """Apply a lighter paper-style axis treatment."""
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.10, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.35)
    ax.spines["bottom"].set_alpha(0.35)
    ax.tick_params(labelsize=11)


# ---------------------------------------------------------------------------
# Plot: Regret vs. Scale
# ---------------------------------------------------------------------------
def _plot_sweep(
    scales: List[float],
    all_results: Dict,
    method_names: List[str],
    args,
    save_dir: str,
):
    colors = _PAPER_COLORS
    markers = _PAPER_MARKERS
    linestyles = _PAPER_LINESTYLES

    fig, ax = plt.subplots(1, 1, figsize=(10.8, 5.8))

    for m in method_names:
        means = [all_results[str(s)][m]["mean"] for s in scales]
        stds = [all_results[str(s)][m]["std"] for s in scales]
        means = np.array(means)
        stds = np.array(stds)

        color = colors.get(m, "#777777")
        marker = markers.get(m, "o")
        ls = linestyles.get(m, "-")
        lw = 2.5 if m == CAP_PPO_NAME else 1.5
        zorder = 10 if m == CAP_PPO_NAME else 5
        alpha_fill = 0.04 if m == "Random" else 0.07
        alpha_line = 0.5 if m == "Random" else 1.0

        ax.plot(scales, means, marker=marker, label=m, color=color,
                linewidth=lw, linestyle=ls, markersize=4.8, zorder=zorder,
                alpha=alpha_line, markeredgewidth=0.5, markeredgecolor="white")
        ax.fill_between(scales, means - stds, means + stds,
                        alpha=alpha_fill, color=color)

    ax.set_xlabel("Variant Scale", fontsize=12)
    ax.set_ylabel("Simple Regret (mean ± std)", fontsize=12)
    ax.set_title(_paper_task_title(args.task, args.surrogate), fontsize=13, pad=10)
    ax.legend(fontsize=9, loc="upper left", ncol=2,
              frameon=True, framealpha=0.92, facecolor="white",
              edgecolor="#d7d7d7", handlelength=2.0, columnspacing=1.0)
    _style_paper_axes(ax)

    if getattr(args, "show_transform_params", False):
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(scales)
        tick_labels = []
        for s in scales:
            spec = make_spec_for_scale(args.task, s)
            dx_hi = spec["dx"][0][1]
            rot_hi = spec["rot"][0][1]
            sx_lo = spec["sx"][0][0]
            tick_labels.append(f"dx±{dx_hi:.2f}\nrot±{rot_hi:.0f}°\nsx≥{sx_lo:.2f}")
        ax_top.set_xticklabels(tick_labels, fontsize=7)
        ax_top.set_xlabel("Variant transformation parameters", fontsize=10)

    plt.tight_layout()
    _savefig(fig, save_dir, "scale_sweep_regret")
    plt.close(fig)

    # Also plot a version highlighting only CAP-PPO vs top baselines
    fig2, ax2 = plt.subplots(1, 1, figsize=(10.2, 5.6))
    highlight = [CAP_PPO_NAME, "EI", "UCB", "TAF_ranking"]

    for m in highlight:
        if m not in method_names:
            continue
        means = [all_results[str(s)][m]["mean"] for s in scales]
        stds = [all_results[str(s)][m]["std"] for s in scales]
        means = np.array(means)
        stds = np.array(stds)

        color = colors.get(m, "#777777")
        marker = markers.get(m, "o")
        ls = linestyles.get(m, "-")
        lw = 2.5 if m == CAP_PPO_NAME else 1.5

        ax2.plot(scales, means, marker=marker, label=m, color=color,
                 linewidth=lw, linestyle=ls, markersize=5.2,
                 markeredgewidth=0.6, markeredgecolor="white")
        ax2.fill_between(scales, means - stds, means + stds, alpha=0.10, color=color)

    ax2.set_xlabel("Variant Scale", fontsize=12)
    ax2.set_ylabel("Simple Regret (mean ± std)", fontsize=12)
    ax2.set_title(
        f"CAP-PPO vs Key Baselines — {_paper_task_title(args.task, args.surrogate)}",
        fontsize=13,
        pad=10,
    )
    ax2.legend(fontsize=10, frameon=True, framealpha=0.92,
               facecolor="white", edgecolor="#d7d7d7")
    _style_paper_axes(ax2)

    # Annotate crossover point if any
    if CAP_PPO_NAME in method_names:
        cap_means = np.array([all_results[str(s)][CAP_PPO_NAME]["mean"] for s in scales])
        for baseline in ["EI", "UCB", "TAF_ranking"]:
            if baseline not in method_names:
                continue
            bl_means = np.array([all_results[str(s)][baseline]["mean"] for s in scales])
            # Find where CAP-PPO crosses above baseline
            diff = cap_means - bl_means
            for i in range(len(diff) - 1):
                if diff[i] <= 0 < diff[i + 1]:
                    frac = -diff[i] / (diff[i + 1] - diff[i])
                    cross_scale = scales[i] + frac * (scales[i + 1] - scales[i])
                    cross_regret = cap_means[i] + frac * (cap_means[i + 1] - cap_means[i])
                    ax2.axvline(x=cross_scale, color=colors.get(baseline, "gray"),
                               linestyle="--", alpha=0.5, linewidth=1)
                    ax2.annotate(
                        f"×{baseline}\n@{cross_scale:.2f}",
                        xy=(cross_scale, cross_regret),
                        xytext=(cross_scale + 0.05, cross_regret + 2),
                        fontsize=8, color=colors.get(baseline, "gray"),
                        arrowprops=dict(arrowstyle="->", color=colors.get(baseline, "gray"), alpha=0.7),
                    )
                    break

    plt.tight_layout()
    _savefig(fig2, save_dir, "scale_sweep_highlight")
    plt.close(fig2)


# ---------------------------------------------------------------------------
# Plot: Per-scale optimization trajectory grid
# ---------------------------------------------------------------------------
def _plot_trajectory_grid(
    scales: List[float],
    all_trajectories: Dict,
    method_names: List[str],
    args,
    save_dir: str,
):
    """Plot per-scale optimization curves (regret vs. step) in a grid layout.

    Uses linear y-axis with median + 30-70% band (not log scale).
    """
    colors = _PAPER_COLORS
    linestyles = _PAPER_LINESTYLES

    n_scales = len(scales)
    n_cols = min(4, n_scales)
    n_rows = (n_scales + n_cols - 1) // n_cols

    # --- Pass 1: compute global y-max across all scales (excluding Random) ---
    global_y_max = 0
    for scale in scales:
        sk = str(scale)
        for m in method_names:
            trajs = all_trajectories[sk][m]  # [n_variants, n_runs, n_steps+1]
            # Average over runs → [n_variants, n_steps+1]
            unit_curves = trajs.mean(axis=1)
            q70 = np.quantile(unit_curves, 0.70, axis=0)
            candidate_max = float(q70.max())
            if m == "Random":
                # Don't let Random stretch the y-axis too much
                candidate_max = min(candidate_max, float(np.median(unit_curves, axis=0)[0]))
            global_y_max = max(global_y_max, candidate_max)
    global_y_max *= 1.1  # 10% padding

    # --- Pass 2: plot ---
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False, sharey=True,
    )

    n_total_evals = args.n_init + args.max_steps
    x = np.arange(n_total_evals + 1)  # 0 .. n_init+max_steps

    for idx, scale in enumerate(scales):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row][col]
        sk = str(scale)
        spec = make_spec_for_scale(args.task, scale)

        for m in method_names:
            trajs = all_trajectories[sk][m]  # [n_variants, n_runs, n_steps+1]
            unit_curves = trajs.mean(axis=1)  # [n_variants, n_steps+1]
            median = np.median(unit_curves, axis=0)
            q30 = np.quantile(unit_curves, 0.30, axis=0)
            q70 = np.quantile(unit_curves, 0.70, axis=0)

            color = colors.get(m, "#777777")
            ls = linestyles.get(m, "-")
            lw = 2.0 if m == CAP_PPO_NAME else 1.2
            zorder = 10 if m == CAP_PPO_NAME else 5
            alpha_line = 0.5 if m == "Random" else (1.0 if m == CAP_PPO_NAME else 0.8)

            ax.plot(x[:len(median)], median, label=m, color=color,
                    linewidth=lw, linestyle=ls, zorder=zorder, alpha=alpha_line)
            ax.fill_between(x[:len(median)], q30, q70, alpha=0.12, color=color)

        dx_hi = spec["dx"][0][1]
        rot_hi = spec["rot"][0][1]
        sx_lo = spec["sx"][0][0]
        ax.set_title(f"scale={scale:.2f}\ndx±{dx_hi:.2f}  rot±{rot_hi:.0f}°  sx≥{sx_lo:.2f}",
                      fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0, top=global_y_max)
        if row == n_rows - 1:
            ax.set_xlabel("Evaluations", fontsize=10)
        if col == 0:
            ax.set_ylabel("Simple Regret", fontsize=10)

    # Turn off unused subplots
    for idx in range(n_scales, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row][col].set_visible(False)

    # Shared legend at bottom
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(10, len(method_names)), fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Optimization Curves — {_paper_task_title(args.task, args.surrogate)}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])

    _savefig(fig, save_dir, "scale_sweep_trajectories")
    plt.close(fig)


if __name__ == "__main__":
    main()
