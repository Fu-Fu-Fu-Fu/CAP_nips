"""
Prescreen candidate HPLC base specs for single-in-range + scale-sweep training.

For each candidate spec:
1. Sample a small set of training variants
2. Check finite evaluations on Sobol probes
3. Measure pre-clip affine-transform severity
4. Run a lightweight GP+EI quick test to estimate task difficulty

Outputs a JSON summary for downstream experiment planning.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import Any, Dict, List, Tuple

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import norm
from scipy.stats.qmc import Sobol
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.normpath(os.path.join(_this_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from myrl.tasks.hplc_emulator import (  # noqa: E402
    HplcEmulatorTask,
    _rotation_matrix_from_givens_deg,
    sample_variant_from_spec,
)


CANDIDATE_SPECS: Dict[str, Dict[str, List[Tuple[float, float]]]] = {
    "conservative": {
        "dx": [(-0.03, 0.03)],
        "rot": [(-10.0, 10.0)],
        "sx": [(0.92, 1.00)],
    },
    "recommended": {
        "dx": [(-0.04, 0.04)],
        "rot": [(-14.0, 14.0)],
        "sx": [(0.88, 1.00)],
    },
    "aggressive": {
        "dx": [(-0.05, 0.05)],
        "rot": [(-18.0, 18.0)],
        "sx": [(0.85, 1.00)],
    },
}


def spec_summary(spec: Dict[str, List[Tuple[float, float]]]) -> str:
    dx_lo, dx_hi = spec["dx"][0]
    rot_lo, rot_hi = spec["rot"][0]
    sx_lo, sx_hi = spec["sx"][0]
    return f"dx ±{dx_hi:.3f}, rot ±{rot_hi:.1f}°, sx [{sx_lo:.2f}, {sx_hi:.2f}]"


def transform_without_clip(X: np.ndarray, variant_params: Dict[str, float]) -> np.ndarray:
    X = np.atleast_2d(X).astype(np.float64)
    center = np.full(6, 0.5, dtype=np.float64)
    Xc = X - center[None, :]
    dx = np.array([float(variant_params.get(f"dx{i+1}", 0.0)) for i in range(6)], dtype=np.float64)
    sx = np.array([float(variant_params.get(f"sx{i+1}", 1.0)) for i in range(6)], dtype=np.float64)
    S = np.diag(sx)
    R = _rotation_matrix_from_givens_deg(
        float(variant_params.get("r01", 0.0)),
        float(variant_params.get("r23", 0.0)),
        float(variant_params.get("r45", 0.0)),
        float(variant_params.get("r03", 0.0)),
        float(variant_params.get("r14", 0.0)),
        float(variant_params.get("r25", 0.0)),
    )
    return center[None, :] + (Xc @ S.T) @ R.T + dx[None, :]


def clip_metrics(X: np.ndarray, variant_params: Dict[str, float]) -> Dict[str, float]:
    X_raw = transform_without_clip(X, variant_params)
    under = X_raw < 0.0
    over = X_raw > 1.0
    viol = under | over
    point_clip_frac = float(np.mean(np.any(viol, axis=1)))
    coord_clip_frac = float(np.mean(viol))
    if np.any(viol):
        under_mag = np.where(under, -X_raw, 0.0)
        over_mag = np.where(over, X_raw - 1.0, 0.0)
        clip_mag = under_mag + over_mag
        mean_clip_mag = float(np.mean(clip_mag[viol]))
        max_clip_mag = float(np.max(clip_mag[viol]))
    else:
        mean_clip_mag = 0.0
        max_clip_mag = 0.0
    return {
        "point_clip_frac": point_clip_frac,
        "coord_clip_frac": coord_clip_frac,
        "mean_clip_mag": mean_clip_mag,
        "max_clip_mag": max_clip_mag,
    }


def run_ei_quick_test(
    evaluate,
    dim: int,
    *,
    n_init: int,
    n_steps: int,
    n_candidates: int,
    seed: int,
    gp_restarts: int,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    X = rng.rand(n_init, dim)
    y = evaluate(X).reshape(-1, 1)
    best_ys = [float(y.min())]

    for step in range(n_steps):
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=np.ones(dim, dtype=np.float64),
            length_scale_bounds=(1e-5, 1e5),
            nu=2.5,
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=gp_restarts,
            random_state=seed + step,
        )
        try:
            gp.fit(X, y.ravel())
        except Exception:
            gp = GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * Matern(
                    length_scale=np.ones(dim, dtype=np.float64),
                    nu=2.5,
                ),
                alpha=1e-6,
                normalize_y=True,
                optimizer=None,
            )
            gp.fit(X, y.ravel())

        sobol = Sobol(d=dim, scramble=True, seed=seed + step)
        X_cand = sobol.random(n_candidates)
        mu_pred, sigma_pred = gp.predict(X_cand, return_std=True)
        mu_pred = mu_pred.ravel()
        sigma_pred = np.maximum(sigma_pred.ravel(), 1e-18)
        best_y = float(y.min())
        z = (best_y - mu_pred) / sigma_pred
        ei = (best_y - mu_pred) * norm.cdf(z) + sigma_pred * norm.pdf(z)

        idx = int(np.argmax(ei))
        x_new = X_cand[idx:idx + 1]
        y_new = evaluate(x_new).reshape(-1, 1)

        X = np.vstack([X, x_new])
        y = np.vstack([y, y_new])
        best_ys.append(min(best_ys[-1], float(y_new[0, 0])))

    return np.array(best_ys, dtype=np.float64)


def estimate_global_stats(evaluate, dim: int, *, n_sobol: int, seed: int) -> Dict[str, float]:
    sobol = Sobol(d=dim, scramble=True, seed=seed)
    X = sobol.random(n_sobol)
    y = evaluate(X).astype(np.float64).reshape(-1)
    return {
        "global_min": float(np.min(y)),
        "global_max": float(np.max(y)),
        "y_mean": float(np.mean(y)),
        "y_std": float(np.std(y)),
        "y_range": float(np.max(y) - np.min(y)),
    }


def classify_difficulty(norm_regret: float) -> str:
    if norm_regret < 0.01:
        return "EASY"
    if norm_regret < 0.05:
        return "MODERATE"
    if norm_regret < 0.15:
        return "HARD"
    return "VERY_HARD"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_variants", type=int, default=5)
    parser.add_argument("--probe_sobol", type=int, default=4096)
    parser.add_argument("--global_sobol", type=int, default=4096)
    parser.add_argument("--n_runs", type=int, default=2)
    parser.add_argument("--n_init", type=int, default=2)
    parser.add_argument("--n_steps", type=int, default=28)
    parser.add_argument("--n_candidates", type=int, default=1024)
    parser.add_argument("--gp_restarts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results_policies/hplc_base_spec_prescreen",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    task = HplcEmulatorTask()
    dim = int(task.dim)
    master_rng = np.random.default_rng(args.seed)
    probe_X = Sobol(d=dim, scramble=True, seed=args.seed).random(args.probe_sobol)

    all_results: Dict[str, Any] = {
        "settings": vars(args),
        "specs": {},
    }

    print("=" * 80)
    print("HPLC Base Spec Prescreen")
    print("=" * 80)
    print(f"Variants/spec: {args.n_variants}")
    print(f"Probe Sobol: {args.probe_sobol}, Global Sobol: {args.global_sobol}")
    print(f"EI quick test: {args.n_runs} runs, {args.n_init}+{args.n_steps} evals, {args.n_candidates} candidates")
    print()

    for spec_name, spec in CANDIDATE_SPECS.items():
        print(f"[{spec_name}] {spec_summary(spec)}")
        spec_rng = np.random.default_rng(master_rng.integers(0, 10**9))
        spec_result: Dict[str, Any] = {
            "summary": spec_summary(spec),
            "variants": [],
        }

        clip_point_fracs = []
        clip_coord_fracs = []
        clip_mean_mags = []
        finite_failures = 0
        ei_regrets = []
        ei_norm_regrets = []

        for vidx in range(args.n_variants):
            variant_params = sample_variant_from_spec(spec_rng, spec)
            y_probe = task.evaluate_numpy(probe_X, variant_params).astype(np.float64).reshape(-1)
            finite_mask = np.isfinite(y_probe)
            finite_ok = bool(np.all(finite_mask))
            finite_ratio = float(np.mean(finite_mask))
            if not finite_ok:
                finite_failures += 1

            cm = clip_metrics(probe_X, variant_params)
            clip_point_fracs.append(cm["point_clip_frac"])
            clip_coord_fracs.append(cm["coord_clip_frac"])
            clip_mean_mags.append(cm["mean_clip_mag"])

            if finite_ok:
                evaluate = lambda X, vp=variant_params: task.evaluate_numpy(X, vp).astype(np.float64)  # noqa: E731
                g = estimate_global_stats(
                    evaluate,
                    dim,
                    n_sobol=args.global_sobol,
                    seed=int(spec_rng.integers(0, 10**9)),
                )
                run_regrets = []
                for run in range(args.n_runs):
                    best_ys = run_ei_quick_test(
                        evaluate,
                        dim,
                        n_init=args.n_init,
                        n_steps=args.n_steps,
                        n_candidates=args.n_candidates,
                        seed=int(spec_rng.integers(0, 10**9)),
                        gp_restarts=args.gp_restarts,
                    )
                    regret = float(best_ys[-1] - g["global_min"])
                    run_regrets.append(regret)
                    ei_regrets.append(regret)
                    ei_norm_regrets.append(regret / max(g["y_range"], 1e-8))
            else:
                g = {
                    "global_min": None,
                    "global_max": None,
                    "y_mean": None,
                    "y_std": None,
                    "y_range": None,
                }
                run_regrets = []

            spec_result["variants"].append({
                "variant_index": vidx,
                "variant_params": variant_params,
                "finite_ok": finite_ok,
                "finite_ratio": finite_ratio,
                "probe_y_min": None if not finite_ok else float(np.min(y_probe)),
                "probe_y_max": None if not finite_ok else float(np.max(y_probe)),
                "probe_y_std": None if not finite_ok else float(np.std(y_probe)),
                "clip_metrics": cm,
                "global_stats": g,
                "ei_quick_regrets": run_regrets,
            })

            status = "OK" if finite_ok else "NONFINITE"
            print(
                f"  v{vidx:02d}  {status}  "
                f"clip(point)={cm['point_clip_frac']:.3f}  "
                f"clip(coord)={cm['coord_clip_frac']:.3f}  "
                f"ei_runs={len(run_regrets)}"
            )

        mean_regret = float(np.mean(ei_regrets)) if ei_regrets else None
        std_regret = float(np.std(ei_regrets)) if ei_regrets else None
        mean_norm_regret = float(np.mean(ei_norm_regrets)) if ei_norm_regrets else None
        std_norm_regret = float(np.std(ei_norm_regrets)) if ei_norm_regrets else None

        spec_result["aggregate"] = {
            "finite_failure_count": finite_failures,
            "finite_failure_rate": float(finite_failures / max(args.n_variants, 1)),
            "mean_point_clip_frac": float(np.mean(clip_point_fracs)),
            "std_point_clip_frac": float(np.std(clip_point_fracs)),
            "mean_coord_clip_frac": float(np.mean(clip_coord_fracs)),
            "std_coord_clip_frac": float(np.std(clip_coord_fracs)),
            "mean_clip_mag": float(np.mean(clip_mean_mags)),
            "std_clip_mag": float(np.std(clip_mean_mags)),
            "ei_regret_mean": mean_regret,
            "ei_regret_std": std_regret,
            "ei_norm_regret_mean": mean_norm_regret,
            "ei_norm_regret_std": std_norm_regret,
            "difficulty": None if mean_norm_regret is None else classify_difficulty(mean_norm_regret),
        }
        all_results["specs"][spec_name] = spec_result

        print(
            "  => "
            f"finite_fail={finite_failures}/{args.n_variants}, "
            f"point_clip={spec_result['aggregate']['mean_point_clip_frac']:.3f}, "
            f"coord_clip={spec_result['aggregate']['mean_coord_clip_frac']:.3f}, "
            f"ei_norm={mean_norm_regret:.4f} ({spec_result['aggregate']['difficulty']})"
            if mean_norm_regret is not None else
            "  => no valid EI result"
        )
        print()

    out_path = os.path.join(args.save_dir, "hplc_base_spec_prescreen.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
