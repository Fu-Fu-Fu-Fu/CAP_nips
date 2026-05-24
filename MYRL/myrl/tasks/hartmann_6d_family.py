from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import TaskSpec
from .hartmann_6d import hartmann_6d_numpy

# ---------- float64 helpers for Hartmann 6D (used by L-BFGS-B) ----------

_H6_ALPHA = np.array([1.0, 1.2, 3.0, 3.2], dtype=np.float64)
_H6_A = np.array(
    [
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
    ],
    dtype=np.float64,
)
_H6_P = 1e-4 * np.array(
    [
        [1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0],
        [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
        [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
        [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0],
    ],
    dtype=np.float64,
)


def _hartmann_6d_f64(X: np.ndarray) -> np.ndarray:
    """Hartmann 6D in full float64 (for optimizer gradient accuracy)."""
    X = np.atleast_2d(X).astype(np.float64)
    Xb = X[:, None, :]
    diff2 = (Xb - _H6_P[None, :, :]) ** 2
    exponent = np.sum(_H6_A[None, :, :] * diff2, axis=-1)
    return -np.sum(_H6_ALPHA[None, :] * np.exp(-exponent), axis=1)


def _evaluate_f64(X: np.ndarray, variant_params: Dict[str, float]) -> np.ndarray:
    """Family evaluation in float64 for L-BFGS-B optimization."""
    X = np.atleast_2d(X).astype(np.float64)

    dx = np.array([float(variant_params.get(f"dx{i+1}", 0.0)) for i in range(6)], dtype=np.float64)
    sx = np.array([float(variant_params.get(f"sx{i+1}", 1.0)) for i in range(6)], dtype=np.float64)
    r01 = float(variant_params.get("r01", 0.0))
    r23 = float(variant_params.get("r23", 0.0))
    r45 = float(variant_params.get("r45", 0.0))
    r03 = float(variant_params.get("r03", 0.0))
    r14 = float(variant_params.get("r14", 0.0))
    r25 = float(variant_params.get("r25", 0.0))
    alpha = float(variant_params.get("alpha", 1.0))
    beta = float(variant_params.get("beta", 0.0))

    center = np.full(6, 0.5, dtype=np.float64)
    Xc = X - center[None, :]
    S = np.diag(sx)
    has_rotation = any(abs(a) > 1e-12 for a in [r01, r23, r45, r03, r14, r25])
    R = _rotation_matrix_from_givens_deg(r01, r23, r45, r03, r14, r25) if has_rotation else np.eye(6, dtype=np.float64)
    Xp = center[None, :] + (Xc @ S.T) @ R.T + dx[None, :]
    Xp = np.clip(Xp, 0.0, 1.0)
    y = _hartmann_6d_f64(Xp)
    return alpha * y + beta


def _validate_segments(segments: List[Tuple[float, float]]) -> None:
    for (lo, hi) in segments:
        if not (float(lo) < float(hi)):
            raise ValueError(f"Invalid segment: ({lo}, {hi})")


def _sample_from_segments(rng: np.random.Generator, segments: List[Tuple[float, float]]) -> float:
    _validate_segments(segments)
    lengths = np.array([float(hi) - float(lo) for (lo, hi) in segments], dtype=np.float64)
    probs = lengths / lengths.sum()
    idx = int(rng.choice(len(segments), p=probs))
    lo, hi = segments[idx]
    return float(rng.uniform(float(lo), float(hi)))


def _givens_rotation(dim: int, i: int, j: int, theta: float) -> np.ndarray:
    """Givens rotation matrix in the (i, j) plane."""
    R = np.eye(dim, dtype=np.float64)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    R[i, i] = c
    R[i, j] = -s
    R[j, i] = s
    R[j, j] = c
    return R


def _rotation_matrix_from_givens_deg(
    r01: float, r23: float, r45: float,
    r03: float, r14: float, r25: float,
) -> np.ndarray:
    """
    Compose 6 Givens rotations in 6D space.

    Rotation planes: (0,1), (2,3), (4,5), (0,3), (1,4), (2,5).
    Angles are in degrees.
    """
    dim = 6
    planes = [(0, 1), (2, 3), (4, 5), (0, 3), (1, 4), (2, 5)]
    angles_deg = [r01, r23, r45, r03, r14, r25]
    R = np.eye(dim, dtype=np.float64)
    for (i, j), a in zip(planes, angles_deg):
        theta = float(a) * np.pi / 180.0
        if abs(theta) > 1e-12:
            R = R @ _givens_rotation(dim, i, j, theta)
    return R


def hartmann_6d_family_numpy(X: np.ndarray, variant_params: Dict[str, float]) -> np.ndarray:
    """
    Hartmann-6D variant family on [0,1]^6.

    We generate variants by applying an affine transform to the input coordinates:
      x' = clip( center + R * (S * (x - center)) + d, 0, 1 )
    then evaluate the standard Hartmann-6D on x'.
    """
    X = np.atleast_2d(X).astype(np.float64)
    if X.shape[1] != 6:
        raise ValueError(f"Hartmann-6D family expects dim=6, got X.shape={X.shape}")

    dx1 = float(variant_params.get("dx1", 0.0))
    dx2 = float(variant_params.get("dx2", 0.0))
    dx3 = float(variant_params.get("dx3", 0.0))
    dx4 = float(variant_params.get("dx4", 0.0))
    dx5 = float(variant_params.get("dx5", 0.0))
    dx6 = float(variant_params.get("dx6", 0.0))
    sx1 = float(variant_params.get("sx1", 1.0))
    sx2 = float(variant_params.get("sx2", 1.0))
    sx3 = float(variant_params.get("sx3", 1.0))
    sx4 = float(variant_params.get("sx4", 1.0))
    sx5 = float(variant_params.get("sx5", 1.0))
    sx6 = float(variant_params.get("sx6", 1.0))
    r01 = float(variant_params.get("r01", 0.0))
    r23 = float(variant_params.get("r23", 0.0))
    r45 = float(variant_params.get("r45", 0.0))
    r03 = float(variant_params.get("r03", 0.0))
    r14 = float(variant_params.get("r14", 0.0))
    r25 = float(variant_params.get("r25", 0.0))
    alpha = float(variant_params.get("alpha", 1.0))
    beta = float(variant_params.get("beta", 0.0))

    center = np.full(6, 0.5, dtype=np.float64)
    Xc = X - center[None, :]

    S = np.diag([sx1, sx2, sx3, sx4, sx5, sx6]).astype(np.float64)

    has_rotation = any(abs(a) > 1e-12 for a in [r01, r23, r45, r03, r14, r25])
    if has_rotation:
        R = _rotation_matrix_from_givens_deg(r01, r23, r45, r03, r14, r25)
    else:
        R = np.eye(6, dtype=np.float64)

    d = np.array([dx1, dx2, dx3, dx4, dx5, dx6], dtype=np.float64)
    Xp = center[None, :] + (Xc @ S.T) @ R.T + d[None, :]
    Xp = np.clip(Xp, 0.0, 1.0)

    y = hartmann_6d_numpy(Xp.astype(np.float64))
    y = alpha * y + beta
    return np.asarray(y, dtype=np.float32)


# 6D space is larger — use smaller perturbations than 3D to avoid excessive clipping.
_VARIANT_SUITE_SPECS: Dict[str, Dict[str, List[Tuple[float, float]]]] = {
    "in_range": {
        "dx": [(-0.03, 0.03)],
        "rot": [(-10.0, 10.0)],
        "sx": [(0.92, 1.08)],
    },
    "ood_level_1": {
        "dx": [(-0.06, -0.03), (0.03, 0.06)],
        "rot": [(-20.0, -10.0), (10.0, 20.0)],
        "sx": [(0.85, 0.92), (1.08, 1.15)],
    },
    "ood_level_2": {
        "dx": [(-0.10, -0.06), (0.06, 0.10)],
        "rot": [(-30.0, -20.0), (20.0, 30.0)],
        "sx": [(0.78, 0.85), (1.15, 1.22)],
    },
    "ood_level_3": {
        "dx": [(-0.15, -0.10), (0.10, 0.15)],
        "rot": [(-40.0, -30.0), (30.0, 40.0)],
        "sx": [(0.70, 0.78), (1.22, 1.30)],
    },
}


def sample_variant_from_spec(rng: np.random.Generator, spec: Dict[str, List[Tuple[float, float]]]) -> Dict[str, float]:
    dx = [_sample_from_segments(rng, spec["dx"]) for _ in range(6)]
    sx = [_sample_from_segments(rng, spec["sx"]) for _ in range(6)]
    rot_keys = ["r01", "r23", "r45", "r03", "r14", "r25"]
    rot = [_sample_from_segments(rng, spec["rot"]) for _ in range(6)]
    params: Dict[str, float] = {}
    for i in range(6):
        params[f"dx{i+1}"] = float(dx[i])
    for i in range(6):
        params[f"sx{i+1}"] = float(sx[i])
    for key, val in zip(rot_keys, rot):
        params[key] = float(val)
    params["alpha"] = 1.0
    params["beta"] = 0.0
    return params


class Hartmann6DFamilyTask(TaskSpec):
    @property
    def task_name(self) -> str:
        return "hartmann_6d_family"

    @property
    def dim(self) -> int:
        return 6

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.zeros(6, dtype=np.float32)
        upper = np.ones(6, dtype=np.float32)
        return lower, upper

    def evaluate_numpy(self, X: np.ndarray, variant_params: Optional[Dict[str, float]] = None) -> np.ndarray:
        return hartmann_6d_family_numpy(X, variant_params or {})

    def sample_train_variants(self, *, k: int, seed: int) -> List[Dict[str, float]]:
        rng = np.random.default_rng(int(seed))
        return [sample_variant_from_spec(rng, _VARIANT_SUITE_SPECS["in_range"]) for _ in range(int(k))]

    def default_variant_suite(self) -> Dict[str, Any]:
        return _VARIANT_SUITE_SPECS

    def sample_eval_suite(
        self,
        *,
        n_per_group: int,
        seed: int,
        suite_specs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Dict[str, float]]]:
        suite_specs = suite_specs or _VARIANT_SUITE_SPECS
        rng = np.random.default_rng(int(seed))
        suite: Dict[str, List[Dict[str, float]]] = {}
        for group_name, spec in suite_specs.items():
            suite[group_name] = [sample_variant_from_spec(rng, spec) for _ in range(int(n_per_group))]
        return suite

    def estimate_global_min(self, variant_params: Optional[Dict[str, float]] = None, *, grid_size: int = 100) -> float:
        """
        Dim=6 global-min estimator:
        - Sobol random search (2^17 = 131072 points)
        - multi-start L-BFGS-B refinement in float64
        """
        variant_params = variant_params or {}
        lower, upper = self.bounds
        dim = int(self.dim)
        n = 2 ** 17  # 131072 — higher density needed for 6D

        from scipy.stats.qmc import Sobol
        from scipy.optimize import minimize

        sobol = Sobol(d=dim, scramble=True, seed=0)
        X = sobol.random(n)
        X = (X * (upper - lower) + lower).astype(np.float64)
        y = self.evaluate_numpy(X, variant_params).astype(np.float64).reshape(-1)

        best_min = float(np.min(y))
        n_starts = int(min(30, len(y)))
        start_points = X[np.argsort(y)[:n_starts]]

        # Use float64 evaluation for L-BFGS-B to get accurate gradients.
        def func_f64(x):
            x = np.asarray(x, dtype=np.float64).reshape(1, dim)
            return float(_evaluate_f64(x, variant_params))

        bounds = [(float(lower[i]), float(upper[i])) for i in range(dim)]
        for x0 in start_points:
            try:
                res = minimize(func_f64, np.asarray(x0, dtype=np.float64), bounds=bounds, method="L-BFGS-B")
                best_min = min(best_min, float(res.fun))
            except Exception:
                pass
        return float(best_min)
