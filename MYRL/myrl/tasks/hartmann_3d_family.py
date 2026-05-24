from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import TaskSpec
from .hartmann_3d import hartmann_3d_numpy


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


def _rot_x(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rotation_matrix_from_euler_deg(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx = float(rx_deg) * np.pi / 180.0
    ry = float(ry_deg) * np.pi / 180.0
    rz = float(rz_deg) * np.pi / 180.0
    # Apply in Z-Y-X order to match common conventions.
    return _rot_z(rz) @ _rot_y(ry) @ _rot_x(rx)


def hartmann_3d_family_numpy(X: np.ndarray, variant_params: Dict[str, float]) -> np.ndarray:
    """
    Hartmann-3D variant family on [0,1]^3.

    We generate variants by applying an affine transform to the input coordinates:
      x' = clip( center + R * (S * (x - center)) + d, 0, 1 )
    then evaluate the standard Hartmann-3D on x'.

    This matches the Branin-family design philosophy:
    - training/eval variants are different functions (not just different RNG seeds)
    - candidate generation/BO loop remains unchanged
    """
    X = np.atleast_2d(X).astype(np.float64)
    if X.shape[1] != 3:
        raise ValueError(f"Hartmann-3D family expects dim=3, got X.shape={X.shape}")

    dx1 = float(variant_params.get("dx1", 0.0))
    dx2 = float(variant_params.get("dx2", 0.0))
    dx3 = float(variant_params.get("dx3", 0.0))
    sx1 = float(variant_params.get("sx1", 1.0))
    sx2 = float(variant_params.get("sx2", 1.0))
    sx3 = float(variant_params.get("sx3", 1.0))
    rx = float(variant_params.get("rx", 0.0))
    ry = float(variant_params.get("ry", 0.0))
    rz = float(variant_params.get("rz", 0.0))
    alpha = float(variant_params.get("alpha", 1.0))
    beta = float(variant_params.get("beta", 0.0))

    center = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    Xc = X - center[None, :]

    S = np.diag([sx1, sx2, sx3]).astype(np.float64)
    if abs(rx) > 1e-12 or abs(ry) > 1e-12 or abs(rz) > 1e-12:
        R = _rotation_matrix_from_euler_deg(rx, ry, rz)
    else:
        R = np.eye(3, dtype=np.float64)

    d = np.array([dx1, dx2, dx3], dtype=np.float64)
    Xp = center[None, :] + (Xc @ S.T) @ R.T + d[None, :]
    Xp = np.clip(Xp, 0.0, 1.0)

    y = hartmann_3d_numpy(Xp.astype(np.float64))
    y = alpha * y + beta
    return np.asarray(y, dtype=np.float32)


_VARIANT_SUITE_SPECS: Dict[str, Dict[str, List[Tuple[float, float]]]] = {
    # Keep "in_range" relatively small so transformed inputs stay mostly inside [0,1]^3 without heavy clipping.
    "in_range": {
        "dx": [(-0.05, 0.05)],
        "rot": [(-15.0, 15.0)],
        "sx": [(0.9, 1.1)],
    },
    "ood_level_1": {
        "dx": [(-0.10, -0.05), (0.05, 0.10)],
        "rot": [(-30.0, -15.0), (15.0, 30.0)],
        "sx": [(0.8, 0.9), (1.1, 1.2)],
    },
    "ood_level_2": {
        "dx": [(-0.15, -0.10), (0.10, 0.15)],
        "rot": [(-45.0, -30.0), (30.0, 45.0)],
        "sx": [(0.7, 0.8), (1.2, 1.3)],
    },
    "ood_level_3": {
        "dx": [(-0.20, -0.15), (0.15, 0.20)],
        "rot": [(-60.0, -45.0), (45.0, 60.0)],
        "sx": [(0.6, 0.7), (1.3, 1.4)],
    },
}


def sample_variant_from_spec(rng: np.random.Generator, spec: Dict[str, List[Tuple[float, float]]]) -> Dict[str, float]:
    dx1 = _sample_from_segments(rng, spec["dx"])
    dx2 = _sample_from_segments(rng, spec["dx"])
    dx3 = _sample_from_segments(rng, spec["dx"])
    rx = _sample_from_segments(rng, spec["rot"])
    ry = _sample_from_segments(rng, spec["rot"])
    rz = _sample_from_segments(rng, spec["rot"])
    sx1 = _sample_from_segments(rng, spec["sx"])
    sx2 = _sample_from_segments(rng, spec["sx"])
    sx3 = _sample_from_segments(rng, spec["sx"])
    return {
        "dx1": float(dx1),
        "dx2": float(dx2),
        "dx3": float(dx3),
        "rx": float(rx),
        "ry": float(ry),
        "rz": float(rz),
        "sx1": float(sx1),
        "sx2": float(sx2),
        "sx3": float(sx3),
        "alpha": 1.0,
        "beta": 0.0,
    }


class Hartmann3DFamilyTask(TaskSpec):
    @property
    def task_name(self) -> str:
        return "hartmann_3d_family"

    @property
    def dim(self) -> int:
        return 3

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        upper = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        return lower, upper

    def evaluate_numpy(self, X: np.ndarray, variant_params: Optional[Dict[str, float]] = None) -> np.ndarray:
        return hartmann_3d_family_numpy(X, variant_params or {})

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
        Dim=3 global-min estimator:
        - Sobol random search (n = grid_size^2 points)
        - multi-start L-BFGS-B refinement
        """
        variant_params = variant_params or {}
        lower, upper = self.bounds
        dim = int(self.dim)
        n = int(max(2048, min(65536, int(grid_size) ** 2)))
        # Sobol balance properties prefer powers of 2.
        n = int(2 ** int(np.ceil(np.log2(max(2, n)))))

        from scipy.stats.qmc import Sobol
        from scipy.optimize import minimize

        sobol = Sobol(d=dim, scramble=True, seed=0)
        X = sobol.random(n)
        X = (X * (upper - lower) + lower).astype(np.float64)
        y = self.evaluate_numpy(X, variant_params).astype(np.float64).reshape(-1)

        best_min = float(np.min(y))
        n_starts = int(min(20, len(y)))
        start_points = X[np.argsort(y)[:n_starts]]

        def func(x):
            x = np.asarray(x, dtype=np.float64).reshape(1, dim)
            return float(self.evaluate_numpy(x, variant_params)[0])

        bounds = [(float(lower[i]), float(upper[i])) for i in range(dim)]
        for x0 in start_points:
            try:
                res = minimize(func, np.asarray(x0, dtype=np.float64), bounds=bounds, method="L-BFGS-B")
                best_min = min(best_min, float(res.fun))
            except Exception:
                pass
        return float(best_min)
