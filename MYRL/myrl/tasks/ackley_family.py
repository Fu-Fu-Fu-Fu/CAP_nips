from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import TaskSpec


def ackley_numpy(X: np.ndarray) -> np.ndarray:
    """
    Ackley function for minimization on normalized [0, 1]^d inputs.

    The normalized coordinates are mapped to z in [-5, 5]^d before evaluating
    the standard Ackley formula.  The global minimum is 0 at x=(0.5, ..., 0.5).
    """
    X = np.atleast_2d(X).astype(np.float64)
    z = 10.0 * (X - 0.5)
    dim = int(z.shape[1])
    sum_sq = np.sum(z ** 2, axis=1)
    sum_cos = np.sum(np.cos(2.0 * np.pi * z), axis=1)
    y = (
        -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / dim))
        - np.exp(sum_cos / dim)
        + 20.0
        + np.e
    )
    return y.astype(np.float32)


def _validate_segments(segments: List[Tuple[float, float]]) -> None:
    for lo, hi in segments:
        if not (float(lo) < float(hi)):
            raise ValueError(f"Invalid segment: ({lo}, {hi})")


def _sample_from_segments(rng: np.random.Generator, segments: List[Tuple[float, float]]) -> float:
    _validate_segments(segments)
    lengths = np.array([float(hi) - float(lo) for lo, hi in segments], dtype=np.float64)
    probs = lengths / lengths.sum()
    idx = int(rng.choice(len(segments), p=probs))
    lo, hi = segments[idx]
    return float(rng.uniform(float(lo), float(hi)))


def _givens_rotation(dim: int, i: int, j: int, theta: float) -> np.ndarray:
    R = np.eye(dim, dtype=np.float64)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    R[i, i] = c
    R[i, j] = -s
    R[j, i] = s
    R[j, j] = c
    return R


def rotation_planes(dim: int) -> List[Tuple[int, int]]:
    if dim < 2:
        return []
    adjacent = [(i, i + 1) for i in range(dim - 1)]
    cross = [(i, i + dim // 2) for i in range(dim // 2)]
    return adjacent + cross


def rotation_matrix_from_params(dim: int, variant_params: Dict[str, float]) -> np.ndarray:
    R = np.eye(dim, dtype=np.float64)
    for ridx, (i, j) in enumerate(rotation_planes(dim)):
        angle_deg = float(variant_params.get(f"r{ridx}", 0.0))
        theta = angle_deg * np.pi / 180.0
        if abs(theta) > 1e-12:
            R = R @ _givens_rotation(dim, i, j, theta)
    return R


def ackley_family_numpy(X: np.ndarray, variant_params: Dict[str, float], *, dim: int) -> np.ndarray:
    X = np.atleast_2d(X).astype(np.float64)
    if X.shape[1] != int(dim):
        raise ValueError(f"Ackley-{dim}D family expects dim={dim}, got X.shape={X.shape}")

    dx = np.array([float(variant_params.get(f"dx{i + 1}", 0.0)) for i in range(dim)], dtype=np.float64)
    sx = np.array([float(variant_params.get(f"sx{i + 1}", 1.0)) for i in range(dim)], dtype=np.float64)
    alpha = float(variant_params.get("alpha", 1.0))
    beta = float(variant_params.get("beta", 0.0))

    center = np.full(dim, 0.5, dtype=np.float64)
    Xc = X - center[None, :]
    S = np.diag(sx)
    R = rotation_matrix_from_params(dim, variant_params)
    Xp = center[None, :] + (Xc @ S.T) @ R.T + dx[None, :]
    Xp = np.clip(Xp, 0.0, 1.0)

    y = ackley_numpy(Xp)
    return (alpha * y + beta).astype(np.float32)


_VARIANT_SUITE_SPECS_5D: Dict[str, Dict[str, List[Tuple[float, float]]]] = {
    "in_range": {
        "dx": [(-0.04, 0.04)],
        "rot": [(-12.0, 12.0)],
        "sx": [(0.92, 1.08)],
    },
    "ood_level_1": {
        "dx": [(-0.08, -0.04), (0.04, 0.08)],
        "rot": [(-24.0, -12.0), (12.0, 24.0)],
        "sx": [(0.84, 0.92), (1.08, 1.16)],
    },
    "ood_level_2": {
        "dx": [(-0.12, -0.08), (0.08, 0.12)],
        "rot": [(-36.0, -24.0), (24.0, 36.0)],
        "sx": [(0.76, 0.84), (1.16, 1.24)],
    },
    "ood_level_3": {
        "dx": [(-0.16, -0.12), (0.12, 0.16)],
        "rot": [(-48.0, -36.0), (36.0, 48.0)],
        "sx": [(0.68, 0.76), (1.24, 1.32)],
    },
}


_VARIANT_SUITE_SPECS_10D: Dict[str, Dict[str, List[Tuple[float, float]]]] = {
    "in_range": {
        "dx": [(-0.025, 0.025)],
        "rot": [(-8.0, 8.0)],
        "sx": [(0.95, 1.05)],
    },
    "ood_level_1": {
        "dx": [(-0.05, -0.025), (0.025, 0.05)],
        "rot": [(-16.0, -8.0), (8.0, 16.0)],
        "sx": [(0.90, 0.95), (1.05, 1.10)],
    },
    "ood_level_2": {
        "dx": [(-0.08, -0.05), (0.05, 0.08)],
        "rot": [(-24.0, -16.0), (16.0, 24.0)],
        "sx": [(0.84, 0.90), (1.10, 1.16)],
    },
    "ood_level_3": {
        "dx": [(-0.12, -0.08), (0.08, 0.12)],
        "rot": [(-32.0, -24.0), (24.0, 32.0)],
        "sx": [(0.76, 0.84), (1.16, 1.24)],
    },
}


def variant_suite_specs(dim: int) -> Dict[str, Dict[str, List[Tuple[float, float]]]]:
    if int(dim) == 5:
        return _VARIANT_SUITE_SPECS_5D
    if int(dim) == 10:
        return _VARIANT_SUITE_SPECS_10D
    raise ValueError(f"Unsupported Ackley family dim: {dim}")


def sample_variant_from_spec(
    rng: np.random.Generator,
    spec: Dict[str, List[Tuple[float, float]]],
    *,
    dim: int,
) -> Dict[str, float]:
    params: Dict[str, float] = {}
    for i in range(int(dim)):
        params[f"dx{i + 1}"] = _sample_from_segments(rng, spec["dx"])
        params[f"sx{i + 1}"] = _sample_from_segments(rng, spec["sx"])
    for ridx, _ in enumerate(rotation_planes(int(dim))):
        params[f"r{ridx}"] = _sample_from_segments(rng, spec["rot"])
    params["alpha"] = 1.0
    params["beta"] = 0.0
    return params


class AckleyFamilyTask(TaskSpec):
    def __init__(self, dim: int):
        if int(dim) not in {5, 10}:
            raise ValueError("AckleyFamilyTask currently supports dim=5 or dim=10")
        self._dim = int(dim)

    @property
    def task_name(self) -> str:
        return f"ackley_{self._dim}d_family"

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros(self._dim, dtype=np.float32), np.ones(self._dim, dtype=np.float32)

    def evaluate_numpy(self, X: np.ndarray, variant_params: Optional[Dict[str, float]] = None) -> np.ndarray:
        return ackley_family_numpy(X, variant_params or {}, dim=self._dim)

    def sample_train_variants(self, *, k: int, seed: int) -> List[Dict[str, float]]:
        rng = np.random.default_rng(int(seed))
        spec = variant_suite_specs(self._dim)["in_range"]
        return [sample_variant_from_spec(rng, spec, dim=self._dim) for _ in range(int(k))]

    def default_variant_suite(self) -> Dict[str, Any]:
        return variant_suite_specs(self._dim)

    def sample_eval_suite(
        self,
        *,
        n_per_group: int,
        seed: int,
        suite_specs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Dict[str, float]]]:
        suite_specs = suite_specs or variant_suite_specs(self._dim)
        rng = np.random.default_rng(int(seed))
        suite: Dict[str, List[Dict[str, float]]] = {}
        for group_name, spec in suite_specs.items():
            suite[group_name] = [
                sample_variant_from_spec(rng, spec, dim=self._dim)
                for _ in range(int(n_per_group))
            ]
        return suite

    def optimal_value(self, variant_params: Optional[Dict[str, float]] = None) -> Optional[float]:
        _ = variant_params
        return None

    def estimate_global_min(self, variant_params: Optional[Dict[str, float]] = None, *, grid_size: int = 100) -> float:
        variant_params = variant_params or {}
        lower, upper = self.bounds
        dim = int(self.dim)

        dx = np.array([float(variant_params.get(f"dx{i + 1}", 0.0)) for i in range(dim)], dtype=np.float64)
        sx = np.array([float(variant_params.get(f"sx{i + 1}", 1.0)) for i in range(dim)], dtype=np.float64)
        R = rotation_matrix_from_params(dim, variant_params)
        x_opt = 0.5 + ((-dx) @ R) / np.maximum(sx, 1e-12)
        if np.all(x_opt >= lower.astype(np.float64) - 1e-10) and np.all(x_opt <= upper.astype(np.float64) + 1e-10):
            return 0.0

        n = int(max(4096, min(131072, int(grid_size) ** 2)))
        n = int(2 ** int(np.ceil(np.log2(max(2, n)))))

        from scipy.optimize import minimize
        from scipy.stats.qmc import Sobol

        sobol = Sobol(d=dim, scramble=True, seed=0)
        X = sobol.random(n)
        X = (X * (upper - lower) + lower).astype(np.float64)
        y = self.evaluate_numpy(X, variant_params).astype(np.float64).reshape(-1)

        best_min = float(np.min(y))
        start_points = X[np.argsort(y)[: min(30, len(y))]]

        def func(x: np.ndarray) -> float:
            x2d = np.asarray(x, dtype=np.float64).reshape(1, dim)
            return float(self.evaluate_numpy(x2d, variant_params)[0])

        bounds = [(float(lower[i]), float(upper[i])) for i in range(dim)]
        for x0 in start_points:
            try:
                res = minimize(func, np.asarray(x0, dtype=np.float64), bounds=bounds, method="L-BFGS-B")
                best_min = min(best_min, float(res.fun))
            except Exception:
                pass
        return float(best_min)
