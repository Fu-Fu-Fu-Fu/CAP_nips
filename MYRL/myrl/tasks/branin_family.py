from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import TaskSpec


CENTER_X1 = 2.5
CENTER_X2 = 7.5


def branin_family_numpy(X: np.ndarray, variant_params: Dict[str, float]) -> np.ndarray:
    X = np.atleast_2d(X).astype(np.float64)
    x1 = X[:, 0]
    x2 = X[:, 1]

    dx1 = float(variant_params.get("dx1", 0.0))
    dx2 = float(variant_params.get("dx2", 0.0))
    sx1 = float(variant_params.get("sx1", 1.0))
    sx2 = float(variant_params.get("sx2", 1.0))
    rotation = float(variant_params.get("rotation", 0.0))
    alpha = float(variant_params.get("alpha", 1.0))
    beta = float(variant_params.get("beta", 0.0))

    if abs(rotation) > 1e-12:
        theta = rotation * np.pi / 180.0
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        x1c = x1 - CENTER_X1
        x2c = x2 - CENTER_X2
        x1r = cos_t * x1c - sin_t * x2c + CENTER_X1
        x2r = sin_t * x1c + cos_t * x2c + CENTER_X2
    else:
        x1r = x1
        x2r = x2

    x1t = sx1 * x1r + dx1
    x2t = sx2 * x2r + dx2

    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)

    y = a * (x2t - b * x1t**2 + c * x1t - r) ** 2 + s * (1.0 - t) * np.cos(x1t) + s
    y = alpha * y + beta
    return y.astype(np.float32)


_VARIANT_SUITE_SPECS: Dict[str, Dict[str, List[Tuple[float, float]]]] = {
    "in_range": {
        "dx": [(-1.0, 1.0)],
        "rotation": [(-15.0, 15.0)],
        "sx": [(0.9, 1.1)],
    },
    "ood_level_1": {
        "dx": [(-2.0, -1.0), (1.0, 2.0)],
        "rotation": [(-30.0, -15.0), (15.0, 30.0)],
        "sx": [(0.8, 0.9), (1.1, 1.2)],
    },
    "ood_level_2": {
        "dx": [(-3.0, -2.0), (2.0, 3.0)],
        "rotation": [(-45.0, -30.0), (30.0, 45.0)],
        "sx": [(0.7, 0.8), (1.2, 1.3)],
    },
    "ood_level_3": {
        "dx": [(-4.0, -3.0), (3.0, 4.0)],
        "rotation": [(-60.0, -45.0), (45.0, 60.0)],
        "sx": [(0.6, 0.7), (1.3, 1.4)],
    },
}


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


def sample_variant_from_spec(rng: np.random.Generator, spec: Dict[str, List[Tuple[float, float]]]) -> Dict[str, float]:
    dx1 = _sample_from_segments(rng, spec["dx"])
    dx2 = _sample_from_segments(rng, spec["dx"])
    rotation = _sample_from_segments(rng, spec["rotation"])
    sx1 = _sample_from_segments(rng, spec["sx"])
    sx2 = _sample_from_segments(rng, spec["sx"])
    return {
        "dx1": float(dx1),
        "dx2": float(dx2),
        "rotation": float(rotation),
        "sx1": float(sx1),
        "sx2": float(sx2),
        "alpha": 1.0,
        "beta": 0.0,
    }


class BraninFamilyTask(TaskSpec):
    @property
    def task_name(self) -> str:
        return "branin_family"

    @property
    def dim(self) -> int:
        return 2

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.array([-5.0, 0.0], dtype=np.float32)
        upper = np.array([10.0, 15.0], dtype=np.float32)
        return lower, upper

    def evaluate_numpy(self, X: np.ndarray, variant_params: Optional[Dict[str, float]] = None) -> np.ndarray:
        return branin_family_numpy(X, variant_params or {})

    def sample_train_variants(self, *, k: int, seed: int) -> List[Dict[str, float]]:
        rng = np.random.default_rng(int(seed))
        variants: List[Dict[str, float]] = []
        for _ in range(int(k)):
            variants.append(sample_variant_from_spec(rng, _VARIANT_SUITE_SPECS["in_range"]))
        return variants

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
        Match the original BraninVariantFunction._find_global_min:
        - coarse grid search
        - multi-start L-BFGS-B refinement
        """
        variant_params = variant_params or {}

        lower, upper = self.bounds
        x1 = np.linspace(float(lower[0]), float(upper[0]), int(grid_size), dtype=np.float64)
        x2 = np.linspace(float(lower[1]), float(upper[1]), int(grid_size), dtype=np.float64)
        X1, X2 = np.meshgrid(x1, x2)
        grid_points = np.stack([X1.flatten(), X2.flatten()], axis=1).astype(np.float32)
        y_grid = self.evaluate_numpy(grid_points, variant_params)

        sorted_indices = np.argsort(y_grid)
        n_starts = 20
        start_points = grid_points[sorted_indices[:n_starts]]

        from scipy.optimize import minimize

        def func(x):
            x = np.array(x, dtype=np.float32).reshape(1, 2)
            return float(self.evaluate_numpy(x, variant_params)[0])

        bounds = [(float(lower[0]), float(upper[0])), (float(lower[1]), float(upper[1]))]
        best_min = float("inf")
        for x0 in start_points:
            try:
                res = minimize(func, x0, bounds=bounds, method="L-BFGS-B")
                if res.fun < best_min:
                    best_min = float(res.fun)
            except Exception:
                pass

        best_min = min(best_min, float(np.min(y_grid)))
        return float(best_min)
