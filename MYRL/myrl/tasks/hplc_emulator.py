from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import TaskSpec

# ---------------------------------------------------------------------------
# Real-scale bounds for ALL 6 dimensions (emulator column order)
# Order: sample_loop, additional_volume, tubing_volume,
#        sample_flow, push_speed, wait_time
# ---------------------------------------------------------------------------
_ALL_LOWER_REAL = np.array([0.0, 0.0, 0.1, 0.5, 80.0, 0.5], dtype=np.float64)
_ALL_UPPER_REAL = np.array([0.08, 0.06, 0.9, 2.5, 150.0, 10.0], dtype=np.float64)


# ---------------------------------------------------------------------------
# Lazy emulator singleton — avoids loading TF model on every call / import
# ---------------------------------------------------------------------------
_EMULATOR_INSTANCE = None


def _get_emulator():
    global _EMULATOR_INSTANCE
    if _EMULATOR_INSTANCE is not None:
        return _EMULATOR_INSTANCE

    # Olympus NeuralNet uses TF1-style code; keras 3 breaks it.
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

    from olympus.datasets import Dataset
    from olympus.emulators.emulator import Emulator

    # Load pre-trained emulator (pickle + TF weights).
    emu = Emulator(dataset="hplc", model="NeuralNet")

    # The pickled Dataset is from an older olympus version and lacks modern
    # attributes (task, known_constraints, aux_param_space, ...).  Replace
    # it with a fresh Dataset that has all current attributes.
    emu.dataset = Dataset(kind="hplc")

    # Old pickle's DataTransformers have _stddev but not _stable_stddev
    # (added later to avoid division by zero).  Patch them.
    for tf in (emu.feature_transformer, emu.target_transformer):
        if hasattr(tf, "_stddev") and not hasattr(tf, "_stable_stddev"):
            tf._stable_stddev = np.where(tf._stddev == 0.0, 1.0, tf._stddev)
        if hasattr(tf, "_min") and not hasattr(tf, "_stable_min"):
            tf._stable_min = tf._min
        if hasattr(tf, "_max") and not hasattr(tf, "_stable_max"):
            tf._stable_max = tf._max

    _EMULATOR_INSTANCE = emu
    return _EMULATOR_INSTANCE


# ---------------------------------------------------------------------------
# Givens rotation helpers (6D: 6 planes, same as Hartmann 6D)
# ---------------------------------------------------------------------------
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
    Each dimension participates in exactly 2 planes.
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


# ---------------------------------------------------------------------------
# Variant suite specs  (affine transform ranges)
#   sx capped at 1.0 (no expansion) to reduce clip near boundary optima
#   6D space — same dx/rot ranges as Hartmann 6D, sx one-sided
# ---------------------------------------------------------------------------
_VARIANT_SUITE_SPECS: Dict[str, Dict[str, List[Tuple[float, float]]]] = {
    "in_range": {
        "dx": [(-0.04, 0.04)],
        "rot": [(-14.0, 14.0)],
        "sx": [(0.88, 1.00)],
    },
}


# ---------------------------------------------------------------------------
# Helpers (same pattern as hartmann_6d_family)
# ---------------------------------------------------------------------------
def _validate_segments(segments: List[Tuple[float, float]]) -> None:
    for lo, hi in segments:
        if not (float(lo) < float(hi)):
            raise ValueError(f"Invalid segment: ({lo}, {hi})")


def _sample_from_segments(
    rng: np.random.Generator, segments: List[Tuple[float, float]]
) -> float:
    _validate_segments(segments)
    lengths = np.array(
        [float(hi) - float(lo) for lo, hi in segments], dtype=np.float64
    )
    probs = lengths / lengths.sum()
    idx = int(rng.choice(len(segments), p=probs))
    lo, hi = segments[idx]
    return float(rng.uniform(float(lo), float(hi)))


def sample_variant_from_spec(
    rng: np.random.Generator,
    spec: Dict[str, List[Tuple[float, float]]],
) -> Dict[str, float]:
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
    return params


# ---------------------------------------------------------------------------
# Core evaluation: affine transform in [0,1]^6 → clip → real scale → emulator
# ---------------------------------------------------------------------------
def hplc_emulator_family_numpy(
    X: np.ndarray, variant_params: Dict[str, float]
) -> np.ndarray:
    """
    HPLC emulator variant family on [0,1]^6.

    We generate variants by applying an affine transform to the input:
      x' = clip( center + R * (S * (x - center)) + d, 0, 1 )
    then map x' from [0,1]^6 to real scale and call the emulator.

    Emulator columns: [sample_loop, additional_volume, tubing_volume,
                       sample_flow, push_speed, wait_time].

    The emulator's default goal is **maximize** peak_area, but our framework
    always **minimizes**.  Therefore we negate the emulator output.
    """
    X = np.atleast_2d(X).astype(np.float64)
    if X.shape[1] != 6:
        raise ValueError(
            f"HplcEmulatorTask expects dim=6, got X.shape={X.shape}"
        )

    # --- extract variant parameters ---
    dx = np.array(
        [float(variant_params.get(f"dx{i+1}", 0.0)) for i in range(6)],
        dtype=np.float64,
    )
    sx = np.array(
        [float(variant_params.get(f"sx{i+1}", 1.0)) for i in range(6)],
        dtype=np.float64,
    )
    r01 = float(variant_params.get("r01", 0.0))
    r23 = float(variant_params.get("r23", 0.0))
    r45 = float(variant_params.get("r45", 0.0))
    r03 = float(variant_params.get("r03", 0.0))
    r14 = float(variant_params.get("r14", 0.0))
    r25 = float(variant_params.get("r25", 0.0))

    # --- affine transform in [0,1]^6 ---
    center = np.full(6, 0.5, dtype=np.float64)
    Xc = X - center[None, :]
    S = np.diag(sx)

    has_rotation = any(abs(a) > 1e-12 for a in [r01, r23, r45, r03, r14, r25])
    if has_rotation:
        R = _rotation_matrix_from_givens_deg(r01, r23, r45, r03, r14, r25)
    else:
        R = np.eye(6, dtype=np.float64)

    Xp = center[None, :] + (Xc @ S.T) @ R.T + dx[None, :]
    Xp = np.clip(Xp, 0.0, 1.0)

    # --- map [0,1]^6 → real scale ---
    X_real = _ALL_LOWER_REAL[None, :] + Xp * (
        _ALL_UPPER_REAL[None, :] - _ALL_LOWER_REAL[None, :]
    )

    # --- call emulator ---
    emulator = _get_emulator()
    out = emulator.run(X_real, num_samples=1)
    y_preds = out[0] if isinstance(out, tuple) else out

    result = np.asarray(y_preds, dtype=np.float64).reshape(-1)

    # Negate: emulator maximizes peak_area, framework minimizes
    return (-result).astype(np.float32)


# ---------------------------------------------------------------------------
# Float64 evaluation for L-BFGS-B (global min estimation)
# ---------------------------------------------------------------------------
def _evaluate_f64(X: np.ndarray, variant_params: Dict[str, float]) -> np.ndarray:
    """Family evaluation in float64 for L-BFGS-B optimization."""
    return hplc_emulator_family_numpy(X, variant_params).astype(np.float64)


# ---------------------------------------------------------------------------
# Task implementation
# ---------------------------------------------------------------------------
class HplcEmulatorTask(TaskSpec):
    """HPLC emulator task (6D, transform-based variants).

    The optimizer sees a **[0,1]^6 normalized** search space (sample_loop,
    additional_volume, tubing_volume, sample_flow, push_speed, wait_time).

    Variants are generated via affine input transforms (rotation + scaling +
    translation) in [0,1]^6, identical to the approach used for Hartmann 6D.
    ``evaluate_numpy`` applies the transform, clips to [0,1]^6, maps to real
    scale, and calls the olympus emulator.

    The emulator's default goal is **maximize** peak_area, but our framework
    always **minimizes**.  Therefore we negate the emulator output.
    """

    @property
    def task_name(self) -> str:
        return "hplc_emulator"

    @property
    def dim(self) -> int:
        return 6

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.zeros(6, dtype=np.float32)
        upper = np.ones(6, dtype=np.float32)
        return lower, upper

    # ----- evaluation --------------------------------------------------------

    def evaluate_numpy(
        self,
        X: np.ndarray,
        variant_params: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        return hplc_emulator_family_numpy(X, variant_params or {})

    # ----- variant sampling --------------------------------------------------

    def sample_train_variants(
        self, *, k: int, seed: int
    ) -> List[Dict[str, float]]:
        rng = np.random.default_rng(int(seed))
        return [
            sample_variant_from_spec(rng, _VARIANT_SUITE_SPECS["in_range"])
            for _ in range(int(k))
        ]

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
            suite[group_name] = [
                sample_variant_from_spec(rng, spec)
                for _ in range(int(n_per_group))
            ]
        return suite

    # ----- global min estimation ---------------------------------------------

    def estimate_global_min(
        self,
        variant_params: Optional[Dict[str, float]] = None,
        *,
        grid_size: int = 100,
    ) -> float:
        """Sobol random search + multi-start L-BFGS-B on [0,1]^6."""
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

        def func(x):
            x = np.asarray(x, dtype=np.float64).reshape(1, dim)
            return float(_evaluate_f64(x, variant_params)[0])

        opt_bounds = [(float(lower[i]), float(upper[i])) for i in range(dim)]
        for x0 in start_points:
            try:
                res = minimize(
                    func,
                    np.asarray(x0, dtype=np.float64),
                    bounds=opt_bounds,
                    method="L-BFGS-B",
                )
                best_min = min(best_min, float(res.fun))
            except Exception:
                pass
        return float(best_min)
