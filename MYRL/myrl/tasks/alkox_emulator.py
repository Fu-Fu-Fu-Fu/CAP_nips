from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import TaskSpec

# ---------------------------------------------------------------------------
# Real-scale bounds for ALL 4 dimensions (emulator column order)
# Order: catalase, peroxidase, alcohol_oxidase, ph
# ---------------------------------------------------------------------------
_ALL_LOWER_REAL = np.array([0.05, 0.5, 2.0, 6.0], dtype=np.float64)
_ALL_UPPER_REAL = np.array([1.0, 10.0, 8.0, 8.0], dtype=np.float64)


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
    emu = Emulator(dataset="alkox", model="NeuralNet")

    # The pickled Dataset is from an older olympus version and lacks modern
    # attributes (task, known_constraints, aux_param_space, ...).  Replace
    # it with a fresh Dataset that has all current attributes.
    emu.dataset = Dataset(kind="alkox")

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
# Givens rotation helpers (4D: 4 planes, each dim in exactly 2 planes)
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
    r01: float, r23: float, r02: float, r13: float,
) -> np.ndarray:
    """
    Compose 4 Givens rotations in 4D space.

    Rotation planes: (0,1), (2,3), (0,2), (1,3).
    Each dimension participates in exactly 2 planes.
    Angles are in degrees.
    """
    dim = 4
    planes = [(0, 1), (2, 3), (0, 2), (1, 3)]
    angles_deg = [r01, r23, r02, r13]
    R = np.eye(dim, dtype=np.float64)
    for (i, j), a in zip(planes, angles_deg):
        theta = float(a) * np.pi / 180.0
        if abs(theta) > 1e-12:
            R = R @ _givens_rotation(dim, i, j, theta)
    return R


# ---------------------------------------------------------------------------
# Variant suite specs  (affine transform ranges)
#   sx capped at 1.0 (no expansion) to reduce clip near boundary optima
#   4D space — slightly larger perturbations than 6D
# ---------------------------------------------------------------------------
_VARIANT_SUITE_SPECS: Dict[str, Dict[str, List[Tuple[float, float]]]] = {
    "in_range": {
        "dx": [(-0.08, 0.08)],
        "rot": [(-28.0, 28.0)],
        "sx": [(0.84, 1.00)],
    },
    "ood_level_1": {
        "dx": [(-0.12, -0.08), (0.08, 0.12)],
        "rot": [(-38.0, -28.0), (28.0, 38.0)],
        "sx": [(0.76, 0.84)],
    },
    "ood_level_2": {
        "dx": [(-0.16, -0.12), (0.12, 0.16)],
        "rot": [(-48.0, -38.0), (38.0, 48.0)],
        "sx": [(0.68, 0.76)],
    },
    "ood_level_3": {
        "dx": [(-0.20, -0.16), (0.16, 0.20)],
        "rot": [(-58.0, -48.0), (48.0, 58.0)],
        "sx": [(0.60, 0.68)],
    },
}


# ---------------------------------------------------------------------------
# Helpers (same pattern as benzylation_emulator / hartmann_6d_family)
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
    dx = [_sample_from_segments(rng, spec["dx"]) for _ in range(4)]
    sx = [_sample_from_segments(rng, spec["sx"]) for _ in range(4)]
    rot_keys = ["r01", "r23", "r02", "r13"]
    rot = [_sample_from_segments(rng, spec["rot"]) for _ in range(4)]
    params: Dict[str, float] = {}
    for i in range(4):
        params[f"dx{i+1}"] = float(dx[i])
    for i in range(4):
        params[f"sx{i+1}"] = float(sx[i])
    for key, val in zip(rot_keys, rot):
        params[key] = float(val)
    return params


# ---------------------------------------------------------------------------
# Core evaluation: affine transform in [0,1]^4 → clip → real scale → emulator
# ---------------------------------------------------------------------------
def alkox_emulator_family_numpy(
    X: np.ndarray, variant_params: Dict[str, float]
) -> np.ndarray:
    """
    Alkox emulator variant family on [0,1]^4.

    We generate variants by applying an affine transform to the input:
      x' = clip( center + R * (S * (x - center)) + d, 0, 1 )
    then map x' from [0,1]^4 to real scale and call the emulator.

    Emulator columns: [catalase, peroxidase, alcohol_oxidase, ph].

    The emulator's default goal is **maximize** conversion, but our framework
    always **minimizes**.  Therefore we negate the emulator output.
    """
    X = np.atleast_2d(X).astype(np.float64)
    if X.shape[1] != 4:
        raise ValueError(
            f"AlkoxEmulatorTask expects dim=4, got X.shape={X.shape}"
        )

    # --- extract variant parameters ---
    dx = np.array(
        [float(variant_params.get(f"dx{i+1}", 0.0)) for i in range(4)],
        dtype=np.float64,
    )
    sx = np.array(
        [float(variant_params.get(f"sx{i+1}", 1.0)) for i in range(4)],
        dtype=np.float64,
    )
    r01 = float(variant_params.get("r01", 0.0))
    r23 = float(variant_params.get("r23", 0.0))
    r02 = float(variant_params.get("r02", 0.0))
    r13 = float(variant_params.get("r13", 0.0))

    # --- affine transform in [0,1]^4 ---
    center = np.full(4, 0.5, dtype=np.float64)
    Xc = X - center[None, :]
    S = np.diag(sx)

    has_rotation = any(abs(a) > 1e-12 for a in [r01, r23, r02, r13])
    if has_rotation:
        R = _rotation_matrix_from_givens_deg(r01, r23, r02, r13)
    else:
        R = np.eye(4, dtype=np.float64)

    Xp = center[None, :] + (Xc @ S.T) @ R.T + dx[None, :]
    Xp = np.clip(Xp, 0.0, 1.0)

    # --- map [0,1]^4 → real scale ---
    X_real = _ALL_LOWER_REAL[None, :] + Xp * (
        _ALL_UPPER_REAL[None, :] - _ALL_LOWER_REAL[None, :]
    )

    # --- call emulator (columns: catalase, peroxidase, alcohol_oxidase, ph) ---
    emulator = _get_emulator()
    out = emulator.run(X_real, num_samples=1)
    y_preds = out[0] if isinstance(out, tuple) else out

    result = np.asarray(y_preds, dtype=np.float64).reshape(-1)

    # Negate: emulator maximizes conversion, framework minimizes
    return (-result).astype(np.float32)


# ---------------------------------------------------------------------------
# Float64 evaluation for L-BFGS-B (global min estimation)
# ---------------------------------------------------------------------------
def _evaluate_f64(X: np.ndarray, variant_params: Dict[str, float]) -> np.ndarray:
    """Family evaluation in float64 for L-BFGS-B optimization."""
    return alkox_emulator_family_numpy(X, variant_params).astype(np.float64)


# ---------------------------------------------------------------------------
# Task implementation
# ---------------------------------------------------------------------------
class AlkoxEmulatorTask(TaskSpec):
    """Alkox reaction emulator task (4D, transform-based variants).

    The optimizer sees a **[0,1]^4 normalized** search space
    (catalase, peroxidase, alcohol_oxidase, ph).

    Variants are generated via affine input transforms (rotation + scaling +
    translation) in [0,1]^4, identical to the approach used for synthetic
    function families (Hartmann 6D, Branin) and the benzylation emulator.
    ``evaluate_numpy`` applies the transform, clips to [0,1]^4, maps to real
    scale, calls the olympus emulator, and **negates** the output (maximize
    conversion → minimize -conversion).
    """

    @property
    def task_name(self) -> str:
        return "alkox_emulator"

    @property
    def dim(self) -> int:
        return 4

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.zeros(4, dtype=np.float32)
        upper = np.ones(4, dtype=np.float32)
        return lower, upper

    # ----- evaluation --------------------------------------------------------

    def evaluate_numpy(
        self,
        X: np.ndarray,
        variant_params: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        return alkox_emulator_family_numpy(X, variant_params or {})

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
        """Sobol random search + multi-start L-BFGS-B on [0,1]^4."""
        variant_params = variant_params or {}
        lower, upper = self.bounds
        dim = int(self.dim)
        n = int(max(2048, min(65536, int(grid_size) ** 2)))
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
