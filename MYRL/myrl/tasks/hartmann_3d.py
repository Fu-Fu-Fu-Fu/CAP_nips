from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .base import TaskSpec


def hartmann_3d_numpy(X: np.ndarray) -> np.ndarray:
    """
    Hartmann 3D function (minimization) on [0, 1]^3.

    Reference: https://www.sfu.ca/~ssurjano/hart3.html

    f(x) = - sum_{i=1..4} alpha_i * exp( - sum_{j=1..3} A_{ij} (x_j - P_{ij})^2 )
    Global minimum: approximately -3.86278
    """
    X = np.atleast_2d(X).astype(np.float64)
    if X.shape[1] != 3:
        raise ValueError(f"Hartmann-3D expects dim=3, got X.shape={X.shape}")

    alpha = np.array([1.0, 1.2, 3.0, 3.2], dtype=np.float64)
    A = np.array(
        [
            [3.0, 10.0, 30.0],
            [0.1, 10.0, 35.0],
            [3.0, 10.0, 30.0],
            [0.1, 10.0, 35.0],
        ],
        dtype=np.float64,
    )
    P = 1e-4 * np.array(
        [
            [3689.0, 1170.0, 2673.0],
            [4699.0, 4387.0, 7470.0],
            [1091.0, 8732.0, 5547.0],
            [381.0, 5743.0, 8828.0],
        ],
        dtype=np.float64,
    )

    # X: (N, 1, 3) and P: (1, 4, 3) => broadcast to (N, 4, 3)
    Xb = X[:, None, :]
    diff2 = (Xb - P[None, :, :]) ** 2
    exponent = np.sum(A[None, :, :] * diff2, axis=-1)  # (N, 4)
    y = -np.sum(alpha[None, :] * np.exp(-exponent), axis=1)  # (N,)
    return y.astype(np.float32)


class Hartmann3DTask(TaskSpec):
    @property
    def task_name(self) -> str:
        return "hartmann_3d"

    @property
    def dim(self) -> int:
        return 3

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        upper = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        return lower, upper

    def evaluate_numpy(self, X: np.ndarray, variant_params: Optional[Dict[str, float]] = None) -> np.ndarray:
        _ = variant_params
        return hartmann_3d_numpy(X)

    def optimal_value(self, variant_params: Optional[Dict[str, float]] = None) -> Optional[float]:
        _ = variant_params
        # Known global minimum value on [0,1]^3.
        return -3.86278

