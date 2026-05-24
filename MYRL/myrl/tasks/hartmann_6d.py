from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .base import TaskSpec


def hartmann_6d_numpy(X: np.ndarray) -> np.ndarray:
    """
    Hartmann 6D function (minimization) on [0, 1]^6.

    Reference: https://www.sfu.ca/~ssurjano/hart6.html

    f(x) = - sum_{i=1..4} alpha_i * exp( - sum_{j=1..6} A_{ij} (x_j - P_{ij})^2 )
    Global minimum: approximately -3.32237
    """
    X = np.atleast_2d(X).astype(np.float64)
    if X.shape[1] != 6:
        raise ValueError(f"Hartmann-6D expects dim=6, got X.shape={X.shape}")

    alpha = np.array([1.0, 1.2, 3.0, 3.2], dtype=np.float64)
    A = np.array(
        [
            [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
            [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
            [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
            [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
        ],
        dtype=np.float64,
    )
    P = 1e-4 * np.array(
        [
            [1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0],
            [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
            [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
            [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0],
        ],
        dtype=np.float64,
    )

    # X: (N, 1, 6) and P: (1, 4, 6) => broadcast to (N, 4, 6)
    Xb = X[:, None, :]
    diff2 = (Xb - P[None, :, :]) ** 2
    exponent = np.sum(A[None, :, :] * diff2, axis=-1)  # (N, 4)
    y = -np.sum(alpha[None, :] * np.exp(-exponent), axis=1)  # (N,)
    return y.astype(np.float32)


class Hartmann6DTask(TaskSpec):
    @property
    def task_name(self) -> str:
        return "hartmann_6d"

    @property
    def dim(self) -> int:
        return 6

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.zeros(6, dtype=np.float32)
        upper = np.ones(6, dtype=np.float32)
        return lower, upper

    def evaluate_numpy(self, X: np.ndarray, variant_params: Optional[Dict[str, float]] = None) -> np.ndarray:
        _ = variant_params
        return hartmann_6d_numpy(X)

    def optimal_value(self, variant_params: Optional[Dict[str, float]] = None) -> Optional[float]:
        _ = variant_params
        # Known global minimum value on [0,1]^6.
        return -3.32237
