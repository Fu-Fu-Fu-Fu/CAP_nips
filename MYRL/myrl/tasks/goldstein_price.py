from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .base import TaskSpec


def goldstein_price_numpy(X: np.ndarray) -> np.ndarray:
    """
    Goldstein-Price function (minimization), standard domain: x1,x2 in [-2, 2].
    Reference form:
      f(x,y) = [1 + (x+y+1)^2*(19-14x+3x^2-14y+6xy+3y^2)]
               * [30 + (2x-3y)^2*(18-32x+12x^2+48y-36xy+27y^2)]
    """
    X = np.atleast_2d(X).astype(np.float64)
    x = X[:, 0]
    y = X[:, 1]

    a = 1.0 + (x + y + 1.0) ** 2 * (19.0 - 14.0 * x + 3.0 * x**2 - 14.0 * y + 6.0 * x * y + 3.0 * y**2)
    b = 30.0 + (2.0 * x - 3.0 * y) ** 2 * (18.0 - 32.0 * x + 12.0 * x**2 + 48.0 * y - 36.0 * x * y + 27.0 * y**2)
    return (a * b).astype(np.float32)


class GoldsteinPriceTask(TaskSpec):
    @property
    def task_name(self) -> str:
        return "goldstein_price"

    @property
    def dim(self) -> int:
        return 2

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.array([-2.0, -2.0], dtype=np.float32)
        upper = np.array([2.0, 2.0], dtype=np.float32)
        return lower, upper

    def evaluate_numpy(self, X: np.ndarray, variant_params: Optional[Dict[str, float]] = None) -> np.ndarray:
        _ = variant_params
        return goldstein_price_numpy(X)

    def optimal_value(self, variant_params: Optional[Dict[str, float]] = None) -> Optional[float]:
        _ = variant_params
        # Standard Goldstein-Price global minimum is 3 at (0, -1).
        return 3.0
