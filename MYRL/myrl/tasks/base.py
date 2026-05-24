from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ObjectiveFunction(ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def optimal_value(self) -> Optional[float]:
        return None


@dataclass(frozen=True)
class VariantSuiteSpec:
    name: str
    spec: Dict[str, Any]


class TaskSpec(ABC):
    """
    Task adapter.

    To support a new objective/function family, implement this interface and register it.
    """

    @property
    @abstractmethod
    def task_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def evaluate_numpy(self, X: np.ndarray, variant_params: Optional[Dict[str, float]] = None) -> np.ndarray:
        raise NotImplementedError

    def sample_train_variants(self, *, k: int, seed: int) -> List[Dict[str, float]]:
        _ = seed
        return [{} for _ in range(int(k))]

    def optimal_value(self, variant_params: Optional[Dict[str, float]] = None) -> Optional[float]:
        _ = variant_params
        return None

    def estimate_global_min(self, variant_params: Optional[Dict[str, float]] = None, *, grid_size: int = 100) -> float:
        """
        Default global-min estimator (2D only): grid search on bounds.
        If a task has an analytic/known optimum, override `optimal_value()` or this method.
        """
        opt = self.optimal_value(variant_params)
        if opt is not None:
            return float(opt)

        if int(self.dim) != 2:
            raise ValueError(f"{self.task_name}: default estimate_global_min only supports dim=2")

        lower, upper = self.bounds
        x1 = np.linspace(float(lower[0]), float(upper[0]), int(grid_size), dtype=np.float64)
        x2 = np.linspace(float(lower[1]), float(upper[1]), int(grid_size), dtype=np.float64)
        X1, X2 = np.meshgrid(x1, x2)
        X = np.stack([X1.reshape(-1), X2.reshape(-1)], axis=1).astype(np.float64)
        y = np.asarray(self.evaluate_numpy(X, variant_params), dtype=np.float64).reshape(-1)
        return float(np.min(y))

    def default_variant_suite(self) -> Dict[str, Any]:
        return {"in_range": {"n": 1}}

    def sample_eval_suite(self, *, n_per_group: int, seed: int, suite_specs: Optional[Dict[str, Any]] = None) -> Dict[str, List[Dict[str, float]]]:
        _ = seed
        _ = suite_specs
        return {"in_range": [{} for _ in range(int(n_per_group))]}
