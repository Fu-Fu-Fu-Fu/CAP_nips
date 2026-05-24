from __future__ import annotations

from typing import Dict, List

from .base import TaskSpec

_TASKS: Dict[str, TaskSpec] = {}


def register_task(task: TaskSpec) -> None:
    name = str(task.task_name)
    if name in _TASKS:
        raise KeyError(f"Task already registered: {name}")
    _TASKS[name] = task


def get_task(name: str) -> TaskSpec:
    if name not in _TASKS:
        raise KeyError(f"Unknown task '{name}'. Available: {', '.join(list_tasks())}")
    return _TASKS[name]


def list_tasks() -> List[str]:
    return sorted(_TASKS.keys())

