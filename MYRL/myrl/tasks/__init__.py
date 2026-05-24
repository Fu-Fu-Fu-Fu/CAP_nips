from .registry import get_task, list_tasks, register_task

# Register built-in tasks
from . import builtin as _builtin  # noqa: F401

__all__ = ["get_task", "list_tasks", "register_task"]
