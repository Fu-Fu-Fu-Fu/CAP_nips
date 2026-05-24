from __future__ import annotations

import os
import sys


def bootstrap_project_root() -> None:
    """
    Allow running `python MYRL/scripts/*.py` from anywhere by ensuring `MYRL/` is on sys.path.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(this_dir, ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

