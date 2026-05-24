"""
Ablation 3 training wrapper: w/o Cross-Attention.

Sets CAP_NO_CROSS_ATTN=1 env var, which triggers auto-patching in
myrl/rl/__init__.py (replaces ImprovedDualTowerSelector with mean-pooling).
Works for both main process and spawned multiprocessing workers.
"""
from __future__ import annotations

import os
os.environ["CAP_NO_CROSS_ATTN"] = "1"

from _bootstrap import bootstrap_project_root
bootstrap_project_root()

if __name__ == "__main__":
    from myrl.rl.train_rl import main
    print("[ABLATION] Cross-attention replaced with mean-pooling (CAP_NO_CROSS_ATTN=1)")
    main()
