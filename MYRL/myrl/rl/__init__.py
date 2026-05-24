__all__ = []

import os as _os

# Auto-patch for ablation 3 (w/o cross-attention).
# When CAP_NO_CROSS_ATTN=1, replace ImprovedDualTowerSelector with mean-pooling variant.
# This works in both main process and spawned multiprocessing workers because
# __init__.py runs whenever the package is imported.
if _os.environ.get("CAP_NO_CROSS_ATTN") == "1":
    from . import train_rl as _trl
    from .ablation_no_crossattn import NoCrossAttnDualTowerSelector
    _trl.ImprovedDualTowerSelector = NoCrossAttnDualTowerSelector
    del _trl, NoCrossAttnDualTowerSelector
