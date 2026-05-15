"""Reproducibility seed helpers."""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - hardware dependent
        torch.cuda.manual_seed_all(seed)
