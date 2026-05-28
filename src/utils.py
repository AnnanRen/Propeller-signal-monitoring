from __future__ import annotations

from pathlib import Path

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def zscore_safe(x: np.ndarray) -> np.ndarray:
    std = np.std(x)
    if std <= 0:
        return np.zeros_like(x, dtype=np.float64)
    return (x - np.mean(x)) / std
