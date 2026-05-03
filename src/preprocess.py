from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import signal


COMPONENTS = ("BH1", "BH2", "BHZ", "HYD")


def _rotate_horizontal(
    bh1: np.ndarray,
    bh2: np.ndarray,
    orientation_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(float(orientation_deg))
    c = np.cos(theta)
    s = np.sin(theta)
    n = bh1 * c - bh2 * s
    e = bh1 * s + bh2 * c
    return n, e


def preprocess_signals(
    signals: Dict[str, np.ndarray],
    fs: float,
    enable_demean: bool = True,
    enable_detrend: bool = True,
    apply_orientation: bool = True,
    orientation_deg: float = 0.0,
) -> Tuple[Dict[str, np.ndarray], dict]:
    out = {k: np.asarray(v, dtype=np.float64).copy() for k, v in signals.items()}

    if enable_demean:
        for k in COMPONENTS:
            out[k] = out[k] - np.mean(out[k])

    if enable_detrend:
        for k in COMPONENTS:
            out[k] = signal.detrend(out[k], type="linear")

    if apply_orientation:
        out["BH1"], out["BH2"] = _rotate_horizontal(out["BH1"], out["BH2"], orientation_deg=orientation_deg)

    report = {
        "enable_demean": bool(enable_demean),
        "enable_detrend": bool(enable_detrend),
        "apply_orientation": bool(apply_orientation),
        "orientation_deg": float(orientation_deg),
    }
    return out, report
