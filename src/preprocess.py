from __future__ import annotations

from fractions import Fraction
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


def downsample_signals(
    t_sec: np.ndarray,
    signals: Dict[str, np.ndarray],
    fs: float,
    target_fs: float,
    max_denominator: int = 100000,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], float, dict]:
    fs_in = float(fs)
    fs_target = float(target_fs)
    if fs_target <= 0:
        raise ValueError(f"target_fs must be > 0, got: {target_fs}")
    if fs_target >= fs_in:
        raise ValueError(
            f"target_fs must be lower than original fs for downsampling, got target_fs={fs_target}, fs={fs_in}"
        )

    ratio = fs_target / fs_in
    frac = Fraction(ratio).limit_denominator(int(max_denominator))
    up = int(frac.numerator)
    down = int(frac.denominator)
    fs_actual = fs_in * up / down

    out: Dict[str, np.ndarray] = {}
    expected_len = None
    for comp in COMPONENTS:
        y = signal.resample_poly(np.asarray(signals[comp], dtype=np.float64), up=up, down=down)
        if expected_len is None:
            expected_len = int(y.size)
        elif int(y.size) != expected_len:
            raise RuntimeError(
                f"Resampling produced inconsistent lengths across components: {comp}={y.size}, expected={expected_len}"
            )
        out[comp] = y.astype(np.float64)

    if expected_len is None or expected_len <= 0:
        raise RuntimeError("Downsampling produced empty output.")

    t_new = np.arange(expected_len, dtype=np.float64) / float(fs_actual)
    report = {
        "enabled": True,
        "input_fs_hz": fs_in,
        "target_fs_hz": fs_target,
        "actual_fs_hz": float(fs_actual),
        "ratio_up": up,
        "ratio_down": down,
        "abs_error_hz": float(abs(fs_actual - fs_target)),
    }
    return t_new, out, float(fs_actual), report
