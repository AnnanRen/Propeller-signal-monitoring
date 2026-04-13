from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import signal


COMPONENTS = ("BH1", "BH2", "BHZ", "HYD")


def _build_band(
    fs: float,
    custom_band: Tuple[float, float] | None,
) -> Tuple[float, float]:
    nyquist = fs / 2.0
    default_low = 2.0
    default_high = 0.8 * nyquist

    if custom_band is None:
        low, high = default_low, default_high
    else:
        low, high = float(custom_band[0]), float(custom_band[1])

    low = max(0.01, low)
    high = min(0.95 * nyquist, high)
    if high <= low:
        raise ValueError(f"Invalid bandpass range after bounds check: ({low}, {high})")
    return low, high


def _bandpass_filter(x: np.ndarray, fs: float, band: Tuple[float, float], order: int = 4) -> np.ndarray:
    low, high = band
    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


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
    enable_bandpass: bool = True,
    preprocess_band: Tuple[float, float] | None = None,
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

    band_used = None
    if enable_bandpass:
        band_used = _build_band(fs, preprocess_band)
        for k in COMPONENTS:
            out[k] = _bandpass_filter(out[k], fs=fs, band=band_used, order=4)

    if apply_orientation:
        out["BH1"], out["BH2"] = _rotate_horizontal(out["BH1"], out["BH2"], orientation_deg=orientation_deg)

    report = {
        "enable_demean": bool(enable_demean),
        "enable_detrend": bool(enable_detrend),
        "enable_bandpass": bool(enable_bandpass),
        "preprocess_band_hz": band_used,
        "apply_orientation": bool(apply_orientation),
        "orientation_deg": float(orientation_deg),
    }
    return out, report
