from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import interpolate


def compute_azimuth_spectrogram(
    s_p: np.ndarray,
    s_vn: np.ndarray,
    s_ve: np.ndarray,
    eps: float = 1e-12,
) -> Dict[str, np.ndarray]:
    i_n = 0.5 * np.real(np.conj(s_p) * s_vn)
    i_e = 0.5 * np.real(np.conj(s_p) * s_ve)
    intensity = np.sqrt(i_n ** 2 + i_e ** 2)

    azimuth_deg = np.degrees(np.arctan2(i_e, i_n))
    azimuth_deg = np.mod(azimuth_deg, 360.0)

    weak = intensity <= eps
    azimuth_deg = azimuth_deg.astype(np.float64)
    azimuth_deg[weak] = np.nan

    return {
        "azimuth_deg": azimuth_deg,
        "intensity": intensity,
        "I_N": i_n,
        "I_E": i_e,
    }


def compute_azimuth_stability(
    azimuth_deg_tf: np.ndarray,
    t_spec: np.ndarray,
    window_size: int = 15,
    step_size: int = 5,
) -> Dict[str, np.ndarray]:
    azimuth_rad = np.deg2rad(azimuth_deg_tf)
    n_freq, n_time = azimuth_rad.shape

    if n_time < window_size:
        return {
            "R_windows": np.zeros((n_freq, 0), dtype=np.float64),
            "R_interp": np.zeros((n_freq, n_time), dtype=np.float64),
            "window_times": np.array([], dtype=np.float64),
        }

    n_windows = (n_time - window_size) // step_size + 1
    r_windows = np.zeros((n_freq, n_windows), dtype=np.float64)

    for fi in range(n_freq):
        series = azimuth_rad[fi, :]
        for wi in range(n_windows):
            t0 = wi * step_size
            t1 = t0 + window_size
            seg = series[t0:t1]
            seg = seg[~np.isnan(seg)]

            if seg.size == 0:
                r_windows[fi, wi] = 0.0
                continue

            cos_sum = np.sum(np.cos(seg))
            sin_sum = np.sum(np.sin(seg))
            r_windows[fi, wi] = np.sqrt(cos_sum ** 2 + sin_sum ** 2) / seg.size

    if t_spec.size > 1:
        dt_spec = t_spec[1] - t_spec[0]
    else:
        dt_spec = 1.0

    centers = np.arange(n_windows) * step_size + window_size / 2.0
    window_times = t_spec[0] + centers * dt_spec

    r_interp = np.zeros((n_freq, n_time), dtype=np.float64)
    for fi in range(n_freq):
        if window_times.size > 1:
            f_interp = interpolate.interp1d(
                window_times,
                r_windows[fi, :],
                kind="linear",
                bounds_error=False,
                fill_value=(r_windows[fi, 0], r_windows[fi, -1]),
            )
            r_interp[fi, :] = f_interp(t_spec)
        elif window_times.size == 1:
            r_interp[fi, :] = r_windows[fi, 0]

    return {
        "R_windows": r_windows,
        "R_interp": np.clip(r_interp, 0.0, 1.0),
        "window_times": window_times,
    }


def compute_confidence_map(
    s_p: np.ndarray,
    s_vn: np.ndarray,
    s_ve: np.ndarray,
    intensity: np.ndarray,
) -> np.ndarray:
    eps = np.finfo(float).eps

    v_h = np.sqrt(np.abs(s_vn) ** 2 + np.abs(s_ve) ** 2)
    coh = np.abs(np.conj(s_p) * (s_vn + 1j * s_ve)) / (np.abs(s_p) * v_h + eps)
    coh = np.clip(coh, 0.0, 1.0)

    p95 = np.percentile(intensity, 95) + eps
    i_score = np.clip(intensity / p95, 0.0, 1.0)

    conf = 0.6 * coh + 0.4 * i_score
    return np.clip(conf, 0.0, 1.0)


def apply_confidence_mask(
    azimuth_deg_tf: np.ndarray,
    confidence_tf: np.ndarray,
    threshold: float = 0.6,
) -> np.ndarray:
    masked = azimuth_deg_tf.copy().astype(np.float64)
    masked[confidence_tf < threshold] = np.nan
    return masked
