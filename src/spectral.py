from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy import signal
from scipy.signal.windows import hamming


@dataclass(frozen=True)
class SpectralParams:
    window_length_s: float = 2.0
    overlap: float = 0.5


def compute_stft(
    x: np.ndarray,
    fs: float,
    params: SpectralParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nfft = max(8, int(round(params.window_length_s * fs)))
    noverlap = int(round(nfft * params.overlap))
    window = hamming(nfft)

    f_hz, t_spec, s_complex = signal.spectrogram(
        x,
        fs=fs,
        window=window,
        nperseg=nfft,
        noverlap=noverlap,
        nfft=nfft,
        mode="complex",
    )
    return f_hz, t_spec, s_complex


def power_db(s_complex: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.abs(s_complex) ** 2 + np.finfo(float).eps)


def lofar_from_spectrogram(s_db: np.ndarray) -> np.ndarray:
    out = s_db.copy().astype(np.float64)
    for i in range(out.shape[0]):
        std = np.std(out[i, :])
        if std > 0:
            out[i, :] = (out[i, :] - np.mean(out[i, :])) / std
        else:
            out[i, :] = 0.0
    return out


def _find_contiguous_true_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []

    regions: List[Tuple[int, int]] = []
    start = idx[0]
    prev = idx[0]
    for cur in idx[1:]:
        if cur == prev + 1:
            prev = cur
            continue
        regions.append((start, prev))
        start = cur
        prev = cur
    regions.append((start, prev))
    return regions


def suggest_frequency_bands(
    f_hz: np.ndarray,
    s_db: np.ndarray,
    fs: float,
    window_length_s: float,
    max_candidates: int = 3,
) -> Dict[str, object]:
    nyquist = fs / 2.0
    df = 1.0 / max(window_length_s, 1e-6)

    f_min_allowed = max(5.0, 2.0 * df)
    f_max_allowed = min(0.8 * nyquist, float(f_hz.max()))

    valid = (f_hz >= f_min_allowed) & (f_hz <= f_max_allowed)
    f_valid = f_hz[valid]
    s_valid = s_db[valid, :]

    if f_valid.size < 4:
        return {
            "constraints": {
                "nyquist_hz": nyquist,
                "recommended_upper_hz": f_max_allowed,
                "frequency_resolution_hz": df,
                "minimum_reliable_hz": f_min_allowed,
            },
            "candidates": [],
            "recommended": (f_min_allowed, f_max_allowed),
        }

    mean_spec = np.median(s_valid, axis=1)
    med = np.median(mean_spec)
    mad = np.median(np.abs(mean_spec - med)) + 1e-12
    threshold = med + 1.5 * mad

    active = mean_spec >= threshold
    regions = _find_contiguous_true_regions(active)

    min_bandwidth = max(30.0, 8.0 * df)
    candidates = []
    for left, right in regions:
        lo = float(f_valid[left])
        hi = float(f_valid[right])
        bw = hi - lo
        if bw < min_bandwidth:
            continue

        strength = float(np.mean(mean_spec[left : right + 1]) - med)
        candidates.append((lo, hi, strength))

    candidates.sort(key=lambda x: x[2], reverse=True)
    candidates = candidates[:max_candidates]

    if candidates:
        recommended = (candidates[0][0], candidates[0][1])
    else:
        span = max(120.0, 20.0 * df)
        lo = max(f_min_allowed, 20.0)
        hi = min(f_max_allowed, lo + span)
        recommended = (lo, hi)

    return {
        "constraints": {
            "nyquist_hz": nyquist,
            "recommended_upper_hz": f_max_allowed,
            "frequency_resolution_hz": df,
            "minimum_reliable_hz": f_min_allowed,
        },
        "candidates": [(float(lo), float(hi)) for lo, hi, _ in candidates],
        "recommended": (float(recommended[0]), float(recommended[1])),
    }
