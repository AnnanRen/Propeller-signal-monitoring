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


def _window_to_mask(
    t_spec: np.ndarray,
    window_s: Tuple[float, float],
    label: str,
) -> np.ndarray:
    start_s = float(window_s[0])
    end_s = float(window_s[1])
    if end_s <= start_s:
        raise ValueError(f"{label} end must be greater than start, got: {window_s}")

    t = np.asarray(t_spec, dtype=float)
    if t.size == 0:
        raise ValueError("STFT time axis is empty.")

    t_min = float(np.min(t))
    t_max = float(np.max(t))
    if start_s > t_max or end_s < t_min:
        raise ValueError(f"{label} window {window_s} is outside STFT time range [{t_min:.3f}, {t_max:.3f}] s.")

    mask = (t >= start_s) & (t <= end_s)
    if not np.any(mask):
        raise ValueError(f"{label} window {window_s} has no valid STFT bins in [{t_min:.3f}, {t_max:.3f}] s.")
    return mask


def compute_snr_from_spectrograms(
    spectrograms_sel: Dict[str, np.ndarray],
    t_spec: np.ndarray,
    noise_window_s: Tuple[float, float],
) -> Dict[str, object]:
    noise_mask = _window_to_mask(t_spec, noise_window_s, "noise")

    snr_db_by_component: Dict[str, float] = {}
    snr_series_by_component: Dict[str, np.ndarray] = {}
    eps = np.finfo(float).eps

    for comp, s_complex in spectrograms_sel.items():
        power_tf = np.abs(np.asarray(s_complex, dtype=np.complex128)) ** 2
        rms_t = np.sqrt(np.mean(power_tf, axis=0))

        noise_rms = float(np.sqrt(np.mean(np.square(rms_t[noise_mask]))))
        if not np.isfinite(noise_rms) or noise_rms <= eps:
            raise ValueError(
                f"Noise RMS is near zero or invalid for component {comp}. "
                "Please choose another noise window with effective background energy."
            )

        snr_t_db = 20.0 * np.log10((rms_t + eps) / noise_rms)
        # Whole-segment SNR summary from STFT-bin sequence.
        snr_db = float(np.mean(snr_t_db))

        snr_db_by_component[comp] = snr_db
        snr_series_by_component[comp] = snr_t_db.astype(np.float64)

    return {
        "snr": snr_db_by_component,
        "snr_hyd_db": float(snr_db_by_component.get("HYD")) if "HYD" in snr_db_by_component else None,
        "snr_series": snr_series_by_component,
        "snr_windows": {
            "signal_window_s": None,
            "noise_window_s": (float(noise_window_s[0]), float(noise_window_s[1])),
        },
    }


def suggest_noise_window(
    hyd_spectrogram_sel: np.ndarray,
    t_spec: np.ndarray,
    window_length_s: float = 60.0,
) -> Tuple[float, float]:
    t = np.asarray(t_spec, dtype=float)
    if t.size < 2:
        raise ValueError("Not enough STFT time bins for auto noise-window suggestion.")

    power_tf = np.abs(np.asarray(hyd_spectrogram_sel, dtype=np.complex128)) ** 2
    rms_t = np.sqrt(np.mean(power_tf, axis=0))
    if rms_t.size != t.size:
        raise ValueError("HYD spectrogram time axis is inconsistent with STFT time bins.")

    target_len = float(window_length_s)
    if target_len <= 0:
        raise ValueError(f"window_length_s must be > 0, got: {window_length_s}")

    dt_bins = np.median(np.diff(t))
    if not np.isfinite(dt_bins) or dt_bins <= 0:
        raise ValueError("Invalid STFT time spacing for auto noise-window suggestion.")
    bins = max(2, int(round(target_len / dt_bins)))
    bins = min(bins, t.size)

    means = []
    stds = []
    starts = []
    for i in range(0, t.size - bins + 1):
        seg = rms_t[i : i + bins]
        means.append(float(np.mean(seg)))
        stds.append(float(np.std(seg)))
        starts.append(i)

    if not starts:
        raise ValueError("Cannot build candidate windows for auto noise-window suggestion.")

    mean_arr = np.asarray(means, dtype=np.float64)
    std_arr = np.asarray(stds, dtype=np.float64)
    mean_norm = (mean_arr - np.min(mean_arr)) / (np.ptp(mean_arr) + 1e-12)
    std_norm = (std_arr - np.min(std_arr)) / (np.ptp(std_arr) + 1e-12)
    score = 0.5 * mean_norm + 0.5 * std_norm

    best_idx = int(np.argmin(score))
    i0 = starts[best_idx]
    i1 = i0 + bins - 1
    return float(t[i0]), float(t[i1])
