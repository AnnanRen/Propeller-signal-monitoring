from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .azimuth import (
    apply_confidence_mask,
    compute_azimuth_spectrogram,
    compute_azimuth_stability,
    compute_confidence_map,
)
from .data_io import SACBundle, find_sac_bundles, load_bundle
from .segment import crop_signals_by_time
from .spectral import SpectralParams, compute_stft, lofar_from_spectrogram, power_db, suggest_frequency_bands


@dataclass
class PipelineParams:
    data_dir: str | Path = "."
    window_length_s: float = 2.0
    overlap: float = 0.5
    selected_band: Tuple[float, float] | None = None
    time_slice_s: Tuple[float, float] | None = None
    stability_window: int = 15
    stability_step: int = 5
    confidence_threshold: float = 0.6


COMPONENTS = ("BH1", "BH2", "BHZ", "HYD")


def process_event(bundle: SACBundle, params: PipelineParams) -> Dict[str, object]:
    payload = load_bundle(bundle)
    fs = float(payload["fs"])
    t_sec = payload["t_sec"]
    signals = {k: payload[k] for k in COMPONENTS}
    crop_info = None

    if params.time_slice_s is not None:
        t_sec, signals, crop_info = crop_signals_by_time(
            t_sec=t_sec,
            signals=signals,
            start_s=params.time_slice_s[0],
            end_s=params.time_slice_s[1],
        )

    spec_params = SpectralParams(window_length_s=params.window_length_s, overlap=params.overlap)

    spectrograms = {}
    db_maps = {}
    lofar_maps = {}

    f_hz = None
    t_spec = None

    for comp in COMPONENTS:
        f_hz_i, t_spec_i, s_complex = compute_stft(signals[comp], fs, spec_params)
        spectrograms[comp] = s_complex
        db_maps[comp] = power_db(s_complex)
        lofar_maps[comp] = lofar_from_spectrogram(db_maps[comp])
        if f_hz is None:
            f_hz = f_hz_i
            t_spec = t_spec_i

    if f_hz is None or t_spec is None:
        raise RuntimeError("Failed to compute spectrograms.")

    band_info = suggest_frequency_bands(f_hz, db_maps["HYD"], fs, params.window_length_s)
    if params.selected_band is None:
        freq_min, freq_max = band_info["recommended"]
    else:
        freq_min, freq_max = params.selected_band

    mask = (f_hz >= freq_min) & (f_hz <= freq_max)
    f_sel = f_hz[mask]

    spectrograms_sel = {k: v[mask, :] for k, v in spectrograms.items()}
    db_sel = {k: v[mask, :] for k, v in db_maps.items()}
    lofar_sel = {k: v[mask, :] for k, v in lofar_maps.items()}

    azi_info = compute_azimuth_spectrogram(
        spectrograms_sel["HYD"],
        spectrograms_sel["BH1"],
        spectrograms_sel["BH2"],
    )

    stability = compute_azimuth_stability(
        azi_info["azimuth_deg"],
        t_spec,
        window_size=params.stability_window,
        step_size=params.stability_step,
    )

    conf = compute_confidence_map(
        spectrograms_sel["HYD"],
        spectrograms_sel["BH1"],
        spectrograms_sel["BH2"],
        azi_info["intensity"],
    )
    azi_masked = apply_confidence_mask(azi_info["azimuth_deg"], conf, params.confidence_threshold)

    return {
        "event_id": payload["event_id"],
        "t_sec": t_sec,
        "signals": signals,
        "fs": fs,
        "f_hz": f_sel,
        "t_spec": t_spec,
        "spectrogram_db": db_sel,
        "lofar": lofar_sel,
        "azimuth_deg": azi_info["azimuth_deg"],
        "azimuth_stability": stability["R_interp"],
        "confidence": conf,
        "azimuth_masked": azi_masked,
        "band_info": band_info,
        "selected_band": (float(freq_min), float(freq_max)),
        "time_slice_s": params.time_slice_s,
        "crop_info": crop_info,
    }


def list_events(data_dir: str | Path = "."):
    return find_sac_bundles(data_dir)
