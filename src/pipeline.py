from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Dict, List, Tuple

from .azimuth import (
    apply_confidence_mask,
    compute_azimuth_spectrogram,
    compute_azimuth_stability,
    compute_confidence_map,
)
from .data_io import SACBundle, find_sac_bundles, load_bundle, parse_event_component_from_filename
from .plotting import (
    PlotParams,
    SaveOptions,
    plot_azimuth_mask as plot_azimuth_mask_fn,
    plot_azimuth_spectrogram as plot_azimuth_spectrogram_fn,
    plot_azimuth_stability as plot_azimuth_stability_fn,
    plot_confidence_map as plot_confidence_map_fn,
    plot_lofar as plot_lofar_fn,
    plot_snr_curve as plot_snr_curve_fn,
    plot_merged_panels as plot_merged_panels_fn,
    plot_spectrogram as plot_spectrogram_fn,
    plot_waveform as plot_waveform_fn,
)
from .preprocess import downsample_signals, preprocess_signals
from .segment import crop_signals_by_time
from .spectral import SpectralParams, compute_stft, lofar_from_spectrogram, power_db, suggest_frequency_bands
from .spectral import compute_snr_from_spectrograms, suggest_noise_window


@dataclass
class PipelineParams:
    data_dir: str | Path = "data"
    window_length_s: float = 2.0
    overlap: float = 0.5
    selected_band: Tuple[float, float] | None = None
    time_slice_s: Tuple[float, float] | None = None
    auto_slice_length_s: float | None = None
    enable_demean: bool = True
    enable_detrend: bool = True
    apply_orientation: bool = True
    orientation_deg: float = 0.0
    stability_window: int = 15
    stability_step: int = 5
    confidence_threshold: float = 0.6
    snr_noise_window_s: Tuple[float, float] | None = None
    compute_snr: bool = False
    snr_auto_noise_window_s: float = 60.0
    target_fs: float | None = None
    timezone_offset_hours: int = 8


COMPONENTS = ("BH1", "BH2", "BHZ", "HYD")


def _clean_name(name: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    return "".join(ch if ch in allowed else "_" for ch in str(name))


def _resolve_bundle(input_path: str | Path | SACBundle, bundles: List[SACBundle]) -> SACBundle:
    if isinstance(input_path, SACBundle):
        return input_path

    text = str(input_path)
    by_event_id = {b.event_id: b for b in bundles}
    if text in by_event_id:
        return by_event_id[text]

    p = Path(text)
    filename = p.name
    parsed = parse_event_component_from_filename(filename)
    if parsed is not None:
        event_id, _, _ = parsed
        if event_id in by_event_id:
            return by_event_id[event_id]

    if filename in by_event_id:
        return by_event_id[filename]

    available = ", ".join(sorted(by_event_id.keys()))
    raise ValueError(f"Cannot resolve input '{input_path}' to an event. Available events: {available}")


def _build_utc_start_from_meta(meta_start: Dict[str, int]) -> dt.datetime:
    base = dt.datetime(
        int(meta_start["nzyear"]),
        1,
        1,
        tzinfo=dt.timezone.utc,
    )
    return base + dt.timedelta(
        days=int(meta_start["nzjday"]) - 1,
        hours=int(meta_start["nzhour"]),
        minutes=int(meta_start["nzmin"]),
        seconds=int(meta_start["nzsec"]),
        milliseconds=int(meta_start["nzmsec"]),
    )


def _resolve_effective_time_slice(
    t_sec: object,
    manual_time_slice_s: Tuple[float, float] | None,
    auto_slice_length_s: float | None,
) -> Tuple[Tuple[float, float] | None, dict | None]:
    if manual_time_slice_s is not None and auto_slice_length_s is not None:
        raise ValueError("time_slice_s and auto_slice_length_s are mutually exclusive.")

    if manual_time_slice_s is not None:
        return (float(manual_time_slice_s[0]), float(manual_time_slice_s[1])), {"mode": "manual"}

    if auto_slice_length_s is None:
        return None, None

    total_duration_s = float(t_sec[-1]) if len(t_sec) > 0 else 0.0
    requested = float(auto_slice_length_s)
    if requested <= 0:
        raise ValueError(f"auto_slice_length_s must be > 0, got: {auto_slice_length_s}")
    effective_end = min(requested, total_duration_s)
    if effective_end <= 0:
        raise ValueError("Input signal has zero duration; cannot apply auto slice.")

    info: dict = {"mode": "auto", "requested_range_s": (0.0, requested)}
    if requested > total_duration_s:
        info["warning"] = (
            f"auto_slice_length_s={requested:.3f}s exceeds data duration "
            f"{total_duration_s:.3f}s; using full available duration."
        )
    return (0.0, float(effective_end)), info


def _validate_manual_selected_band(selected_band: Tuple[float, float], nyquist_hz: float) -> Tuple[float, float]:
    freq_min = float(selected_band[0])
    freq_max = float(selected_band[1])
    if not (isfinite(freq_min) and isfinite(freq_max)):
        raise ValueError(f"selected_band must be finite numbers, got: {selected_band}")
    if freq_min <= 0:
        raise ValueError(f"selected_band lower bound must be > 0, got: {selected_band}")
    if freq_max <= freq_min:
        raise ValueError(f"selected_band upper bound must be greater than lower bound, got: {selected_band}")
    if freq_max > float(nyquist_hz):
        raise ValueError(
            f"selected_band upper bound {freq_max:.3f} Hz exceeds Nyquist {nyquist_hz:.3f} Hz "
            "(after optional downsampling)."
        )
    return freq_min, freq_max


def process_event(bundle: SACBundle, params: PipelineParams) -> Dict[str, object]:
    payload = load_bundle(bundle)
    fs = float(payload["fs"])
    t_sec = payload["t_sec"]
    signals = {k: payload[k] for k in COMPONENTS}
    crop_info = None
    resample_report = {"enabled": False}
    utc_start = _build_utc_start_from_meta(payload["meta"]["start"])
    effective_time_slice_s, auto_slice_info = _resolve_effective_time_slice(
        t_sec=t_sec,
        manual_time_slice_s=params.time_slice_s,
        auto_slice_length_s=params.auto_slice_length_s,
    )

    if effective_time_slice_s is not None:
        t_sec, signals, crop_info = crop_signals_by_time(
            t_sec=t_sec,
            signals=signals,
            start_s=effective_time_slice_s[0],
            end_s=effective_time_slice_s[1],
        )
        if crop_info is not None:
            crop_info["requested_time_range_s"] = (
                float(effective_time_slice_s[0]),
                float(effective_time_slice_s[1]),
            )
            if auto_slice_info and auto_slice_info.get("warning"):
                crop_info["warning"] = auto_slice_info["warning"]
            if auto_slice_info:
                crop_info["mode"] = auto_slice_info["mode"]
        if auto_slice_info is None or auto_slice_info.get("mode") == "manual":
            utc_start = utc_start + dt.timedelta(seconds=float(crop_info["source_time_range_s"][0]))

    if params.target_fs is not None:
        t_sec, signals, fs, resample_report = downsample_signals(
            t_sec=t_sec,
            signals=signals,
            fs=fs,
            target_fs=float(params.target_fs),
        )

    signals, preprocess_report = preprocess_signals(
        signals=signals,
        fs=fs,
        enable_demean=params.enable_demean,
        enable_detrend=params.enable_detrend,
        apply_orientation=params.apply_orientation,
        orientation_deg=params.orientation_deg,
    )
    preprocess_report["resample_report"] = resample_report

    spec_params = SpectralParams(window_length_s=params.window_length_s, overlap=params.overlap)

    spectrograms = {}
    db_maps = {}
    lofar_maps = {}
    stft_report_by_component: Dict[str, Dict[str, object]] = {}
    stft_warnings: List[str] = []

    f_hz = None
    t_spec = None

    for comp in COMPONENTS:
        input_samples = int(len(signals[comp]))
        if input_samples < 2:
            raise ValueError(
                f"STFT input too short for component {comp}: requires at least 2 samples, got {input_samples}."
            )
        requested_nperseg = max(2, int(round(float(spec_params.window_length_s) * float(fs))))
        effective_nperseg = min(requested_nperseg, input_samples)
        requested_seconds = float(requested_nperseg) / float(fs)
        effective_seconds = float(effective_nperseg) / float(fs)
        adjusted = effective_nperseg != requested_nperseg
        stft_report_by_component[comp] = {
            "input_samples": input_samples,
            "requested_window_samples": requested_nperseg,
            "effective_window_samples": effective_nperseg,
            "requested_window_s": requested_seconds,
            "effective_window_s": effective_seconds,
            "window_adjusted": adjusted,
        }
        if adjusted:
            stft_warnings.append(
                f"{comp}: STFT window adjusted {requested_nperseg} samples ({requested_seconds:.3f}s) -> "
                f"{effective_nperseg} samples ({effective_seconds:.3f}s) due to short segment."
            )
        f_hz_i, t_spec_i, s_complex = compute_stft(signals[comp], fs, spec_params)
        spectrograms[comp] = s_complex
        db_maps[comp] = power_db(s_complex)
        lofar_maps[comp] = lofar_from_spectrogram(db_maps[comp])
        if f_hz is None:
            f_hz = f_hz_i
            t_spec = t_spec_i

    preprocess_report["stft_report"] = {
        "window_adjusted": bool(stft_warnings),
        "warnings": stft_warnings,
        "by_component": stft_report_by_component,
    }

    if f_hz is None or t_spec is None:
        raise RuntimeError("Failed to compute spectrograms.")

    nyquist_hz = float(fs) / 2.0
    frequency_guard = {
        "nyquist_hz": nyquist_hz,
        "manual_band_checked": False,
        "auto_band_clipped": False,
        "warnings": [],
    }
    band_info = suggest_frequency_bands(f_hz, db_maps["HYD"], fs, params.window_length_s)
    if params.selected_band is None:
        freq_min, freq_max = band_info["recommended"]
        if float(freq_max) > nyquist_hz:
            freq_max = nyquist_hz
            frequency_guard["auto_band_clipped"] = True
            frequency_guard["warnings"].append("Auto-recommended band upper bound clipped to Nyquist.")
    else:
        freq_min, freq_max = _validate_manual_selected_band(params.selected_band, nyquist_hz)
        frequency_guard["manual_band_checked"] = True

    if float(freq_max) > nyquist_hz:
        raise ValueError(
            f"Effective frequency upper bound {float(freq_max):.3f} Hz exceeds Nyquist {nyquist_hz:.3f} Hz."
        )

    mask = (f_hz >= freq_min) & (f_hz <= freq_max)
    if not mask.any():
        raise ValueError(
            f"No frequency bins available in selected band [{float(freq_min):.3f}, {float(freq_max):.3f}] Hz."
        )
    f_sel = f_hz[mask]
    selected_max_hz = float(f_sel.max())
    if selected_max_hz > nyquist_hz:
        raise ValueError(
            f"Selected frequency bins reach {selected_max_hz:.3f} Hz, exceeding Nyquist {nyquist_hz:.3f} Hz."
        )
    frequency_guard["checked_limits_hz"] = {
        "selected_band_max_hz": selected_max_hz,
        "stability_band_max_hz": selected_max_hz,
        "snr_band_max_hz": selected_max_hz,
        "plot_fmax_hz": selected_max_hz,
    }

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
    snr_payload = {
        "snr": {},
        "snr_hyd_db": None,
        "snr_series": {},
        "snr_windows": {"signal_window_s": None, "noise_window_s": None},
        "snr_noise_window_source": None,
    }
    if params.compute_snr:
        if params.snr_noise_window_s is not None:
            noise_window_s = params.snr_noise_window_s
            noise_source = "manual"
        else:
            noise_window_s = suggest_noise_window(
                spectrograms_sel["HYD"],
                t_spec=t_spec,
                window_length_s=float(params.snr_auto_noise_window_s),
            )
            noise_source = "auto"
        snr_payload = compute_snr_from_spectrograms(
            spectrograms_sel=spectrograms_sel,
            t_spec=t_spec,
            noise_window_s=noise_window_s,
        )
        snr_payload["snr_noise_window_source"] = noise_source

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
        "time_slice_s": effective_time_slice_s,
        "crop_info": crop_info,
        "preprocess_report": preprocess_report,
        "resample_report": resample_report,
        "frequency_guard": frequency_guard,
        "snr": snr_payload["snr"],
        "snr_hyd_db": snr_payload["snr_hyd_db"],
        "snr_series": snr_payload["snr_series"],
        "snr_windows": snr_payload["snr_windows"],
        "snr_noise_window_source": snr_payload["snr_noise_window_source"],
        "utc_start": utc_start,
        "utc_start_iso": utc_start.isoformat(),
        "timezone_offset_hours": int(params.timezone_offset_hours),
    }


def list_events(data_dir: str | Path = "data"):
    return find_sac_bundles(data_dir)


def preview_auto_band(
    input_path,
    data_dir="data",
    window_length_s=2.0,
    overlap=0.5,
    time_slice_s=None,
    auto_slice_length_s=None,
    enable_demean=True,
    enable_detrend=True,
    apply_orientation=True,
    orientation_deg=0.0,
    target_fs=None,
):
    """
    Lightweight preview for auto frequency-band recommendation.
    Computes only HYD spectrogram and suggested frequency-band info.
    """
    data_dir = Path(data_dir)
    bundles = list_events(data_dir)
    if not bundles:
        raise ValueError(f"No complete SAC bundles found in data directory: {data_dir}")
    bundle = _resolve_bundle(input_path, bundles)

    payload = load_bundle(bundle)
    fs = float(payload["fs"])
    t_sec = payload["t_sec"]
    signals = {k: payload[k] for k in COMPONENTS}

    effective_time_slice_s, _ = _resolve_effective_time_slice(
        t_sec=t_sec,
        manual_time_slice_s=time_slice_s,
        auto_slice_length_s=auto_slice_length_s,
    )
    if effective_time_slice_s is not None:
        t_sec, signals, _ = crop_signals_by_time(
            t_sec=t_sec,
            signals=signals,
            start_s=effective_time_slice_s[0],
            end_s=effective_time_slice_s[1],
        )

    if target_fs is not None:
        t_sec, signals, fs, _ = downsample_signals(
            t_sec=t_sec,
            signals=signals,
            fs=fs,
            target_fs=float(target_fs),
        )

    signals, _ = preprocess_signals(
        signals=signals,
        fs=fs,
        enable_demean=bool(enable_demean),
        enable_detrend=bool(enable_detrend),
        apply_orientation=bool(apply_orientation),
        orientation_deg=float(orientation_deg),
    )

    spec_params = SpectralParams(window_length_s=float(window_length_s), overlap=float(overlap))
    if int(len(signals["HYD"])) < 2:
        raise ValueError(
            f"STFT input too short for HYD in preview: requires at least 2 samples, got {int(len(signals['HYD']))}."
        )
    f_hz, _, s_hyd = compute_stft(signals["HYD"], fs, spec_params)
    s_hyd_db = power_db(s_hyd)
    band_info = suggest_frequency_bands(f_hz, s_hyd_db, fs, float(window_length_s))

    nyquist_hz = float(fs) / 2.0
    recommended = tuple(float(x) for x in band_info["recommended"])
    if recommended[1] > nyquist_hz:
        recommended = (recommended[0], nyquist_hz)

    return {
        "event_id": bundle.event_id,
        "recommended": recommended,
        "candidates": band_info.get("candidates", []),
        "constraints": band_info.get("constraints", {}),
        "effective_time_slice_s": effective_time_slice_s,
        "fs_hz": float(fs),
    }


def _build_auto_segments(total_duration_s: float, segment_length_s: float) -> List[Tuple[float, float]]:
    total = float(total_duration_s)
    length = float(segment_length_s)
    if total <= 0:
        raise ValueError("Input signal has zero duration; cannot build auto segments.")
    if length <= 0:
        raise ValueError(f"auto_slice_length_s must be > 0, got: {segment_length_s}")

    segments: List[Tuple[float, float]] = []
    start = 0.0
    while start < total:
        end = min(start + length, total)
        segments.append((float(start), float(end)))
        if end >= total:
            break
        start = end
    return segments


def _format_segment_tag(index: int, start_s: float, end_s: float) -> str:
    return _clean_name(f"seg{int(index):03d}_t{float(start_s):.3f}_t{float(end_s):.3f}")


def run_pipeline(
    input_path,
    output_dir="results",
    data_dir="data",
    component="BHZ",
    selected_band=None,
    time_slice_s=None,
    auto_slice_length_s=None,
    window_length_s=2.0,
    overlap=0.5,
    enable_demean=True,
    enable_detrend=True,
    apply_orientation=True,
    orientation_deg=0.0,
    stability_window=15,
    stability_step=5,
    confidence_threshold=0.6,
    save_plots=True,
    formats=("png", "pdf"),
    plot_waveform=True,
    plot_spectrogram=True,
    plot_lofar=True,
    plot_azimuth=True,
    plot_azimuth_stability=True,
    plot_azimuth_mask=True,
    plot_confidence=True,
    plot_azimuth_confidence=None,
    plot_snr=True,
    compute_snr=False,
    snr_noise_window_s=None,
    snr_auto_noise_window_s=60.0,
    target_fs=None,
    merge_all_plots=True,
    normalize_waveform=True,
    plot_font_name="Helvetica",
    plot_dpi=300,
    plot_fig_width=7.2,
    plot_fig_height=3.2,
    plot_cmap_spec="viridis",
    plot_cmap_lofar="plasma",
    plot_cmap_azi="hsv",
    plot_cmap_stability="RdYlBu_r",
    plot_cmap_confidence="magma",
    plot_linewidth_waveform=0.4,
    plot_grid_alpha=0.2,
    timezone_offset_hours=8,
):
    """
    Read one event, run full processing, save result figures, and return output info.
    """
    import matplotlib.pyplot as plt

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    component = str(component).upper()
    if component not in COMPONENTS:
        raise ValueError(f"component must be one of {COMPONENTS}, got: {component}")

    confidence_flag = bool(plot_confidence) if plot_azimuth_confidence is None else bool(plot_azimuth_confidence)
    plot_flags = {
        "waveform": bool(plot_waveform),
        "spectrogram": bool(plot_spectrogram),
        "lofar": bool(plot_lofar),
        "snr": bool(plot_snr),
        "azimuth_mask": bool(plot_azimuth_mask),
        "azimuth": bool(plot_azimuth),
        "confidence": confidence_flag,
        "azimuth_stability": bool(plot_azimuth_stability),
    }
    bundles = list_events(data_dir)
    if not bundles:
        raise ValueError(f"No complete SAC bundles found in data directory: {data_dir}")
    bundle = _resolve_bundle(input_path, bundles)
    if time_slice_s is not None and auto_slice_length_s is not None:
        raise ValueError("time_slice_s and auto_slice_length_s are mutually exclusive.")

    base_kwargs = {
        "data_dir": data_dir,
        "window_length_s": float(window_length_s),
        "overlap": float(overlap),
        "selected_band": selected_band,
        "enable_demean": bool(enable_demean),
        "enable_detrend": bool(enable_detrend),
        "apply_orientation": bool(apply_orientation),
        "orientation_deg": float(orientation_deg),
        "stability_window": int(stability_window),
        "stability_step": int(stability_step),
        "confidence_threshold": float(confidence_threshold),
        "compute_snr": bool(compute_snr),
        "snr_noise_window_s": snr_noise_window_s,
        "snr_auto_noise_window_s": float(snr_auto_noise_window_s),
        "target_fs": target_fs,
        "timezone_offset_hours": int(timezone_offset_hours),
        "auto_slice_length_s": None,
    }

    segmented_mode = auto_slice_length_s is not None
    payload_for_segment = None
    if segmented_mode:
        payload_for_segment = load_bundle(bundle)
        total_duration_s = float(payload_for_segment["t_sec"][-1]) if len(payload_for_segment["t_sec"]) > 0 else 0.0
        segment_ranges = _build_auto_segments(total_duration_s=total_duration_s, segment_length_s=float(auto_slice_length_s))
    else:
        segment_ranges = [time_slice_s]

    panel_order = [
        "waveform",
        "snr",
        "spectrogram",
        "lofar",
        "azimuth_mask",
        "azimuth",
        "confidence",
        "azimuth_stability",
    ]
    selected_panels = [p for p in panel_order if plot_flags.get(p, False)]

    all_output_files: List[Path] = []
    segments_info: List[Dict[str, object]] = []
    skipped_segment_warnings: List[str] = []
    stft_adjust_logs: List[str] = []
    first_result = None
    first_selected_band = None
    first_utc_start_iso = None

    for seg_idx, seg_range in enumerate(segment_ranges, start=1):
        seg_tag = None
        seg_slice = None
        if seg_range is not None:
            seg_slice = (float(seg_range[0]), float(seg_range[1]))
            if segmented_mode:
                seg_tag = _format_segment_tag(seg_idx, seg_slice[0], seg_slice[1])

        seg_params = dict(base_kwargs)
        seg_params["time_slice_s"] = seg_slice

        segment_sample_count = None
        if segmented_mode and payload_for_segment is not None and seg_slice is not None:
            t_all = payload_for_segment["t_sec"]
            segment_sample_count = int(((t_all >= seg_slice[0]) & (t_all <= seg_slice[1])).sum())

        try:
            result = process_event(bundle, PipelineParams(**seg_params))
        except ValueError as exc:
            err_msg = str(exc)
            if segmented_mode and "STFT input too short" in err_msg:
                clean_msg = err_msg.rstrip(".")
                warning = (
                    f"Segment {seg_idx} skipped ({seg_slice}): {clean_msg}. "
                    f"samples={segment_sample_count if segment_sample_count is not None else 'unknown'}"
                )
                skipped_segment_warnings.append(warning)
                segments_info.append(
                    {
                        "segment_index": seg_idx,
                        "segment_tag": seg_tag if seg_tag is not None else "full",
                        "time_slice_s": seg_slice,
                        "utc_start_iso": None,
                        "selected_band": None,
                        "output_files": [],
                        "warning": warning,
                    }
                )
                continue
            raise

        stft_report = result.get("preprocess_report", {}).get("stft_report", {})
        for warning_text in stft_report.get("warnings", []):
            stft_adjust_logs.append(f"segment={seg_idx}: {warning_text}")
        if first_result is None:
            first_result = result
            first_selected_band = result["selected_band"]
            first_utc_start_iso = result["utc_start_iso"]

        event_id_for_save = result["event_id"]
        if seg_tag is not None:
            event_id_for_save = f"{event_id_for_save}_{seg_tag}"

        plot_params = PlotParams(
            font_name=str(plot_font_name),
            dpi=int(plot_dpi),
            figsize=(float(plot_fig_width), float(plot_fig_height)),
            cmap_spec=str(plot_cmap_spec),
            cmap_lofar=str(plot_cmap_lofar),
            cmap_azi=str(plot_cmap_azi),
            cmap_stability=str(plot_cmap_stability),
            cmap_confidence=str(plot_cmap_confidence),
            freq_min=result["selected_band"][0],
            freq_max=result["selected_band"][1],
            linewidth_waveform=float(plot_linewidth_waveform),
            grid_alpha=float(plot_grid_alpha),
            timezone_offset_hours=int(timezone_offset_hours),
        )
        save_opts = SaveOptions(
            save=bool(save_plots),
            outdir=output_dir,
            event_id=event_id_for_save,
            formats=tuple(formats),
        )

        module_component_pairs: List[Tuple[str, str]] = []
        if bool(merge_all_plots):
            if selected_panels:
                fig, _ = plot_merged_panels_fn(
                    selected_panels=selected_panels,
                    t_sec=result["t_sec"],
                    signal=result["signals"][component],
                    component_name=component,
                    t_spec=result["t_spec"],
                    f_hz=result["f_hz"],
                    spectrogram_db=result["spectrogram_db"][component],
                    lofar_map=result["lofar"][component],
                    azimuth_masked_tf=result["azimuth_masked"],
                    azimuth_deg_tf=result["azimuth_deg"],
                    confidence_tf=result["confidence"],
                    r_tf=result["azimuth_stability"],
                    threshold=float(confidence_threshold),
                    plot_params=plot_params,
                    normalize_waveform=bool(normalize_waveform),
                    utc_start=result["utc_start"],
                    snr_series_hyd=result.get("snr_series", {}).get("HYD"),
                    snr_noise_window_s=result.get("snr_windows", {}).get("noise_window_s"),
                    snr_noise_window_source=result.get("snr_noise_window_source"),
                )
                if bool(save_plots):
                    event_clean = _clean_name(event_id_for_save)
                    for fmt in formats:
                        fig.savefig(
                            output_dir / f"{event_clean}_all.{fmt}",
                            dpi=fig.dpi,
                            bbox_inches="tight",
                            facecolor="white",
                        )
                plt.close(fig)
                module_component_pairs.append(("all", ""))
        else:
            if plot_flags["waveform"]:
                fig, _ = plot_waveform_fn(
                    result["t_sec"],
                    result["signals"][component],
                    component,
                    plot_params,
                    save_opts,
                    normalize=bool(normalize_waveform),
                    utc_start=result["utc_start"],
                    noise_window_s=result.get("snr_windows", {}).get("noise_window_s"),
                    noise_window_source=result.get("snr_noise_window_source"),
                    show_noise_window=True,
                )
                plt.close(fig)
                module_component_pairs.append(("waveform", component))

            if plot_flags["spectrogram"]:
                fig, _ = plot_spectrogram_fn(
                    result["t_spec"],
                    result["f_hz"],
                    result["spectrogram_db"][component],
                    component,
                    plot_params,
                    save_opts,
                    utc_start=result["utc_start"],
                )
                plt.close(fig)
                module_component_pairs.append(("spectrogram", component))

            if plot_flags["lofar"]:
                fig, _ = plot_lofar_fn(
                    result["t_spec"],
                    result["f_hz"],
                    result["lofar"][component],
                    component,
                    plot_params,
                    save_opts,
                    utc_start=result["utc_start"],
                )
                plt.close(fig)
                module_component_pairs.append(("lofar", component))

            if plot_flags["azimuth_mask"]:
                fig, _ = plot_azimuth_mask_fn(
                    result["t_spec"],
                    result["f_hz"],
                    result["azimuth_masked"],
                    threshold=float(confidence_threshold),
                    plot_params=plot_params,
                    save_opts=save_opts,
                    utc_start=result["utc_start"],
                )
                plt.close(fig)
                module_component_pairs.append(("azimuth_mask", "ALL"))

            if plot_flags["azimuth"]:
                fig, _ = plot_azimuth_spectrogram_fn(
                    result["t_spec"],
                    result["f_hz"],
                    result["azimuth_deg"],
                    plot_params,
                    save_opts,
                    utc_start=result["utc_start"],
                )
                plt.close(fig)
                module_component_pairs.append(("azimuth", "ALL"))

            if plot_flags["confidence"]:
                fig, _ = plot_confidence_map_fn(
                    result["t_spec"],
                    result["f_hz"],
                    result["confidence"],
                    plot_params=plot_params,
                    save_opts=save_opts,
                    utc_start=result["utc_start"],
                )
                plt.close(fig)
                module_component_pairs.append(("confidence", "ALL"))

            if plot_flags["azimuth_stability"]:
                fig, _ = plot_azimuth_stability_fn(
                    result["t_spec"],
                    result["f_hz"],
                    result["azimuth_stability"],
                    plot_params,
                    save_opts,
                    utc_start=result["utc_start"],
                )
                plt.close(fig)
                module_component_pairs.append(("azimuth_stability", "ALL"))

            if plot_flags["snr"] and result["snr_series"] and ("HYD" in result["snr_series"]):
                fig, _ = plot_snr_curve_fn(
                    result["t_spec"],
                    result["snr_series"]["HYD"],
                    "HYD",
                    plot_params,
                    save_opts,
                    utc_start=result["utc_start"],
                    noise_window_s=result.get("snr_windows", {}).get("noise_window_s"),
                    noise_window_source=result.get("snr_noise_window_source"),
                    show_noise_window=True,
                )
                plt.close(fig)
                module_component_pairs.append(("snr", "HYD"))

        segment_output_files: List[Path] = []
        event_clean = _clean_name(event_id_for_save)
        for module_name, comp_name in module_component_pairs:
            for fmt in formats:
                if module_name == "all":
                    fp = output_dir / f"{event_clean}_all.{fmt}"
                else:
                    module_clean = _clean_name(module_name)
                    comp_clean = _clean_name(comp_name)
                    fp = output_dir / f"{event_clean}_{module_clean}_{comp_clean}.{fmt}"
                if fp.exists():
                    segment_output_files.append(fp)
                    all_output_files.append(fp)

        segments_info.append(
            {
                "segment_index": seg_idx,
                "segment_tag": seg_tag if seg_tag is not None else "full",
                "time_slice_s": result["time_slice_s"],
                "utc_start_iso": result["utc_start_iso"],
                "selected_band": result["selected_band"],
                "output_files": [str(p) for p in segment_output_files],
                "warning": None,
            }
        )

    if first_result is None or first_selected_band is None or first_utc_start_iso is None:
        if skipped_segment_warnings:
            raise RuntimeError(
                "No segment result generated. All segments were skipped. "
                f"First warning: {skipped_segment_warnings[0]}"
            )
        raise RuntimeError("No segment result generated.")

    logs = [
        f"event={first_result['event_id']}",
        f"segment_count={len(segments_info)}",
        f"segmented_mode={segmented_mode}",
        f"selected_band={first_selected_band}",
        f"utc_start={first_utc_start_iso}",
        f"component={component}",
        f"time_slice_s={first_result['time_slice_s']}",
        f"window_length_s={window_length_s}",
        f"overlap={overlap}",
        f"orientation_deg={orientation_deg}",
        f"compute_snr={bool(compute_snr)}",
        f"snr_noise_window_s={snr_noise_window_s}",
        f"snr_noise_window_source={first_result['snr_noise_window_source']}",
        f"snr_hyd_db={first_result['snr_hyd_db']}",
        f"target_fs={target_fs}",
        f"resample_report={first_result.get('resample_report')}",
        f"timezone_offset_hours={int(timezone_offset_hours)}",
        f"saved_files={len(all_output_files)}",
    ]
    if skipped_segment_warnings:
        logs.append(f"skipped_segments={len(skipped_segment_warnings)}")
        logs.extend(skipped_segment_warnings)
    if stft_adjust_logs:
        logs.append(f"stft_window_adjustments={len(stft_adjust_logs)}")
        logs.extend(stft_adjust_logs)

    return {
        "event_id": first_result["event_id"],
        "component": component,
        "selected_band": first_selected_band,
        "utc_start_iso": first_utc_start_iso,
        "output_files": [str(p) for p in all_output_files],
        "logs": logs,
        "result": first_result,
        "segments": segments_info,
    }
