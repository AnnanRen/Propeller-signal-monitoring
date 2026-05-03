from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .azimuth import (
    apply_confidence_mask,
    compute_azimuth_spectrogram,
    compute_azimuth_stability,
    compute_confidence_map,
)
from .data_io import SACBundle, find_sac_bundles, load_bundle
from .plotting import (
    PlotParams,
    SaveOptions,
    plot_azimuth_confidence_mask as plot_azimuth_confidence_mask_fn,
    plot_azimuth_spectrogram as plot_azimuth_spectrogram_fn,
    plot_azimuth_stability as plot_azimuth_stability_fn,
    plot_lofar as plot_lofar_fn,
    plot_spectrogram as plot_spectrogram_fn,
    plot_waveform as plot_waveform_fn,
)
from .preprocess import preprocess_signals
from .segment import crop_signals_by_time
from .spectral import SpectralParams, compute_stft, lofar_from_spectrogram, power_db, suggest_frequency_bands


@dataclass
class PipelineParams:
    data_dir: str | Path = "data"
    window_length_s: float = 2.0
    overlap: float = 0.5
    selected_band: Tuple[float, float] | None = None
    time_slice_s: Tuple[float, float] | None = None
    enable_demean: bool = True
    enable_detrend: bool = True
    apply_orientation: bool = True
    orientation_deg: float = 0.0
    stability_window: int = 15
    stability_step: int = 5
    confidence_threshold: float = 0.6


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
    for suffix in (".bh1.sac", ".bh2.sac", ".bhz.sac", ".hyd.sac"):
        if filename.lower().endswith(suffix):
            event_id = filename[: -len(suffix)]
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


def process_event(bundle: SACBundle, params: PipelineParams) -> Dict[str, object]:
    payload = load_bundle(bundle)
    fs = float(payload["fs"])
    t_sec = payload["t_sec"]
    signals = {k: payload[k] for k in COMPONENTS}
    crop_info = None
    utc_start = _build_utc_start_from_meta(payload["meta"]["start"])

    if params.time_slice_s is not None:
        t_sec, signals, crop_info = crop_signals_by_time(
            t_sec=t_sec,
            signals=signals,
            start_s=params.time_slice_s[0],
            end_s=params.time_slice_s[1],
        )
        utc_start = utc_start + dt.timedelta(seconds=float(crop_info["source_time_range_s"][0]))

    signals, preprocess_report = preprocess_signals(
        signals=signals,
        fs=fs,
        enable_demean=params.enable_demean,
        enable_detrend=params.enable_detrend,
        apply_orientation=params.apply_orientation,
        orientation_deg=params.orientation_deg,
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
        "preprocess_report": preprocess_report,
        "utc_start": utc_start,
        "utc_start_iso": utc_start.isoformat(),
    }


def list_events(data_dir: str | Path = "data"):
    return find_sac_bundles(data_dir)


def run_pipeline(
    input_path,
    output_dir="results",
    data_dir="data",
    component="BHZ",
    selected_band=None,
    time_slice_s=None,
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
    plot_azimuth_confidence=True,
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

    plot_flags = {
        "waveform": bool(plot_waveform),
        "spectrogram": bool(plot_spectrogram),
        "lofar": bool(plot_lofar),
        "azimuth": bool(plot_azimuth),
        "azimuth_stability": bool(plot_azimuth_stability),
        "azimuth_confidence": bool(plot_azimuth_confidence),
    }
    pipeline_kwargs = {
        "data_dir": data_dir,
        "window_length_s": float(window_length_s),
        "overlap": float(overlap),
        "selected_band": selected_band,
        "time_slice_s": time_slice_s,
        "enable_demean": bool(enable_demean),
        "enable_detrend": bool(enable_detrend),
        "apply_orientation": bool(apply_orientation),
        "orientation_deg": float(orientation_deg),
        "stability_window": int(stability_window),
        "stability_step": int(stability_step),
        "confidence_threshold": float(confidence_threshold),
    }

    bundles = list_events(data_dir)
    if not bundles:
        raise ValueError(f"No complete SAC bundles found in data directory: {data_dir}")
    bundle = _resolve_bundle(input_path, bundles)

    result = process_event(bundle, PipelineParams(**pipeline_kwargs))

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
    )
    save_opts = SaveOptions(
        save=bool(save_plots),
        outdir=output_dir,
        event_id=result["event_id"],
        formats=tuple(formats),
    )

    module_component_pairs = []
    if plot_flags["waveform"]:
        fig, _ = plot_waveform_fn(
            result["t_sec"],
            result["signals"][component],
            component,
            plot_params,
            save_opts,
            normalize=bool(normalize_waveform),
            utc_start=result["utc_start"],
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

    if plot_flags["azimuth_confidence"]:
        fig, _ = plot_azimuth_confidence_mask_fn(
            result["t_spec"],
            result["f_hz"],
            result["azimuth_masked"],
            result["confidence"],
            threshold=float(confidence_threshold),
            plot_params=plot_params,
            save_opts=save_opts,
            utc_start=result["utc_start"],
        )
        plt.close(fig)
        module_component_pairs.append(("azimuth_confidence", "ALL"))

    output_files = []
    event_clean = _clean_name(result["event_id"])
    for module_name, comp_name in module_component_pairs:
        module_clean = _clean_name(module_name)
        comp_clean = _clean_name(comp_name)
        for fmt in formats:
            fp = output_dir / f"{event_clean}_{module_clean}_{comp_clean}.{fmt}"
            if fp.exists():
                output_files.append(fp)

    logs = [
        f"event={result['event_id']}",
        f"selected_band={result['selected_band']}",
        f"utc_start={result['utc_start_iso']}",
        f"component={component}",
        f"time_slice_s={time_slice_s}",
        f"window_length_s={window_length_s}",
        f"overlap={overlap}",
        f"orientation_deg={orientation_deg}",
        f"saved_files={len(output_files)}",
    ]

    return {
        "event_id": result["event_id"],
        "component": component,
        "selected_band": result["selected_band"],
        "utc_start_iso": result["utc_start_iso"],
        "output_files": [str(p) for p in output_files],
        "logs": logs,
        "result": result,
    }
