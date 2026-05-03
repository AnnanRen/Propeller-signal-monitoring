from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from .utils import ensure_dir, zscore_safe


@dataclass
class PlotParams:
    font_name: str = "Helvetica"
    dpi: int = 300
    figsize: Tuple[float, float] = (7.2, 3.2)
    cmap_spec: str = "viridis"
    cmap_lofar: str = "plasma"
    cmap_azi: str = "hsv"
    cmap_stability: str = "RdYlBu_r"
    cmap_confidence: str = "magma"
    freq_min: float = 150.0
    freq_max: float = 400.0
    linewidth_waveform: float = 0.4
    grid_alpha: float = 0.2


@dataclass
class SaveOptions:
    save: bool = False
    outdir: str | Path = "results"
    event_id: str = "event"
    formats: Tuple[str, ...] = ("png", "pdf")


def _clean_name(name: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    return "".join(ch if ch in allowed else "_" for ch in str(name))


def _robust_limits(x: np.ndarray, q_low: float, q_high: float) -> Tuple[float, float]:
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(finite, q_low))
    hi = float(np.percentile(finite, q_high))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _style_axes(ax: plt.Axes, font_name: str, xlabel: str, ylabel: str, grid_alpha: float) -> None:
    ax.set_xlabel(xlabel, fontname=font_name)
    ax.set_ylabel(ylabel, fontname=font_name)
    ax.grid(alpha=grid_alpha)


def _to_x_axis_values(t_axis: np.ndarray, utc_start: dt.datetime | None) -> np.ndarray:
    t = np.asarray(t_axis, dtype=float)
    if utc_start is None:
        return t
    base = mdates.date2num(utc_start)
    return base + t / 86400.0


def _configure_utc_axis(ax: plt.Axes, use_utc: bool) -> None:
    if not use_utc:
        return
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator, tz=dt.timezone.utc)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def _save_figure(fig: plt.Figure, module_name: str, component_name: str, opts: SaveOptions) -> None:
    if not opts.save:
        return

    outdir = ensure_dir(opts.outdir)
    stem = f"{_clean_name(opts.event_id)}_{_clean_name(module_name)}_{_clean_name(component_name)}"
    for fmt in opts.formats:
        fig.savefig(outdir / f"{stem}.{fmt}", dpi=fig.dpi, bbox_inches="tight", facecolor="white")


def plot_waveform(
    t_sec: np.ndarray,
    signal: np.ndarray,
    component_name: str,
    plot_params: PlotParams,
    save_opts: SaveOptions,
    normalize: bool = True,
    utc_start: dt.datetime | None = None,
):
    fig, ax = plt.subplots(1, 1, figsize=plot_params.figsize, dpi=plot_params.dpi)
    x = zscore_safe(signal) if normalize else signal
    t_axis = _to_x_axis_values(t_sec, utc_start)
    ax.plot(t_axis, x, color="tab:blue", linewidth=plot_params.linewidth_waveform)
    ax.set_title(f"{component_name} Waveform", fontname=plot_params.font_name)
    xlabel = "UTC Time" if utc_start is not None else "Time (s)"
    _style_axes(ax, plot_params.font_name, xlabel, "Amplitude", plot_params.grid_alpha)
    ax.set_xlim(float(np.min(t_axis)), float(np.max(t_axis)))
    _configure_utc_axis(ax, utc_start is not None)
    fig.tight_layout()

    _save_figure(fig, "waveform", component_name, save_opts)
    return fig, ax


def plot_spectrogram(
    t_spec: np.ndarray,
    f_hz: np.ndarray,
    s_db: np.ndarray,
    component_name: str,
    plot_params: PlotParams,
    save_opts: SaveOptions,
    vmin: float | None = None,
    vmax: float | None = None,
    utc_start: dt.datetime | None = None,
):
    fig, ax = plt.subplots(1, 1, figsize=plot_params.figsize, dpi=plot_params.dpi)
    lo, hi = _robust_limits(s_db, 5, 98)
    vmin = lo if vmin is None else vmin
    vmax = hi if vmax is None else vmax
    t_axis = _to_x_axis_values(t_spec, utc_start)

    im = ax.imshow(
        s_db,
        aspect="auto",
        origin="lower",
        extent=[float(np.min(t_axis)), float(np.max(t_axis)), float(np.min(f_hz)), float(np.max(f_hz))],
        cmap=plot_params.cmap_spec,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(f"{component_name} Spectrogram", fontname=plot_params.font_name)
    xlabel = "UTC Time" if utc_start is not None else "Time (s)"
    _style_axes(ax, plot_params.font_name, xlabel, "Frequency (Hz)", plot_params.grid_alpha)
    _configure_utc_axis(ax, utc_start is not None)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("Magnitude (dB)", fontname=plot_params.font_name)
    fig.tight_layout()

    _save_figure(fig, "spectrogram", component_name, save_opts)
    return fig, ax


def plot_lofar(
    t_spec: np.ndarray,
    f_hz: np.ndarray,
    lofar_map: np.ndarray,
    component_name: str,
    plot_params: PlotParams,
    save_opts: SaveOptions,
    vmin: float | None = None,
    vmax: float | None = None,
    utc_start: dt.datetime | None = None,
):
    fig, ax = plt.subplots(1, 1, figsize=plot_params.figsize, dpi=plot_params.dpi)
    lo, hi = _robust_limits(lofar_map, 2, 99)
    vmin = lo if vmin is None else vmin
    vmax = hi if vmax is None else vmax
    t_axis = _to_x_axis_values(t_spec, utc_start)

    im = ax.imshow(
        lofar_map,
        aspect="auto",
        origin="lower",
        extent=[float(np.min(t_axis)), float(np.max(t_axis)), float(np.min(f_hz)), float(np.max(f_hz))],
        cmap=plot_params.cmap_lofar,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(f"{component_name} LOFAR", fontname=plot_params.font_name)
    xlabel = "UTC Time" if utc_start is not None else "Time (s)"
    _style_axes(ax, plot_params.font_name, xlabel, "Frequency (Hz)", plot_params.grid_alpha)
    _configure_utc_axis(ax, utc_start is not None)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("Normalized Power", fontname=plot_params.font_name)
    fig.tight_layout()

    _save_figure(fig, "lofar", component_name, save_opts)
    return fig, ax


def plot_azimuth_spectrogram(
    t_spec: np.ndarray,
    f_hz: np.ndarray,
    azimuth_deg_tf: np.ndarray,
    plot_params: PlotParams,
    save_opts: SaveOptions,
    utc_start: dt.datetime | None = None,
):
    fig, ax = plt.subplots(1, 1, figsize=plot_params.figsize, dpi=plot_params.dpi)
    t_axis = _to_x_axis_values(t_spec, utc_start)

    im = ax.imshow(
        azimuth_deg_tf,
        aspect="auto",
        origin="lower",
        extent=[float(np.min(t_axis)), float(np.max(t_axis)), float(np.min(f_hz)), float(np.max(f_hz))],
        cmap=plot_params.cmap_azi,
        vmin=0,
        vmax=360,
    )
    ax.set_title("Azimuth Spectrogram", fontname=plot_params.font_name)
    xlabel = "UTC Time" if utc_start is not None else "Time (s)"
    _style_axes(ax, plot_params.font_name, xlabel, "Frequency (Hz)", plot_params.grid_alpha)
    _configure_utc_axis(ax, utc_start is not None)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("Azimuth (deg)", fontname=plot_params.font_name)
    fig.tight_layout()

    _save_figure(fig, "azimuth", "ALL", save_opts)
    return fig, ax


def plot_azimuth_stability(
    t_spec: np.ndarray,
    f_hz: np.ndarray,
    r_tf: np.ndarray,
    plot_params: PlotParams,
    save_opts: SaveOptions,
    utc_start: dt.datetime | None = None,
):
    fig, ax = plt.subplots(1, 1, figsize=plot_params.figsize, dpi=plot_params.dpi)
    t_axis = _to_x_axis_values(t_spec, utc_start)

    im = ax.imshow(
        r_tf,
        aspect="auto",
        origin="lower",
        extent=[float(np.min(t_axis)), float(np.max(t_axis)), float(np.min(f_hz)), float(np.max(f_hz))],
        cmap=plot_params.cmap_stability,
        vmin=0,
        vmax=1,
    )
    ax.set_title("Azimuth Stability (R)", fontname=plot_params.font_name)
    xlabel = "UTC Time" if utc_start is not None else "Time (s)"
    _style_axes(ax, plot_params.font_name, xlabel, "Frequency (Hz)", plot_params.grid_alpha)
    _configure_utc_axis(ax, utc_start is not None)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("R", fontname=plot_params.font_name)
    fig.tight_layout()

    _save_figure(fig, "azimuth_stability", "ALL", save_opts)
    return fig, ax


def plot_azimuth_confidence_mask(
    t_spec: np.ndarray,
    f_hz: np.ndarray,
    azimuth_masked_tf: np.ndarray,
    confidence_tf: np.ndarray,
    threshold: float,
    plot_params: PlotParams,
    save_opts: SaveOptions,
    utc_start: dt.datetime | None = None,
):
    fig, axes = plt.subplots(2, 1, figsize=(plot_params.figsize[0], plot_params.figsize[1] * 1.6), dpi=plot_params.dpi)
    t_axis = _to_x_axis_values(t_spec, utc_start)

    im1 = axes[0].imshow(
        azimuth_masked_tf,
        aspect="auto",
        origin="lower",
        extent=[float(np.min(t_axis)), float(np.max(t_axis)), float(np.min(f_hz)), float(np.max(f_hz))],
        cmap=plot_params.cmap_azi,
        vmin=0,
        vmax=360,
    )
    axes[0].set_title(f"Azimuth (confidence >= {threshold:.2f})", fontname=plot_params.font_name)
    xlabel = "UTC Time" if utc_start is not None else "Time (s)"
    _style_axes(axes[0], plot_params.font_name, xlabel, "Frequency (Hz)", plot_params.grid_alpha)
    _configure_utc_axis(axes[0], utc_start is not None)
    cb1 = fig.colorbar(im1, ax=axes[0], pad=0.01)
    cb1.set_label("Azimuth (deg)", fontname=plot_params.font_name)

    c_lo, c_hi = _robust_limits(confidence_tf, 1, 99)
    cmin = min(0.0, c_lo)
    cmax = max(1.0, c_hi)
    im2 = axes[1].imshow(
        confidence_tf,
        aspect="auto",
        origin="lower",
        extent=[float(np.min(t_axis)), float(np.max(t_axis)), float(np.min(f_hz)), float(np.max(f_hz))],
        cmap=plot_params.cmap_confidence,
        vmin=cmin,
        vmax=cmax,
    )
    axes[1].set_title("Confidence Map", fontname=plot_params.font_name)
    _style_axes(axes[1], plot_params.font_name, xlabel, "Frequency (Hz)", plot_params.grid_alpha)
    _configure_utc_axis(axes[1], utc_start is not None)
    cb2 = fig.colorbar(im2, ax=axes[1], pad=0.01)
    cb2.set_label("Confidence", fontname=plot_params.font_name)

    fig.tight_layout()
    _save_figure(fig, "azimuth_confidence", "ALL", save_opts)
    return fig, axes


def plot_azimuth_mask(
    t_spec: np.ndarray,
    f_hz: np.ndarray,
    azimuth_masked_tf: np.ndarray,
    threshold: float,
    plot_params: PlotParams,
    save_opts: SaveOptions,
    utc_start: dt.datetime | None = None,
):
    fig, ax = plt.subplots(1, 1, figsize=plot_params.figsize, dpi=plot_params.dpi)
    t_axis = _to_x_axis_values(t_spec, utc_start)
    im = ax.imshow(
        azimuth_masked_tf,
        aspect="auto",
        origin="lower",
        extent=[float(np.min(t_axis)), float(np.max(t_axis)), float(np.min(f_hz)), float(np.max(f_hz))],
        cmap=plot_params.cmap_azi,
        vmin=0,
        vmax=360,
    )
    ax.set_title(f"Azimuth (confidence >= {threshold:.2f})", fontname=plot_params.font_name)
    xlabel = "UTC Time" if utc_start is not None else "Time (s)"
    _style_axes(ax, plot_params.font_name, xlabel, "Frequency (Hz)", plot_params.grid_alpha)
    _configure_utc_axis(ax, utc_start is not None)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("Azimuth (deg)", fontname=plot_params.font_name)
    fig.tight_layout()
    _save_figure(fig, "azimuth_mask", "ALL", save_opts)
    return fig, ax


def plot_confidence_map(
    t_spec: np.ndarray,
    f_hz: np.ndarray,
    confidence_tf: np.ndarray,
    plot_params: PlotParams,
    save_opts: SaveOptions,
    utc_start: dt.datetime | None = None,
):
    fig, ax = plt.subplots(1, 1, figsize=plot_params.figsize, dpi=plot_params.dpi)
    t_axis = _to_x_axis_values(t_spec, utc_start)
    c_lo, c_hi = _robust_limits(confidence_tf, 1, 99)
    cmin = min(0.0, c_lo)
    cmax = max(1.0, c_hi)
    im = ax.imshow(
        confidence_tf,
        aspect="auto",
        origin="lower",
        extent=[float(np.min(t_axis)), float(np.max(t_axis)), float(np.min(f_hz)), float(np.max(f_hz))],
        cmap=plot_params.cmap_confidence,
        vmin=cmin,
        vmax=cmax,
    )
    ax.set_title("Confidence Map", fontname=plot_params.font_name)
    xlabel = "UTC Time" if utc_start is not None else "Time (s)"
    _style_axes(ax, plot_params.font_name, xlabel, "Frequency (Hz)", plot_params.grid_alpha)
    _configure_utc_axis(ax, utc_start is not None)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("Confidence", fontname=plot_params.font_name)
    fig.tight_layout()
    _save_figure(fig, "confidence", "ALL", save_opts)
    return fig, ax


def plot_merged_panels(
    selected_panels: list[str],
    t_sec: np.ndarray,
    signal: np.ndarray,
    component_name: str,
    t_spec: np.ndarray,
    f_hz: np.ndarray,
    spectrogram_db: np.ndarray,
    lofar_map: np.ndarray,
    azimuth_masked_tf: np.ndarray,
    azimuth_deg_tf: np.ndarray,
    confidence_tf: np.ndarray,
    r_tf: np.ndarray,
    threshold: float,
    plot_params: PlotParams,
    normalize_waveform: bool = True,
    utc_start: dt.datetime | None = None,
):
    panels = [p for p in selected_panels if p != "snr"]
    n = len(panels)
    if n == 0:
        raise ValueError("No panels selected for merged plotting.")

    ncols = 1
    nrows = n
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(plot_params.figsize[0] * ncols, plot_params.figsize[1] * nrows),
        dpi=plot_params.dpi,
    )
    axes_arr = np.array(axes).reshape(-1)

    for i, panel in enumerate(panels):
        ax = axes_arr[i]
        if panel == "waveform":
            x = zscore_safe(signal) if normalize_waveform else signal
            t_axis = _to_x_axis_values(t_sec, utc_start)
            ax.plot(t_axis, x, color="tab:blue", linewidth=plot_params.linewidth_waveform)
            ax.set_title(f"{component_name} Waveform", fontname=plot_params.font_name)
            xlabel = "UTC Time" if utc_start is not None else "Time (s)"
            _style_axes(ax, plot_params.font_name, xlabel, "Amplitude", plot_params.grid_alpha)
            ax.set_xlim(float(np.min(t_axis)), float(np.max(t_axis)))
            _configure_utc_axis(ax, utc_start is not None)
        elif panel == "spectrogram":
            lo, hi = _robust_limits(spectrogram_db, 5, 98)
            t_axis = _to_x_axis_values(t_spec, utc_start)
            im = ax.imshow(
                spectrogram_db,
                aspect="auto",
                origin="lower",
                extent=[float(np.min(t_axis)), float(np.max(t_axis)), float(np.min(f_hz)), float(np.max(f_hz))],
                cmap=plot_params.cmap_spec,
                vmin=lo,
                vmax=hi,
            )
            ax.set_title(f"{component_name} Spectrogram", fontname=plot_params.font_name)
            xlabel = "UTC Time" if utc_start is not None else "Time (s)"
            _style_axes(ax, plot_params.font_name, xlabel, "Frequency (Hz)", plot_params.grid_alpha)
            _configure_utc_axis(ax, utc_start is not None)
            cb = fig.colorbar(im, ax=ax, pad=0.01)
            cb.set_label("Magnitude (dB)", fontname=plot_params.font_name)
        elif panel == "lofar":
            lo, hi = _robust_limits(lofar_map, 2, 99)
            t_axis = _to_x_axis_values(t_spec, utc_start)
            im = ax.imshow(
                lofar_map,
                aspect="auto",
                origin="lower",
                extent=[float(np.min(t_axis)), float(np.max(t_axis)), float(np.min(f_hz)), float(np.max(f_hz))],
                cmap=plot_params.cmap_lofar,
                vmin=lo,
                vmax=hi,
            )
            ax.set_title(f"{component_name} LOFAR", fontname=plot_params.font_name)
            xlabel = "UTC Time" if utc_start is not None else "Time (s)"
            _style_axes(ax, plot_params.font_name, xlabel, "Frequency (Hz)", plot_params.grid_alpha)
            _configure_utc_axis(ax, utc_start is not None)
            cb = fig.colorbar(im, ax=ax, pad=0.01)
            cb.set_label("Normalized Power", fontname=plot_params.font_name)
        elif panel == "azimuth_mask":
            t_axis = _to_x_axis_values(t_spec, utc_start)
            im = ax.imshow(
                azimuth_masked_tf,
                aspect="auto",
                origin="lower",
                extent=[float(np.min(t_axis)), float(np.max(t_axis)), float(np.min(f_hz)), float(np.max(f_hz))],
                cmap=plot_params.cmap_azi,
                vmin=0,
                vmax=360,
            )
            ax.set_title(f"Azimuth (confidence >= {threshold:.2f})", fontname=plot_params.font_name)
            xlabel = "UTC Time" if utc_start is not None else "Time (s)"
            _style_axes(ax, plot_params.font_name, xlabel, "Frequency (Hz)", plot_params.grid_alpha)
            _configure_utc_axis(ax, utc_start is not None)
            cb = fig.colorbar(im, ax=ax, pad=0.01)
            cb.set_label("Azimuth (deg)", fontname=plot_params.font_name)
        elif panel == "azimuth":
            t_axis = _to_x_axis_values(t_spec, utc_start)
            im = ax.imshow(
                azimuth_deg_tf,
                aspect="auto",
                origin="lower",
                extent=[float(np.min(t_axis)), float(np.max(t_axis)), float(np.min(f_hz)), float(np.max(f_hz))],
                cmap=plot_params.cmap_azi,
                vmin=0,
                vmax=360,
            )
            ax.set_title("Azimuth Spectrogram", fontname=plot_params.font_name)
            xlabel = "UTC Time" if utc_start is not None else "Time (s)"
            _style_axes(ax, plot_params.font_name, xlabel, "Frequency (Hz)", plot_params.grid_alpha)
            _configure_utc_axis(ax, utc_start is not None)
            cb = fig.colorbar(im, ax=ax, pad=0.01)
            cb.set_label("Azimuth (deg)", fontname=plot_params.font_name)
        elif panel == "confidence":
            t_axis = _to_x_axis_values(t_spec, utc_start)
            c_lo, c_hi = _robust_limits(confidence_tf, 1, 99)
            cmin = min(0.0, c_lo)
            cmax = max(1.0, c_hi)
            im = ax.imshow(
                confidence_tf,
                aspect="auto",
                origin="lower",
                extent=[float(np.min(t_axis)), float(np.max(t_axis)), float(np.min(f_hz)), float(np.max(f_hz))],
                cmap=plot_params.cmap_confidence,
                vmin=cmin,
                vmax=cmax,
            )
            ax.set_title("Confidence Map", fontname=plot_params.font_name)
            xlabel = "UTC Time" if utc_start is not None else "Time (s)"
            _style_axes(ax, plot_params.font_name, xlabel, "Frequency (Hz)", plot_params.grid_alpha)
            _configure_utc_axis(ax, utc_start is not None)
            cb = fig.colorbar(im, ax=ax, pad=0.01)
            cb.set_label("Confidence", fontname=plot_params.font_name)
        elif panel == "azimuth_stability":
            t_axis = _to_x_axis_values(t_spec, utc_start)
            im = ax.imshow(
                r_tf,
                aspect="auto",
                origin="lower",
                extent=[float(np.min(t_axis)), float(np.max(t_axis)), float(np.min(f_hz)), float(np.max(f_hz))],
                cmap=plot_params.cmap_stability,
                vmin=0,
                vmax=1,
            )
            ax.set_title("Azimuth Stability (R)", fontname=plot_params.font_name)
            xlabel = "UTC Time" if utc_start is not None else "Time (s)"
            _style_axes(ax, plot_params.font_name, xlabel, "Frequency (Hz)", plot_params.grid_alpha)
            _configure_utc_axis(ax, utc_start is not None)
            cb = fig.colorbar(im, ax=ax, pad=0.01)
            cb.set_label("R", fontname=plot_params.font_name)

    for j in range(n, len(axes_arr)):
        axes_arr[j].axis("off")

    fig.tight_layout()
    return fig, axes_arr
