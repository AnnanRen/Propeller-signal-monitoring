# %%
from pathlib import Path

import matplotlib.pyplot as plt

from src.pipeline import PipelineParams, list_events, process_event
from src.plotting import (
    PlotParams,
    SaveOptions,
    plot_azimuth_confidence_mask,
    plot_azimuth_spectrogram,
    plot_azimuth_stability,
    plot_lofar,
    plot_spectrogram,
    plot_waveform,
)

# %%
# User parameters
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT  # or PROJECT_ROOT / "data" / "raw"

WINDOW_LENGTH_S = 2.0
OVERLAP = 0.5
SELECTED_BAND = None  # e.g. (150.0, 400.0)
TIME_SLICE_S = None  # e.g. (600.0, 780.0)
STABILITY_WINDOW = 15
STABILITY_STEP = 5
CONFIDENCE_THRESHOLD = 0.6

SAVE_FIGURES = False
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
FORMATS = ("png", "pdf")

# Toggle modules
PLOT_WAVEFORM = True
PLOT_SPECTROGRAM = True
PLOT_LOFAR = True
PLOT_AZIMUTH = True
PLOT_AZIMUTH_STABILITY = True
PLOT_AZIMUTH_CONFIDENCE = True

# %%
# Event selection
events = list_events(DATA_DIR)
if not events:
    raise RuntimeError(f"No complete SAC bundles found in: {DATA_DIR}")

print(f"Found {len(events)} complete events:")
for i, ev in enumerate(events):
    print(f"  [{i}] {ev.event_id}")

EVENT_IDX = 0
bundle = events[EVENT_IDX]

# %%
# Process
params = PipelineParams(
    data_dir=DATA_DIR,
    window_length_s=WINDOW_LENGTH_S,
    overlap=OVERLAP,
    selected_band=SELECTED_BAND,
    time_slice_s=TIME_SLICE_S,
    stability_window=STABILITY_WINDOW,
    stability_step=STABILITY_STEP,
    confidence_threshold=CONFIDENCE_THRESHOLD,
)
res = process_event(bundle, params)

print("Event:", res["event_id"])
print("Selected band:", res["selected_band"])
print("Band candidates:", res["band_info"]["candidates"])
print("Time slice:", res["time_slice_s"])
print("Crop info:", res["crop_info"])

# %%
# Plot settings
plot_params = PlotParams(
    freq_min=res["selected_band"][0],
    freq_max=res["selected_band"][1],
)
save_opts = SaveOptions(
    save=SAVE_FIGURES,
    outdir=OUTPUT_DIR,
    event_id=res["event_id"],
    formats=FORMATS,
)
COMPONENT = "HYD"  # BH1 / BH2 / BHZ / HYD

# %%
# Waveform
if PLOT_WAVEFORM:
    fig, _ = plot_waveform(
        res["t_sec"],
        res["signals"][COMPONENT],
        COMPONENT,
        plot_params,
        save_opts,
        normalize=True,
    )
    plt.show()

# %%
# Spectrogram
if PLOT_SPECTROGRAM:
    fig, _ = plot_spectrogram(
        res["t_spec"],
        res["f_hz"],
        res["spectrogram_db"][COMPONENT],
        COMPONENT,
        plot_params,
        save_opts,
    )
    plt.show()

# %%
# LOFAR
if PLOT_LOFAR:
    fig, _ = plot_lofar(
        res["t_spec"],
        res["f_hz"],
        res["lofar"][COMPONENT],
        COMPONENT,
        plot_params,
        save_opts,
    )
    plt.show()

# %%
# Azimuth spectrogram
if PLOT_AZIMUTH:
    fig, _ = plot_azimuth_spectrogram(
        res["t_spec"],
        res["f_hz"],
        res["azimuth_deg"],
        plot_params,
        save_opts,
    )
    plt.show()

# %%
# Azimuth stability
if PLOT_AZIMUTH_STABILITY:
    fig, _ = plot_azimuth_stability(
        res["t_spec"],
        res["f_hz"],
        res["azimuth_stability"],
        plot_params,
        save_opts,
    )
    plt.show()

# %%
# Azimuth confidence
if PLOT_AZIMUTH_CONFIDENCE:
    fig, _ = plot_azimuth_confidence_mask(
        res["t_spec"],
        res["f_hz"],
        res["azimuth_masked"],
        res["confidence"],
        threshold=CONFIDENCE_THRESHOLD,
        plot_params=plot_params,
        save_opts=save_opts,
    )
    plt.show()

# %%
# TODO
# - Structured result tables/exports
# - Feature extraction
# - Truth-based validation
# - Batch processing workflow
# - GUI (app.py)
