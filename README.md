# SAC Azimuth Spectral Analysis

## Project layout

- `run.py`: notebook-style entrypoint with `# %%` cells (parameter tuning + module calls only)
- `notebooks/run.ipynb`: optional notebook entrypoint
- `src/data_io.py`: SAC discovery and reading
- `src/spectral.py`: STFT, LOFAR normalization, automatic frequency-band suggestion
- `src/segment.py`: time-slice cropping for focused analysis
- `src/azimuth.py`: azimuth map, stability (R), confidence mask
- `src/plotting.py`: independent plotting modules
- `src/pipeline.py`: end-to-end analysis assembly

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Put SAC files in the project root or set `data_dir` in notebook cells.

3. Run with script or notebook:

```bash
python run.py
```

## Notes

- Plots support both PNG and PDF outputs.
- The notebook controls which modules to render and save.
- Use `PipelineParams(time_slice_s=(start_s, end_s))` to analyze only an interested segment.
- Table outputs, feature extraction, truth-based validation, batch processing, and GUI are tracked as TODOs.
