from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from obspy import read


@dataclass(frozen=True)
class SACBundle:
    event_id: str
    bh1: Path
    bh2: Path
    bhz: Path
    hyd: Path


COMPONENT_SUFFIX_CANDIDATES = {
    "BH1": (".bh1.sac", ".bh1"),
    "BH2": (".bh2.sac", ".bh2"),
    "BHZ": (".bhz.sac", ".bhz"),
    "HYD": (".hyd.sac", ".hyd"),
}


def is_ignored_system_file(filename: str) -> bool:
    name = str(filename).strip()
    if not name:
        return True
    lower_name = name.lower()
    return bool(
        name.startswith(".")
        or name.startswith("._")
        or lower_name in {".ds_store"}
    )


def read_sac_header(file_path: str | Path) -> dict:
    tr = read(str(file_path), headonly=True)[0]
    dt = float(tr.stats.delta)
    fs = 1.0 / dt
    npts = int(tr.stats.npts)
    return {
        "delta": dt,
        "fs": fs,
        "npts": npts,
    }


def parse_event_component_from_filename(filename: str) -> Tuple[str, str, int] | None:
    """
    Parse filename into (event_id, component, priority).
    Lower priority value means higher precedence.
    """
    name = str(filename)
    if is_ignored_system_file(name):
        return None

    lower_name = name.lower()
    for comp, suffixes in COMPONENT_SUFFIX_CANDIDATES.items():
        for priority, suffix in enumerate(suffixes):
            if not lower_name.endswith(suffix):
                continue
            event_id = name[: -len(suffix)]
            if not event_id:
                return None
            return event_id, comp, priority
    return None


def find_sac_bundles(data_dir: str | Path) -> List[SACBundle]:
    data_path = Path(data_dir)
    bundles: List[SACBundle] = []
    grouped: Dict[str, Dict[str, Tuple[int, Path]]] = {}

    for file_path in sorted(data_path.iterdir()):
        if not file_path.is_file():
            continue
        if is_ignored_system_file(file_path.name):
            continue
        parsed = parse_event_component_from_filename(file_path.name)
        if parsed is None:
            continue

        event_id, component, priority = parsed
        comp_map = grouped.setdefault(event_id, {})
        existing = comp_map.get(component)
        if existing is None or priority < existing[0]:
            comp_map[component] = (priority, file_path)

    for event_id in sorted(grouped.keys()):
        comp_map = grouped[event_id]
        if not all(comp in comp_map for comp in ("BH1", "BH2", "BHZ", "HYD")):
            continue
        bundles.append(
            SACBundle(
                event_id=event_id,
                bh1=comp_map["BH1"][1],
                bh2=comp_map["BH2"][1],
                bhz=comp_map["BHZ"][1],
                hyd=comp_map["HYD"][1],
            )
        )
    return bundles


def read_sac_trace(file_path: str | Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    tr = read(str(file_path))[0]
    dt = float(tr.stats.delta)
    fs = 1.0 / dt
    npts = int(tr.stats.npts)

    t_sec = np.arange(npts, dtype=float) * dt
    data = tr.data.astype(np.float64)

    sac = getattr(tr.stats, "sac", None)
    meta = {
        "delta": dt,
        "fs": fs,
        "npts": npts,
        "start": {
            "nzyear": int(getattr(sac, "nzyear", 2026)),
            "nzjday": int(getattr(sac, "nzjday", 1)),
            "nzhour": int(getattr(sac, "nzhour", 0)),
            "nzmin": int(getattr(sac, "nzmin", 0)),
            "nzsec": int(getattr(sac, "nzsec", 0)),
            "nzmsec": int(getattr(sac, "nzmsec", 0)),
        },
    }
    return t_sec, data, meta


def load_bundle(bundle: SACBundle) -> Dict[str, np.ndarray | float | np.ndarray]:
    t_sec, bh1, meta1 = read_sac_trace(bundle.bh1)
    _, bh2, _ = read_sac_trace(bundle.bh2)
    _, bhz, _ = read_sac_trace(bundle.bhz)
    _, hyd, _ = read_sac_trace(bundle.hyd)

    lengths = {len(bh1), len(bh2), len(bhz), len(hyd)}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent component lengths for event {bundle.event_id}: {sorted(lengths)}")

    return {
        "event_id": bundle.event_id,
        "t_sec": t_sec,
        "fs": float(meta1["fs"]),
        "meta": meta1,
        "BH1": bh1,
        "BH2": bh2,
        "BHZ": bhz,
        "HYD": hyd,
    }


def get_bundle_overview(bundle: SACBundle) -> dict:
    component_paths = {
        "BH1": bundle.bh1,
        "BH2": bundle.bh2,
        "BHZ": bundle.bhz,
        "HYD": bundle.hyd,
    }
    headers = {comp: read_sac_header(path) for comp, path in component_paths.items()}
    warnings: List[str] = []

    fs_values = {comp: float(h["fs"]) for comp, h in headers.items()}
    npts_values = {comp: int(h["npts"]) for comp, h in headers.items()}

    ref_component = "BHZ" if "BHZ" in headers else next(iter(headers.keys()))
    ref_fs = fs_values[ref_component]
    ref_npts = npts_values[ref_component]

    fs_set = {round(v, 12) for v in fs_values.values()}
    if len(fs_set) > 1:
        warnings.append(f"四分量采样率不一致：{fs_values}。当前按 {ref_component} 显示事件概况。")

    npts_set = set(npts_values.values())
    if len(npts_set) > 1:
        warnings.append(f"四分量采样点数不一致：{npts_values}。当前按 {ref_component} 显示事件概况。")

    duration_s = max(0.0, (float(ref_npts) - 1.0) / float(ref_fs)) if ref_npts > 0 else 0.0
    return {
        "event_id": bundle.event_id,
        "fs_hz": float(ref_fs),
        "duration_s": float(duration_s),
        "duration_hours": float(duration_s / 3600.0),
        "duration_days": float(duration_s / 86400.0),
        "warnings": warnings,
        "reference_component": ref_component,
        "component_headers": headers,
    }
