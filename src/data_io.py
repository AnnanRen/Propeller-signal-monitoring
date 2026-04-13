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


COMPONENT_FILE_SUFFIX = {
    "BH1": ".bh1.sac",
    "BH2": ".bh2.sac",
    "BHZ": ".bhz.sac",
    "HYD": ".hyd.sac",
}


def find_sac_bundles(data_dir: str | Path) -> List[SACBundle]:
    data_path = Path(data_dir)
    bundles: List[SACBundle] = []

    for bhz_file in sorted(data_path.glob("*.bhz.sac")):
        event_id = bhz_file.name[: -len(".bhz.sac")]
        candidate = {
            comp: data_path / f"{event_id}{suffix}"
            for comp, suffix in COMPONENT_FILE_SUFFIX.items()
        }

        if all(path.exists() for path in candidate.values()):
            bundles.append(
                SACBundle(
                    event_id=event_id,
                    bh1=candidate["BH1"],
                    bh2=candidate["BH2"],
                    bhz=candidate["BHZ"],
                    hyd=candidate["HYD"],
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
