from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class TimeSlice:
    start_s: float
    end_s: float

    def normalized(self, total_duration_s: float) -> "TimeSlice":
        start = max(0.0, float(self.start_s))
        end = min(float(self.end_s), float(total_duration_s))
        if end <= start:
            raise ValueError(f"Invalid time slice: start={self.start_s}, end={self.end_s}")
        return TimeSlice(start_s=start, end_s=end)


def crop_signals_by_time(
    t_sec: np.ndarray,
    signals: Dict[str, np.ndarray],
    start_s: float,
    end_s: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], dict]:
    if t_sec.size == 0:
        raise ValueError("Empty time axis cannot be cropped.")

    dt = float(np.median(np.diff(t_sec))) if t_sec.size > 1 else 1.0
    total_duration = float(t_sec[-1]) if t_sec.size > 0 else 0.0
    ts = TimeSlice(start_s=start_s, end_s=end_s).normalized(total_duration)

    i0 = int(np.floor(ts.start_s / dt))
    i1 = int(np.ceil(ts.end_s / dt)) + 1
    i0 = max(0, i0)
    i1 = min(len(t_sec), i1)

    t_crop = t_sec[i0:i1]
    if t_crop.size == 0:
        raise ValueError(f"No samples in selected interval [{start_s}, {end_s}] s")

    t_crop = t_crop - t_crop[0]
    out = {k: v[i0:i1] for k, v in signals.items()}
    info = {
        "index_range": (i0, i1),
        "source_time_range_s": (float(t_sec[i0]), float(t_sec[i1 - 1])),
        "duration_s": float(t_crop[-1]) if t_crop.size > 0 else 0.0,
    }
    return t_crop, out, info
