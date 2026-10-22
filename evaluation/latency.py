"""Inference latency tracking and summary statistics."""

from __future__ import annotations

import numpy as np


class LatencyTracker:
    """Records per-inference wall-clock latency in seconds."""

    def __init__(self) -> None:
        self._records: list[float] = []

    def record(self, seconds: float) -> None:
        """Record one inference latency observation."""
        self._records.append(float(seconds))

    def summary(self) -> dict[str, float | int]:
        """Return ``{count, mean, p50, p99}`` of the recorded latencies."""
        if not self._records:
            return {"count": 0, "mean": 0.0, "p50": 0.0, "p99": 0.0}

        arr = np.asarray(self._records, dtype=float)
        return {
            "count": len(self._records),
            "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p99": float(np.percentile(arr, 99)),
        }
