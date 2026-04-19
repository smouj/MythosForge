"""
MythosForge API — In-process metrics collector.

Lightweight Prometheus-style metrics without external dependencies.
Exposes counters, histograms, and gauges as a JSON endpoint.
"""

from __future__ import annotations

import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricPoint:
    value: float
    timestamp: float = field(default_factory=time.time)


class MetricsRegistry:
    """Simple in-process metrics registry.

    Thread-safe, no external dependencies.  Use for dev/staging.
    For production, replace with prometheus_client.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)

    # ── Counters ─────────────────────────────────────

    def inc(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += value

    # ── Gauges ───────────────────────────────────────

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    # ── Histograms ───────────────────────────────────

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._histograms[name].append(value)
            # Keep last 1000 observations
            if len(self._histograms[name]) > 1000:
                self._histograms[name] = self._histograms[name][-500:]

    def observe_latency(self, endpoint: str, duration_ms: float) -> None:
        self.observe(f"http_request_duration_ms_{endpoint}", duration_ms)
        self.observe("http_request_duration_ms_all", duration_ms)

    # ── Serialization ────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            result: dict[str, Any] = {"counters": {}, "gauges": {}, "histograms": {}}
            for k, v in self._counters.items():
                result["counters"][k] = v
            for k, v in self._gauges.items():
                result["gauges"][k] = v
            for k, values in self._histograms.items():
                if values:
                    sorted_vals = sorted(values)
                    result["histograms"][k] = {
                        "count": len(sorted_vals),
                        "min": round(sorted_vals[0], 3),
                        "max": round(sorted_vals[-1], 3),
                        "avg": round(sum(sorted_vals) / len(sorted_vals), 3),
                        "p50": round(sorted_vals[len(sorted_vals) // 2], 3),
                        "p95": round(sorted_vals[int(len(sorted_vals) * 0.95)], 3),
                        "p99": round(sorted_vals[int(len(sorted_vals) * 0.99)], 3),
                    }
            return result


# Singleton
metrics = MetricsRegistry()
