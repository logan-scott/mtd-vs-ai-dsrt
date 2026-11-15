"""Operational-overhead metrics.

Definitions: ``docs/metrics_specification.md`` §3.1–§3.3. All overhead is
relative to the matching static-baseline reference (see A4 in
``docs/open_items.md``); ``operational_event_volume`` is an absolute count.
"""

from __future__ import annotations

from statistics import fmean

from measurement import clock
from measurement.models import (
    BaselineReference,
    MonitoringSample,
    ReconfigEvent,
    WorkloadSample,
)


def _mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


def _pct_increase(mean_condition: float | None, mean_baseline: float) -> float | None:
    """Percent increase vs. baseline; None if inputs are unusable."""

    if mean_condition is None or mean_baseline == 0:
        return None
    return 100.0 * (mean_condition - mean_baseline) / mean_baseline


def cpu_overhead_pct(
    samples: tuple[MonitoringSample, ...], baseline: BaselineReference
) -> float | None:
    return _pct_increase(_mean([s.cpu_pct for s in samples]), baseline.mean_cpu_pct)


def memory_overhead_pct(
    samples: tuple[MonitoringSample, ...], baseline: BaselineReference
) -> float | None:
    return _pct_increase(_mean([s.memory_pct for s in samples]), baseline.mean_memory_pct)


def network_overhead_pct(
    samples: tuple[MonitoringSample, ...], baseline: BaselineReference
) -> float | None:
    return _pct_increase(_mean([s.network_pct for s in samples]), baseline.mean_network_pct)


def latency_overhead_ms(
    samples: tuple[WorkloadSample, ...], baseline: BaselineReference
) -> float | None:
    """Δ mean response time (ms) vs. baseline, over answered requests.

    Returns None if no answered request carried a latency measurement.
    """

    latencies = [
        s.latency_ms for s in samples if s.answered_within_timeout and s.latency_ms is not None
    ]
    mean_condition = _mean(latencies)
    if mean_condition is None:
        return None
    return mean_condition - baseline.mean_response_ms


def operational_event_volume(
    events: tuple[ReconfigEvent, ...], trial_start: str, trial_end: str
) -> int:
    """Count of reconfiguration events within [trial_start, trial_end] inclusive."""

    lo = clock.parse(trial_start)
    hi = clock.parse(trial_end)
    return sum(1 for e in events if lo <= clock.parse(e.timestamp) <= hi)
