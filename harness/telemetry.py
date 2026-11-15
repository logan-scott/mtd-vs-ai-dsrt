"""Synthetic operational telemetry for OFFLINE dataset generation and fixture determinism.

These functions are deterministic stand-ins for real monitoring/workload collection. Their
sole legitimate roles are (1) OFFLINE dataset generation and (2) the determinism guarantees
the offline pipeline and its fixtures depend on. They are **not** a measurement of any real
system: overhead is a linear function of the reconfiguration-event count
(``BASE_* + PER_EVENT_* * n_events``), which is why the synthetic cost side of a run is
near-uniform across MTD conditions at equal cadence, near-perfectly collinear across the four
overhead metrics, and (with a 30-sample workload cap against a ~35 s cadence over an 1800 s
window) saturates ``service_availability_impact`` — see ``docs/metrics_specification.md`` §3.

**Division of labour.** ``harness.trial_runner`` attaches these offline, but gates them in a
LIVE run: a live run uses the synthetic stand-in only when ``config.live.synthetic_telemetry``
is explicitly set, and otherwise records the five synthetic cost columns as not-measured
(null, ``operational_cost_source="not_measured"``). **Every run archived under
``docs/results/`` (the pilot and the five full runs, including the live confirmatory run
``full_haiku_20260902``) carries these synthetic values in its five cost columns** — they
predate the gate and are labeled in ``docs/metrics_specification.md`` §3; their *security*
outcomes are genuine live/offline measurements and are unaffected. **Measured** operational
cost is produced at the TECHNIQUE level by the reconfiguration benchmark
(``harness.reconfig_benchmark``), which times a real ``reconfigure.sh`` on the targets node —
that is what "replaced by real collectors" now means in practice.
"""

from __future__ import annotations

import random

from measurement import clock
from measurement.models import BaselineReference, MonitoringSample, WorkloadSample

# Static-baseline reference means (the comparator).
BASE_RESPONSE_MS = 100.0
BASE_CPU = 20.0
BASE_MEM = 40.0
BASE_NET = 10.0

# Per-reconfiguration-event overhead increments.
PER_EVENT_CPU = 0.3
PER_EVENT_MEM = 0.4
PER_EVENT_NET = 0.2
PER_EVENT_LATENCY_MS = 1.0

_MONITORING_SAMPLES = 6
_MAX_WORKLOAD_REQUESTS = 30


def baseline_reference(vuln_profile_version: str) -> BaselineReference:
    """The static-baseline reference all overhead is measured against."""

    return BaselineReference(
        vuln_profile_version=vuln_profile_version,
        mean_response_ms=BASE_RESPONSE_MS,
        mean_cpu_pct=BASE_CPU,
        mean_memory_pct=BASE_MEM,
        mean_network_pct=BASE_NET,
    )


def synth_monitoring(
    n_events: int, start_ts: str, horizon_s: int, interval_s: int, seed: int
) -> tuple[MonitoringSample, ...]:
    """Utilization samples whose mean rises with reconfiguration burden."""

    rng = random.Random((seed ^ 0xA5A5) & 0xFFFFFFFF)
    cpu = BASE_CPU + PER_EVENT_CPU * n_events
    mem = BASE_MEM + PER_EVENT_MEM * n_events
    net = BASE_NET + PER_EVENT_NET * n_events
    step = max(horizon_s // _MONITORING_SAMPLES, interval_s)

    samples: list[MonitoringSample] = []
    for i in range(_MONITORING_SAMPLES):
        ts = clock.shift(start_ts, min(i * step, horizon_s))
        samples.append(
            MonitoringSample(
                timestamp=ts,
                cpu_pct=cpu + rng.uniform(-0.5, 0.5),
                memory_pct=mem + rng.uniform(-0.5, 0.5),
                network_pct=net + rng.uniform(-0.3, 0.3),
            )
        )
    return tuple(samples)


def synth_workload(
    event_timestamps: tuple[str, ...],
    start_ts: str,
    horizon_s: int,
    rps: float,
    seed: int,
) -> tuple[WorkloadSample, ...]:
    """Legitimate-request stream (subsampled across the window).

    Requests are spread evenly over the trial window; the request closest to each
    reconfiguration event is failed to model the brief service disruption during a
    handoff. The static baseline (no events) answers every request.
    """

    rng = random.Random((seed ^ 0x5A5A) & 0xFFFFFFFF)
    n = max(min(int(horizon_s * rps), _MAX_WORKLOAD_REQUESTS), 1)
    interval = horizon_s / n
    latency_mean = BASE_RESPONSE_MS + PER_EVENT_LATENCY_MS * len(event_timestamps)

    # The request index nearest each reconfiguration event is disrupted.
    failed: set[int] = set()
    for e in event_timestamps:
        eo = clock.elapsed_seconds(start_ts, e)
        failed.add(min(int(round(eo / interval)), n - 1))

    samples: list[WorkloadSample] = []
    for i in range(n):
        ts = clock.shift(start_ts, i * interval)
        if i in failed:
            samples.append(WorkloadSample(ts, answered_within_timeout=False, latency_ms=None))
        else:
            samples.append(
                WorkloadSample(
                    ts,
                    answered_within_timeout=True,
                    latency_ms=latency_mean + rng.uniform(-2.0, 2.0),
                )
            )
    return tuple(samples)
