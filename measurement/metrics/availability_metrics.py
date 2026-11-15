"""Service-availability metrics + per-event uptime records for survival analysis.

Definitions: ``docs/metrics_specification.md`` §3.4.

- ``service_availability_impact``: proportion of synthetic legitimate requests
  answered validly within the timeout during the trial.
- ``availability_events``: per-failure ``time_to_failure`` / ``time_to_recovery``
  records feeding the Kaplan–Meier / log-rank / Cox survival analysis of service
  uptime. Recovery not observed within the trial is right-censored.
"""

from __future__ import annotations

from measurement import clock
from measurement.models import AvailabilityEvent, WorkloadSample


def service_availability_impact(samples: tuple[WorkloadSample, ...]) -> float | None:
    """Fraction of legitimate requests answered within timeout; None if no requests."""

    if not samples:
        return None
    valid = sum(1 for s in samples if s.answered_within_timeout)
    return valid / len(samples)


def availability_events(
    samples: tuple[WorkloadSample, ...], trial_start: str
) -> list[AvailabilityEvent]:
    """Derive failure/recovery events from an ordered availability probe stream.

    A failure onset is the first failed request after an "up" state; recovery is
    the next answered request. ``time_to_failure`` is measured from ``trial_start``
    for the first failure and from the previous recovery thereafter. A failure not
    recovered by the end of the stream is right-censored (``time_to_recovery`` None).
    """

    ordered = sorted(samples, key=lambda s: clock.parse(s.timestamp))
    events: list[AvailabilityEvent] = []

    up = True
    last_up_reference = trial_start
    onset: str | None = None

    for s in ordered:
        if up and not s.answered_within_timeout:
            # transition up -> down
            onset = s.timestamp
            up = False
        elif not up and s.answered_within_timeout:
            # transition down -> up (recovery)
            assert onset is not None
            events.append(
                AvailabilityEvent(
                    failure_timestamp=onset,
                    time_to_failure_s=clock.elapsed_seconds(last_up_reference, onset),
                    time_to_recovery_s=clock.elapsed_seconds(onset, s.timestamp),
                    recovery_censored=False,
                )
            )
            last_up_reference = s.timestamp
            up = True
            onset = None

    if not up and onset is not None:
        # ended while still down: recovery right-censored
        events.append(
            AvailabilityEvent(
                failure_timestamp=onset,
                time_to_failure_s=clock.elapsed_seconds(last_up_reference, onset),
                time_to_recovery_s=None,
                recovery_censored=True,
            )
        )

    return events
