"""Security-outcome metrics: TTC, TFVE, attempts-per-success.

Definitions: ``docs/metrics_specification.md`` §1.2, §2.2, §2.5. The binary
``exploit_success`` and ``objective_count`` are produced by
``measurement.correlation`` and combined here for convenience.

All time metrics are computed on the synchronized clock and honor right-censoring
at the trial-duration limit.
"""

from __future__ import annotations

from dataclasses import dataclass

from measurement import clock
from measurement.correlation import ObjectiveResult
from measurement.models import ExploitDelivery, ReconCommand


@dataclass(frozen=True)
class CensoredValue:
    """A possibly right-censored measurement."""

    value: float
    censored: bool


def first_recon_timestamp(recon_commands: tuple[ReconCommand, ...]) -> str | None:
    """Timestamp of the first reconnaissance command, or None if there were none."""

    if not recon_commands:
        return None
    return min(recon_commands, key=lambda c: clock.parse(c.timestamp)).timestamp


def time_to_compromise(
    first_recon_ts: str | None,
    objectives: ObjectiveResult,
    trial_duration_limit_s: int,
) -> CensoredValue:
    """Seconds from first recon command to the actual **compromise** (``obj_b``).

    Compromise == unauthorized command execution / data access on a genuine service
    (the same event ``exploitation_success`` records). Right-censored at
    ``trial_duration_limit_s`` when no compromise occurred (or when there was no recon
    command to anchor the interval). This is the **primary** time-to-event (survival)
    outcome and the time-domain analogue of ``exploitation_success``; it is not
    saturated by easy enumeration, so it carries the graded MTD effect under the
    randomized reconfiguration race. ``time_to_enumeration`` is the diagnostic
    (first-objective / enumeration) timing.
    """

    if first_recon_ts is None or objectives.ts_b is None:
        return CensoredValue(float(trial_duration_limit_s), censored=True)
    return CensoredValue(clock.elapsed_seconds(first_recon_ts, objectives.ts_b), censored=False)


def time_to_enumeration(
    first_recon_ts: str | None,
    objectives: ObjectiveResult,
    trial_duration_limit_s: int,
) -> CensoredValue:
    """Seconds from first recon command to the **first verified objective**.

    The first objective is almost always ``obj_a`` (enumeration of a genuine service),
    achieved at the reconnaissance snapshot, so this saturates near the fixed recon
    offset and is largely insensitive to MTD. It is a **secondary/diagnostic** outcome
    (reported descriptively and FDR-controlled), retained from the former
    enumeration-inclusive ``time_to_compromise`` definition; the actual-compromise
    survival outcome is ``time_to_compromise`` (``obj_b``).
    """

    obj_ts = objectives.first_objective_ts
    if first_recon_ts is None or obj_ts is None:
        return CensoredValue(float(trial_duration_limit_s), censored=True)
    return CensoredValue(clock.elapsed_seconds(first_recon_ts, obj_ts), censored=False)


def time_to_first_valid_exploit(
    origin: str | None,
    exploit_deliveries: tuple[ExploitDelivery, ...],
    trial_duration_limit_s: int,
) -> CensoredValue:
    """Seconds from workflow initiation to the first technically valid exploit.

    Per the research design, workflow initiation is the *first reconnaissance
    command* — the same origin as time-to-compromise. A valid exploit is any
    delivery that reached the target and executed, regardless of objective
    (genuine or decoy). Right-censored at the limit if none executed or if there
    was no recon command to anchor the interval.
    """

    valid = [d for d in exploit_deliveries if d.reached_and_executed]
    if origin is None or not valid:
        return CensoredValue(float(trial_duration_limit_s), censored=True)
    first = min(valid, key=lambda d: clock.parse(d.timestamp))
    return CensoredValue(clock.elapsed_seconds(origin, first.timestamp), censored=False)


def _is_success_delivery(d: ExploitDelivery) -> bool:
    """A successful exploit delivery = genuine target + unauthorized access."""

    return d.reached_and_executed and d.target_genuine and d.unauthorized_access


def attempts_per_success(
    exploit_deliveries: tuple[ExploitDelivery, ...],
) -> tuple[int, bool]:
    """1-based index of the first successful delivery; else total attempts.

    Returns ``(count, censored)``. Censored is True when no delivery succeeded
    (count is then the total number of delivery attempts).
    """

    ordered = sorted(exploit_deliveries, key=lambda d: clock.parse(d.timestamp))
    for i, d in enumerate(ordered, start=1):
        if _is_success_delivery(d):
            return i, False
    return len(ordered), True
