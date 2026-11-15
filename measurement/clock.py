"""Clock synchronization helpers.

All timestamped sources share one common clock (see
``docs/metrics_specification.md`` §5). Elapsed-time metrics are only valid on
that shared base. This module centralizes timestamp parsing and elapsed-time
computation so every metric uses identical, deterministic logic, and provides an
offset-bound assertion used before each trial set.
"""

from __future__ import annotations

from datetime import datetime, timedelta


class ClockError(ValueError):
    """Raised when timestamps are malformed or clock offset exceeds tolerance."""


def parse(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp deterministically.

    Accepts a trailing ``Z`` (UTC) as an alias for ``+00:00``.
    """

    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError) as exc:  # pragma: no cover - defensive
        raise ClockError(f"invalid ISO-8601 timestamp: {ts!r}") from exc


def elapsed_seconds(t0: str, t1: str) -> float:
    """Return ``t1 - t0`` in seconds (may be negative if out of order)."""

    return (parse(t1) - parse(t0)).total_seconds()


def shift(ts: str, seconds: float) -> str:
    """Return ``ts`` advanced by ``seconds``, as an ISO-8601 ``...Z`` string."""

    return (parse(ts) + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def assert_offset_within(offset_s: float, bound_s: float) -> None:
    """Assert a measured clock offset is within ``bound_s`` seconds.

    Called before each trial set to document that all sources agree on the
    common clock to within the recorded tolerance.
    """

    if abs(offset_s) > bound_s:
        raise ClockError(f"clock offset {offset_s:.6f}s exceeds tolerance {bound_s:.6f}s")
