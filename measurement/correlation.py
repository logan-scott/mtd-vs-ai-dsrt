"""Objective verification via log correlation against ground truth.

Codes the three Cyber-Kill-Chain attacker objectives from offensive output
(recon snapshots + exploit deliveries) correlated with the genuine
(non-deceptive) ground-truth inventory. See
``docs/metrics_specification.md`` §1.1.

- ``obj_a`` — enumerate a genuine service.
- ``obj_b`` — unauthorized command execution / data access on a genuine service.
- ``obj_c`` — multi-step chain across ≥ 2 genuine services.

Each objective is timestamped so the earliest achieved objective can drive
time-to-compromise. All logic is deterministic and side-effect free.
"""

from __future__ import annotations

from dataclasses import dataclass

from measurement import clock
from measurement.models import ExploitDelivery, GroundTruthInventory, ReconSnapshot


@dataclass(frozen=True)
class ObjectiveResult:
    """Verified objectives and the timestamp at which each was first achieved."""

    obj_a: bool
    obj_b: bool
    obj_c: bool
    ts_a: str | None
    ts_b: str | None
    ts_c: str | None

    @property
    def objective_count(self) -> int:
        return int(self.obj_a) + int(self.obj_b) + int(self.obj_c)

    @property
    def exploit_success(self) -> bool:
        return self.obj_a or self.obj_b or self.obj_c

    @property
    def first_objective_ts(self) -> str | None:
        """Earliest-in-time achieved-objective timestamp, or None if none."""

        stamps = [t for t in (self.ts_a, self.ts_b, self.ts_c) if t is not None]
        if not stamps:
            return None
        return min(stamps, key=clock.parse)


def verify_objectives(
    ground_truth: GroundTruthInventory,
    recon_snapshots: tuple[ReconSnapshot, ...],
    exploit_deliveries: tuple[ExploitDelivery, ...],
) -> ObjectiveResult:
    """Correlate offensive output with ground truth to code obj_a/b/c."""

    genuine = ground_truth.genuine

    # obj_a: earliest recon snapshot that discovered any genuine service.
    ts_a: str | None = None
    for snap in sorted(recon_snapshots, key=lambda s: clock.parse(s.timestamp)):
        if snap.discovered & genuine:
            ts_a = snap.timestamp
            break

    # Deliveries in chronological order; a genuine compromise = reached the target,
    # it is genuine, and it yielded unauthorized command execution / data access.
    ordered = sorted(exploit_deliveries, key=lambda d: clock.parse(d.timestamp))

    ts_b: str | None = None
    ts_c: str | None = None
    compromised: set = set()
    for d in ordered:
        if d.target_genuine and d.unauthorized_access:
            if ts_b is None:
                ts_b = d.timestamp
            compromised.add(d.target)
            if len(compromised) >= 2 and ts_c is None:
                ts_c = d.timestamp

    return ObjectiveResult(
        obj_a=ts_a is not None,
        obj_b=ts_b is not None,
        obj_c=ts_c is not None,
        ts_a=ts_a,
        ts_b=ts_b,
        ts_c=ts_c,
    )
