"""MTD reconfiguration orchestrator.

Drives a condition's reconfiguration over a trial window at a fixed cadence,
emitting the **reconfiguration event log** (source of ``operational_event_volume``)
and **ground-truth inventory snapshots** (regenerated on every change). Given a
seed it is fully deterministic — identical inputs produce an identical event log
and identical snapshots — which is what lets the offline pipeline reproduce a live
trial's telemetry.

A *reconfiguration event* is one mutation cycle in which at least one observable
attribute changed. The static baseline produces none.

**Randomized rotation phase (Method C).** By default the tick grid is anchored at
the window start (``fixed`` phase): ticks fall at ``start + k*cadence``. When
``rotation_phase="randomized"``, a per-trial phase ``φ`` is drawn (seeded, uniform
on ``[0, cadence)``) and the grid is shifted to ``start + φ + k*cadence``. This
models MTD efficacy as a *race*: the reconfiguration is applied on a randomized
phase relative to the attacker's deterministic action timeline, so across a
condition's trials the attacker's exploit sometimes lands **before** the next tick
(recon-time intel is fresh → the attacker wins → compromise) and sometimes
**after** (intel is stale → the defense wins → no compromise). The binary
``exploitation_success`` therefore takes intermediate proportions with real
variance instead of the degenerate 0/1 step function of a fixed phase. Cadence
stays a controlled parameter; the grading comes from the phase, not from tuning
cadence per run. The phase is a pure function of the seed, so the whole pipeline
stays reproducible and the offline path replays a live trial exactly.

**Distinct host ports (2026-10-31).** No state the orchestrator emits puts two services on
one host port, or any service on a reserved one (:data:`RESERVED_HOST_PORTS`: vsftpd's backdoor
port and the targets node's administrative SSH port). The techniques
draw each moved port independently (``mtd.techniques._new_port``), so a tick could do either;
the testbed cannot publish such a state (``render_override.py`` refuses it), and the next
attempt starts from this window's final state. The October 2026 main study lost a whole
sequence that way (see :func:`_distinct_host_ports`). The redraw that prevents it uses its own
sub-stream and runs only on a clash, so every schedule without one is unchanged.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import timedelta

from measurement import clock
from measurement.models import GroundTruthInventory, ReconfigEvent
from mtd.ground_truth import generate_ground_truth
from mtd.models import ConfigState
from mtd.techniques import _new_port, get_technique

# Dedicated sub-stream constant for the phase draw: derived from the trial seed but
# kept separate from the technique-mutation RNG so enabling the randomized phase
# never perturbs the mutation sequence for a given tick count (reproducibility).
_PHASE_SALT = 0x9E3779B9

# Dedicated sub-stream for the redraws that keep host ports distinct
# (:func:`_distinct_host_ports`), separate from the technique and phase streams for the same
# reason: it is drawn from only when a tick puts two services on one host port, so it never
# perturbs the mutation sequence and leaves every schedule without such a clash unchanged.
_PORT_SALT = 0x27D4EB2F

#: Host ports no service may be moved onto, because something else on the targets host holds
#: them and ``mtd.techniques._new_port`` draws from a range that contains them: vsftpd's backdoor
#: opens its root shell on 6200, a fixed auxiliary port the testbed publishes beside the
#: observable ones (``testbed/roster.yaml``, ``auxiliary_ports``), and the targets node's
#: administrative sshd listens on 2222, moved there so the SSH decoy can own 22
#: (``infra/ansible/site.yml``). ``tests/test_distinct_host_ports.py`` keeps this in step with both.
RESERVED_HOST_PORTS: frozenset[int] = frozenset({6200, 2222})

# Accepted values for the ``rotation_phase`` schedule mode.
ROTATION_PHASE_MODES = ("fixed", "randomized")


@dataclass(frozen=True)
class GroundTruthSnapshot:
    """A ground-truth inventory tied to the timestamp it took effect."""

    timestamp: str
    inventory: GroundTruthInventory


@dataclass(frozen=True)
class OrchestrationResult:
    """Everything the orchestrator produced for one trial window."""

    condition: str
    initial_state: ConfigState
    final_state: ConfigState
    events: tuple[ReconfigEvent, ...]
    snapshots: tuple[GroundTruthSnapshot, ...]

    def ground_truth_at(self, timestamp: str) -> GroundTruthInventory:
        """The inventory in force at ``timestamp`` (latest snapshot at or before it)."""

        target = clock.parse(timestamp)
        chosen = self.snapshots[0]
        for snap in self.snapshots:
            if clock.parse(snap.timestamp) <= target:
                chosen = snap
            else:
                break
        return chosen.inventory


def _fmt(dt) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def rotation_phase_offset(seed: int, cadence_s: int) -> float:
    """The seeded per-trial rotation phase ``φ`` ∈ ``[0, cadence)`` (Method C).

    Drawn from a dedicated sub-stream of ``seed`` so it is reproducible via the
    trial seed yet independent of the technique-mutation RNG. Returns ``0.0`` when
    ``cadence_s <= 0`` (no schedule).
    """

    if cadence_s <= 0:
        return 0.0
    return random.Random((seed ^ _PHASE_SALT) & 0xFFFFFFFF).uniform(0.0, float(cadence_s))


def _tick_offsets(cadence_s: int, horizon_s: int, phase_s: float) -> list[float]:
    """Reconfiguration tick offsets (seconds into the window), ascending.

    Grid ``φ + k*cadence`` for k = 0, 1, 2, … within ``(0, horizon]`` — ticks at or
    before the window start (offset ``<= 0``) are skipped (the initial snapshot
    already covers ``start``). With ``phase_s == 0`` this reproduces the legacy
    ``start + k*cadence`` (k ≥ 1) schedule exactly.
    """

    if cadence_s <= 0:
        return []
    offsets: list[float] = []
    k = 0
    while True:
        offset = phase_s + k * cadence_s
        k += 1
        if offset <= 0.0:
            continue
        if offset > horizon_s:
            break
        offsets.append(offset)
    return offsets


def _distinct_host_ports(
    previous: ConfigState, state: ConfigState, rng: random.Random
) -> ConfigState:
    """``state`` with no two services on one host port and none on a reserved one.

    ``mtd.techniques._new_port`` draws each moved service's port independently, so a tick can
    put two services on the same host port, or one on a port something else on the targets host
    holds (:data:`RESERVED_HOST_PORTS`). The testbed cannot publish that: ``render_override.py``
    refuses a duplicate host port, Docker cannot bind a port the host's sshd holds, the attempt's
    reconfiguration fails, and the whole sequence is excluded as an infrastructure fault (a new
    attempt would start from the same state). The October 2026 main study lost ``combined_mtd_deception-s37``
    that way, DVWA and the FTP decoy both on 62181 at the start of its seventh attempt, and the
    same seeds would lose it again. A real MTD controller would not assign one port twice.

    A service whose port this tick left as it was keeps it (``previous`` is already distinct,
    so those ports are). Each service whose port changed, in the state's order, keeps its new
    port unless a service already placed holds it or it is reserved; it then draws again from
    ``rng``, the same range the techniques use, until the port is free. ``rng`` is used only on
    a clash, and ``state`` itself is returned when there is none.
    """

    before = {s.service_id: s.port for s in previous.services}
    unmoved = [s.port for s in state.services if before.get(s.service_id) == s.port]
    taken = set(RESERVED_HOST_PORTS) | set(unmoved)
    services = []
    redrawn = False
    for s in state.services:
        if before.get(s.service_id) == s.port:
            services.append(s)
            continue
        port = s.port
        while port in taken:
            port = _new_port(rng)
            redrawn = True
        taken.add(port)
        services.append(s if port == s.port else s.with_attrs(port=port))
    return state.with_services(tuple(services)) if redrawn else state


def run_orchestration(
    condition: str,
    initial_state: ConfigState,
    start_ts: str,
    cadence_s: int,
    horizon_s: int,
    seed: int,
    rotation_phase: str = "fixed",
) -> OrchestrationResult:
    """Run the reconfiguration loop for one trial window.

    Ticks occur at ``start + φ + k*cadence`` while within ``(start, start +
    horizon]``. With the default ``rotation_phase="fixed"`` the phase ``φ`` is 0 and
    the grid reduces to ``start + k*cadence`` for k = 1, 2, … (the legacy schedule).
    With ``rotation_phase="randomized"`` a seeded per-trial ``φ`` ∈ ``[0, cadence)``
    shifts the grid so the reconfiguration lands on a randomized phase relative to
    the attacker's deterministic timeline (see the module docstring — this is what
    makes ``exploitation_success`` graded and varianced). The initial ground-truth
    snapshot is recorded at ``start_ts``; a new snapshot + event are recorded at each
    tick that changes the state. Every tick's state keeps host ports distinct and off
    :data:`RESERVED_HOST_PORTS` (:func:`_distinct_host_ports`).
    """

    if rotation_phase not in ROTATION_PHASE_MODES:
        raise ValueError(
            f"unknown rotation_phase {rotation_phase!r} (use one of {ROTATION_PHASE_MODES})"
        )

    technique = get_technique(condition)
    rng = random.Random(seed)
    port_rng = random.Random((seed ^ _PORT_SALT) & 0xFFFFFFFF)
    start = clock.parse(start_ts)

    phase_s = rotation_phase_offset(seed, cadence_s) if rotation_phase == "randomized" else 0.0

    state = initial_state
    events: list[ReconfigEvent] = []
    snapshots: list[GroundTruthSnapshot] = [
        GroundTruthSnapshot(start_ts, generate_ground_truth(state))
    ]

    for offset in _tick_offsets(cadence_s, horizon_s, phase_s):
        ts = _fmt(start + timedelta(seconds=offset))
        new_state, changed = technique(state, rng)
        if changed:
            state = _distinct_host_ports(state, new_state, port_rng)
            events.append(ReconfigEvent(timestamp=ts, technique=condition))
            snapshots.append(GroundTruthSnapshot(ts, generate_ground_truth(state)))

    return OrchestrationResult(
        condition=condition,
        initial_state=initial_state,
        final_state=state,
        events=tuple(events),
        snapshots=tuple(snapshots),
    )
