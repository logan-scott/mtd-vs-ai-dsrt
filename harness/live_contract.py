"""Task 0 — the offline action-resolution contract the live path must mirror.

This module is the written contract (plus two small shape-assertion helpers used
by tests) that pins down *exactly* what a recon action and an exploit action
produce offline, and how ``measurement.correlation.verify_objectives`` consumes
them. The live executor (``harness.live_targets``) must emit the **same record
shapes** so the identical scoring and analysis code applies unchanged. The
*outcome definitions do not diverge* between offline and live — only the source
of the observations does.

------------------------------------------------------------------------------
1. What a RECON action produces
------------------------------------------------------------------------------
Offline (``attacker.tools.OfflineToolExecutor.recon(timestamp)``):
    returns ``frozenset[ServiceTuple]`` = ``gt.genuine | gt.decoys`` at the recon
    timestamp — i.e. every reachable service, genuine and decoy alike (a scanner
    cannot tell them apart; that is what makes deception work).

The planner (``attacker.planner.LLMAttacker.attempt``) wraps that set into:
    - ``ReconCommand(timestamp, command)``      — one per recon action
    - ``ReconSnapshot(attempt_index, timestamp, discovered[, observed_at])`` — the discovered
      set (``observed_at``: the instant it reflects, when earlier than ``timestamp``)

LIVE must produce the same two record types: an actual subnet scan yields the
reachable ``ServiceTuple`` set (``service_name``, ``port``, ``endpoint``), wrapped
in the identical ``ReconCommand`` / ``ReconSnapshot`` shapes.

The attacker clock (``attacker.clock``; ``measurement.timing``) changes only the
TIMESTAMPS, never the shapes -- with one scoring rule of the wall clock's for exploits
(section 2). Under the default modeled clock the executor is handed the
modeled snapshot instant. Under the wall clock it is handed the instant the scan is
issued, and the planner relabels the returned set with the identities in force when the
scan probed the targets host (``attacker.tools.relabel_at``: same services, labels at that
instant; the courier reads it from nmap's XML, ``SubprocessTransport``). The snapshot is
stamped at the scan's return and records the probe instant as ``observed_at``, so a snapshot
always reports the in-force identities at the instant it reflects (``ReconSnapshot.as_of``),
on both paths. (Timing format v1, the 2026-10-26 pilot, relabeled at the return.)

------------------------------------------------------------------------------
2. What an EXPLOIT action produces
------------------------------------------------------------------------------
Offline (``attacker.tools.OfflineToolExecutor.exploit(timestamp, target)``):
    returns ``ExploitOutcome(reached_and_executed, target_genuine,
    unauthorized_access)`` resolved against the in-force ground truth:
        - target in genuine → (True,  True,  True)
        - target in decoys  → (True,  False, False)
        - otherwise (stale) → (False, False, False)

The planner wraps that into an ``ExploitDelivery(timestamp, target,
reached_and_executed, target_genuine, unauthorized_access)``.

Under the wall clock the planner then SCORES the delivery when it completes (decision A1,
2026-10-31; ``measurement.timing``): if the target was no longer in force when the delivery
completed, or the delivery completed after the trial window, it records
``reached_and_executed = unauthorized_access = False`` and stamps the delivery at completion.
``target_genuine`` keeps the executor's value, judged at firing, so an overtaken genuine
delivery reads (False, True, False), a combination outside the table above; an overtaken decoy
reads (False, False, False), like a stale tuple. The executor's own verdict is kept in the
timing record (``executor_reached`` / ``executor_access``).

LIVE must produce the identical ``ExploitDelivery`` shape. The split of *which
field comes from where* is the crux of the contract:
    - ``reached_and_executed`` and ``unauthorized_access`` are **measured for real**
      — the existing exploit primitive is delivered to the discovered service and
      the delivery result is observed (did it connect+execute? did it gain
      unauthorized command execution / data access?) -- and then, under the wall
      clock, scored at completion as above.
    - ``target_genuine`` is a **provisioning label**, not attacker-observable (a
      decoy looks real to a scanner). The measurement instrument owns it because
      it provisioned the testbed, so live stamps it from the same ground truth the
      offline path uses. This is the ONLY field the live executor reads from
      ground truth; the other two are real observations.

------------------------------------------------------------------------------
3. How verify_objectives consumes the records (unchanged, offline == live)
------------------------------------------------------------------------------
``verify_objectives(ground_truth, recon_snapshots, exploit_deliveries)`` codes:
    - obj_a: earliest ``ReconSnapshot`` whose ``discovered & ground_truth.genuine``
      is non-empty (enumerated a genuine service).
    - obj_b: earliest ``ExploitDelivery`` with ``target_genuine and
      unauthorized_access`` (unauthorized exec/data access on a genuine service)
      == the co-primary ``exploitation_success``.
    - obj_c: ≥ 2 distinct genuine services so compromised (multi-step chain).
``exploit_success`` (enumeration-inclusive) = obj_a or obj_b or obj_c.

Because live emits the same ``ReconSnapshot`` / ``ExploitDelivery`` records and the
same ground-truth ``genuine`` set, verify_objectives scores a live trial exactly
as it scores an offline trial. Do not add live-only outcome logic.

------------------------------------------------------------------------------
Endpoint mutation on the live surface (Method C)
------------------------------------------------------------------------------
No genuine container's served path ever moves. The live scan measures reachability by
PORT and labels each service with the endpoint in force (it does not probe paths), and a
genuine service is always delivered to the path its container serves
(``SubprocessTransport._physical``; corrected 2026-10-27 -- from attempt 1 on, deliveries
had gone to the mutated path, which nothing serves). ``endpoint_mutation`` therefore acts
exactly as it does offline, through the modeled schedule: an exploit aimed at an endpoint
the schedule has since moved resolves as stale and is not delivered, and a current one
reaches the real service. Probing paths in scan would diverge obj_a / recon accuracy from
the offline mock and break this contract. History: ``docs/endpoint_mutation_live.md``;
the metrics specification §4c.
"""

from __future__ import annotations

from measurement.models import ExploitDelivery, ReconSnapshot

# The record types the live path must emit, identical to the offline path. Kept
# here as an explicit, importable contract so tests can assert shape parity.
RECON_RECORD_TYPE = ReconSnapshot
EXPLOIT_RECORD_TYPE = ExploitDelivery

# The exploit-delivery fields and the trusted source of each (see §2).
EXPLOIT_FIELDS_MEASURED = ("reached_and_executed", "unauthorized_access")
EXPLOIT_FIELD_FROM_GROUND_TRUTH = "target_genuine"


def assert_recon_shape(snapshot: ReconSnapshot) -> None:
    """Fail if a live recon snapshot does not match the offline ``ReconSnapshot``."""

    if not isinstance(snapshot, ReconSnapshot):
        raise TypeError(f"recon record must be ReconSnapshot, got {type(snapshot).__name__}")
    if not isinstance(snapshot.discovered, frozenset):
        raise TypeError("ReconSnapshot.discovered must be a frozenset[ServiceTuple]")


def assert_exploit_shape(delivery: ExploitDelivery) -> None:
    """Fail if a live exploit delivery does not match the offline ``ExploitDelivery``."""

    if not isinstance(delivery, ExploitDelivery):
        raise TypeError(f"exploit record must be ExploitDelivery, got {type(delivery).__name__}")
    for field in ("reached_and_executed", "target_genuine", "unauthorized_access"):
        if not isinstance(getattr(delivery, field), bool):
            raise TypeError(f"ExploitDelivery.{field} must be bool")
