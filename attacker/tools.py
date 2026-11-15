"""Offline tool executor for the LLM attacker.

Bridges the attacker's abstract actions (scan, exploit) to the offline
``Environment`` (the mock target backed by the MTD orchestrator). These stand in
for the real scanning/exploitation tool adapters once the live testbed exists;
crucially they perform **no live network activity** — they resolve actions against
the ground-truth inventory in force at the action's timestamp.

A reconnaissance scan returns every reachable service (genuine and decoy look the
same to a scanner — that is what makes deception work). An exploit against a
genuine service succeeds (its vulnerability is real); against a decoy it reaches
but never compromises; against a stale tuple it cannot even connect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from attacker.base import Environment
from measurement.models import ServiceTuple


@dataclass(frozen=True)
class ExploitOutcome:
    """Verified outcome of one exploit delivery."""

    reached_and_executed: bool
    target_genuine: bool
    unauthorized_access: bool


@runtime_checkable
class TargetExecutor(Protocol):
    """Resolves attacker actions into observations (offline or live).

    The planner is agnostic to how actions are carried out: the ``OfflineToolExecutor``
    resolves them against the mock ground truth, while the live executor
    (``harness.live_targets.LiveTargetExecutor``) carries them to real testbed
    services. Both MUST return the same shapes so the identical measurement core
    scores either — see ``harness.live_contract``.
    """

    def recon(self, timestamp: str) -> frozenset[ServiceTuple]:
        """Discover reachable services as of ``timestamp``."""
        ...

    def exploit(self, timestamp: str, target: ServiceTuple) -> ExploitOutcome:
        """Attempt to exploit ``target`` and return the verified outcome."""
        ...


@dataclass(frozen=True)
class OfflineToolExecutor:
    """Executes attacker actions against the offline environment."""

    environment: Environment

    def recon(self, timestamp: str) -> frozenset[ServiceTuple]:
        """A scan returns all reachable services (genuine + decoys, indistinguishable)."""

        gt = self.environment.ground_truth_at(timestamp)
        return gt.genuine | gt.decoys

    def exploit(self, timestamp: str, target: ServiceTuple) -> ExploitOutcome:
        """Attempt to exploit ``target`` against the in-force ground truth."""

        gt = self.environment.ground_truth_at(timestamp)
        if target in gt.genuine:
            return ExploitOutcome(True, True, True)
        if target in gt.decoys:
            return ExploitOutcome(True, False, False)
        return ExploitOutcome(False, False, False)


# --------------------------------------------------------------------------- #
# Identity at an instant (the attacker clock's bookkeeping)
# --------------------------------------------------------------------------- #
#
# A service's ``service`` name is invariant across every MTD technique (techniques mutate only
# its address, port, endpoint -- see ``mtd.techniques``), and names are unique within any one
# in-force inventory, so "the same service at another instant" is well defined. These helpers
# are the instrument's clock model, not attacker knowledge: the planner uses them to label a
# real-time scan at the instant its results returned (``clock: wall``) and to score the paired
# staleness counterfactual (``measurement.timing``).


def identity_at(environment: Environment, service: str, timestamp: str) -> ServiceTuple | None:
    """The observable identity ``service`` has in force at ``timestamp`` (None if absent)."""

    gt = environment.ground_truth_at(timestamp)
    for tup in gt.genuine | gt.decoys:
        if tup.service == service:
            return tup
    return None


def relabel_at(
    discovered: frozenset[ServiceTuple], environment: Environment, timestamp: str
) -> frozenset[ServiceTuple]:
    """Relabel discovered services with the identities in force at ``timestamp``.

    Reachability is what a scan measures; the observable identity it reports is a label from
    the in-force configuration (``harness.live_targets`` Method C). Relabeling a scan taken at
    one instant to another instant keeps exactly the services it found and reports each under
    the identity in force at the new instant -- the same set ``executor.recon(timestamp)``
    returns for that instant, because the physical surface probed does not depend on the
    instant. A service with no identity in force at ``timestamp`` is dropped (cannot occur
    under the current techniques, which never add or remove a service).
    """

    gt = environment.ground_truth_at(timestamp)
    by_name = {tup.service: tup for tup in gt.genuine | gt.decoys}
    return frozenset(by_name[tup.service] for tup in discovered if tup.service in by_name)


def service_is_genuine(environment: Environment, service: str, timestamp: str) -> bool:
    """Whether ``service`` is a genuine (not decoy) service in the inventory at ``timestamp``.

    By NAME, so a genuine service whose stale tuple the attacker aimed at still counts as
    genuine -- unlike ``ExploitDelivery.target_genuine``, which is False for any tuple no longer
    in force.
    """

    gt = environment.ground_truth_at(timestamp)
    return any(tup.service == service for tup in gt.genuine)
