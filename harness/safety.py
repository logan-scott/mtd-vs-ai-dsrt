"""Fail-closed safety guards for live testbed runs.

Two independent controls, both of which must hold before any offensive action
touches the network:

1. **CIDR allowlist** — live mode refuses to start unless a ``testbed_cidr`` is
   configured, and every target address must resolve *inside* that CIDR. Anything
   outside aborts the run. This is what keeps offensive actions off production.
2. **Egress split** — the planner (Anthropic API calls) runs on the control host;
   the offensive executor runs from a no-egress testbed node. The control host is
   **dual-homed by design** (a testbed leg on ``vmbr9`` + a LAN leg for the model
   API), so it *does* have a route into ``testbed_cidr`` — that is expected and not
   a violation. The security property is that the executor and every other
   non-control testbed node have **no route off the testbed**, and that the control
   host **does not forward** the testbed to the internet (``net.ipv4.ip_forward``
   must be ``0``). The preflight asserts this via an injectable ``RouteChecker``.
   Modeled as data (``EgressMap``) + assertions so the split is structural, not an
   honor system.

The pure pieces (``EgressMap``, ``assert_egress_split``) are unit-tested with a fake
``RouteChecker``; the real ``SshRouteChecker`` (active probing over SSH) is selected
only under ``--live``. ``NullRouteChecker`` stays the offline default.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from harness.live_ssh import RemoteRunner

# The model API host the planner (control host) is allowed to reach, and which the
# executor node must NOT be able to reach.
ANTHROPIC_API_HOST = "api.anthropic.com"


class SafetyError(RuntimeError):
    """Base class for fail-closed safety violations (aborts the run)."""


class ConfigurationError(SafetyError):
    """A required safety setting (e.g. testbed_cidr) is missing."""


class CidrViolation(SafetyError):
    """A target address resolved outside the testbed CIDR allowlist."""


class EgressViolation(SafetyError):
    """The egress split is violated (executor can reach off-testbed, etc.)."""


class ProvisioningError(SafetyError):
    """A live reconfiguration (Ansible) failed, so the testbed state is untrusted."""


class LiveMeasurementError(ProvisioningError):
    """A live scan or delivery produced no usable measurement, so the trial cannot be scored.

    Raised when the executor's command failed (SSH error, timeout) or its output was not a
    complete report or verdict, or (under ``live.surface_check: strict``) an in-force service
    did not answer. A scan or connect probe is retried once first; a delivery is not, because
    an exploit has side effects. A subclass of ``ProvisioningError``, so the experiment runner
    excludes the sequence as an ``infrastructure_fault`` instead of scoring the empty result as
    an attacker failure -- which would credit the defense with an outage.
    """


def parse_cidr(testbed_cidr: str | None) -> ipaddress.IPv4Network | ipaddress.IPv6Network:
    """Parse and require the testbed CIDR; raise ConfigurationError if unset/blank.

    Live mode fails closed: without an allowlist there is nothing constraining
    targets to the isolated subnet, so we refuse to proceed.
    """

    if not testbed_cidr or not str(testbed_cidr).strip():
        raise ConfigurationError(
            "live mode requires a testbed_cidr allowlist; refusing to start without one"
        )
    try:
        return ipaddress.ip_network(str(testbed_cidr).strip(), strict=False)
    except ValueError as exc:  # malformed CIDR
        raise ConfigurationError(f"invalid testbed_cidr {testbed_cidr!r}: {exc}") from exc


def is_within_cidr(address: str, testbed_cidr: str) -> bool:
    """True iff ``address`` is a host address inside ``testbed_cidr``."""

    network = parse_cidr(testbed_cidr)
    try:
        return ipaddress.ip_address(address) in network
    except ValueError:
        return False


def assert_within_cidr(addresses: Iterable[str], testbed_cidr: str) -> None:
    """Assert every address is inside the testbed CIDR; raise CidrViolation otherwise."""

    network = parse_cidr(testbed_cidr)
    outside = []
    for address in addresses:
        try:
            if ipaddress.ip_address(address) not in network:
                outside.append(address)
        except ValueError:
            outside.append(address)
    if outside:
        raise CidrViolation(
            f"target(s) outside testbed CIDR {testbed_cidr}: {', '.join(map(str, outside))}"
        )


@runtime_checkable
class RouteChecker(Protocol):
    """Probes reachability and forwarding state, so the egress preflight can enforce.

    Injected so the preflight is testable offline. The real implementation
    (``SshRouteChecker``) actively probes from each node over SSH; tests use a fake
    with fixed maps.
    """

    def can_reach(self, source: str, dest: str) -> bool:
        """True iff ``source`` has a network route that reaches ``dest``."""
        ...

    def ip_forwarding_enabled(self, host: str) -> bool:
        """True iff ``host`` has IPv4 forwarding enabled (``net.ipv4.ip_forward``)."""
        ...


@dataclass(frozen=True)
class EgressMap:
    """The intended egress topology asserted before a live run.

    ``control_host`` runs the planner + harness and is **dual-homed** (it reaches the
    model API *and* the testbed subnet), but must not *forward* the testbed to the
    internet. ``executor_node`` runs the offensive actions; ``testbed_nodes`` is every
    non-control node (executor included) — each must have no route off the testbed. If
    ``testbed_nodes`` is left empty it defaults to just the executor node.
    """

    control_host: str
    executor_node: str
    testbed_cidr: str
    api_host: str = ANTHROPIC_API_HOST
    testbed_nodes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.testbed_nodes:
            object.__setattr__(self, "testbed_nodes", (self.executor_node,))

    def as_dict(self) -> dict:
        return {
            "control_host": {
                "role": "planner (LLM) + harness — dual-homed (API leg + testbed leg)",
                "may_reach": [self.api_host, self.testbed_cidr],
                "must_not_reach": ["internet via forwarding (net.ipv4.ip_forward must be 0)"],
            },
            "executor_node": {
                "role": f"offensive executor (of {len(self.testbed_nodes)} no-egress nodes)",
                "may_reach": [self.testbed_cidr],
                "must_not_reach": [self.api_host, "* (any off-testbed host)"],
            },
        }


# A couple of off-testbed probe destinations no testbed node must be able to reach.
_OFF_TESTBED_PROBES = (ANTHROPIC_API_HOST, "1.1.1.1")


def assert_egress_split(egress: EgressMap, checker: RouteChecker) -> None:
    """Assert the dual-homed egress split holds; raise EgressViolation otherwise.

    Fail-closed checks for the built topology (see the module docstring):
      - every non-control testbed node (executor included) has NO route off the
        testbed — it cannot reach the model API or any generic internet host;
      - the planner (control host) reaches the model API (it needs it); and
      - the control host does NOT forward the testbed to the internet
        (``net.ipv4.ip_forward`` must be ``0``).

    The control host *does* have a route into ``testbed_cidr`` by design (it is
    dual-homed), so that is deliberately not asserted against — the security
    property is executor/target egress-lessness plus no forwarding.
    """

    # 1. No non-control testbed node has any egress off the testbed.
    for node in egress.testbed_nodes:
        for dest in _OFF_TESTBED_PROBES:
            if checker.can_reach(node, dest):
                raise EgressViolation(
                    f"testbed node {node} can reach off-testbed host {dest} "
                    "— every non-control node must have no route off the isolated subnet"
                )

    # 2. Planner reaches the model API (it needs it).
    if not checker.can_reach(egress.control_host, egress.api_host):
        raise EgressViolation(
            f"planner host {egress.control_host} cannot reach {egress.api_host} "
            "— the LLM planner needs the model API"
        )

    # 3. Control host must not forward the testbed to the internet.
    if checker.ip_forwarding_enabled(egress.control_host):
        raise EgressViolation(
            f"control host {egress.control_host} has IP forwarding enabled — it must "
            "not route the testbed to the internet (net.ipv4.ip_forward must be 0)"
        )


def _tcp_probe_argv(dest: str, port: int, timeout_s: float) -> list[str]:
    """A bounded TCP-connect probe: exits 0 iff ``dest:port`` accepts a connection."""

    seconds = max(1, int(timeout_s))
    return ["timeout", str(seconds), "bash", "-c", f"exec 3<>/dev/tcp/{dest}/{port}"]


@dataclass(frozen=True)
class SshRouteChecker:
    """Active-probing route checker: runs bounded probes on each node via SSH.

    Selected only under ``--live``. Uses the shared ``RemoteRunner`` seam, so it is
    the only network I/O it performs and it is unit-testable with a fake runner.

    - ``can_reach(source, dest)`` runs a bounded TCP connect from ``source`` to
      ``dest:443`` and reports whether it succeeded.
    - ``ip_forwarding_enabled(host)`` reads ``/proc/sys/net/ipv4/ip_forward``.
    - ``has_default_route(host)`` reports whether the node has an IPv4 default route.
    - ``resolves_external_dns(host)`` reports whether the node can still resolve an
      external name (a residual provisioning DNS leak) — must be false when sealed.

    The last two are read-only probes used by the standalone isolation preflight
    (``harness.preflight``); none of these probes need root.
    """

    runner: RemoteRunner
    probe_port: int = 443
    connect_timeout_s: float = 5.0
    read_timeout_s: float = 10.0
    dns_check_host: str = "archive.ubuntu.com"

    def can_reach(self, source: str, dest: str) -> bool:
        argv = _tcp_probe_argv(dest, self.probe_port, self.connect_timeout_s)
        result = self.runner.run(source, argv, timeout=self.connect_timeout_s + 5.0)
        return result.returncode == 0

    def ip_forwarding_enabled(self, host: str) -> bool:
        result = self.runner.run(
            host, ["cat", "/proc/sys/net/ipv4/ip_forward"], timeout=self.read_timeout_s
        )
        return result.stdout.strip() == "1"

    def has_default_route(self, host: str) -> bool:
        result = self.runner.run(
            host, ["ip", "-4", "route", "show", "default"], timeout=self.read_timeout_s
        )
        return bool(result.stdout.strip())

    def resolves_external_dns(self, host: str) -> bool:
        result = self.runner.run(
            host, ["getent", "hosts", self.dns_check_host], timeout=self.read_timeout_s
        )
        return result.returncode == 0


class NullRouteChecker:
    """Offline-default checker: refuses to answer (no live probing without deployment).

    Kept as the default so nothing probes the network unless the real
    ``SshRouteChecker`` is explicitly selected under ``--live``.
    """

    def can_reach(self, source: str, dest: str) -> bool:
        raise NotImplementedError(
            "live route checking is staged: select SshRouteChecker under --live"
        )

    def ip_forwarding_enabled(self, host: str) -> bool:
        raise NotImplementedError(
            "live route checking is staged: select SshRouteChecker under --live"
        )
