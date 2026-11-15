"""Live target executor: carries the attacker's actions to real testbed services.

``LiveTargetExecutor`` implements the ``attacker.tools.TargetExecutor`` interface
against the real testbed, emitting the *same* ``ReconSnapshot`` / ``ExploitDelivery``
records the offline executor emits (see ``harness.live_contract``) so the identical
measurement core scores it. It authors **no** offensive capability: recon is a
subnet scan and exploit is delivery of the instrument's *existing* primitive,
carried by an injectable ``TargetTransport``.

Two transports are provided:
  - ``MockTransport`` — resolves actions against the orchestrator's ground truth
    (no network), producing records identical to the offline path. Used by tests
    and ``--dry-run``.
  - ``SubprocessTransport`` — the real SSH/scan-backed courier. It runs standard
    tooling (``nmap``) and the instrument's *pre-staged* exploit primitives (staged
    onto the executor node by ``infra/ansible``; see ``testbed/roster.yaml``) from
    the no-egress executor node via an injectable ``RemoteRunner``. It authors no
    payload — it carries and parses.

Method C — the reconfiguration STATE is modeled, so both transports resolve
recon/deliver against the *same* modeled in-force inventory (the orchestrator's
scheduled ``ground_truth_at``), which is what restores offline↔live parity under
port rotation. The live testbed is provisioned once per attempt at the attempt's
``initial_state`` (the physical host is frozen for the window — Method A's
intra-window physical re-binding is out of scope), so ``SubprocessTransport``
measures real reachability against the frozen **physical** host + ports and
*relabels* each reachable physical service with its modeled observable identity
(address, port, endpoint). The modeled **address** is a label in exactly the way the
modeled port already is: every service is provisioned on the single ``targets_host``,
so the address the attacker observes moves under ``network_shuffling`` while delivery
keeps going to the frozen physical host. Staleness
(a recon-time tuple rotated out of the in-force inventory before delivery) is thus
scored deliver-side from the shared model — identically offline and live — while
the real primitive is still delivered to the real service at its physical port
(the validation value: real attacker, real services, real exploitation dynamics).

Field split (the contract): ``reached_and_executed`` and ``unauthorized_access`` are
measured by the transport for real; ``target_genuine`` is stamped from ground truth
(a decoy is indistinguishable to a scanner, so the measurement instrument — which
provisioned the testbed — owns that label).

Address split (safety): ``LiveService.ip`` is the modeled **observable** address and is
never connected to (offline addresses are 10.10.x.x, outside the live testbed CIDR);
``LiveService.delivery_host()`` is the address every probe and delivery actually uses,
and it is the ONLY address the ``assert_within_cidr`` allowlist is asserted against —
asserting the modeled label instead would abort every live run in every condition.

What the live scan measures vs. labels (the honest reading of the contract): the
scan *measures reachability* (which provisioned ports actually answer on the
targets host) and *labels* each reachable service with its canonical observable
identity from the in-force configuration (the instrument just provisioned those
ports/endpoints when it applied the condition — see ``AnsibleProvisioner``). The
genuine/decoy distinction is never used by the scan; both are discovered and named
identically, so a scanner still cannot tell them apart.
"""

from __future__ import annotations

import ipaddress
import json
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from attacker.tools import ExploitOutcome
from harness.config import SCAN_SCOPE_SUBNET, normalize_scan_scope
from harness.live_ssh import RemoteRunner
from harness.safety import LiveMeasurementError, assert_within_cidr
from measurement.models import ServiceTuple
from mtd.orchestrator import OrchestrationResult
from testbed.profile import default_profile
from testbed.roster import PrimitiveSpec

# First host offset used when laying services out within the testbed CIDR (matches
# the infra convention of assigning nodes at .10+).
_HOST_BASE = 10


@dataclass(frozen=True)
class LiveService:
    """A service on the testbed: its modeled observable identity plus where it answers.

    ``ip``/``port``/``endpoint`` are the **modeled observable** identity (what the
    attacker fingerprints and what parity is scored on — the modeled schedule may have
    shuffled the address, rotated the port or mutated the endpoint).
    ``physical_host``/``physical_port``/``physical_endpoint`` are where the real service
    actually answers on the frozen testbed host (each defaults to its observable
    counterpart when they coincide, e.g. the static baseline, decoys, or the offline
    mock); the real primitive is delivered there so it reaches the real service even when
    the observable identity has been modeled away from the physical one.

    Only the ``delivery_*`` accessors may be used to touch the network or to check the
    CIDR allowlist: ``ip`` is a modeled label (offline addresses are 10.10.0.x, outside
    the live testbed CIDR) and is never connected to.
    """

    ip: str
    port: int
    service_name: str
    endpoint: str = ""
    physical_port: int | None = None
    physical_endpoint: str | None = None
    physical_host: str | None = None

    def observable(self) -> ServiceTuple:
        """The attacker-observable identity: service–port–endpoint–address."""

        return ServiceTuple(self.service_name, self.port, self.endpoint, self.ip)

    def delivery_host(self) -> str:
        """The concrete host to deliver the primitive to (physical if set, else observable)."""

        return self.physical_host if self.physical_host is not None else self.ip

    def delivery_port(self) -> int:
        """The concrete port to deliver the primitive to (physical if set, else observable)."""

        return self.physical_port if self.physical_port is not None else self.port

    def delivery_endpoint(self) -> str:
        """The concrete path to deliver the primitive to (physical if set, else observable)."""

        return self.physical_endpoint if self.physical_endpoint is not None else self.endpoint


@dataclass(frozen=True)
class DeliveryResult:
    """What the transport observed when delivering the exploit primitive.

    Both fields are *measured* from the real delivery — the genuine/decoy label is
    NOT here (it is stamped from ground truth by the executor).
    """

    reached_and_executed: bool
    unauthorized_access: bool


@runtime_checkable
class TargetTransport(Protocol):
    """Carries recon/exploit to the testbed. The only place real network I/O lives."""

    def scan(self, testbed_cidr: str, timestamp: str) -> list[LiveService]:
        """Scan the testbed subnet and return the reachable services."""
        ...

    def resolve(self, target: ServiceTuple, timestamp: str) -> LiveService | None:
        """Address the exploit would be delivered to, or None if not present."""
        ...

    def deliver(self, service: LiveService, timestamp: str) -> DeliveryResult:
        """Deliver the existing exploit primitive and observe the outcome."""
        ...


@dataclass(frozen=True)
class MockTransport:
    """Offline transport backed by the orchestrator's ground truth (no network).

    Lays the in-force services out at deterministic in-CIDR addresses and resolves
    delivery outcomes from ground truth exactly as the offline executor does, so a
    trial run through ``LiveTargetExecutor`` + ``MockTransport`` scores identically
    to the offline path.
    """

    orchestration: OrchestrationResult
    testbed_cidr: str

    def _network(self):
        return ipaddress.ip_network(self.testbed_cidr, strict=False)

    def _ordered(self, timestamp: str) -> list[ServiceTuple]:
        inv = self.orchestration.ground_truth_at(timestamp)
        return sorted(inv.genuine | inv.decoys)

    def _service_at(self, index: int, target: ServiceTuple, network) -> LiveService:
        """Modeled observable identity + the deterministic in-CIDR host it is laid out on.

        The observable address comes from the modeled tuple (network shuffling moves it),
        exactly as the live courier reads it; the laid-out in-CIDR address is the
        *physical* host, so the executor's allowlist assertions check a real deliverable
        address here just as they do live.
        """

        return LiveService(
            ip=target.address,
            port=target.port,
            service_name=target.service,
            endpoint=target.endpoint,
            physical_host=str(network[_HOST_BASE + index]),
        )

    def scan(self, testbed_cidr: str, timestamp: str) -> list[LiveService]:
        network = ipaddress.ip_network(testbed_cidr, strict=False)
        return [self._service_at(i, t, network) for i, t in enumerate(self._ordered(timestamp))]

    def resolve(self, target: ServiceTuple, timestamp: str) -> LiveService | None:
        ordered = self._ordered(timestamp)
        if target not in ordered:
            return None  # not present in the in-force inventory (stale after MTD)
        return self._service_at(ordered.index(target), target, self._network())

    def deliver(self, service: LiveService, timestamp: str) -> DeliveryResult:
        inv = self.orchestration.ground_truth_at(timestamp)
        obs = service.observable()
        if obs in inv.genuine:
            return DeliveryResult(reached_and_executed=True, unauthorized_access=True)
        if obs in inv.decoys:
            return DeliveryResult(reached_and_executed=True, unauthorized_access=False)
        return DeliveryResult(reached_and_executed=False, unauthorized_access=False)


@dataclass
class LiveTargetExecutor:
    """Resolves attacker actions against the real testbed via a ``TargetTransport``.

    Enforces the CIDR allowlist on every address it would touch (fail closed) and
    stamps ``target_genuine`` from ground truth while measuring reach/unauthorized
    for real. Emits offline-identical record content.
    """

    orchestration: OrchestrationResult
    transport: TargetTransport
    testbed_cidr: str

    def recon(self, timestamp: str) -> frozenset[ServiceTuple]:
        services = self.transport.scan(self.testbed_cidr, timestamp)
        # Defense in depth: a scan must only ever surface services whose DELIVERY host is
        # in-CIDR (``.ip`` is the modeled observable label, which nothing connects to).
        assert_within_cidr([s.delivery_host() for s in services], self.testbed_cidr)
        return frozenset(s.observable() for s in services)

    @property
    def last_observed_offset_s(self) -> float | None:
        """Seconds after its issue at which the last scan probed the targets host.

        Reported by the real courier from nmap's per-host XML times; None for a transport
        that reports none (``MockTransport``). The wall clock dates the snapshot's
        identities to this instant (``attacker.planner``).
        """

        return getattr(self.transport, "last_observed_offset_s", None)

    def exploit(self, timestamp: str, target: ServiceTuple) -> ExploitOutcome:
        genuine = target in self.orchestration.ground_truth_at(timestamp).genuine
        service = self.transport.resolve(target, timestamp)
        if service is None:
            # Nothing to reach (MTD moved/removed it): unreachable, no access.
            return ExploitOutcome(
                reached_and_executed=False, target_genuine=genuine, unauthorized_access=False
            )
        # Fail closed: never deliver to an address outside the testbed allowlist. The
        # DELIVERY host is what the transport will connect to; ``.ip`` is the modeled
        # observable label and an in-CIDR-looking label must not launder an off-testbed
        # delivery.
        assert_within_cidr([service.delivery_host()], self.testbed_cidr)
        result = self.transport.deliver(service, timestamp)
        return ExploitOutcome(
            reached_and_executed=result.reached_and_executed,
            target_genuine=genuine,
            unauthorized_access=result.unauthorized_access,
        )


# nmap greppable "Host:" line, e.g. "Host: 172.31.66.10 () \tPorts: 80/open/tcp//http///"
_NMAP_HOST_RE = re.compile(r"^Host:\s+(\S+)\s")


def parse_nmap_greppable(text: str) -> dict[str, set[int]]:
    """Parse ``nmap -oG -`` output into ``host -> {open tcp ports}``.

    Pure and deterministic so the courier's discovery logic is unit-tested without
    a network. Only ports in the ``open`` state are returned; hosts with no open
    ports are omitted.
    """

    result: dict[str, set[int]] = {}
    for line in text.splitlines():
        if not line.startswith("Host:") or "Ports:" not in line:
            continue
        match = _NMAP_HOST_RE.match(line)
        if not match:
            continue
        host = match.group(1)
        ports_field = line.partition("Ports:")[2].split("\t")[0]
        open_ports: set[int] = set()
        for spec in ports_field.split(","):
            fields = spec.strip().split("/")
            if len(fields) >= 2 and fields[1] == "open":
                try:
                    open_ports.add(int(fields[0]))
                except ValueError:
                    continue
        if open_ports:
            result[host] = result.get(host, set()) | open_ports
    return result


@dataclass(frozen=True)
class NmapHostResult:
    """One host of an nmap XML report: its open TCP ports and when nmap scanned it.

    ``start_s``/``end_s`` are the host's ``starttime``/``endtime`` minus the run's ``start``
    (whole seconds, both from the scanning node's clock, so no cross-host clock sync is
    needed). None when the report omits them.
    """

    open_ports: frozenset[int]
    start_s: float | None = None
    end_s: float | None = None

    @property
    def observed_offset_s(self) -> float | None:
        """The midpoint of the host's scan window, seconds after the run started."""

        if self.start_s is None or self.end_s is None:
            return None
        return max(0.0, (self.start_s + self.end_s) / 2.0)


def parse_nmap_xml(text: str) -> dict[str, NmapHostResult] | None:
    """Parse ``nmap -oX -`` output into ``host -> NmapHostResult``.

    Returns None when ``text`` is not an nmap XML report (the courier then treats the scan as
    failed; there is no greppable fallback). Pure and deterministic, like
    :func:`parse_nmap_greppable`: only ports in
    the ``open`` state are returned. The report comes from the instrument's own scanner on the
    no-egress executor node (trusted input), so the stdlib parser is adequate.
    """

    body = text.strip()
    if not body.startswith("<"):
        return None
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return None
    if root.tag != "nmaprun":
        return None
    try:
        run_start = float(root.get("start", ""))
    except ValueError:
        run_start = None
    out: dict[str, NmapHostResult] = {}
    for host in root.iter("host"):
        addr = None
        for a in host.findall("address"):
            if a.get("addrtype", "ipv4") == "ipv4":
                addr = a.get("addr")
                break
        if not addr:
            continue
        ports: set[int] = set()
        for port in host.iter("port"):
            state = port.find("state")
            if port.get("protocol", "tcp") != "tcp" or state is None:
                continue
            if state.get("state") == "open":
                try:
                    ports.add(int(port.get("portid", "")))
                except ValueError:
                    continue

        def _offset(attr: str, host=host) -> float | None:
            raw = host.get(attr)
            if raw is None or run_start is None:
                return None
            try:
                return float(raw) - run_start
            except ValueError:
                return None

        out[addr] = NmapHostResult(
            open_ports=frozenset(ports), start_s=_offset("starttime"), end_s=_offset("endtime")
        )
    return out


def parse_nmap_elapsed(text: str) -> float | None:
    """nmap's own run time (``runstats/finished@elapsed``, seconds) from an XML report.

    None when ``text`` is not a complete nmap XML report (a report cut off by a timeout has no
    ``finished`` element).
    """

    body = text.strip()
    if not body.startswith("<"):
        return None
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return None
    finished = root.find("runstats/finished")
    if root.tag != "nmaprun" or finished is None:
        return None
    try:
        return float(finished.get("elapsed", ""))
    except ValueError:
        return None


def nmap_greppable_complete(text: str) -> bool:
    """True when ``nmap -oG -`` output ran to completion (it ends with a ``# Nmap done`` line)."""

    return any(line.startswith("# Nmap done") for line in text.splitlines())


def parse_primitive_result(stdout: str) -> DeliveryResult | None:
    """Parse a pre-staged primitive's structured JSON line into a ``DeliveryResult``.

    Each primitive emits one JSON object ``{"reached_and_executed": bool,
    "unauthorized_access": bool}`` (last such line wins). Returns ``None`` if no
    valid result line is present — a measurement failure the courier fails closed on
    (it will not claim reach/access it could not substantiate).
    """

    for line in reversed(stdout.strip().splitlines()):
        candidate = line.strip()
        if not candidate.startswith("{"):
            continue
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if "reached_and_executed" in obj and "unauthorized_access" in obj:
            return DeliveryResult(
                reached_and_executed=bool(obj["reached_and_executed"]),
                unauthorized_access=bool(obj["unauthorized_access"]),
            )
    return None


def served_endpoints() -> dict[str, str]:
    """The path each GENUINE service's container actually serves: its canonical profile endpoint.

    Endpoints are never realized physically for genuine services: ``render_override.py``
    republishes PORTS, and each container serves the paths of its image (DVWA ``/login.php``,
    Struts ``/struts2/index.action``, ...). Only the deception responder serves rendered paths,
    for the decoys.
    """

    return {s.service_name: s.endpoint for s in default_profile().services if s.genuine}


@dataclass
class SubprocessTransport:
    """Real SSH/scan-backed courier, executed from the no-egress executor node.

    All network I/O goes through an injectable ``RemoteRunner`` (the single seam in
    ``harness.live_ssh``), so this class is fully unit-tested offline with a fake
    runner and driven against the real testbed only under ``MTD_TESTBED=1``.

    Method C — the reconfiguration state is modeled (the physical host is frozen at
    the attempt's ``initial_state``; intra-window physical re-binding is out of
    scope), so recon/deliver resolve against the *modeled* in-force inventory and the
    real network I/O measures the frozen **physical** surface:

    - ``scan`` runs ``nmap`` against the frozen physical ports on ``targets_host``,
      then relabels each physically-reachable service with its **modeled** observable
      identity (rotated port/endpoint) from the in-force configuration. This measures
      real reachability yet reports the modeled identity, so the discovered set is
      parity-identical to the offline mock at any timestamp. The genuine/decoy label
      is never consulted.
    - ``resolve`` maps a modeled ``ServiceTuple`` to its concrete address + physical
      delivery port from the most recent scan cache (``None`` if it was not
      discovered, or has since rotated out of the in-force inventory — stale after a
      tick; deliver-side staleness, identical offline and live).
    - ``deliver`` invokes the service's **pre-staged** primitive against the physical
      port (``testbed/roster.yaml``, staged by ``infra/ansible``) and parses its
      structured result. A service with no genuine primitive (a decoy or unknown
      fingerprint) gets a connect probe only: it can be reached, never accessed.

    No exploit payload is authored here; this is a courier. Every address it would
    touch is checked against the CIDR allowlist first (fail closed).
    """

    orchestration: OrchestrationResult
    runner: RemoteRunner
    executor_node: str
    testbed_cidr: str
    targets_host: str
    primitives: Mapping[str, PrimitiveSpec]
    ssh_user: str = "labadmin"
    scan_timeout_s: float = 300.0
    deliver_timeout_s: float = 120.0
    probe_timeout_s: float = 30.0
    # What the scan targets (``SCAN_SCOPES``): the whole testbed CIDR, or the targets host only.
    scan_scope: str = SCAN_SCOPE_SUBNET
    # ``live.surface_check: strict``: a scan that cannot reach every in-force service is an
    # infrastructure fault (under Method C they are all physically up), not an observation.
    require_full_surface: bool = False
    # One retry after a failed scan or connect probe, this long after it (transient SSH faults).
    # A delivery is never retried: an exploit has side effects on the target and the executor.
    retry_delay_s: float = 2.0
    # Wrap every executor command in a remote ``timeout`` that fires before the local one, so a
    # command the local timeout gives up on is dead on the executor too (``_remote``). Off by
    # default so the offline fakes see the bare commands; the live factory turns it on.
    remote_timeouts: bool = False
    clock: Callable[[], float] = field(default=time.monotonic, repr=False, compare=False)
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False, compare=False)
    # service_name -> the path a genuine service's container serves (``served_endpoints``).
    served_paths: Mapping[str, str] = field(default_factory=served_endpoints, repr=False)
    _cache: dict[ServiceTuple, LiveService] = field(default_factory=dict, init=False, repr=False)
    # Seconds after the scan call began at which the last scan probed the targets host; None
    # before any scan or when the report omits the times. Anchored at the scan's RETURN with
    # nmap's own elapsed time, so the SSH connect and nmap start-up before the probe are not
    # counted as time the target was already being observed (``scan``).
    last_observed_offset_s: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.scan_scope = normalize_scan_scope(self.scan_scope)

    def _physical(self, service_name: str, default_port: int, default_endpoint: str):
        """Where a service physically answers during this attempt: (port, endpoint).

        **Port:** the attempt's ``initial_state`` port. The host is re-provisioned at every
        attempt and the override republishes every service's ``initial_state`` port, so a port
        rotated during attempt N is physically real from attempt N+1 (metrics spec §4c); the
        modeled schedule's intra-attempt rotations are not (Method C).

        **Endpoint:** for a GENUINE service, the path its container serves (``served_paths``).
        Genuine paths are never realized physically, so the threaded ``initial_state`` endpoint
        -- mutated since attempt 0 under endpoint_mutation and combined_mtd_deception -- is a
        path nothing serves; delivering there failed every path-dependent primitive (Struts)
        from attempt 1 on, whether or not the attacker's information was current (corrected
        2026-10-27). For a decoy, the ``initial_state`` endpoint: the responder serves the
        rendered one. Staleness is decided by the model before delivery (``resolve``), so a
        current target is delivered where the vulnerable service really is.

        Mapped by the stable ``service_name`` (invariant across mutations). Falls back to the
        modeled values for a fingerprint not in the initial state.
        """

        for s in self.orchestration.initial_state.services:
            if s.service_name == service_name:
                if s.genuine:
                    return s.port, self.served_paths.get(service_name, s.endpoint)
                return s.port, s.endpoint
        return default_port, default_endpoint

    def _live_service(self, target: ServiceTuple) -> LiveService:
        """Build the concrete LiveService for a modeled tuple: observable identity from
        the tuple (address included — network shuffling moves it), physical delivery
        target from the frozen initial state on the provisioned targets host.

        ``_physical`` looks up port/endpoint per service; there is no physical HOST to
        look up — every service is provisioned on the single ``targets_host``.
        """

        phys_port, phys_endpoint = self._physical(target.service, target.port, target.endpoint)
        return LiveService(
            ip=target.address,
            port=target.port,
            service_name=target.service,
            endpoint=target.endpoint,
            physical_port=phys_port,
            physical_endpoint=phys_endpoint,
            physical_host=self.targets_host,
        )

    def _remote(self, argv: list[str], timeout_s: float) -> list[str]:
        """``argv`` as the executor runs it: under a remote ``timeout`` when ``remote_timeouts``.

        The local runner's timeout kills only the local ``ssh``; the remote command (no PTY)
        would keep running -- a timed-out Metasploit primitive still holding its handler port
        would then make the next vsftpd delivery fail as if the defense had stopped it. GNU
        ``timeout`` signals the command's whole process group 10 s before the local limit and
        kills it 5 s later, so nothing outlives the call.
        """

        if not self.remote_timeouts:
            return list(argv)
        limit = max(1, int(timeout_s) - 10)
        return ["timeout", "-k", "5", str(limit), *argv]

    def _candidates(self, timestamp: str) -> list[ServiceTuple]:
        inv = self.orchestration.ground_truth_at(timestamp)
        return sorted(inv.genuine | inv.decoys)

    def scan(self, testbed_cidr: str, timestamp: str) -> list[LiveService]:
        # Fail closed: the provisioning host must be inside the allowlist we scan.
        assert_within_cidr([self.targets_host], testbed_cidr)
        self.last_observed_offset_s = None
        candidates = self._candidates(timestamp)
        if not candidates:
            self._cache = {}
            return []

        # Probe the frozen PHYSICAL ports (the host answers on its initial-state
        # ports); the modeled schedule may have rotated the observable ports away.
        built = {t: self._live_service(t) for t in candidates}
        wanted = {s.delivery_port() for s in built.values()}
        ports = ",".join(str(p) for p in sorted(wanted))
        # XML output: besides the open ports it gives each host's scan window, which dates
        # what the scan saw (the wall clock labels the snapshot at that instant).
        target = testbed_cidr if self.scan_scope == SCAN_SCOPE_SUBNET else self.targets_host
        argv = self._remote(
            ["nmap", "-Pn", "-n", "-oX", "-", "-p", ports, target], self.scan_timeout_s
        )
        # A scan that produced no usable report is NOT an observation of an empty testbed: it is
        # retried once and then raised as an infrastructure fault, so an SSH drop or a timeout
        # can never be scored as the defense hiding every service.
        began = self.clock()
        problem = ""
        for attempt in range(2):
            if attempt:
                self.sleep(self.retry_delay_s)
            tried = self.clock()
            result = self.runner.run(self.executor_node, argv, timeout=self.scan_timeout_s)
            returned = self.clock()
            hosts = parse_nmap_xml(result.stdout)
            elapsed = parse_nmap_elapsed(result.stdout)
            host = None if hosts is None else hosts.get(self.targets_host)
            if hosts is None or elapsed is None:
                problem = f"no complete nmap report (rc={result.returncode}{_stderr_tail(result)})"
                continue
            if host is None:
                problem = f"the report has no entry for the targets host {self.targets_host}"
                continue
            missing = sorted(wanted - set(host.open_ports))
            if missing and self.require_full_surface:
                problem = f"in-force service ports {missing} did not answer on {self.targets_host}"
                continue
            break
        else:
            raise LiveMeasurementError(f"live scan failed twice: {problem}")
        open_ports = set(host.open_ports)
        offset = host.observed_offset_s  # seconds after nmap's own start
        if offset is not None:
            # nmap started (return - elapsed) on this clock: the SSH connect and nmap's start-up
            # precede it, and anchoring at the return keeps them out of the probe instant.
            nmap_started = max(tried, returned - elapsed) - began
            self.last_observed_offset_s = round(
                min(max(0.0, nmap_started + offset), returned - began), 3
            )

        services: list[LiveService] = []
        cache: dict[ServiceTuple, LiveService] = {}
        for tup in candidates:
            svc = built[tup]
            # Reachable iff its physical port answers; the identity is already modeled.
            if svc.delivery_port() in open_ports:
                services.append(svc)
                cache[tup] = svc
        # Defense in depth: a scan must only ever surface services whose DELIVERY host is
        # in-CIDR (``.ip`` is the modeled observable label, which nothing connects to).
        assert_within_cidr([s.delivery_host() for s in services], testbed_cidr)
        self._cache = cache
        return services

    def resolve(self, target: ServiceTuple, timestamp: str) -> LiveService | None:
        # Parity with MockTransport/offline (both stateless): a tuple that has mutated out
        # of the in-force inventory is unreachable (stale after a tick); anything in force
        # resolves, whether or not it was in the last scan cache — so an exploit issued
        # without a covering scan (e.g. from retained cross-attempt knowledge) scores the
        # same live as offline instead of falsely returning unreachable.
        inv = self.orchestration.ground_truth_at(timestamp)
        if target not in inv.genuine and target not in inv.decoys:
            return None
        # Prefer the cached LiveService (carries the measured physical mapping); fall back
        # to resolving the same mapping from the model for an un-scanned in-force target.
        return self._cache.get(target) or self._live_service(target)

    def deliver(self, service: LiveService, timestamp: str) -> DeliveryResult:
        # Fail closed: never deliver to an address outside the testbed allowlist. This is
        # the guard that stops a LiveService fabricated with an out-of-testbed PHYSICAL
        # host, and it is also the guard ``_reachable``'s nmap probe sits behind.
        assert_within_cidr([service.delivery_host()], self.testbed_cidr)
        primitive = self.primitives.get(service.service_name)
        if primitive is None:
            # No genuine primitive staged for this fingerprint (a decoy or unknown
            # service): a connect probe can only reach it, never gain access.
            return DeliveryResult(
                reached_and_executed=self._reachable(service), unauthorized_access=False
            )
        # Deliver to the physical host + port + path (where the frozen service answers).
        argv = self._remote(
            primitive.render(
                service.delivery_host(), service.delivery_port(), service.delivery_endpoint()
            ),
            self.deliver_timeout_s,
        )
        # Every primitive prints a verdict on every path, so a missing verdict means the
        # primitive never completed (SSH failure, timeout, killed). That is an infrastructure
        # fault, never a failed exploit. It is NOT retried: an exploit has side effects (the
        # vsftpd backdoor arms a listener, Metasploit binds a handler port), so a second run
        # could fail because of the first and print a verdict that blames the defense.
        result = self.runner.run(self.executor_node, argv, timeout=self.deliver_timeout_s)
        parsed = parse_primitive_result(result.stdout)
        if parsed is not None:
            return parsed
        what = (
            f"timed out after {self.deliver_timeout_s:g} s"
            if result.returncode in (124, 137)
            else "returned no verdict"
        )
        raise LiveMeasurementError(
            f"the {service.service_name} primitive {what} "
            f"(rc={result.returncode}{_stderr_tail(result)})"
        )

    def _reachable(self, service: LiveService) -> bool:
        # Probe the PHYSICAL host: ``service.ip`` is the modeled 10.10.x.x label, and a
        # probe aimed there fires a real scan off the testbed (the ``deliver`` allowlist
        # assertion above checks ``delivery_host()``, so it would NOT catch that).
        host = service.delivery_host()
        port = service.delivery_port()
        argv = self._remote(
            ["nmap", "-Pn", "-n", "-oG", "-", "-p", str(port), host], self.probe_timeout_s
        )
        problem = ""
        for attempt in range(2):
            if attempt:
                self.sleep(self.retry_delay_s)
            result = self.runner.run(self.executor_node, argv, timeout=self.probe_timeout_s)
            if nmap_greppable_complete(result.stdout):
                return port in parse_nmap_greppable(result.stdout).get(host, set())
            problem = f"rc={result.returncode}{_stderr_tail(result)}"
        raise LiveMeasurementError(
            f"connect probe of {service.service_name} failed twice ({problem})"
        )


def _stderr_tail(result) -> str:
    """A short, safe excerpt of a failed command's stderr for the exclusion message."""

    err = (getattr(result, "stderr", "") or "").strip()
    return f", stderr: {err[-160:]}" if err else ""
