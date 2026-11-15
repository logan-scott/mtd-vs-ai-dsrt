"""Live-mode wiring: env/executor factories, provisioning, preflight, dry-run.

Offline stays the default everywhere. This module assembles the pieces the harness
needs to run against the real testbed when live mode is opted into:

  - the **env factory** the trial runner uses to build a ``LiveEnvironment`` per
    attempt — it also drives live MTD by applying each attempt's orchestration to
    the testbed via a ``TestbedProvisioner`` (same ``mtd.orchestrator`` output that
    the offline path replays);
  - the **executor factory** the planner uses to obtain the live executor;
  - the fail-closed **preflight** (CIDR set + egress split asserted); and
  - a **dry-run plan** that exercises the whole path with the mock executor and no
    network, returning the plan (conditions, trials, targets, egress map).

The real transport/provisioner/route-checker are clearly-marked scaffolds; the mock
ones make the entire path testable and dry-runnable offline.
"""

from __future__ import annotations

import ipaddress
import json
import re
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from attacker.base import Environment
from attacker.tools import TargetExecutor
from harness.config import SURFACE_CHECK_STRICT, ExperimentConfig, LiveSettings
from harness.live_environment import LiveEnvironment
from harness.live_ssh import RemoteRunner
from harness.live_targets import (
    LiveTargetExecutor,
    MockTransport,
    SubprocessTransport,
    TargetTransport,
    nmap_greppable_complete,
    parse_nmap_greppable,
)
from harness.safety import (
    ANTHROPIC_API_HOST,
    EgressMap,
    ProvisioningError,
    RouteChecker,
    assert_egress_split,
    assert_within_cidr,
    parse_cidr,
)
from mtd.checksums import state_checksum
from mtd.orchestrator import OrchestrationResult
from testbed.profile import default_profile
from testbed.roster import PrimitiveSpec

EnvFactory = Callable[[OrchestrationResult], Environment]
TransportFactory = Callable[[OrchestrationResult], TargetTransport]


# --------------------------------------------------------------------------- #
# Deployment-fact resolution (derive the infra convention from testbed_cidr)
# --------------------------------------------------------------------------- #


def resolve_targets_host(settings: LiveSettings) -> str:
    """The host the genuine + decoy services are provisioned on (.10 by convention)."""

    if settings.targets_host:
        return settings.targets_host
    return str(ipaddress.ip_network(settings.testbed_cidr, strict=False)[10])


def resolve_testbed_nodes(settings: LiveSettings) -> tuple[str, ...]:
    """Every non-control node the egress preflight probes (.10/.20/.30/.40 by convention)."""

    if settings.testbed_nodes:
        return tuple(settings.testbed_nodes)
    network = ipaddress.ip_network(settings.testbed_cidr, strict=False)
    return tuple(str(network[octet]) for octet in (10, 20, 30, 40))


def resolve_node_ssh_ports(settings: LiveSettings) -> dict[str, int]:
    """Per-host admin SSH ports; defaults the targets node to :2222 (SSH decoy owns :22)."""

    if settings.node_ssh_ports:
        return {str(h): int(p) for h, p in settings.node_ssh_ports}
    return {resolve_targets_host(settings): 2222}


def resolve_control_testbed_ip(settings: LiveSettings) -> str:
    """The control host's address on the isolated subnet (.5 by infra convention).

    This is the temporary default gateway the nodes point at during provisioning
    egress — never during a trial (the deploy flow tears it down and re-seals).
    """

    return str(ipaddress.ip_network(settings.testbed_cidr, strict=False)[5])


def build_remote_runner(settings: LiveSettings):
    """The one place a live ``SubprocessRemoteRunner`` is constructed from settings.

    Treats the control host as local (its own probes/Ansible run locally) and reaches
    every other node over SSH on its admin port.
    """

    from harness.live_ssh import SubprocessRemoteRunner

    return SubprocessRemoteRunner(
        ssh_user=settings.ssh_user,
        local_host=settings.control_host,
        host_ports=resolve_node_ssh_ports(settings),
    )


# --------------------------------------------------------------------------- #
# Provisioning (drive live MTD from the orchestrator)
# --------------------------------------------------------------------------- #

# The conditions whose technique rotates DECOY observables, i.e. the only two whose
# deceptive surface has to be re-published on the wire after a reconfiguration. Every
# other condition leaves the decoys at their provisioned 8090/22/2121, which the base
# compose file already publishes, so the manipulation check below does not apply to them
# (and is not run for them — see ``tests/test_decoy_port_publication.py``, which derives
# this set from ``mtd.techniques`` so the literal cannot drift).
DECEPTION_CONDITIONS: frozenset[str] = frozenset(
    {"deceptive_presentation", "combined_mtd_deception"}
)


@runtime_checkable
class TestbedProvisioner(Protocol):
    """Applies an orchestration's reconfiguration to the live testbed."""

    def apply(self, orchestration: OrchestrationResult) -> None:
        """Reconfigure the testbed to match ``orchestration`` (network/ports/etc.)."""
        ...


@dataclass
class MockProvisioner:
    """Records the orchestrations it would apply (no side effects) — for tests/dry-run.

    That it is invoked once per attempt with the trial's orchestration is the proof
    that live MTD is driven from ``mtd.orchestrator`` on the configured cadence.
    """

    applied: list[OrchestrationResult] = field(default_factory=list)

    def apply(self, orchestration: OrchestrationResult) -> None:
        self.applied.append(orchestration)


def desired_config_for_state(condition: str, state) -> dict:
    """Serialize one ``ConfigState`` into the reconfigure desired-state payload.

    This is the exact ``--extra-vars`` contract ``reconfigure.yml`` / ``reconfigure.sh``
    consume: ``mtd_condition``, ``vuln_profile_version``, ``config_checksum`` (the
    idempotency key the reconfigure no-ops on), and one service dict per service carrying
    its observable identity plus the two live deployment facts from the roster
    (``container_port``, ``auxiliary_ports``) that are not part of ``ConfigState``.
    Factored out so anything that applies a state to the substrate (the per-attempt
    provisioner AND the reconfiguration-cost benchmark, ``harness.reconfig_benchmark``)
    serializes it identically — no drift between what is benchmarked and what is run.
    """

    from testbed.roster import load_roster

    roster = load_roster()
    container_ports = roster.container_ports()  # service_id -> in-container port
    auxiliary_ports = roster.auxiliary_ports()  # service_id -> fixed, non-observable ports
    services = [
        {
            "service_id": s.service_id,
            "service_name": s.service_name,
            "ip": s.ip,
            "port": s.port,
            "endpoint": s.endpoint,
            "genuine": s.genuine,
            "software_version": s.software_version,
            # The port to publish the observable port TO inside the container; the
            # reconfigure override publishes observable_port:container_port so a service
            # whose internal port differs (Struts 8080 vs published 8081) stays reachable.
            "container_port": container_ports.get(s.service_id, s.port),
            # Fixed ports the service must ALSO publish which are not its observable
            # identity and never rotate (vsftpd's CVE-2011-2523 backdoor shell on 6200).
            # They are not in ConfigState at all — they are a live deployment fact — so
            # they are read from the roster here, the same way container_port is.
            "auxiliary_ports": list(auxiliary_ports.get(s.service_id, ())),
        }
        for s in state.services
    ]
    return {
        "mtd_condition": condition,
        "vuln_profile_version": state.version,
        "config_checksum": state_checksum(state),
        "services": services,
    }


def _desired_config(orchestration: OrchestrationResult) -> dict:
    """Serialize the orchestration's in-force observable configuration for Ansible.

    Uses the attempt's ``initial_state`` (the configuration the attacker faces at the
    window start); the state threads forward across attempts, so applying it per
    attempt tracks the condition's evolving layout. The ``mtd`` node runs the
    intra-window cadence; this asserts the per-attempt baseline idempotently.
    """

    return desired_config_for_state(orchestration.condition, orchestration.initial_state)


_ANSIBLE_CHANGED_RE = re.compile(r"changed=(\d+)")


def _parse_ansible_changed(stdout: str) -> int | None:
    """Sum ``changed=N`` across the Ansible PLAY RECAP (``None`` if not present)."""

    counts = [int(m) for m in _ANSIBLE_CHANGED_RE.findall(stdout)]
    return sum(counts) if counts else None


#: The reconfigure playbook's task that re-creates DVWA's training database when a recreated
#: container lost it (``infra/ansible/reconfigure.yml``).
DVWA_DB_TASK = "Ensure the DVWA training database exists"

_TASK_HEADER_RE = re.compile(r"^TASK \[(?P<name>.*)\]")
_TASK_STATUS_RE = re.compile(r"^(?P<status>ok|changed|failed|fatal|skipping): \[")


def _ansible_task_status(stdout: str, task_prefix: str) -> str | None:
    """The status (``ok``/``changed``/...) the named task reported in an ansible-playbook log.

    Reads the default stdout callback: the task's ``TASK [...]`` header, then its first status
    line (``FAILED - RETRYING`` lines are not statuses). None when the task is not in the log.
    """

    in_task = False
    for line in stdout.splitlines():
        header = _TASK_HEADER_RE.match(line)
        if header:
            in_task = header.group("name").startswith(task_prefix)
            continue
        if in_task:
            status = _TASK_STATUS_RE.match(line)
            if status:
                return status.group("status")
    return None


def _ansible_failure(result) -> str:
    """The failing task and why, from an ansible-playbook run, for the fault record.

    ansible-playbook reports a task failure on stdout as ``fatal: [host]: FAILED! => {json}``,
    so stderr alone usually says nothing about what failed. The JSON is printed with sorted
    keys, so a shell task's whole script (``cmd``) comes before ``msg`` and ``stdout``: the
    reason is read from the parsed result, not from the start of the line.
    """

    task = None
    for line in (result.stdout or "").splitlines():
        header = _TASK_HEADER_RE.match(line)
        if header:
            task = header.group("name")
            continue
        if not line.startswith(("fatal:", "failed:")):
            continue
        reason = line.strip()[:240]
        _, sep, payload = line.partition("=> ")
        if sep:
            try:
                parsed = json.loads(payload)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                parts = [str(parsed.get("msg") or "").strip()]
                for key in ("stdout", "stderr"):
                    text = str(parsed.get(key) or "").strip()
                    if text:
                        parts.append(f"{key}: {text[-160:]}")
                reason = "; ".join(p for p in parts if p) or reason
        return (f"{task}: " if task else "") + reason
    return (result.stderr or "").strip()[:200]


def _probe_note(probe) -> str:
    """Why a probe saw nothing, when the probe itself failed (else empty): for the fault message."""

    if nmap_greppable_complete(probe.stdout):
        return ""
    err = (probe.stderr or "").strip()
    tail = f", stderr: {err[-160:]}" if err else ""
    return f" (the probe itself failed: rc={probe.returncode}{tail})"


@dataclass
class AnsibleProvisioner:
    """Real provisioner: reconfigure the testbed to each orchestration via infra/ansible.

    Called once per attempt from ``build_env_factory``. It renders the orchestration's
    in-force observable configuration to ``--extra-vars`` and runs the idempotent
    ``reconfigure.yml`` playbook on the control host through the shared ``RemoteRunner``
    seam (so it is unit-tested with a fake runner and driven live only under
    ``MTD_TESTBED=1``). It records what was applied for the run sidecar and fails
    closed (``ProvisioningError``) if the reconfiguration does not converge — a wrong
    testbed state must never be measured against.

    **Deception manipulation check.** For the two conditions that rotate decoy
    observables (``DECEPTION_CONDITIONS``), a converged Ansible run is not evidence that
    the manipulation reached the wire: the responder can rebind inside its container
    while the host publishes a different port, in which case every decoy vanishes from
    the scannable surface and the arm silently degenerates to "no deception at all" —
    the exact failure that produced a full 33-hour study of unusable
    ``deceptive_presentation`` data. After the playbook converges, ``apply`` therefore
    probes the testbed from the executor node (the attacker's own vantage point) and
    requires EVERY in-force decoy to answer on its modeled observable port. A decoy that
    does not answer raises ``ProvisioningError``, which the experiment runner handles as an
    infrastructure fault: the sequence is excluded, with the failure's message, and the run
    continues (or halts under ``on_trial_failure='halt'``; metrics spec §8a). The check does
    NOT run for any other condition — their
    decoys stay on the ports the base compose file already publishes, so nothing about
    those arms changes.
    """

    runner: RemoteRunner
    control_host: str
    playbook: str = "infra/ansible/reconfigure.yml"
    inventory: str = "infra/ansible/inventory.ini"
    ssh_user: str = "labadmin"
    timeout_s: float = 300.0
    # Manipulation-check wiring; required only for the deception conditions. A deception
    # apply with these unset fails LOUDLY rather than skipping the check silently.
    executor_node: str = ""
    targets_host: str = ""
    testbed_cidr: str = ""
    probe_timeout_s: float = 60.0
    # A rotated decoy port changes the `decoys` container's published ports, so compose
    # recreates it; give it a bounded moment to bind before declaring the manipulation
    # failed. Tests inject a zero delay.
    probe_attempts: int = 3
    probe_delay_s: float = 2.0
    # ``live.surface_check: strict``: after the playbook converges, wait (up to this long) until
    # EVERY in-force service answers on its published port before the attempt starts. 0 = off,
    # as in every archived run, where containers recreated by a port change could still be
    # starting when the attacker's first scan arrived.
    readiness_timeout_s: float = 0.0
    readiness_poll_s: float = 2.0
    clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    records: list[dict] = field(default_factory=list, init=False)

    def apply(self, orchestration: OrchestrationResult) -> None:
        desired = _desired_config(orchestration)
        argv = [
            "env",
            "ANSIBLE_CONFIG=infra/ansible/ansible.cfg",
            "ansible-playbook",
            "-i",
            self.inventory,
            self.playbook,
            "--extra-vars",
            json.dumps(desired, sort_keys=True),
        ]
        result = self.runner.run(self.control_host, argv, timeout=self.timeout_s)
        record = {
            "condition": orchestration.condition,
            "config_checksum": desired["config_checksum"],
            "reconfig_events": len(orchestration.events),
            "changed": _parse_ansible_changed(result.stdout),
            "returncode": result.returncode,
            # Which decoy ports were verified reachable (None = check not applicable).
            "decoy_ports_verified": None,
            # The readiness gate's wait (None = gate off).
            "ready": None,
            # DVWA's training database at this attempt: "present", "initialized" (a recreated
            # container had lost it), or None (the playbook did not report the task).
            "dvwa_db": {"ok": "present", "changed": "initialized"}.get(
                _ansible_task_status(result.stdout or "", DVWA_DB_TASK) or ""
            ),
        }
        self.records.append(record)
        if result.returncode != 0:
            raise ProvisioningError(
                f"ansible reconfigure for condition {orchestration.condition!r} failed "
                f"(rc={result.returncode}): {_ansible_failure(result)}"
            )
        if self.readiness_timeout_s > 0:
            record["ready"] = self._wait_until_served(orchestration)
        if orchestration.condition in DECEPTION_CONDITIONS:
            record["decoy_ports_verified"] = self._verify_decoys_published(orchestration)

    def _verify_decoys_published(self, orchestration: OrchestrationResult) -> list[int]:
        """Assert every in-force decoy answers on its modeled observable port (or raise).

        The substrate was just re-provisioned to ``orchestration.initial_state``, so this
        attempt's modeled decoy ports ARE the ports the host must now be publishing.
        Probed from the executor node, so the check measures exactly the surface the
        courier's own ``nmap`` scan will measure.
        """

        decoys = {
            s.service_name: s.port for s in orchestration.initial_state.services if not s.genuine
        }
        if not decoys:
            return []
        if not (self.executor_node and self.targets_host and self.testbed_cidr):
            raise ProvisioningError(
                f"condition {orchestration.condition!r} rotates decoy ports but this "
                "provisioner has no executor_node/targets_host/testbed_cidr, so the decoy "
                "manipulation check cannot run — refusing to measure an unverified "
                "deceptive surface"
            )
        # Fail closed: never aim a probe outside the testbed allowlist.
        assert_within_cidr([self.targets_host], self.testbed_cidr)

        wanted = sorted(set(decoys.values()))
        argv = [
            "nmap",
            "-Pn",
            "-n",
            "-oG",
            "-",
            "-p",
            ",".join(str(p) for p in wanted),
            self.targets_host,
        ]
        missing: dict[str, int] = {}
        probe_note = ""
        for attempt in range(max(1, self.probe_attempts)):
            if attempt:
                self.sleep(self.probe_delay_s)
            probe = self.runner.run(self.executor_node, argv, timeout=self.probe_timeout_s)
            open_ports = parse_nmap_greppable(probe.stdout).get(self.targets_host, set())
            missing = {name: port for name, port in decoys.items() if port not in open_ports}
            if not missing:
                return wanted
            probe_note = _probe_note(probe)
        raise ProvisioningError(
            f"decoy manipulation check FAILED for condition {orchestration.condition!r}: "
            f"{sorted(missing.items())} not reachable on {self.targets_host}{probe_note} — the "
            "rotated deceptive surface was not published, so the deception manipulation never "
            "reached the wire"
        )

    def _wait_until_served(self, orchestration: OrchestrationResult) -> dict:
        """Block until every in-force service answers on its published port (or raise).

        The attempt's ``initial_state`` ports are what the host must now publish, and exactly what
        the courier's scan probes. A port change recreates the containers, and some services
        (MySQL-backed DVWA, Juice Shop, WebGoat) take tens of seconds to bind. Without this wait the
        attacker's first scan arrived while they were still starting, so the condition was
        credited with an outage, not a moving target. Probed from the executor node, the
        attacker's own vantage point.
        """

        wanted = {s.service_name: s.port for s in orchestration.initial_state.services}
        if not (self.executor_node and self.targets_host and self.testbed_cidr):
            raise ProvisioningError(
                "the readiness gate needs executor_node/targets_host/testbed_cidr; refusing to "
                "start an attempt against an unverified surface"
            )
        assert_within_cidr([self.targets_host], self.testbed_cidr)
        argv = [
            "nmap",
            "-Pn",
            "-n",
            "-oG",
            "-",
            "-p",
            ",".join(str(p) for p in sorted(set(wanted.values()))),
            self.targets_host,
        ]
        began = self.clock()
        polls = 0
        while True:
            probe = self.runner.run(self.executor_node, argv, timeout=self.probe_timeout_s)
            polls += 1
            open_ports = parse_nmap_greppable(probe.stdout).get(self.targets_host, set())
            missing = {name: port for name, port in wanted.items() if port not in open_ports}
            waited = self.clock() - began
            if not missing:
                return {"wait_s": round(waited, 3), "polls": polls}
            if waited >= self.readiness_timeout_s:
                break
            self.sleep(self.readiness_poll_s)
        raise ProvisioningError(
            f"readiness gate: {sorted(missing.items())} still not answering on "
            f"{self.targets_host}{_probe_note(probe)} {waited:.0f} s after reconfiguring "
            f"{orchestration.condition!r} ({polls} probes)"
        )

    def summary(self) -> dict:
        """A compact record of what was applied, for the run sidecar."""

        waits: dict[str, list[float]] = {}
        for r in self.records:
            if r.get("ready"):
                waits.setdefault(r["condition"], []).append(float(r["ready"]["wait_s"]))
        every = [w for ws in waits.values() for w in ws]
        readiness = None
        if every:
            readiness = {
                "checks": len(every),
                "wait_s_median": round(statistics.median(every), 3),
                "wait_s_max": round(max(every), 3),
                "wait_s_median_by_condition": {
                    c: round(statistics.median(ws), 3) for c, ws in sorted(waits.items())
                },
            }
        return {
            "provisioner": "ansible",
            "applies": len(self.records),
            "conditions": sorted({r["condition"] for r in self.records}),
            "total_changed": sum(r["changed"] or 0 for r in self.records),
            # How many applies had their rotated deceptive surface verified on the wire.
            "decoy_checks": sum(1 for r in self.records if r.get("decoy_ports_verified")),
            # How long attempts waited for every in-force service (None = gate off).
            "readiness": readiness,
            # DVWA's database checks: how many attempts checked it, and how many re-created it.
            "dvwa_db": (
                {
                    "checked": sum(1 for r in self.records if r.get("dvwa_db")),
                    "initialized": sum(
                        1 for r in self.records if r.get("dvwa_db") == "initialized"
                    ),
                }
                if any(r.get("dvwa_db") for r in self.records)
                else None
            ),
        }


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #


def live_executor_factory(env: Environment) -> TargetExecutor:
    """Planner-side factory: use the live environment's carried executor."""

    executor = getattr(env, "executor", None)
    if executor is None:  # pragma: no cover - guarded misuse
        raise TypeError("live_executor_factory requires a LiveEnvironment with an .executor")
    return executor


def build_env_factory(
    settings: LiveSettings,
    transport_factory: TransportFactory,
    provisioner: TestbedProvisioner,
) -> EnvFactory:
    """Build the per-attempt ``LiveEnvironment`` factory used by the trial runner."""

    def factory(orchestration: OrchestrationResult) -> LiveEnvironment:
        # Drive live MTD: reconfigure the testbed to this attempt's orchestration.
        provisioner.apply(orchestration)
        transport = transport_factory(orchestration)
        executor = LiveTargetExecutor(orchestration, transport, settings.testbed_cidr)
        return LiveEnvironment(orchestration, executor)

    return factory


def mock_transport_factory(settings: LiveSettings) -> TransportFactory:
    """A transport factory that resolves against ground truth (offline, no network)."""

    return lambda orchestration: MockTransport(orchestration, settings.testbed_cidr)


def subprocess_transport_factory(
    settings: LiveSettings,
    runner: RemoteRunner,
    primitives: Mapping[str, PrimitiveSpec],
) -> TransportFactory:
    """A transport factory for the real SSH/scan courier (selected under ``--live``).

    ``runner`` is the shared remote-execution seam; ``primitives`` is the roster's
    ``service_name -> primitive`` map. Each per-attempt transport is built with the
    attempt's orchestration so the scan can canonicalize reachable ports.
    """

    targets_host = resolve_targets_host(settings)

    def make(orchestration: OrchestrationResult) -> SubprocessTransport:
        return SubprocessTransport(
            orchestration=orchestration,
            runner=runner,
            executor_node=settings.executor_node,
            testbed_cidr=settings.testbed_cidr,
            targets_host=targets_host,
            primitives=primitives,
            ssh_user=settings.ssh_user,
            scan_scope=settings.scan_scope,
            require_full_surface=settings.surface_check == SURFACE_CHECK_STRICT,
            remote_timeouts=True,
        )

    return make


def build_mock_env_factory(settings: LiveSettings) -> tuple[EnvFactory, MockProvisioner]:
    """Convenience: mock-executor env factory + the provisioner it records into."""

    provisioner = MockProvisioner()
    factory = build_env_factory(settings, mock_transport_factory(settings), provisioner)
    return factory, provisioner


# --------------------------------------------------------------------------- #
# Preflight + dry-run
# --------------------------------------------------------------------------- #


def egress_map(settings: LiveSettings) -> EgressMap:
    return EgressMap(
        control_host=settings.control_host,
        executor_node=settings.executor_node,
        testbed_cidr=settings.testbed_cidr,
        testbed_nodes=resolve_testbed_nodes(settings),
    )


def preflight(settings: LiveSettings, checker: RouteChecker) -> EgressMap:
    """Fail-closed preflight: require the CIDR allowlist and assert the egress split.

    Raises ``ConfigurationError`` if ``testbed_cidr`` is unset and ``EgressViolation``
    if the planner/executor egress split does not hold. Returns the asserted map.
    """

    parse_cidr(settings.testbed_cidr)  # raises ConfigurationError if unset/blank
    emap = egress_map(settings)
    assert_egress_split(emap, checker)
    return emap


@dataclass(frozen=True)
class PlannedEgressChecker:
    """On-paper route checker for ``--dry-run``: reports the *intended* egress map.

    Reflects the built containment design — the control host is dual-homed (reaches
    the model API *and* the testbed subnet) but does not forward; every non-control
    testbed node has no route off the testbed. It validates config-level invariants
    without any live probing, and is NOT evidence the real routes hold — the deployed
    ``harness.safety.SshRouteChecker`` probes for that.
    """

    settings: LiveSettings

    def can_reach(self, source: str, dest: str) -> bool:
        cidr = self.settings.testbed_cidr
        if source == self.settings.control_host:
            # Dual-homed: reaches the model API and the testbed subnet (by design).
            return dest == ANTHROPIC_API_HOST or _in_cidr(dest, cidr)
        # Any non-control testbed node: only in-CIDR destinations are reachable.
        return _in_cidr(dest, cidr)

    def ip_forwarding_enabled(self, host: str) -> bool:
        # The intended design: forwarding is off on the control host.
        return False

    def has_default_route(self, host: str) -> bool:
        # Only the dual-homed control host carries a default route (its LAN leg).
        return host == self.settings.control_host

    def resolves_external_dns(self, host: str) -> bool:
        # Intended sealed state: no non-control node resolves external DNS.
        return False


def _in_cidr(dest: str, cidr: str) -> bool:
    from harness.safety import is_within_cidr

    try:
        return is_within_cidr(dest, cidr)
    except Exception:
        return False


def dry_run_plan(config: ExperimentConfig, mode: str = "pilot") -> dict:
    """Exercise the full live path with the mock executor; return the plan (no network).

    Prints nothing; the caller renders. Validates the CIDR + egress split on paper,
    lists the conditions, the trial count, a sample of resolved in-CIDR targets, and
    the egress map.
    """

    settings = config.live
    checker = PlannedEgressChecker(settings)
    emap = preflight(settings, checker)  # fail closed: CIDR + egress split

    if mode == "pilot":
        conditions = list(config.pilot.conditions)
        sequences = config.pilot.sequences_per_condition
        trials = len(conditions) * sequences * config.attempts_per_sequence
    elif mode == "repeatability":
        conditions = [config.repeatability.condition]
        trials = config.repeatability.n_trials
    elif mode == "clock_validation":
        from harness.config import CLOCK_VALIDATION_ARMS

        conditions = list(config.clock_validation.conditions)
        trials = (
            len(conditions)
            * config.clock_validation.sequences_per_condition
            * config.attempts_per_sequence
            * len(CLOCK_VALIDATION_ARMS)
        )
    else:  # full
        conditions = list(config.conditions)
        trials = len(conditions) * config.sequences_per_condition * config.attempts_per_sequence

    # Resolve a sample target set through the mock transport (no network touched).
    from mtd.orchestrator import run_orchestration

    sample_condition = conditions[0]
    orch = run_orchestration(
        sample_condition,
        default_profile(config.vuln_profile_version),
        "2026-07-01T00:00:00Z",
        config.mtd_cadence_s,
        config.max_trial_duration_s,
        seed=1,
    )
    transport = MockTransport(orch, settings.testbed_cidr)
    sample_targets = [
        {
            "service": s.service_name,
            "port": s.port,
            "endpoint": s.endpoint,
            # ``address`` is the attacker-OBSERVABLE address (network shuffling moves it);
            # ``delivery_host`` is the in-CIDR host anything would actually be delivered to.
            "address": s.ip,
            "delivery_host": s.delivery_host(),
        }
        for s in transport.scan(settings.testbed_cidr, "2026-07-01T00:00:00Z")
    ]

    return {
        "mode": mode,
        "live": True,
        "conditions": conditions,
        "trials": trials,
        "testbed_cidr": settings.testbed_cidr,
        "sample_targets": sample_targets,
        "egress_map": emap.as_dict(),
        "executor": "mock (dry-run: no network touched)",
    }
