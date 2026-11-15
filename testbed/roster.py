"""The version-pinned live-testbed roster: services, weaknesses, and primitives.

This is the single source of truth that ties the *offline* observable identity of
each service (the ``service_name`` fingerprint used by ``testbed/profile.py`` and
the ground-truth inventory) to its *live* deployment facts:

- the pinned software version and the known weakness it represents (reproducibility
  + a documented control variable), and
- for a genuine service, the **pre-staged** exploit primitive that
  ``harness.live_targets.SubprocessTransport.deliver`` invokes as a pure courier
  (standard tooling only, staged onto the attacker node by ``infra/ansible``; no
  payload is authored in this repository).

Decoys carry no primitive: they expose no real vulnerability, so the courier can
only ever *reach* them (a connect probe), never gain access — which is exactly the
offline outcome. The genuine/decoy label is **never** read by the courier; it is
stamped separately from ground truth by the executor. The roster keys primitives
on the observable ``service_name``, which MTD never mutates (only ip/port/endpoint
rotate), so a primitive stays correctly matched to its service across every
condition.

The data lives alongside this module in ``roster.yaml`` and is version-pinned by
``vuln_profile_version`` (mirrors the ``vp-*`` tag on the offline profile).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_DEFAULT_ROSTER_PATH = Path(__file__).with_name("roster.yaml")


@dataclass(frozen=True)
class PrimitiveSpec:
    """A pre-staged exploit primitive: a named, standard tool invocation.

    ``argv`` is the command staged on the attacker node, with ``{host}``,
    ``{port}`` and ``{path}`` placeholders filled at delivery time. The primitive
    emits a single JSON line ``{"reached_and_executed": bool,
    "unauthorized_access": bool}`` which the courier parses — see
    ``harness.live_targets.parse_primitive_result``.
    """

    name: str
    argv: tuple[str, ...]

    def render(self, host: str, port: int, path: str) -> list[str]:
        """Fill the placeholders for a concrete delivery."""

        rendered_path = path or "/"
        return [a.format(host=host, port=port, path=rendered_path) for a in self.argv]


@dataclass(frozen=True)
class RosterEntry:
    """One service in the standardized profile, with its live deployment facts."""

    service_id: str
    service_name: str  # observable fingerprint (matches the ground-truth inventory)
    genuine: bool
    software_version: str
    weakness: str  # CVE id or a short description of the known weakness
    host_role: str  # which testbed node this service is provisioned on
    # The port the service listens on INSIDE its container, which may differ from the
    # observable/published port (Struts serves 8080 but is published as 8081). The
    # reconfigure override must publish ``observable_port:container_port``. 0 = same as
    # observable (decoys, whose responder binds the observable port directly).
    container_port: int = 0
    # Ports the service must ALSO publish that are not its observable identity and are
    # NOT subject to MTD. vsftpd is the only one today: CVE-2011-2523's exploitation path
    # spans two ports — the ``:)`` smiley on the observable FTP port opens a root shell on
    # tcp/6200, which the maintained Metasploit module then reaches. 6200 is a property of
    # the vulnerability, not a binding, so it never rotates and is always published; it is
    # deliberately absent from ``ServiceConfig``/``ServiceTuple`` so it never enters the
    # attacker-observable identity, reconnaissance accuracy, or attack-path churn.
    auxiliary_ports: tuple[int, ...] = ()
    primitive: PrimitiveSpec | None = None


@dataclass(frozen=True)
class Roster:
    """The full version-pinned roster."""

    vuln_profile_version: str
    entries: tuple[RosterEntry, ...]

    def primitives(self) -> dict[str, PrimitiveSpec]:
        """Map ``service_name`` -> primitive for every genuine service that has one."""

        return {e.service_name: e.primitive for e in self.entries if e.primitive is not None}

    def container_ports(self) -> dict[str, int]:
        """Map ``service_id`` -> in-container port for genuine services that declare one."""

        return {e.service_id: e.container_port for e in self.entries if e.container_port}

    def auxiliary_ports(self) -> dict[str, tuple[int, ...]]:
        """Map ``service_id`` -> fixed, non-observable ports that must also be published.

        These travel to the targets node in the desired-state JSON alongside
        ``container_port`` (``harness.live_runner._desired_config``) because
        ``render_override.py`` is stdlib-only and can read neither this roster nor the
        base compose file at render time.
        """

        return {e.service_id: e.auxiliary_ports for e in self.entries if e.auxiliary_ports}

    def genuine(self) -> tuple[RosterEntry, ...]:
        return tuple(e for e in self.entries if e.genuine)

    def decoys(self) -> tuple[RosterEntry, ...]:
        return tuple(e for e in self.entries if not e.genuine)


def load_roster(path: str | Path | None = None) -> Roster:
    """Load and validate the roster from ``roster.yaml`` (or an override path)."""

    p = Path(path) if path is not None else _DEFAULT_ROSTER_PATH
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    entries: list[RosterEntry] = []
    for raw in data.get("services", []):
        prim_raw = raw.get("primitive")
        primitive = None
        if prim_raw:
            primitive = PrimitiveSpec(name=prim_raw["name"], argv=tuple(prim_raw["argv"]))
        entries.append(
            RosterEntry(
                service_id=raw["service_id"],
                service_name=raw["service_name"],
                genuine=bool(raw["genuine"]),
                software_version=raw.get("software_version", ""),
                weakness=raw.get("weakness", ""),
                host_role=raw.get("host_role", "targets"),
                container_port=int(raw.get("container_port", 0)),
                auxiliary_ports=tuple(int(p) for p in (raw.get("auxiliary_ports") or ())),
                primitive=primitive,
            )
        )
    return Roster(
        vuln_profile_version=str(data.get("vuln_profile_version", "vp-0.2.0")),
        entries=tuple(entries),
    )
