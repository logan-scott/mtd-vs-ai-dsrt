#!/usr/bin/env python3
"""Render the compose override + responder config from a desired-state JSON.

Stdlib-only: this runs on the targets node (which has no instrument package),
invoked by reconfigure.sh. Extracted from reconfigure.sh's inline Python so the
port-mapping logic is unit-tested offline (see tests/test_render_override.py).

The override publishes ``observable_port:container_port`` for **every** service in the
desired state — genuine services AND decoys. A service whose in-container port differs
from its published port (Struts serves on 8080 but is published as 8081) would otherwise
be republished as ``8081:8081`` after a reconfigure — mapping to a port nothing listens
on inside the container, silently breaking the target. It also refuses to render a
duplicate host-port binding, which would make ``docker compose up`` fail with an
unhelpful error.

DECOYS ARE ONE COMPOSE SERVICE. The three decoys (``http-admin``, ``ssh-gateway``,
``ftp-backup``) are served by a SINGLE ``decoys`` container (the deterministic deception
responder), so they cannot be emitted as three compose services: they are emitted as one
``decoys:`` entry whose ``ports:`` list carries all three mappings. Publishing them is
what makes ``deceptive_presentation`` (and ``combined_mtd_deception``) real on the wire:
before this, ``render_responder`` rotated the port the responder binds INSIDE the
container while the compose file kept publishing the fixed 8090/22/2121, so a rotated
decoy answered on no reachable port at all. Decoys have no separate in-container port —
the responder binds ``spec.port`` straight from the rendered ``responder.yml`` — so a
decoy publishes ``rotated:rotated`` (``container_port`` defaults to the observable port).

AUXILIARY PORTS ARE PUBLISHED TOO, AND THEY DO NOT ROTATE. A service's published
surface is its observable port PLUS every port in ``auxiliary_ports`` (carried in the
desired state from ``testbed/roster.yaml`` via ``harness.live_runner._desired_config``,
the same route ``container_port`` takes — this script is stdlib-only and runs on the
targets node, so it can read neither the roster nor the base compose file). Today that
is vsftpd's ``6200``: CVE-2011-2523's exploitation path spans TWO ports — the ``:)``
smiley on the observable FTP port opens a root shell on tcp/6200, and the maintained
Metasploit module reaches the shell THERE. Emitting only the observable mapping made
``!override`` (below) DROP the base file's ``6200`` publication, so ``ftp-vsftpd`` could
be reached but never compromised: one of five genuine targets was structurally
un-compromisable in all six arms for the whole ``full_haiku_20260831`` run. An auxiliary
port is a property of the vulnerability, not a binding, so it publishes ``fixed:fixed``
and is NEVER rotated by ``port_rotation``/``combined_mtd_deception``; it is likewise
absent from ``ServiceTuple``, so it never enters the attacker-observable identity.

DUPLICATE GUARD SPANS BOTH CLASSES AND AUXILIARY PORTS. ``mtd.techniques._new_port``
draws ``randint(1024, 65535)`` for genuine ports AND for decoy ports, so under
``combined_mtd_deception`` a rotated decoy can collide with a rotated genuine service —
and either can now land on an auxiliary port (6200 is inside that draw range). That must
FAIL LOUDLY (the reconfigure aborts, the harness records an infrastructure fault and
re-runs the attempt) rather than silently dropping one of the two services from the
published surface. Note port 22 is below 1024, so a rotated decoy can never land back on
the SSH decoy's initial port.

CRITICAL: the ports list is emitted with the Compose ``!override`` merge tag. Docker
Compose merges two ``ports:`` sequences by APPENDING (keyed by published port), so a
plain override would ADD the rotated mapping while the base file's original port stays
published — leaving every service reachable on its ORIGINAL port under port rotation
(defeating the mutation and diverging from the offline model, which replaces the port).
``!override`` makes the override REPLACE the base sequence, so each service publishes
exactly the desired mapping.
"""

from __future__ import annotations

import json
import sys

DEFAULT_BIND_HOST = "172.31.66.10"

# All decoys are served by this ONE compose service (the deception responder container),
# so their port mappings are emitted together under a single entry.
DECOY_COMPOSE_SERVICE = "decoys"


def _claim(seen: dict[int, str], host_port: int, claimant: str) -> None:
    """Record a host-port claim, refusing a duplicate across genuine services, decoys and
    auxiliary ports alike. ``claimant`` is the service id, suffixed ``(auxiliary)`` for a
    non-observable port, so the error names which of a service's two mappings collided."""

    if host_port in seen:
        raise ValueError(
            f"duplicate host port {host_port} for {seen[host_port]} and {claimant} "
            "— docker compose up would fail"
        )
    seen[host_port] = claimant


def _mappings(service: dict, bind_host: str) -> list[tuple[str, int, str]]:
    """Every ``(claimant, host_port, "bind:host:container")`` this service publishes.

    The observable mapping comes first (``observable_port:container_port``), then one
    ``fixed:fixed`` mapping per declared auxiliary port, in declaration order. The
    claimant label is what the duplicate guard reports, so an auxiliary port names itself
    as such rather than colliding anonymously with its own service.
    """

    service_id = service["service_id"]
    host_port = int(service["port"])
    container_port = int(service.get("container_port") or host_port)
    out = [(service_id, host_port, f'"{bind_host}:{host_port}:{container_port}"')]
    for aux in service.get("auxiliary_ports") or ():
        # Auxiliary ports never rotate, so host and container are the same fixed number.
        aux_port = int(aux)
        out.append((f"{service_id} (auxiliary)", aux_port, f'"{bind_host}:{aux_port}:{aux_port}"'))
    return out


def _entry(service: dict, bind_host: str, seen: dict[int, str]) -> list[str]:
    """The rendered ``!override`` mapping list for one service, claiming each host port."""

    mappings = []
    for claimant, host_port, mapping in _mappings(service, bind_host):
        _claim(seen, host_port, claimant)
        mappings.append(mapping)
    return mappings


def render_override(state: dict, bind_host: str = DEFAULT_BIND_HOST) -> str:
    """Render docker-compose.override.yml: every port every service publishes.

    Genuine services are separate compose services keyed by ``service_id``; the decoys
    share the single ``decoys`` service and are emitted as one entry with all of their
    mappings. Each service publishes its observable ``host:container`` mapping plus one
    ``fixed:fixed`` mapping per declared auxiliary port. The host-port uniqueness guard
    spans both classes and both kinds of port.
    """

    genuine = [s for s in state["services"] if s["genuine"]]
    decoys = [s for s in state["services"] if not s["genuine"]]
    seen: dict[int, str] = {}
    lines = ["# generated by reconfigure.sh via render_override.py — do not edit", "services:"]
    for s in genuine:
        mappings = _entry(s, bind_host, seen)
        lines.append(f"  {s['service_id']}:")
        # !override so this REPLACES the base file's ports (Compose appends otherwise —
        # see the module docstring); the base port would else stay published under rotation.
        # Because it REPLACES, the list must carry the service's auxiliary ports as well,
        # or the base file's publication of them is silently dropped.
        lines.append(f"    ports: !override [{', '.join(mappings)}]")
    if decoys:
        mappings = []
        for d in decoys:
            mappings.extend(_entry(d, bind_host, seen))
        # ONE compose service carrying every decoy mapping (see the module docstring).
        lines.append(f"  {DECOY_COMPOSE_SERVICE}:")
        lines.append(f"    ports: !override [{', '.join(mappings)}]")
    return "\n".join(lines) + "\n"


def render_responder(state: dict) -> str:
    """Render the decoy responder banner map (rotated for deceptive_presentation).

    ``port`` here is the port the responder binds INSIDE the container; the override
    above publishes the same number on the host, so host and container agree.
    """

    decoys = [s for s in state["services"] if not s["genuine"]]
    out = ["decoys:"]
    for d in decoys:
        out.append(f"  - service_name: {d['service_name']}")
        out.append(f"    port: {d['port']}")
        if d.get("endpoint"):
            out.append(f"    endpoint: {d['endpoint']}")
    return "\n".join(out) + "\n"


def main(argv: list[str]) -> int:
    state = json.load(open(argv[1], encoding="utf-8"))
    testbed = argv[2]
    bind_host = argv[3] if len(argv) > 3 else DEFAULT_BIND_HOST
    with open(f"{testbed}/docker-compose.override.yml", "w", encoding="utf-8") as f:
        f.write(render_override(state, bind_host))
    with open(f"{testbed}/responder/responder.yml", "w", encoding="utf-8") as f:
        f.write(render_responder(state))
    return 0


if __name__ == "__main__":  # pragma: no cover - node entrypoint
    raise SystemExit(main(sys.argv))
