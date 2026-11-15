"""The single remote-execution seam shared by every live-mode body.

Tasks 1–3 (the scan/exploit courier, the Ansible provisioner, and the active
route checker) all need to run a command *on a specific testbed node* and read its
result. They all do it through the one ``RemoteRunner`` protocol here, so:

- there is exactly **one** place that performs real subprocess / SSH I/O
  (``SubprocessRemoteRunner``), and it is trivial to audit; and
- every consumer is injectable — offline unit tests drive the courier, the
  provisioner, and the route checker with an in-memory fake runner and never
  touch the network, while an integration test (``MTD_TESTBED=1``) swaps in the
  real runner against the deployed testbed.

The runner is deliberately dumb: it runs an ``argv`` on a host and returns the
``CommandResult``. All parsing/decision logic lives in the callers (and is unit
tested), so nothing that matters for correctness is hidden behind the network.
"""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class CommandResult:
    """The outcome of running one command on a node."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@runtime_checkable
class RemoteRunner(Protocol):
    """Runs ``argv`` on ``host`` and returns its result.

    ``host`` is a testbed node address/hostname (or the local control host). The
    real implementation reaches non-local hosts over SSH; a node equal to the
    runner's ``local_host`` runs locally (the control host operates on itself for
    the ``ip_forward`` check and for launching Ansible).

    ``sudo=True`` runs the command as root via passwordless ``sudo -n`` (the deploy
    user has passwordless sudo on every node + the control host). Privileged
    provisioning commands (``sysctl``, ``iptables``, ``ip route``, ``resolvectl``)
    need it; read-only preflight probes do not.
    """

    def run(
        self, host: str, argv: list[str], *, timeout: float, sudo: bool = False
    ) -> CommandResult: ...


@dataclass(frozen=True)
class SubprocessRemoteRunner:
    """Real runner: local exec for ``local_host``, SSH for everything else.

    This is the only class in the codebase that performs live network I/O for the
    testbed path. It is constructed solely under ``--live`` (or the integration
    test) — the offline default never instantiates it.

    SSH is invoked in batch mode (no prompts, no agent forwarding); the remote
    ``argv`` is shell-quoted with :func:`shlex.join` and handed to ``ssh`` as a
    single remote-command string so arguments with spaces survive the round trip.
    """

    ssh_user: str = "labadmin"
    local_host: str | None = None
    ssh_port: int = 22
    # Per-host admin SSH port overrides (host -> port). Needed because the targets
    # node's administrative sshd is moved to :2222 so the SSH decoy can own :22.
    host_ports: Mapping[str, int] = field(default_factory=dict)
    ssh_opts: tuple[str, ...] = (
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=10",
    )

    _LOCAL_ALIASES = frozenset({"localhost", "127.0.0.1", "::1"})

    def _is_local(self, host: str | None) -> bool:
        return host is None or host == self.local_host or host in self._LOCAL_ALIASES

    def _port_for(self, host: str) -> int:
        return self.host_ports.get(host, self.ssh_port)

    def _argv_for(self, host: str, argv: list[str], *, sudo: bool = False) -> list[str]:
        argv = ["sudo", "-n", *argv] if sudo else list(argv)
        if self._is_local(host):
            return argv
        remote = shlex.join(argv)
        return [
            "ssh",
            "-p",
            str(self._port_for(host)),
            *self.ssh_opts,
            f"{self.ssh_user}@{host}",
            "--",
            remote,
        ]

    def run(
        self, host: str, argv: list[str], *, timeout: float, sudo: bool = False
    ) -> CommandResult:
        cmd = self._argv_for(host, argv, sudo=sudo)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout or ""
            err = exc.stderr or ""
            out = out.decode() if isinstance(out, bytes) else out
            err = err.decode() if isinstance(err, bytes) else err
            return CommandResult(returncode=124, stdout=out, stderr=f"{err}\n[timeout]".strip())
        return CommandResult(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


@dataclass
class RecordingRunner:
    """In-memory ``RemoteRunner`` for offline tests: canned replies + a call log.

    Not used in production; it lets the offline suite exercise every real body's
    command construction and output parsing with zero network I/O. ``replies`` maps
    a ``(host, argv_prefix_substring)`` intent to a ``CommandResult``; ``calls``
    records every invocation so tests can assert *what* would have been run.
    """

    responder: object  # Callable[[str, list[str]], CommandResult]
    calls: list[tuple[str, list[str]]] = field(default_factory=list)

    def run(
        self, host: str, argv: list[str], *, timeout: float, sudo: bool = False
    ) -> CommandResult:
        self.calls.append((host, (["sudo", "-n", *argv] if sudo else list(argv))))
        result = self.responder(host, argv)  # type: ignore[operator]
        if result is None:  # pragma: no cover - test misuse guard
            raise AssertionError(f"RecordingRunner has no canned reply for {host!r}: {argv!r}")
        return result
