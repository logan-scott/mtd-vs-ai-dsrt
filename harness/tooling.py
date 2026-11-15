"""Capture the versions of the standard tools each live primitive drives.

Reproducibility/provenance: the study records, per trial, the exact standard tooling used to
carry recon and exploitation (``tooling_versions`` — a control variable; see
``measurement.models`` and ``docs/metrics_specification.md``). Offline the field is a fixed
placeholder; for a **live** run this module queries the executor node once at wiring time for
the real versions of the tools the pre-staged primitives invoke:

  - ``nmap``  — the subnet scan (recon),
  - ``curl``  — the training-app couriers (DVWA / Juice Shop / WebGoat, and the Struts reach
    probe), following each app's documented procedure, and
  - ``metasploit`` (``msfconsole``) — the maintained modules for the two CVE targets
    (vsftpd 2.3.4 backdoor, Struts S2-045), and
  - ``msfd`` — the resident Metasploit daemon those two primitives run their modules in since
    2026-10-31: the framework version its console reports.

Everything in :func:`capture_tooling_versions` is pure/parsed and best-effort: a tool that is
missing or slow yields ``<name>=unknown`` rather than failing the run. The parser is unit-tested
offline against captured real ``--version`` output; the capture path runs only under ``--live``.

:func:`require_msfd` is the one hard check: a live run refuses to start (``ToolingError``)
unless msfd answers on the executor node and loads both modules, because every Struts and
vsftpd delivery would otherwise end its sequence as an infrastructure fault.
"""

from __future__ import annotations

import re

from harness.live_ssh import RemoteRunner

#: Where the resident Metasploit daemon listens on the executor node (loopback only; the
#: primitives' ``MSFD_HOST`` / ``MSFD_PORT`` defaults and ``files/msfd.service`` say the same,
#: pinned by ``tests/test_primitive_scripts.py``).
MSFD_HOST = "127.0.0.1"
MSFD_PORT = 55554

#: Ask msfd's console for its version over the same bash TCP route the primitives use.
MSFD_VERSION_ARGV: list[str] = [
    "bash",
    "-c",
    f"exec 3<>/dev/tcp/{MSFD_HOST}/{MSFD_PORT} && printf 'version\\nexit -y\\n' >&3 "
    "&& timeout 30 cat <&3",
]
_MSFD_VERSION_RE = r"Framework:\s*([0-9][0-9A-Za-z.\-]*)"

#: The two maintained modules the Metasploit primitives run in msfd.
MSFD_MODULES: tuple[str, ...] = (
    "exploit/multi/http/struts2_content_type_ognl",
    "exploit/unix/ftp/vsftpd_234_backdoor",
)

#: The readiness check a live run makes before it starts: load both modules into msfd (so the
#: first delivery after a start is not slowed by loading one), then ask for the version.
MSFD_READY_ARGV: list[str] = [
    "bash",
    "-c",
    f"exec 3<>/dev/tcp/{MSFD_HOST}/{MSFD_PORT} && printf '"
    + "".join(f"use {m}\\n" for m in MSFD_MODULES)
    + "version\\nexit -y\\n' >&3 && timeout 120 cat <&3",
]
_MSFD_LOAD_FAILED_RE = r"Failed to load module|No results from search|Invalid module"


class ToolingError(RuntimeError):
    """A primitive's standard tooling is not ready on the executor node (aborts a live run)."""


# tool label -> (argv to run on the node, regex whose first group is the version)
_TOOL_PROBES: tuple[tuple[str, list[str], str], ...] = (
    ("nmap", ["nmap", "--version"], r"Nmap version\s+([0-9][0-9A-Za-z.]*)"),
    ("curl", ["curl", "--version"], r"curl\s+([0-9][0-9A-Za-z.]*)"),
    (
        "metasploit",
        ["msfconsole", "--version"],
        r"(?:Framework Version:|metasploit\s+v)\s*([0-9][0-9A-Za-z.\-]*)",
    ),
    ("msfd", MSFD_VERSION_ARGV, _MSFD_VERSION_RE),
)


def parse_tool_version(pattern: str, output: str) -> str:
    """Return the first captured version in ``output`` for ``pattern``, else ``unknown``."""

    match = re.search(pattern, output)
    return match.group(1) if match else "unknown"


def capture_tooling_versions(
    runner: RemoteRunner,
    node: str,
    *,
    roster_version: str = "",
    timeout: float = 30.0,
) -> str:
    """Query ``node`` for the versions of the standard tools the live primitives drive.

    Returns a compact, deterministic ``name=version;...`` string suitable for the per-trial
    ``tooling_versions`` control variable. Best-effort: any probe that errors/times out is
    recorded as ``<name>=unknown`` so version capture never breaks a live run. ``roster_version``
    (the ``vuln_profile_version``) is appended when given, tying the tooling to the pinned
    target set.
    """

    parts: list[str] = []
    for name, argv, pattern in _TOOL_PROBES:
        try:
            result = runner.run(node, argv, timeout=timeout)
            output = f"{result.stdout or ''}\n{result.stderr or ''}"
            parts.append(f"{name}={parse_tool_version(pattern, output)}")
        except Exception:  # noqa: BLE001 - version capture must never break a run
            parts.append(f"{name}=unknown")
    if roster_version:
        parts.append(f"vuln_profile={roster_version}")
    return ";".join(parts)


def require_msfd(runner: RemoteRunner, node: str, *, timeout: float = 150.0) -> str:
    """The framework version the resident msfd on ``node`` reports; ``ToolingError`` if it
    does not answer, or cannot load the two modules the Metasploit primitives use.

    The Struts and vsftpd primitives print no verdict when msfd cannot be reached, so a live run
    started without it would lose every sequence that delivers to either target. Called by the
    live wiring after the isolation preflight, before any trial runs; it also loads both
    modules, so the run's first delivery to either target is not slowed by loading one.
    """

    try:
        result = runner.run(node, MSFD_READY_ARGV, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - any failure to ask is a failure to start
        raise ToolingError(f"could not ask msfd on {node} for its version: {exc}") from exc
    output = f"{result.stdout or ''}\n{result.stderr or ''}"
    version = parse_tool_version(_MSFD_VERSION_RE, output)
    if version == "unknown":
        raise ToolingError(
            f"msfd does not answer on {node} ({MSFD_HOST}:{MSFD_PORT}, rc={result.returncode}): "
            "start it with `sudo systemctl start msfd` and wait for the port "
            "(docs/rerun_runbook.md §7q)"
        )
    if re.search(_MSFD_LOAD_FAILED_RE, output):
        raise ToolingError(
            f"msfd on {node} could not load one of {', '.join(MSFD_MODULES)} (send the output of "
            "the version check in docs/rerun_runbook.md §7q)"
        )
    return version
