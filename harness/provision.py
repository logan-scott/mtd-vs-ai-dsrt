"""Fail-closed provisioning orchestration: open temporary egress, deploy, re-seal.

The no-egress rule is a *trial-phase* invariant (it protects data collection, which
is IRB-gated). Provisioning the no-egress nodes — ``apt install``, pulling the five
public target images — needs the internet once. This module opens a **temporary,
audited** egress path (NAT on the dual-homed control host + a default route on each
node via ``172.31.66.5``), runs ``site.yml``, then **always** tears the egress down
and proves the testbed is sealed again via the real isolation preflight before it
returns. If the seal cannot be verified, it raises and the run is blocked.

Two containment guarantees, both tested with a fake runner:

- **The seal always runs.** ``close_egress`` + the preflight execute in a ``finally``,
  so egress is removed even if ``site.yml`` fails midway.
- **Fail closed on a leak.** If the post-deploy preflight finds any node egress,
  control-host forwarding left on, or a node default route, it raises
  ``SafetyError`` and the deploy does not report success.

All I/O goes through the injected ``RemoteRunner`` seam; nothing here weakens the
``--live`` startup preflight (that still runs independently before any trial).
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass

from harness.config import LiveSettings
from harness.live_runner import resolve_control_testbed_ip, resolve_testbed_nodes
from harness.live_ssh import RemoteRunner
from harness.preflight import run_isolation_preflight
from harness.safety import ProvisioningError, SshRouteChecker

# NAT rule that masquerades testbed-sourced, off-testbed traffic out the control
# host's LAN leg. Applied only during provisioning; removed before any trial.
_SYSCTL = "net.ipv4.ip_forward"
_READ_TIMEOUT = 15.0
_SITE_TIMEOUT = 1800.0
# Ship a lab ansible.cfg (host_key_checking off) so the deploy needs no shell export.
_ANSIBLE_CFG = "infra/ansible/ansible.cfg"


def _nat_rule(cidr: str, action: str) -> list[str]:
    """The MASQUERADE rule with ``action`` in {-C (check), -A (add), -D (delete)}."""

    return [
        "iptables",
        "-t",
        "nat",
        action,
        "POSTROUTING",
        "-s",
        cidr,
        "!",
        "-d",
        cidr,
        "-j",
        "MASQUERADE",
    ]


def _forward_src_rule(cidr: str, action: str) -> list[str]:
    """Allow forwarding FROM the testbed (Docker leaves the FORWARD policy at DROP)."""

    return ["iptables", action, "FORWARD", "-s", cidr, "-j", "ACCEPT"]


def _forward_return_rule(cidr: str, action: str) -> list[str]:
    """Allow established/related return traffic back TO the testbed."""

    return [
        "iptables",
        action,
        "FORWARD",
        "-d",
        cidr,
        "-m",
        "conntrack",
        "--ctstate",
        "ESTABLISHED,RELATED",
        "-j",
        "ACCEPT",
    ]


_FORWARD_RULES = (_forward_src_rule, _forward_return_rule)


def parse_resolvectl_dns(text: str) -> str:
    """Extract the first non-loopback IPv4 upstream DNS server from ``resolvectl status``.

    Prefers the active ``Current DNS Server:`` over the configured ``DNS Servers:``;
    skips the 127.0.0.53 stub. Restricted to IPv4 because the provisioning egress path
    (default route + NAT + FORWARD accepts) is IPv4-only — an auto-detected IPv6
    resolver would be unreachable; set ``provisioning_dns`` explicitly for IPv6. Returns
    "" if none found (the caller then fails loudly). Pure + unit-tested.
    """

    for prefix in ("Current DNS Server:", "DNS Servers:"):
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith(prefix):
                continue
            for token in stripped[len(prefix) :].replace(",", " ").split():
                try:
                    address = ipaddress.ip_address(token)
                except ValueError:
                    continue
                if address.version == 4 and not address.is_loopback:
                    return str(address)
    return ""


def _resolve_provisioning_dns(settings: LiveSettings, runner: RemoteRunner) -> str:
    """The provisioning resolver: the configured value, else the control host's upstream.

    A hardcoded public IP (e.g. 1.1.1.1) fails where external port 53 is blocked, so
    when ``provisioning_dns`` is blank we auto-detect the control host's own working
    upstream from ``resolvectl status``.
    """

    if settings.provisioning_dns:
        return settings.provisioning_dns
    result = runner.run(settings.control_host, ["resolvectl", "status"], timeout=_READ_TIMEOUT)
    dns = parse_resolvectl_dns(result.stdout)
    if not dns:
        raise ProvisioningError(
            "could not auto-detect an upstream DNS from `resolvectl status` on the "
            "control host; set live.provisioning_dns explicitly"
        )
    return dns


def open_egress(settings: LiveSettings, runner: RemoteRunner) -> None:
    """Open the temporary provisioning egress path (control NAT + node routes + DNS).

    Every command that changes network state runs as root (``sudo -n``). DNS is
    pointed at ``provisioning_dns`` for the window and resolution is verified, so
    ``apt``/``docker`` do not hang on name lookup. Raises ``ProvisioningError`` if a
    node cannot resolve after the resolver is set (fail loudly, not silently).
    """

    control = settings.control_host
    cidr = settings.testbed_cidr
    gateway = resolve_control_testbed_ip(settings)
    iface = settings.provisioning_iface
    dns = _resolve_provisioning_dns(settings, runner)

    # 1. Control host: enable forwarding + add the MASQUERADE rule (idempotent).
    runner.run(control, ["sysctl", "-w", f"{_SYSCTL}=1"], timeout=_READ_TIMEOUT, sudo=True)
    if runner.run(control, _nat_rule(cidr, "-C"), timeout=_READ_TIMEOUT, sudo=True).returncode != 0:
        runner.run(control, _nat_rule(cidr, "-A"), timeout=_READ_TIMEOUT, sudo=True)

    # 1b. Docker leaves the filter FORWARD policy at DROP + its own rules, so forwarded
    # testbed traffic is dropped even with ip_forward + NAT. Insert explicit accepts.
    for rule_fn in _FORWARD_RULES:
        if (
            runner.run(control, rule_fn(cidr, "-C"), timeout=_READ_TIMEOUT, sudo=True).returncode
            != 0
        ):
            runner.run(control, rule_fn(cidr, "-I"), timeout=_READ_TIMEOUT, sudo=True)

    # 2. Each non-control node: temporary default route + resolver, then verify DNS.
    for node in resolve_testbed_nodes(settings):
        runner.run(
            node,
            ["ip", "route", "replace", "default", "via", gateway],
            timeout=_READ_TIMEOUT,
            sudo=True,
        )
        runner.run(node, ["resolvectl", "dns", iface, dns], timeout=_READ_TIMEOUT, sudo=True)
        check = runner.run(
            node, ["getent", "hosts", settings.provisioning_dns_check_host], timeout=_READ_TIMEOUT
        )
        if check.returncode != 0:
            raise ProvisioningError(
                f"node {node} cannot resolve {settings.provisioning_dns_check_host} via "
                f"{dns} on {iface} — provisioning DNS is not working (apt/docker would hang)"
            )


def close_egress(settings: LiveSettings, runner: RemoteRunner) -> None:
    """Tear the temporary egress down (revert DNS + routes, drop NAT, forwarding off)."""

    control = settings.control_host
    cidr = settings.testbed_cidr
    gateway = resolve_control_testbed_ip(settings)
    iface = settings.provisioning_iface

    # 1. Each node: revert the resolver, flush the DNS cache (systemd-resolved keeps
    #    answering the check host from cache otherwise → spurious seal failure), and
    #    remove the temporary default route.
    for node in resolve_testbed_nodes(settings):
        runner.run(node, ["resolvectl", "revert", iface], timeout=_READ_TIMEOUT, sudo=True)
        runner.run(node, ["resolvectl", "flush-caches"], timeout=_READ_TIMEOUT, sudo=True)
        runner.run(
            node,
            ["ip", "route", "del", "default", "via", gateway],
            timeout=_READ_TIMEOUT,
            sudo=True,
        )

    # 2. Control host: drop the FORWARD accepts + the MASQUERADE rule (all matching)
    #    and turn forwarding off.
    for rule_fn in _FORWARD_RULES:
        while (
            runner.run(control, rule_fn(cidr, "-C"), timeout=_READ_TIMEOUT, sudo=True).returncode
            == 0
        ):
            runner.run(control, rule_fn(cidr, "-D"), timeout=_READ_TIMEOUT, sudo=True)
    while (
        runner.run(control, _nat_rule(cidr, "-C"), timeout=_READ_TIMEOUT, sudo=True).returncode == 0
    ):
        runner.run(control, _nat_rule(cidr, "-D"), timeout=_READ_TIMEOUT, sudo=True)
    runner.run(control, ["sysctl", "-w", f"{_SYSCTL}=0"], timeout=_READ_TIMEOUT, sudo=True)


def verify_sealed(settings: LiveSettings, runner: RemoteRunner):
    """Prove the testbed is sealed via the real isolation preflight; raise on any leak."""

    checker = SshRouteChecker(runner, dns_check_host=settings.provisioning_dns_check_host)
    return run_isolation_preflight(settings, checker)


@dataclass(frozen=True)
class DeployResult:
    """What a deploy did, for the operator log."""

    mode: str
    site_returncode: int
    sealed: bool


def _run_site(settings: LiveSettings, runner: RemoteRunner, inventory: str, site_playbook: str):
    # Point Ansible at the shipped lab ansible.cfg (host_key_checking off) via env so
    # the deploy does not depend on a hand-exported ANSIBLE_HOST_KEY_CHECKING.
    argv = [
        "env",
        f"ANSIBLE_CONFIG={_ANSIBLE_CFG}",
        "ansible-playbook",
        "-i",
        inventory,
        site_playbook,
    ]
    return runner.run(settings.control_host, argv, timeout=_SITE_TIMEOUT)


def deploy_sealed(
    settings: LiveSettings,
    runner: RemoteRunner,
    *,
    mode: str = "egress",
    inventory: str = "infra/ansible/inventory.ini",
    site_playbook: str = "infra/ansible/site.yml",
) -> DeployResult:
    """Deploy the testbed, guaranteeing it is sealed + verified before returning.

    ``mode`` = ``egress`` opens/closes the temporary provisioning egress around
    ``site.yml``; ``mode`` = ``mirror`` assumes the no-egress mirror on the monitoring
    node and opens nothing. Either way the seal is verified afterward (fail closed).
    """

    if mode not in {"egress", "mirror"}:
        raise ValueError(f"unknown provisioning mode {mode!r} (use 'egress' or 'mirror')")

    opened = mode == "egress"
    site_result = None
    try:
        # open_egress is INSIDE the try so a mid-open failure (e.g. DNS verify raising
        # after the FORWARD accepts + some node routes are already in place) still
        # re-seals in the finally — otherwise those accepts leave a live egress path.
        if opened:
            open_egress(settings, runner)
        site_result = _run_site(settings, runner, inventory, site_playbook)
    finally:
        # ALWAYS re-seal + verify, even if open_egress/site.yml raised or failed midway.
        if opened:
            close_egress(settings, runner)  # idempotent: safely unwinds a partial open
        verify_sealed(settings, runner)  # raises SafetyError on any leak (fail closed)

    if site_result is not None and site_result.returncode != 0:
        raise ProvisioningError(
            f"site.yml failed (rc={site_result.returncode}); testbed re-sealed. "
            f"stderr: {site_result.stderr.strip()[:200]}"
        )
    return DeployResult(mode=mode, site_returncode=site_result.returncode, sealed=True)
