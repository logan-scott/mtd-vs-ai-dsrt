"""The canonical initial configuration state (approved target set + decoys).

This is the offline representation of the version-pinned vulnerability profile
confirmed in ``testbed/vulnerability_profile/README.md``: five genuine targets
(DVWA, OWASP Juice Shop, WebGoat, Vulhub/Struts2 S2-045, Metasploitable/vsftpd)
plus three decoys, which ALL SIX conditions carry (they are seeded here
unconditionally and no technique removes them — only `deceptive_presentation` and
`combined_mtd_deception` rotate them, so `static_baseline` is *static deception*,
not a no-deception control). The genuine services' stable identity
and vulnerability profile are held constant across all six conditions (verified by
``profile_checksum``); MTD techniques mutate only their observable attributes.
"""

from __future__ import annotations

from mtd.models import ConfigState, ServiceConfig

DEFAULT_PROFILE_VERSION = "vp-0.2.0"  # digest-pinned, validated image set (see testbed/roster.yaml)


def default_profile(version: str = DEFAULT_PROFILE_VERSION) -> ConfigState:
    """Build the standardized initial configuration state."""

    services = (
        ServiceConfig(
            "dvwa", "http-dvwa", "10.10.0.10", 80, "/login.php", True, "cmd-injection", "dvwa-1.x"
        ),
        ServiceConfig(
            "juice",
            "http-juice",
            "10.10.0.11",
            3000,
            "/rest/products",
            True,
            "sqli",
            "juice-17.1.1",
        ),
        ServiceConfig(
            "webgoat",
            "http-webgoat",
            "10.10.0.12",
            8080,
            "/WebGoat/login",
            True,
            "xxe",
            "webgoat-2023.8",
        ),
        ServiceConfig(
            "struts",
            "http-struts",
            "10.10.0.13",
            8081,
            "/struts2/index.action",
            True,
            "CVE-2017-5638",
            "struts-2.5.10",
        ),
        ServiceConfig(
            "vsftpd", "ftp-vsftpd", "10.10.0.14", 21, "", True, "CVE-2011-2523", "vsftpd-2.3.4"
        ),
        # Decoys (genuine=False). Seeded UNCONDITIONALLY, so every condition —
        # static_baseline included — presents 5 genuine services and these 3 decoys.
        # Only deceptive_presentation / combined_mtd_deception mutate them.
        ServiceConfig("decoy_http", "http-admin", "10.10.0.90", 8090, "/admin", False),
        ServiceConfig("decoy_ssh", "ssh-gateway", "10.10.0.91", 22, "", False),
        ServiceConfig("decoy_ftp", "ftp-backup", "10.10.0.92", 2121, "", False),
    )
    return ConfigState(version=version, services=services)
