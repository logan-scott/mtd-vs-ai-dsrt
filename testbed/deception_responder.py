"""Deterministic deception responder for the decoy services.

Serves the plausible-but-false decoy services that EVERY condition presents (the
profile seeds them unconditionally; only ``deceptive_presentation`` and
``combined_mtd_deception`` rotate them, so elsewhere they are static deception) —
``http-admin`` (:8090), ``ssh-gateway`` (:22), ``ftp-backup`` (:2121) — with fixed,
config-driven banners and **no real vulnerability**. A courier can *reach* a decoy
(a connect probe succeeds) but can never gain access, exactly matching the offline
decoy outcome. This is the program packaged as ``mtd-testbed/deception-responder``
(see ``infra/ansible/roles/targets/files/deception-responder/Dockerfile``).

Two properties matter and are unit-tested:

- **Deterministic.** Banners are a pure function of the config: the same
  ``responder.yml`` always produces byte-identical banners. Canonical per-service
  defaults are baked in so the responder is robust to *either* the full initial
  ``responder.yml`` *or* the reduced one that ``reconfigure.sh`` rewrites (which
  keeps only port/endpoint) — both yield the same banners.
- **Reloadable.** ``SIGHUP`` reloads the config and rebinds, so ``reconfigure.yml``
  can rotate the deceptive surface without recreating the container.

The banner map lives in ``/etc/responder/responder.yml`` (bind-mounted). This
module authors no offensive capability — it only answers with static banners.
"""

from __future__ import annotations

import signal
import socketserver
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import yaml

# Canonical per-service defaults. Keyed by the observable service_name so that a
# reduced responder.yml (port/endpoint only) still yields identical banners.
_CANONICAL: dict[str, dict] = {
    "http-admin": {
        "proto": "http",
        "port": 8090,
        "endpoint": "/admin",
        "banner": "nginx/1.18.0 (admin-console)",
        "body": "<html><title>Admin Console</title><body>401 Unauthorized</body></html>",
    },
    "ssh-gateway": {
        "proto": "ssh",
        "port": 22,
        "endpoint": "",
        "banner": "SSH-2.0-OpenSSH_8.9p1 gateway",
        "body": "",
    },
    "ftp-backup": {
        "proto": "ftp",
        "port": 2121,
        "endpoint": "",
        "banner": "220 backup-ftp ready (vsFTPd 3.0.5)",
        "body": "",
    },
}


@dataclass(frozen=True)
class DecoySpec:
    """One decoy's serving parameters, resolved from config + canonical defaults."""

    service_name: str
    proto: str
    port: int
    endpoint: str
    banner: str
    body: str


def _infer_proto(service_name: str) -> str:
    prefix = service_name.split("-", 1)[0]
    return prefix if prefix in {"http", "ssh", "ftp"} else "tcp"


def resolve_decoy(raw: dict) -> DecoySpec:
    """Resolve one raw config entry against the canonical defaults (deterministic)."""

    name = raw["service_name"]
    base = _CANONICAL.get(name, {})
    proto = raw.get("proto") or base.get("proto") or _infer_proto(name)
    port = int(raw.get("port", base.get("port", 0)))
    endpoint = raw.get("endpoint", base.get("endpoint", ""))
    # A responder.yml banner override wins; else the canonical banner; else a generic one.
    banner = raw.get("banner") or base.get("banner") or f"{proto}-decoy"
    body = raw.get("body", base.get("body", ""))
    return DecoySpec(name, proto, port, endpoint, banner, body)


def load_decoys(config_path: str | Path) -> list[DecoySpec]:
    """Parse ``responder.yml`` into resolved, deterministically-ordered decoy specs."""

    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    specs = [resolve_decoy(raw) for raw in data.get("decoys", [])]
    return sorted(specs, key=lambda s: s.service_name)


def _http_handler(spec: DecoySpec):
    body = spec.body.encode("utf-8")

    class _Handler(BaseHTTPRequestHandler):
        server_version = spec.banner  # the fake Server: header
        sys_version = ""

        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
            self.send_response(401)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # silence
            return

    return _Handler


def _banner_handler(spec: DecoySpec):
    line = (spec.banner + "\r\n").encode("utf-8")

    class _Handler(socketserver.BaseRequestHandler):
        def handle(self):
            try:
                self.request.sendall(line)
            except OSError:
                pass

    return _Handler


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


@dataclass
class DeceptionResponder:
    """Binds one listener per decoy; reloadable, deterministic banners.

    ``bind_host`` defaults to all interfaces (the container publishes the ports);
    tests bind to 127.0.0.1. Decoy ports come from the config, so tests point the
    config at high ports to avoid needing privileges.
    """

    config_path: str | Path
    bind_host: str = "0.0.0.0"
    _servers: list[socketserver.TCPServer] = None  # type: ignore[assignment]
    _threads: list[threading.Thread] = None  # type: ignore[assignment]

    def __post_init__(self):
        self._servers = []
        self._threads = []

    def _handler_for(self, spec: DecoySpec):
        return _http_handler(spec) if spec.proto == "http" else _banner_handler(spec)

    def start(self) -> None:
        for spec in load_decoys(self.config_path):
            server = _Server((self.bind_host, spec.port), self._handler_for(spec))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self._servers.append(server)
            self._threads.append(thread)

    def stop(self) -> None:
        for server in self._servers:
            server.shutdown()
            server.server_close()
        self._servers, self._threads = [], []

    def reload(self, *_args) -> None:
        """Rebind from the (possibly rotated) config — the SIGHUP handler."""

        self.stop()
        self.start()

    def bound_ports(self) -> list[int]:
        return [s.server_address[1] for s in self._servers]

    def __enter__(self) -> DeceptionResponder:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - container entrypoint
    import argparse

    parser = argparse.ArgumentParser(description="Deterministic deception responder")
    parser.add_argument("--config", default="/etc/responder/responder.yml")
    parser.add_argument("--bind", default="0.0.0.0")
    args = parser.parse_args(argv)

    responder = DeceptionResponder(args.config, bind_host=args.bind)
    responder.start()
    signal.signal(signal.SIGHUP, responder.reload)  # rotate the deceptive surface
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    stop.wait()
    responder.stop()
    return 0


if __name__ == "__main__":  # pragma: no cover - container entrypoint
    raise SystemExit(main())
