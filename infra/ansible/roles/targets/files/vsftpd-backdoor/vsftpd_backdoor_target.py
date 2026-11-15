#!/usr/bin/env python3
"""Controlled reproduction of the vsftpd 2.3.4 backdoor (CVE-2011-2523) — a single-service
intentionally-vulnerable target for the isolated, no-egress MTD testbed.

This plays the same role DVWA / WebGoat / Struts play in the study: a researcher-owned
vulnerable target that the pre-staged ``vsftpd_backdoor`` primitive exploits (obj_b). It
replaces ``tleemcjr/metasploitable2`` (a full-system image that boots many services and
crash-loops under plain ``docker compose``) and the vulhub build (whose context was removed
upstream). It reproduces the CVE's OBSERVABLE behavior deterministically:

  - serves the FTP greeting ``220 (vsFTPd 2.3.4)`` on :21;
  - when a ``USER`` command contains the ``:)`` smiley (the historical backdoor trigger), it
    OPENS a shell on :6200 — exactly as the compromised 2.3.4 source did — that runs commands
    in THIS container and returns their output.

Faithful-behavior note (why :6200 is *bound on trigger*, not always): the real backdoor's
:6200 listener does not exist until the ``:)`` sequence opens it, and the maintained
Metasploit module ``exploit/unix/ftp/vsftpd_234_backdoor`` RELIES on that — its check
connects to :6200 first and aborts ("backdoor bind listener already open/in-use") if the
port is already listening. So this target keeps :6200 CLOSED (connection refused) until a
``:)`` arms it, then binds a real ``/bin/sh`` on the socket. That is both more faithful to
the CVE (a scanner sees :6200 only after compromise) and what lets the standard module drive
it to obj_b.

This authors no attacker capability: it is a *target*, and the shell only ever runs inside
this throwaway, no-egress container on 172.31.66.0/24 (structural isolation + the fail-closed
isolation preflight). It is a behavior reproduction, not the vsftpd binary. Unit-tested
offline like the deception responder.
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import threading

BANNER = b"220 (vsFTPd 2.3.4)\r\n"
_BACKDOOR_TRIGGER = ":)"


class VsftpdBackdoorTarget:
    """Serves the FTP banner on ``ftp_port``; arms a shell on ``backdoor_port`` on ``:)``.

    ``bind_host`` defaults to all interfaces (the container publishes the ports); tests bind
    to 127.0.0.1 on high ports. The backdoor port is NOT bound until a ``USER ... :)`` arms
    it, so it refuses connections (like the real CVE) until then.
    """

    def __init__(
        self, bind_host: str = "0.0.0.0", ftp_port: int = 21, backdoor_port: int = 6200
    ) -> None:
        self.bind_host = bind_host
        self.ftp_port = ftp_port
        self.backdoor_port = backdoor_port
        self._armed = threading.Event()
        self._arm_lock = threading.Lock()
        self._backdoor_srv: socket.socket | None = None
        self._servers: list[socket.socket] = []

    def _accept_loop(self, srv: socket.socket, handler) -> None:
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return  # socket closed by stop()
            threading.Thread(target=handler, args=(conn,), daemon=True).start()

    def _ftp_session(self, conn: socket.socket) -> None:
        try:
            conn.sendall(BANNER)
            conn.settimeout(15)
            while True:
                data = conn.recv(256)
                if not data:
                    break
                line = data.decode(errors="replace")
                verb = line.strip().upper()
                if verb.startswith("USER"):
                    if _BACKDOOR_TRIGGER in line:
                        self._arm_backdoor()  # open the shell listener on :6200
                    conn.sendall(b"331 Please specify the password.\r\n")
                elif verb.startswith("PASS"):
                    conn.sendall(b"230 Login successful.\r\n")
                elif verb.startswith("QUIT"):
                    conn.sendall(b"221 Goodbye.\r\n")
                    break
                else:
                    conn.sendall(b"530 Please login with USER and PASS.\r\n")
        except OSError:
            pass
        finally:
            conn.close()

    def _arm_backdoor(self) -> None:
        """Bind the :6200 shell listener the first time a ``:)`` trigger is seen (idempotent).

        The listener is SINGLE-SHOT: it accepts one connection and then closes (see
        ``_backdoor_accept_once``), so :6200 refuses again afterwards. The study exploits this
        target repeatedly (once per trial) and the standard Metasploit module's check aborts if
        :6200 is already open, so each trigger→exploit cycle must start from the closed state —
        exactly as the real one-shot CVE behaves."""

        with self._arm_lock:
            if self._armed.is_set():
                return
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.bind_host, self.backdoor_port))
            srv.listen(16)
            self._backdoor_srv = srv
            self._servers.append(srv)
            threading.Thread(target=self._backdoor_accept_once, args=(srv,), daemon=True).start()
            self._armed.set()

    def _disarm_backdoor(self, srv: socket.socket) -> None:
        """Close the :6200 listener so the port refuses again until the next ``:)`` re-arms it."""

        with self._arm_lock:
            # Close the listener while still holding the lock and clearing _armed, so there is
            # no window where _armed reads clear but the old listener is still bound (which a
            # concurrent _arm_backdoor could hit with EADDRINUSE).
            self._armed.clear()
            if self._backdoor_srv is srv:
                self._backdoor_srv = None
            if srv in self._servers:
                self._servers.remove(srv)
            try:
                srv.close()
            except OSError:
                pass

    def _backdoor_accept_once(self, srv: socket.socket) -> None:
        """Accept a single backdoor connection, close the listener, then serve the shell."""

        try:
            conn, _ = srv.accept()
        except OSError:
            return  # listener closed by stop()
        self._disarm_backdoor(srv)  # one-shot: :6200 refuses again after this connection
        self._backdoor_session(conn)

    def _backdoor_session(self, conn: socket.socket) -> None:
        """Wire a real ``/bin/sh`` to the socket — exactly what the CVE-2011-2523 backdoor did.

        The compromised vsftpd 2.3.4 ``execl``'d ``/bin/sh`` on the :6200 socket, giving a
        persistent root shell. Reproducing that faithfully (rather than a per-command loop) is
        what lets the maintained Metasploit module drive this target end-to-end: it reads
        commands from the socket and writes their output back, so the module's command-shell
        payload — and any command run on the resulting session — behaves as against the real
        bug. The shell stays alive until the socket closes.
        """

        try:
            fd = conn.fileno()
            proc = subprocess.Popen(  # noqa: S603,S607 - intentional: this IS the backdoor
                ["/bin/sh"], stdin=fd, stdout=fd, stderr=fd
            )
            proc.wait()
        except (OSError, ValueError):  # pragma: no cover - defensive
            pass
        finally:
            conn.close()

    def start(self) -> None:
        # Bind the FTP port synchronously so it is listening the moment start() returns; the
        # backdoor port stays CLOSED until a ":)" arms it (see _arm_backdoor).
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.bind_host, self.ftp_port))
        srv.listen(16)
        self._servers.append(srv)
        threading.Thread(
            target=self._accept_loop, args=(srv, self._ftp_session), daemon=True
        ).start()

    def stop(self) -> None:
        for srv in self._servers:
            try:
                srv.close()
            except OSError:
                pass
        self._servers = []
        self._backdoor_srv = None
        self._armed.clear()

    def __enter__(self) -> VsftpdBackdoorTarget:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - container entrypoint
    parser = argparse.ArgumentParser(description="vsftpd 2.3.4 backdoor target (CVE-2011-2523)")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--ftp-port", type=int, default=21)
    parser.add_argument("--backdoor-port", type=int, default=6200)
    args = parser.parse_args(argv)
    target = VsftpdBackdoorTarget(
        bind_host=args.bind, ftp_port=args.ftp_port, backdoor_port=args.backdoor_port
    )
    target.start()
    threading.Event().wait()  # run forever
    return 0


if __name__ == "__main__":  # pragma: no cover - container entrypoint
    raise SystemExit(main())
