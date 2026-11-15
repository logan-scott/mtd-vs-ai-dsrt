#!/usr/bin/env bash
# Build the single-service vsftpd 2.3.4 backdoor target (CVE-2011-2523) OFFLINE on the
# control host and save it as a docker-load tarball for the no-egress targets node — the
# same pattern as the deception responder.
#
# WHY NOT metasploitable2 / vulhub: tleemcjr/metasploitable2 is a full-SYSTEM image that
# boots many services and crash-loops under plain `docker compose`; the vulhub
# CVE-2011-2523 build context was removed upstream. This builds from the LOCAL, committed
# context here (Dockerfile + vsftpd_backdoor_target.py) — a controlled reproduction of the
# CVE's observable behavior (banner on :21, ":)" backdoor shell on :6200). No upstream
# clone, so it cannot break on a moved/deleted remote path.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TAG="mtd-testbed/vsftpd-backdoor:0.1.0"

docker build -t "$TAG" "$HERE"

mkdir -p "$HERE/images"
OUT="$HERE/images/vsftpd-backdoor-0.1.0.tar"
docker save "$TAG" -o "$OUT"
echo "built + saved: $TAG -> $OUT (serves :21; backdoor shell on :6200)"
