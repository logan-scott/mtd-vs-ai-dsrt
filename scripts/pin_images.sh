#!/usr/bin/env bash
# Pin the five public target images by @sha256:<digest> after their first pull, and
# bump vuln_profile_version to record the resolved set (reproducibility control).
#
# Run on the targets node (or any host that has just pulled the images) AFTER a
# deploy. It reads each image's resolved RepoDigest from the local Docker and prints
# the pinned `image:` lines to paste into docker-compose.targets.yml, plus a
# suggested vuln_profile_version bump. Pass --write to rewrite the compose file in
# place (a .bak is kept). Requires docker + the images present locally.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
COMPOSE="$ROOT/infra/ansible/roles/targets/files/docker-compose.targets.yml"
WRITE=0
[ "${1:-}" = "--write" ] && WRITE=1

# The five genuine target image references (must match the compose file). Re-run on
# the testbed after a pull to refresh the @sha256 pins in docker-compose.targets.yml.
IMAGES=(
  "vulnerables/web-dvwa"
  "bkimminich/juice-shop:v17.1.1"
  "webgoat/webgoat:v2023.8"
  "piesecurity/apache-struts2-cve-2017-5638"
)

echo "# Resolved digests (paste into docker-compose.targets.yml):"
for tag in "${IMAGES[@]}"; do
  digest="$(docker inspect --format '{{index .RepoDigests 0}}' "$tag" 2>/dev/null || true)"
  if [ -z "$digest" ]; then
    echo "  # WARNING: $tag not present locally (pull it first)"
    continue
  fi
  echo "  # was: $tag"
  echo "  image: $digest"
  if [ "$WRITE" -eq 1 ]; then
    # Replace the tag reference with the digest ref (keep any trailing comment).
    sed -i.bak "s#image: ${tag}.*#image: ${digest}#" "$COMPOSE"
  fi
done

STAMP="$(date -u +%Y%m%d 2>/dev/null || echo pinned)"
echo
echo "# Current pinned set is vp-0.2.0. If the digests above differ, bump"
echo "#   vuln_profile_version to vp-0.2.0+pinned-${STAMP} in testbed/roster.yaml,"
echo "#   testbed/profile.py, testbed/vulnerability_profile/README.md, and the config."
[ "$WRITE" -eq 1 ] && echo "# Rewrote $COMPOSE (backup at ${COMPOSE}.bak)."
