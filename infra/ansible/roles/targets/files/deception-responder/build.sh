#!/usr/bin/env bash
# Build the deterministic deception responder image OFFLINE on the control host and
# save it as a docker-load tarball for the no-egress targets node.
#
# The image only needs the python base + pyyaml (fetched once on the control host,
# which has egress), so this never requires egress on the testbed nodes. Called by
# the deploy flow (scripts/deploy.sh) before site.yml stands the container up.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
INSTRUMENT_ROOT="$(cd "$HERE/../../../../../.." && pwd)"
if [ ! -f "$INSTRUMENT_ROOT/testbed/deception_responder.py" ]; then
  echo "ERROR: cannot locate testbed/deception_responder.py from $INSTRUMENT_ROOT" >&2
  exit 1
fi

TAG="mtd-testbed/deception-responder:0.1.0"
CTX="$(mktemp -d)"
trap 'rm -rf "$CTX"' EXIT
cp "$HERE/Dockerfile" "$CTX/Dockerfile"
cp "$INSTRUMENT_ROOT/testbed/deception_responder.py" "$CTX/deception_responder.py"

docker build -t "$TAG" "$CTX"

mkdir -p "$HERE/images"
OUT="$HERE/images/deception-responder-0.1.0.tar"
docker save "$TAG" -o "$OUT"
echo "built + saved: $TAG -> $OUT"
