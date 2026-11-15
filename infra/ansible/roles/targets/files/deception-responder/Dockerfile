# Deterministic deception responder image: mtd-testbed/deception-responder:0.1.0
#
# Built OFFLINE on the control host (see build.sh) and staged to the targets node as
# a `docker load` tarball. Serves the decoy banners from /etc/responder/responder.yml
# (bind-mounted); no real vulnerability. Reload the deceptive surface with SIGHUP.
# PIN: pin the base by @sha256:<digest> at first build (record in vuln_profile_version).
# The FROM line must carry NO trailing comment — the legacy Docker builder rejects it.
FROM python:3.11-slim

RUN pip install --no-cache-dir "pyyaml==6.0.2"

COPY deception_responder.py /app/deception_responder.py
WORKDIR /app

# Decoy ports the compose file publishes (http-admin, ssh-gateway, ftp-backup).
EXPOSE 22 2121 8090

# STOPSIGNAL SIGTERM (default); SIGHUP reloads the banner map without a restart.
ENTRYPOINT ["python", "deception_responder.py", "--config", "/etc/responder/responder.yml"]
