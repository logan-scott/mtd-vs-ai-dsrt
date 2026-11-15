# Controlled vsftpd 2.3.4 backdoor target (CVE-2011-2523): mtd-testbed/vsftpd-backdoor:0.1.0
#
# A single-service, intentionally-vulnerable RESEARCH TARGET for the isolated no-egress
# testbed — see vsftpd_backdoor_target.py. Built OFFLINE on the control host from THIS
# local, committed context (no upstream clone) and docker-load'd onto the targets node,
# like the deception responder. Serves FTP on :21; the ":)" backdoor opens a shell on :6200.
# PIN: pin the base by @sha256:<digest> at first build.
FROM python:3.11-slim

COPY vsftpd_backdoor_target.py /app/vsftpd_backdoor_target.py
WORKDIR /app

EXPOSE 21 6200

ENTRYPOINT ["python", "vsftpd_backdoor_target.py"]
