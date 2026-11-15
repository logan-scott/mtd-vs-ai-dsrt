# Proxmox provider. Credentials come from a gitignored secrets file
# (secrets.auto.tfvars, generated from secrets.auto.tfvars.example) or from
# PROXMOX_VE_* environment variables. Never commit a token or password.

provider "proxmox" {
  endpoint  = var.proxmox_endpoint
  api_token = var.proxmox_api_token
  insecure  = var.proxmox_tls_insecure

  # SSH is used by the provider for a few operations (e.g., file uploads). The
  # node must be reachable from the machine running Terraform. Keep this within
  # the management path; the *testbed* network itself has no external egress.
  ssh {
    agent    = true
    username = var.proxmox_ssh_username
  }
}
