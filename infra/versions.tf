terraform {
  required_version = ">= 1.6.0"

  required_providers {
    proxmox = {
      source = "bpg/proxmox"
      # Pinned to the validated version for reproducibility.
      version = "0.111.0"
    }
  }
}
