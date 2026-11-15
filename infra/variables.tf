# ---------------------------------------------------------------------------
# Connection / credentials (values in gitignored secrets.auto.tfvars)
# ---------------------------------------------------------------------------

variable "proxmox_endpoint" {
  description = "Proxmox VE API endpoint, e.g. https://10.5.0.200:8006/"
  type        = string
}

variable "proxmox_api_token" {
  description = "Scoped Proxmox API token in 'user@realm!tokenid=uuid' form (secret)."
  type        = string
  sensitive   = true
}

variable "proxmox_ssh_username" {
  description = "SSH username for provider file operations on the node."
  type        = string
  default     = "amethyst"
}

variable "proxmox_tls_insecure" {
  description = "Skip TLS verification for the Proxmox endpoint (self-signed certs)."
  type        = bool
  default     = true
}

# ---------------------------------------------------------------------------
# Placement / storage (defaults for the study's Proxmox node)
# ---------------------------------------------------------------------------

variable "node_name" {
  description = "Proxmox node to provision on."
  type        = string
  default     = "pve2"
}

variable "vm_datastore" {
  description = "Datastore for VM disks."
  type        = string
  default     = "data2"
}

variable "image_datastore" {
  description = "Datastore holding the cloud image / templates."
  type        = string
  default     = "software-images"
}

variable "snippets_datastore" {
  description = "Datastore that has the 'snippets' content type enabled (cloud-init user-data)."
  type        = string
  default     = "data2"
}

# ---------------------------------------------------------------------------
# Base image (pinned; control variable)
# ---------------------------------------------------------------------------

variable "ubuntu_image_url" {
  description = "URL of the Ubuntu 22.04 LTS cloud image (pinned release)."
  type        = string
  default     = "https://cloud-images.ubuntu.com/releases/22.04/release-20240821/ubuntu-22.04-server-cloudimg-amd64.img"
}

variable "ubuntu_image_filename" {
  description = "Local filename for the imported cloud image. Must end in .qcow2: the 'import' content role rejects .img, and the Ubuntu cloud .img is qcow2-format, so this rename is correct."
  type        = string
  default     = "ubuntu-22.04-server-cloudimg-amd64.qcow2"
}

# ---------------------------------------------------------------------------
# Isolated network (NO uplink / NO external egress)
# ---------------------------------------------------------------------------

variable "isolated_bridge" {
  description = "Name of the internal-only Linux bridge (no physical ports => no egress)."
  type        = string
  default     = "vmbr9"
}

# ---------------------------------------------------------------------------
# Control host (dual-homed: testbed leg + LAN/internet leg). Only VM with egress.
# ---------------------------------------------------------------------------

variable "control_lan_bridge" {
  description = "Existing Proxmox bridge that provides LAN/internet access for the control VM's second NIC."
  type        = string
  default     = "vmbr0"
}

variable "control_lan_address" {
  description = "Static IPv4 (CIDR) for the control VM's LAN leg. Confirm it is FREE on your LAN before applying; this is your SSH target."
  type        = string
  default     = "10.5.0.50/24"
}

variable "control_lan_gateway" {
  description = "Default gateway for the control VM's LAN leg (your LAN router)."
  type        = string
  default     = "10.5.0.1"
}

variable "control_ip_octet" {
  description = "Last octet of the control VM's static address on the isolated testbed subnet."
  type        = number
  default     = 5
}

variable "control_cores" {
  description = "vCPU cores for the control VM."
  type        = number
  default     = 2
}

variable "control_memory_mb" {
  description = "Memory (MiB) for the control VM."
  type        = number
  default     = 4096
}

variable "control_disk_gb" {
  description = "Disk (GiB) for the control VM."
  type        = number
  default     = 32
}

variable "isolated_cidr_prefix" {
  description = "IPv4 prefix length for the isolated testbed subnet."
  type        = number
  default     = 24
}

variable "isolated_subnet_prefix" {
  description = <<-EOT
    First three octets of the isolated /24 (hosts assigned .10+). MUST NOT overlap
    any network you actually use — the fail-closed testbed_cidr allowlist can only
    keep offensive actions off production if the testbed range is unique. The study's
    lab already used 10.0/10.10/10.20; the default deliberately sits in an unusual block.
    Confirm this is free on your network and override in terraform.tfvars if not.
  EOT
  type        = string
  default     = "172.31.66"
}

# ---------------------------------------------------------------------------
# Access
# ---------------------------------------------------------------------------

variable "ssh_public_keys" {
  description = "SSH public keys injected into every VM via cloud-init."
  type        = list(string)
  default     = []
}

variable "ci_username" {
  description = "Default cloud-init user created on every VM."
  type        = string
  default     = "labadmin"
}

# ---------------------------------------------------------------------------
# Node roles + sizing (within the pve2 envelope: ~12 vCPU / 48 GiB / 300 GiB)
# ---------------------------------------------------------------------------

variable "nodes" {
  description = "Testbed nodes to provision: role -> sizing + last IP octet."
  type = map(object({
    cores     = number
    memory_mb = number
    disk_gb   = number
    ip_octet  = number
    tags      = list(string)
  }))
  default = {
    targets    = { cores = 4, memory_mb = 8192, disk_gb = 40, ip_octet = 10, tags = ["testbed", "targets"] }
    attacker   = { cores = 4, memory_mb = 8192, disk_gb = 40, ip_octet = 20, tags = ["testbed", "attacker"] }
    mtd        = { cores = 2, memory_mb = 4096, disk_gb = 20, ip_octet = 30, tags = ["testbed", "mtd"] }
    monitoring = { cores = 2, memory_mb = 4096, disk_gb = 40, ip_octet = 40, tags = ["testbed", "monitoring"] }
  }
}
