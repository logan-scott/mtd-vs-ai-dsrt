output "isolated_bridge" {
  description = "The internal-only bridge all testbed VMs attach to."
  value       = proxmox_network_linux_bridge.isolated.name
}

output "node_ips" {
  description = "Role -> static IPv4 address on the isolated testbed network."
  value       = { for k, v in var.nodes : k => "${var.isolated_subnet_prefix}.${v.ip_octet}" }
}

output "vm_ids" {
  description = "Role -> Proxmox VM id."
  value       = { for k, vm in proxmox_virtual_environment_vm.node : k => vm.vm_id }
}

output "control_vm_id" {
  description = "Proxmox VM id of the dual-homed control host."
  value       = proxmox_virtual_environment_vm.control.vm_id
}

output "control_lan_ip" {
  description = "SSH here (labadmin@) to operate the harness; LAN leg with internet."
  value       = var.control_lan_address
}

output "control_testbed_ip" {
  description = "Control host's address on the isolated testbed subnet."
  value       = "${var.isolated_subnet_prefix}.${var.control_ip_octet}"
}
