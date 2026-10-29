# Lab Environment Setup Guide

## SENTINEL-AI — Isolated Lab for Safe Attack Simulation

> ⚠️ **IMPORTANT:** All attack simulation must be performed in an isolated virtual network. Never run attack tools against production, shared, or third-party networks.

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                  Host Machine (Windows)               │
│                                                       │
│  ┌─────────────┐         ┌─────────────┐             │
│  │ Attacker VM  │ ◄─────► │  Target VM   │            │
│  │ (Kali/Ubuntu)│         │ (Ubuntu)     │            │
│  │              │         │              │            │
│  │ Tools:       │  vboxnet│ Services:    │            │
│  │ - Nmap       │  (host- │ - Scapy      │            │
│  │ - hping3     │  only)  │   Sniffer    │            │
│  │ - Metasploit │         │ - SENTINEL   │            │
│  │   (optional) │         │   Pipeline   │            │
│  └─────────────┘         └─────────────┘             │
│        192.168.56.101         192.168.56.102          │
│                                                       │
│  Network: Host-Only Adapter (vboxnet0 / 192.168.56.0/24)│
└──────────────────────────────────────────────────────┘
```

---

## 2. VirtualBox Setup

### 2.1 Install VirtualBox
- Download: https://www.virtualbox.org/wiki/Downloads
- Install with default settings (include networking components)

### 2.2 Create Host-Only Network
1. Open VirtualBox → File → Host Network Manager
2. Click "Create" to add a new host-only network
3. Configure:
   - IPv4 Address: `192.168.56.1`
   - IPv4 Network Mask: `255.255.255.0`
   - DHCP Server: Enabled (or assign static IPs)

---

## 3. Target VM Setup (Ubuntu Server)

### 3.1 Create VM
- **OS:** Ubuntu Server 22.04 LTS
- **RAM:** 4 GB (minimum 2 GB)
- **Disk:** 20 GB
- **Network:**
  - Adapter 1: NAT (for internet/package installation)
  - Adapter 2: Host-Only Adapter (vboxnet0)

### 3.2 Configure Static IP
```bash
# /etc/netplan/01-host-only.yaml
network:
  version: 2
  ethernets:
    enp0s8:  # Host-only adapter (check with `ip a`)
      addresses:
        - 192.168.56.102/24
```
```bash
sudo netplan apply
```

### 3.3 Install SENTINEL-AI Dependencies
```bash
# System packages
sudo apt update && sudo apt install -y python3.11 python3.11-venv python3-pip \
    libpcap-dev tcpdump iptables git

# Clone project
git clone <your-repo-url>
cd Sentinel-AI

# Python environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r ml/requirements.txt

# Verify Scapy capture works (requires root)
sudo python -c "from scapy.all import sniff; print('Scapy OK')"
```

### 3.4 Verify Capture
```bash
# Terminal 1: Start capture
sudo python -c "
from scapy.all import sniff, IP
def show(pkt):
    if IP in pkt: print(f'{pkt[IP].src} → {pkt[IP].dst}')
sniff(iface='enp0s8', prn=show, count=10)
"

# Terminal 2 (or from Attacker VM): Generate traffic
ping 192.168.56.102
```

---

## 4. Attacker VM Setup (Kali Linux or Ubuntu)

### 4.1 Create VM
- **OS:** Kali Linux 2024.x or Ubuntu 22.04
- **RAM:** 2 GB
- **Disk:** 15 GB
- **Network:** Host-Only Adapter only (vboxnet0) — no NAT needed

### 4.2 Configure Static IP
```bash
# Set IP address
sudo ip addr add 192.168.56.101/24 dev eth0
sudo ip link set eth0 up
```

### 4.3 Install Attack Tools
```bash
# On Kali: already installed
# On Ubuntu:
sudo apt update && sudo apt install -y nmap hping3 tcpdump
```

### 4.4 Verify Connectivity
```bash
ping 192.168.56.102  # Should reach Target VM
```

---

## 5. Attack Simulation Scenarios

### 5.1 Port Scan (Nmap)
```bash
# From Attacker VM:
# SYN scan (most common)
sudo nmap -sS 192.168.56.102

# Full port scan
sudo nmap -sS -p 1-65535 192.168.56.102

# Service detection
sudo nmap -sV 192.168.56.102

# OS detection
sudo nmap -O 192.168.56.102
```

### 5.2 SYN Flood (hping3)
```bash
# From Attacker VM:
# Basic SYN flood to port 80
sudo hping3 -S -p 80 --flood 192.168.56.102

# SYN flood with random source IPs (spoofing)
sudo hping3 -S -p 80 --flood --rand-source 192.168.56.102

# Controlled rate (100 packets/sec)
sudo hping3 -S -p 80 -i u10000 192.168.56.102
```

### 5.3 UDP Flood
```bash
sudo hping3 --udp -p 53 --flood 192.168.56.102
```

### 5.4 Benign Traffic Baseline
```bash
# Generate normal-looking traffic for false positive testing
# Ping
ping -c 100 192.168.56.102

# HTTP requests (if web server is running on target)
for i in $(seq 1 100); do
    curl -s http://192.168.56.102/ > /dev/null
    sleep 0.5
done

# SSH connections (if SSH is running)
ssh -o ConnectTimeout=2 user@192.168.56.102 echo "test" 2>/dev/null
```

---

## 6. Safety Checklist

| # | Check | Status |
|---|-------|--------|
| 1 | Both VMs use Host-Only networking (no bridged/NAT to real network) | ☐ |
| 2 | Attacker VM has NO route to the internet or your LAN | ☐ |
| 3 | Auto-mitigation (iptables) only targets the attacker VM's IP | ☐ |
| 4 | You have a "unblock" script ready in case you lock yourself out | ☐ |
| 5 | Host firewall allows VirtualBox host-only traffic | ☐ |
| 6 | You've tested basic ping between VMs before running attacks | ☐ |

---

## 7. Quick Reference

| Item | Value |
|------|-------|
| Host-Only Network | `192.168.56.0/24` |
| Target VM IP | `192.168.56.102` |
| Attacker VM IP | `192.168.56.101` |
| Host Machine IP | `192.168.56.1` |
| Capture Interface | `enp0s8` (target VM, host-only adapter) |
| Whitelist IPs | `192.168.56.1` (host), `127.0.0.1` |

---

## 8. Troubleshooting

| Issue | Solution |
|-------|----------|
| VMs can't ping each other | Check both are on the same host-only network; run `ip a` to verify IPs |
| Scapy capture requires root | Always run capture with `sudo`; or add `cap_net_raw` to Python binary |
| hping3 not found | `sudo apt install hping3` |
| iptables rule locked you out | SSH from host (192.168.56.1) and run `sudo iptables -F` to flush all rules |
| VirtualBox host-only network missing | File → Host Network Manager → Create; then assign to VM adapters |
