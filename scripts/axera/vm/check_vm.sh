#!/usr/bin/env bash
# Read-only preflight for the AXCL LXD VM: checks every way the card has gone
# missing or silently stopped answering, and prints the fix for each failure.
# Changes nothing. No sudo needed (the user must be in the `lxd` group).
#
#   bash scripts/axera/vm/check_vm.sh [vm-name] [--health]
#
# --health additionally runs axcl_session.health_check (native Relu, <= 1 LSB).
# Exit status: 0 if nothing FAILed (WARNs allowed), 1 otherwise.
set -uo pipefail
VM=axcl-vm
HEALTH=0
for a in "$@"; do
  case "$a" in
    --health) HEALTH=1 ;;
    -h|--help) sed -n 2,9p "$0"; exit 0 ;;
    *) VM=$a ;;
  esac
done
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
FAILS=0
ok()   { echo "  OK    $*"; }
warn() { echo "  WARN  $*"; }
fail() { echo "  FAIL  $*"; FAILS=$((FAILS + 1)); }
fix()  { echo "        fix: $*"; }

echo "host:"
# Never start the VM with the host stack loaded (host panic 2026-09-07, README).
if lsmod | grep -qE '^(axcl_host|ax_pcie_msg|ax_pcie_mmb|ax_pcie_host_dev|ax_pcie_p2p_rc)\b'; then
  fail "host AXCL driver stack is loaded"
  fix "sudo bash scripts/axera/vm/host_bind_vfio.sh"
else
  ok "host AXCL driver stack not loaded"
fi
if [ -f /etc/modprobe.d/blacklist-axcl-host.conf ]; then
  ok "host blacklist present"
else
  fail "/etc/modprobe.d/blacklist-axcl-host.conf missing"
  fix "sudo bash scripts/axera/vm/host_bind_vfio.sh"
fi

# The card sits behind Thunderbolt and can re-enumerate at a new address
# (63:00.0 -> 03:00.0 on 2026-10-24); the VM's pci device then points at nothing.
ADDR=$(lspci -D -d 1f4b:0650 | awk 'NR==1{print $1}')
if [ -z "$ADDR" ]; then
  fail "no 1f4b:0650 card on the host PCI bus"
  fix "reconnect the card (Thunderbolt), then re-run"
else
  ok "card at $ADDR"
  DRV=$(basename "$(readlink "/sys/bus/pci/devices/$ADDR/driver" 2>/dev/null)" 2>/dev/null)
  case "${DRV:-none}" in
    vfio-pci|none) ok "host driver: ${DRV:-none}" ;;
    *) fail "host driver for $ADDR is $DRV (expected vfio-pci or none)"
       fix "sudo bash scripts/axera/vm/host_bind_vfio.sh" ;;
  esac
fi
if ! lxc info "$VM" >/dev/null 2>&1; then
  fail "LXD VM $VM does not exist"; fix "bash scripts/axera/vm/create_vm.sh $VM"
  echo; echo "$FAILS check(s) failed"; exit 1
fi
VMADDR=$(lxc config device get "$VM" ax650 address 2>/dev/null)
if [ -n "$ADDR" ] && [ "$VMADDR" != "$ADDR" ]; then
  fail "VM device ax650 points at '$VMADDR', card is at $ADDR"
  fix "lxc config device set $VM ax650 address=$ADDR && lxc restart --force $VM"
elif [ -n "$ADDR" ]; then
  ok "VM device ax650 address matches"
fi

STATE=$(lxc list "$VM" -c s -f csv)
if [ "$STATE" != RUNNING ]; then
  warn "VM $VM is $STATE -- guest checks skipped"
  fix "lxc start $VM"
  echo; [ "$FAILS" -eq 0 ] && echo "all host checks passed" || echo "$FAILS check(s) failed"
  [ "$FAILS" -eq 0 ]; exit
fi

echo "guest ($VM):"
# One lxc exec; the guest script prints the same OK/WARN/FAIL lines. Login
# shell: axcl-smi's /usr/bin/axcl is only on PATH via /etc/profile.
GUEST=$(lxc exec "$VM" -- bash -ls <<'GUEST_EOF'
ok()   { echo "  OK    $*"; }
warn() { echo "  WARN  $*"; }
fail() { echo "  FAIL  $*"; }
fix()  { echo "        fix: $*"; }
RUN=$(uname -r)

# Unattended upgrades install new kernels; DKMS may then build against the
# *running* (old) kernel's headers and drop that into the new kernel's tree, so
# the module refuses to load after the reboot ("disagrees about version of
# symbol"). Check every installed kernel, not just the running one.
NEWEST=$(ls /boot/vmlinuz-* 2>/dev/null | sed 's|/boot/vmlinuz-||' | sort -V | tail -1)
for k in $(ls /boot/vmlinuz-* 2>/dev/null | sed 's|/boot/vmlinuz-||' | sort -V); do
  f=$(ls /lib/modules/$k/updates/dkms/ax_pcie_host_dev.ko* 2>/dev/null | head -1)
  tag=""; [ "$k" = "$RUN" ] && tag=" (running)"; [ "$k" = "$NEWEST" ] && tag="$tag (next boot)"
  if [ -z "$f" ]; then
    m="no AXCL modules for kernel $k$tag"
  else
    vm=$(modinfo -F vermagic "$f" | awk '{print $1}')
    if [ "$vm" = "$k" ]; then ok "AXCL modules built for $k$tag"; continue; fi
    m="AXCL modules in $k$tag were built for $vm"
  fi
  if [ -n "$tag" ]; then fail "$m"; else warn "$m"; fi
  [ "$k" = "$RUN" ] && fix "dkms build --force axclhost/<ver> -k $k && dkms install --force axclhost/<ver> -k $k" \
                     || fix "boot $k first (DKMS uses the running kernel's headers), then dkms build/install --force -k $k"
done

grep -qw 'iommu=pt' /proc/cmdline && ok "iommu=pt on guest cmdline" \
  || { fail "iommu=pt missing from guest cmdline (card gets unmapped IOVAs)"; fix "/etc/default/grub.d/99-axcl-iommu.cfg + update-grub + reboot"; }

DEV=$(lspci -D -d 1f4b:0650 | awk 'NR==1{print $1}')
if [ -z "$DEV" ]; then
  fail "card not visible in the guest"; fix "lxc restart --force <vm> (check the host section)"
else
  t=$(cat /sys/bus/pci/devices/$DEV/iommu_group/type 2>/dev/null)
  [ "$t" = identity ] && ok "guest IOMMU domain identity" || fail "guest IOMMU domain is '$t' (expected identity)"
fi

# The card's firmware must match the installed runtime: a V2.25.0 pac under the
# 3.6 runtime handshakes and sends heartbeats, but every control command fails
# "invalid response message payload" / 0x80300909 (2026-10-24).
PAC=/lib/firmware/axcl/ax650_card.pac
PKG=$(dpkg-query -W -f '${Version}' axclhost 2>/dev/null)
if [ ! -f "$PAC" ]; then
  fail "$PAC missing"
else
  PACV=$(head -c 4096 "$PAC" | strings -n 4 | grep -m1 -E '^V[0-9]+\.[0-9]+')
  if [ "$(echo "${PACV#V}" | cut -d. -f1,2)" = "$(echo "$PKG" | cut -d. -f1,2)" ]; then
    ok "card firmware $PACV matches axclhost $PKG"
  else
    fail "card firmware is $PACV but axclhost package is $PKG"
    fix "replace $PAC with the one from axclhost_${PKG}*.deb (dpkg-deb -x; only that file, keep the patched DKMS source)"
  fi
  if dpkg --verify axclhost 2>/dev/null | grep -q " $PAC\$"; then
    warn "$PAC differs from the installed package's checksum"
  fi
fi

if ! lsmod | grep -q '^axcl_host '; then
  fail "guest AXCL stack not loaded"
  fix "modprobe ax_pcie_host_dev && modprobe ax_pcie_msg && modprobe ax_pcie_mmb && modprobe axcl_host"
  exit 0
fi
s=$(cat /sys/module/ax_pcie_host_dev/parameters/slot_index_force 2>/dev/null)
[ "$s" = 3 ] && ok "slot_index_force=3" \
  || { fail "slot_index_force is '${s:-unset}' (heartbeats get discarded)"; fix "options ax_pcie_host_dev slot_index_force=3 in /etc/modprobe.d, reload the stack"; }

# `heartbeat_recv_thread ... timeout: 9000` is a per-heartbeat debug trace, not
# an error; "dead!" is the real failure. axcl-smi hangs on a dead card. Only a
# "dead!" after the last firmware load counts: the driver resets the card and
# reloads its firmware ("Firmware loading") after one, and a module reload does
# too, so an earlier "dead!" in the same boot is stale.
if dmesg | awk '/Firmware loading/{d=0} /dead!/{d=1} END{exit !d}'; then
  fail "driver declared the card dead (dmesg 'dead!')"
  fix "modprobe -r axcl_host ax_pcie_mmb ax_pcie_msg ax_pcie_p2p_rc ax_pcie_host_dev, then reload (README 'Use it')"
  exit 0
fi
SMI=$(timeout 30 axcl-smi 2>&1); rc=$?
if [ $rc -eq 124 ]; then
  fail "axcl-smi hung"
elif echo "$SMI" | grep -q 'invalid response message payload'; then
  fail "axcl-smi: invalid response message payload (firmware/runtime mismatch?)"
else
  name=$(echo "$SMI" | awk -F'|' '/^\| +0 /{print $2}' | awk '{print $2, $3}')
  case "$name" in
    ""|UNKNOWN*) fail "axcl-smi does not identify the card ('${name:-no card row}')" ;;
    *) ok "axcl-smi: card $name" ;;
  esac
fi
GUEST_EOF
)
echo "$GUEST"
FAILS=$((FAILS + $(echo "$GUEST" | grep -c '^  FAIL')))

if [ "$HEALTH" = 1 ] && [ "$FAILS" -eq 0 ]; then
  echo "device:"
  if out=$(cd "$REPO/scripts/axera" && AXCL_LXD_VM=$VM timeout 300 python3 -c '
import axcl_session as a
with a.AXSession() as s:
    print(a.health_check(s))' 2>&1); then
    ok "native Relu health check: $(echo "$out" | tail -1) LSB"
  else
    fail "health check: $(echo "$out" | tail -1)"
  fi
fi

echo
if [ "$FAILS" -eq 0 ]; then echo "all checks passed"; else echo "$FAILS check(s) failed"; fi
[ "$FAILS" -eq 0 ]
