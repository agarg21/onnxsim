# Running the AX650N inside an LXD virtual machine

The AXCL host driver is an out-of-tree kernel module, and its faults have taken
this host down (see `../host-driver-patches/NOTES.md`). Putting the card into a
KVM guest via VFIO passthrough bounds the blast radius: a driver fault kills the
guest, and `lxc restart --force <vm>` bus-resets the card without a host reboot.
This must be an LXD **virtual machine** (`--vm`); a container shares the host
kernel and gives no protection.

Everything that only *compiles* (Docker, `pulsar2 build`/`llm_build`) stays on
the host. Only `axcl_run_model` moves into the guest, and the harness routes
it there transparently when `AXCL_LXD_VM=<vm name>` is set:
`pulsar2_docker.axcl_available()`, `run_on_device()` and
`run_on_device_with_inputs()` push the `.axmodel` and input folders with
`lxc file push`, run the tool with `lxc exec`, and pull outputs back. Every
existing device test runs unchanged.

## One-time host prerequisites (sudo)

1. The IOMMU must be on. This host shipped with `amd_iommu=off` on the kernel
   command line; removing it (optionally adding `iommu=pt`) and rebooting gave
   38 IOMMU groups, with the card in a group of its own plus its ASMedia
   downstream bridge -- the friendly case (bridges stay on the host). Check:
   `ls /sys/kernel/iommu_groups/*/devices/ | grep 03:00.0`.
2. Hand the card to `vfio-pci` and keep the host from loading the AXCL stack:
   `sudo bash scripts/axera/vm/host_bind_vfio.sh` (`--undo` reverses it). It
   writes `modprobe.d` overrides, rebuilds the initramfs, and tries to rebind
   immediately; a reboot makes it certain.

**Why the host can no longer run the card once the IOMMU is on** (confirmed
2026-09-07): a Thunderbolt-attached device is treated as untrusted, so the
kernel keeps it in a translated `DMA` domain even with `iommu=pt`
(`/sys/bus/pci/devices/0000:03:00.0/iommu_group/type` reads `DMA`), and the
AXCL driver hands the card raw host physical addresses for its shared memory
-- the card's DMA then faults (`AMD-Vi: Event logged [IO_PAGE_FAULT
domain=0x0003 ...]` from `ax_pcie_dev_host 0000:03:00.0`) and `axcl-smi`
hangs. That is almost certainly why `amd_iommu=off` was on the command line.
Inside a VFIO guest the same driver works, because VFIO installs the guest's
memory map in the IOMMU and the addresses the guest driver hands the card are
exactly the ones mapped. That was fixed the same day by driver fix 5 (`../host-driver-patches/`,
`fix_axcl_iommu_p5.sh`): every card-visible buffer is now allocated and
mapped against the card's own `pci_dev`, and the host runs the card with the
IOMMU on and zero faults. The VM is therefore optional again -- its value is
containment of driver faults, not access to the card.

## Create the guest (no sudo)

```sh
bash scripts/axera/vm/create_vm.sh            # ubuntu:24.04, 8 cores, 16 GiB, the card, /mnt/share
lxc exec axcl-vm -- bash /mnt/share/guest_install_axcl.sh   # deb + DKMS + the four driver fixes
lxc exec axcl-vm -- axcl-smi
```

`create_vm.sh` stages the `axclhost` deb (`AXCL_DEB=` to point at it), the
`host-driver-patches/` tree and the guest installer into
`scripts/axera/vm/share/` (git-ignored), mounted at `/mnt/share` in the guest.
The guest installer registers the driver with DKMS exactly as on the host and
applies the four fixes with `patch -p1`, so a Thunderbolt hiccup inside the
guest is survivable there too.

## Use it

Before device work (and whenever the card stops answering), run the
read-only preflight; it checks each failure mode below and prints the fix:

```sh
bash scripts/axera/vm/check_vm.sh [vm-name] [--health]
```

It catches: the host AXCL stack loaded or the blacklist missing; the card
re-enumerated to a new PCI address the VM's `ax650` device no longer points at
(Thunderbolt, `63:00.0` -> `03:00.0` on 2026-10-24); guest AXCL modules built
for the wrong kernel (DKMS uses the *running* kernel's headers, so a module
built during an unattended kernel upgrade fails to load after the reboot with
"disagrees about version of symbol"); missing guest `iommu=pt` / non-identity
IOMMU domain; card firmware (`/lib/firmware/axcl/ax650_card.pac`) from a
different AXCL release than the installed `axclhost` -- a V2.25.0 pac under the
3.6 runtime handshakes and heartbeats fine but every command fails `invalid
response message payload` / `0x80300909`, and the card shows as `UNKNOWN`;
`slot_index_force` unset; a dead card. `--health` also runs the native Relu
health check. Note `heartbeat_recv_thread ... timeout: 9000` in the guest
`dmesg` is a per-heartbeat debug trace, not an error.

```sh
export AXCL_LXD_VM=axcl-vm
python -m pytest tests/test_axera_mcode_structure.py -k on_device
```

If the guest driver wedges: `lxc restart --force axcl-vm` (the card gets a
secondary bus reset on the way), then re-run. The host is never involved.

If the card's firmware died (guest `dmesg` repeats `device 3: dead!` and
`axcl-smi` lists no card -- seen after mcode fault-injection sweeps),
reloading `axcl_host` alone is not enough: `ax_pcie_host_dev` keeps stale
transfer handles (`transfer handle ... already exists!`). Reload the whole
stack in the guest, in `guest_install_axcl.sh`'s order:

```sh
modprobe -r axcl_host ax_pcie_mmb ax_pcie_msg ax_pcie_p2p_rc ax_pcie_host_dev
modprobe ax_pcie_host_dev && modprobe ax_pcie_msg && modprobe ax_pcie_mmb && modprobe axcl_host
```

`ax_pcie_p2p_rc` also holds `ax_pcie_host_dev`: leave it out of the `-r`
list and `ax_pcie_host_dev` stays loaded ("Module ax_pcie_host_dev is in
use"), the card keeps its stale state and is still dead after the reload.
Do not poll with `axcl-smi` while the card is dead -- it hangs. If the
firmware push then times out (`Handshake timeout!` after ~360 s), run the
same reload once more; the second one came up in 5 s (2026-09-24).

then wait ~30 s and confirm the card in `axcl-smi` plus a control model run
before any further device work.

## A host crash to never repeat (2026-09-07 19:39, kdump `202609071939`)

Starting the VM while the host AXCL modules were still loaded (the blacklist
had been undone to test driver fix 5) took the host down. LXD steals the card
with a per-device `driver_override`; when the guest driver reset the card's
SoC, the card re-enumerated, the override went with the old device instance,
the host's `ax_pcie_dev_host` probed the new one, and driver fix 4's automatic
bring-up pushed firmware into a card the guest was driving -- panic in
`axcl_firmware_load` (`axcl_pcie_device_online_work`). `create_vm.sh` now
refuses to run unless the stack is unloaded *and* the blacklist file exists.
The guest side has a rule too: a VFIO bus reset does not reset the card's
SoC, only the driver's unload path does, so after `lxc restart --force`
always `modprobe -r axcl_host && modprobe axcl_host` inside the guest before
expecting a firmware push to succeed.

## Status: the card runs inside the guest (2026-09-07)

An `.axmodel` compiled on the host now runs on the AX650N inside the VM, and
the repo's own device tests pass through it unchanged
(`test_llm_build_a7_is_a_sync_verb_on_device` via `AXCL_LXD_VM=axcl-vm`,
110 s, hand-patched mcode variants and all). Three things had to be fixed:

1. **The guest's virtual IOMMU was remapping the card's DMA.** The
   `intel-iommu,intremap=on` device needed for interrupt remapping also turns
   on DMA remapping, and the card's group came up as a translated `DMA`
   domain: the guest driver handed it IOVAs VFIO had never mapped, visible on
   the host as `vfio-pci ... IO_PAGE_FAULT address=0xffc00000` (the IOVA
   allocator works down from the top). Adding `iommu=pt` to the *guest's*
   command line gives the card an `identity` domain, and the faults stop.
   `create_vm.sh` does not set this for you; it lives in
   `/etc/default/grub.d/99-axcl-iommu.cfg` in the guest.
2. **A VFIO bus reset does not reset the card's SoC.** Only the driver's
   unload path does (`start reset slave`), so after `lxc restart --force`
   always `modprobe -r axcl_host && modprobe axcl_host` in the guest before
   expecting a firmware push to succeed.
3. **The driver rejected the card's heartbeats** because it compares the id
   the card reports (a fixed 3) against `pdev->bus->number`, which is 3 on
   this host by coincidence and 7 in the guest. Driver fix 7 adds
   `slot_index_force`; load the stack in the guest with
   `modprobe ax_pcie_host_dev slot_index_force=3`. Without it the card boots,
   handshakes and sends heartbeats that are all discarded, and is declared
   dead after 50 s -- which is what made this look like a device-side problem
   for so long.

With those in place: `[ax_pcie_dev_probe]: slot index pinned to 3 (bus 7)`,
no missed heartbeats, `axcl-smi` listing the card with live temperature and
utilisation, and device runs at the same latency as on the host.

## Known limits

- A Thunderbolt link drop while the card is assigned reaches the *guest*
  driver as a hot-unplug -- exactly the path fix 3 hardens -- and VFIO on the
  host handles the surprise removal; untested until it happens.
- `lxc file push` adds ~0.1 s per run for a small model and scales with the
  weight table (a 12 MB resnet18d `.axmodel` is fine; a 250 MB LLM head is
  noticeable). Keep large models in `/mnt/share` and pass that path when it
  matters.

## Many small models per step: plan mode

`AXSession.run` costs about 1.0-1.7 ms of host round trip per call (input and output files on the
share, one command line), on top of about 0.3 ms of device time for a small model. For a chain of
hundreds of small models per step, use plan mode (`axcl_batch_runner.c` `BUF` / `BIND` / `PLAN`):

```python
from axcl_session import AXSession, Plan

with AXSession() as s:
    a, b = s.load(model_a), s.load(model_b)
    x, mid, y = s.buffer(nx), s.buffer(nmid), s.buffer(ny)      # device buffers
    s.bind(a, "I", 0, x); s.bind(a, "O", 0, mid)                 # a writes mid ...
    s.bind(b, "I", 0, mid); s.bind(b, "O", 0, y)                 # ... b reads it: no copy
    plan = Plan(s)
    plan.write(x, inputs).execute(a).execute(b)
    key = plan.read(y, ny)
    s.run_plan(plan)                                             # one command
    out = plan.result(key, np.float32, shape)
```

A binding can point at an offset inside a buffer, so slices of a tensor (for example one attention
head's row) need no copy; `Plan.bind` rebinds inside a plan, and `Plan.copy` does a device-to-device
copy (about 0.16 ms). Inside a plan, a small model costs only its device time (about 0.27 ms). The
SmolLM2-135M Pulsar-free decode (about 1900 model runs per token) went from 4.3 s/token with `run`
to 0.6 s/token with one plan per token. `tests/test_axcl_plan_hardware.py` checks it on the card.
