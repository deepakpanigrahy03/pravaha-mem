"""Find the SMMU counter (PMCG) that sees the GB10 GPU's copy-engine traffic.

How it works, in plain steps:

1. Find the GPU in sysfs: a PCI device from NVIDIA (vendor 0x10de) whose
   class is "display controller" (0x03....).
2. Ask sysfs which IOMMU (address translator) serves that GPU. Its name ends
   in the SMMU's physical address, e.g. smmu3.0x0000000013000000.
3. Linux names each SMMU counter unit smmuv3_pmcg_<page>, where page is its
   register address divided by 4096. The units whose address lies between
   this SMMU and the next SMMU belong to the GPU's SMMU. On the tested GB10
   there are several, and only one counts copies.
4. To pick the right one, measure: count all candidates while idle and while
   the GPU copies data, and keep the one that rises (``confirm`` in cli.py).
   The choice is saved, so this is done once per machine.

Everything takes a ``sysfs`` argument so the tests can use a fake tree.
"""

import os
import re

NVIDIA_VENDOR = "0x10de"
PMCG_PREFIX = "smmuv3_pmcg_"


class DiscoveryError(RuntimeError):
    """Raised when the GPU, its SMMU or a counter unit cannot be found."""


def _read(path):
    with open(path) as f:
        return f.read().strip()


def _address_from_name(name):
    """'smmu3.0x0000000013000000' -> 0x13000000, or None if there is no address."""
    m = re.search(r"0x([0-9a-fA-F]+)$", name)
    return int(m.group(1), 16) if m else None


def find_gpus(sysfs="/sys"):
    """PCI addresses (e.g. '000f:01:00.0') of all NVIDIA display devices."""
    root = os.path.join(sysfs, "bus", "pci", "devices")
    found = []
    for bdf in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        dev = os.path.join(root, bdf)
        try:
            vendor = _read(os.path.join(dev, "vendor")).lower()
            klass = _read(os.path.join(dev, "class")).lower()
        except OSError:
            continue
        if vendor == NVIDIA_VENDOR and klass.startswith("0x03"):
            found.append(bdf)
    return found


def smmu_of(bdf, sysfs="/sys"):
    """Physical base address of the SMMU that translates for this PCI device."""
    link = os.path.join(sysfs, "bus", "pci", "devices", bdf, "iommu")
    if not os.path.exists(link):
        raise DiscoveryError(
            f"{bdf} has no IOMMU in sysfs. Is the IOMMU enabled (not off or passthrough)?")
    base = _address_from_name(os.path.basename(os.path.realpath(link)))
    if base is None:
        raise DiscoveryError(f"cannot read an SMMU address from {os.path.realpath(link)}")
    return base


def all_smmus(sysfs="/sys"):
    """Base addresses of every SMMU listed in /sys/class/iommu, sorted."""
    root = os.path.join(sysfs, "class", "iommu")
    names = os.listdir(root) if os.path.isdir(root) else []
    return sorted(a for a in (_address_from_name(n) for n in names) if a is not None)


def pmcg_units(sysfs="/sys"):
    """Every SMMU counter unit the kernel exposes, as {name: register address}."""
    root = os.path.join(sysfs, "bus", "event_source", "devices")
    units = {}
    for name in os.listdir(root) if os.path.isdir(root) else []:
        if name.startswith(PMCG_PREFIX):
            try:
                units[name] = int(name[len(PMCG_PREFIX):], 16) * 4096
            except ValueError:
                pass
    return units


def candidates(bdf=None, sysfs="/sys"):
    """Counter units inside the GPU's SMMU window, lowest address first.

    The window runs from the GPU's SMMU base up to the next SMMU's base.
    Returns (bdf, list of unit names).
    """
    if bdf is None:
        gpus = find_gpus(sysfs)
        if not gpus:
            raise DiscoveryError("no NVIDIA GPU found in sysfs (is this a GB10?)")
        bdf = gpus[0]
    base = smmu_of(bdf, sysfs)
    above = [a for a in all_smmus(sysfs) if a > base]
    end = min(above) if above else float("inf")
    units = pmcg_units(sysfs)
    inside = sorted((addr, name) for name, addr in units.items() if base <= addr < end)
    if not inside:
        raise DiscoveryError(
            f"no {PMCG_PREFIX}* unit inside the GPU's SMMU at {base:#x}. "
            "Is the arm_smmuv3_pmu kernel module loaded?")
    return bdf, [name for _, name in inside]
