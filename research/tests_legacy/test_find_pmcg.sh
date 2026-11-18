#!/usr/bin/env bash
# test_find_pmcg.sh: gb10-find-pmcg against a fake sysfs tree shaped like the tested GB10.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
mkdir -p "$T/devices/platform/arm-smmu-v3.1.auto/iommu" "$T/class/iommu" "$T/bus/pci/devices/000f:01:00.0" "$T/bus/event_source/devices"
for s in 0x0000000013000000 0x0000000013800000 0x0000000014900000; do
  mkdir -p "$T/devices/platform/x/iommu/smmu3.$s"; ln -s "$T/devices/platform/x/iommu/smmu3.$s" "$T/class/iommu/smmu3.$s"
done
ln -s "$T/devices/platform/x/iommu/smmu3.0x0000000013000000" "$T/bus/pci/devices/000f:01:00.0/iommu"
for u in 13002 13042 13062 13082 138a2 14902; do mkdir -p "$T/bus/event_source/devices/smmuv3_pmcg_$u"; done
got=$(SYSFS=$T "$ROOT/bin/gb10-find-pmcg" --bdf 000f:01:00.0 2>/dev/null | paste -sd' ')
want="smmuv3_pmcg_13002 smmuv3_pmcg_13042 smmuv3_pmcg_13062 smmuv3_pmcg_13082"
if [ "$got" = "$want" ]; then echo "PASS find-pmcg candidates: $got"; else echo "FAIL find-pmcg: got '$got' want '$want'"; exit 1; fi
if SYSFS=$T "$ROOT/bin/gb10-find-pmcg" --bdf 0000:09:00.0 >/dev/null 2>&1; then echo "FAIL: missing GPU accepted"; exit 1; else echo "PASS find-pmcg rejects a device without IOMMU"; fi
