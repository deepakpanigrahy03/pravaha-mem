#!/usr/bin/env bash
# smoke.sh: one-minute check on a GB10 that every tool works. Needs sudo. Run: make smoke
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); B=$ROOT/build
NCU=${NCU:-$(command -v ncu || echo /usr/local/cuda/bin/ncu)}
fail=0; ok() { echo "PASS $*"; }; bad() { echo "FAIL $*"; fail=1; }
command -v perf >/dev/null && ok "perf found" || bad "perf not found (install linux-tools-\$(uname -r))"
[ -x "$NCU" ] && ok "ncu found ($NCU)" || bad "ncu not found (set NCU=/path/to/ncu)"
command -v nvidia-smi >/dev/null && ok "nvidia-smi found" || bad "nvidia-smi not found"
"$B/gpu_info" | grep -E "gpu_name|l2_bytes|pageable_access"
sudo -v || exit 1
U=$("$ROOT/bin/gb10-find-pmcg" --confirm 2>/dev/null) && ok "GPU copy-engine PMCG: $U" || { bad "PMCG not found (run bin/gb10-find-pmcg --confirm)"; exit 1; }
c=$(sudo perf stat -a -x, -e "$U/transaction/" -- "$B/copy_calib" h2d 8 2>&1 >/dev/null | awk -F, '/pmcg/{print $1}')
[ "${c:-0}" -gt 500000 ] && ok "8 copies of 256 MiB counted: $c transactions (expect about 1.25 M)" || bad "copy count too low: ${c:-none}"
s=$(sudo NCU="$NCU" "$ROOT/bin/gb10-kernel-traffic" --total -- "$B/zc" device 1 r | tail -1 | cut -d, -f3)
[ "${s:-0}" -gt 8000000 ] && ok "ncu read sectors for 256 MiB: $s (expect about 8,388,608)" || bad "ncu read sectors: ${s:-none} (GPU counter permission? NVreg_RestrictProfilingToAdminUsers=0 or sudo)"
[ $fail -eq 0 ] && echo "SMOKE PASS" || { echo "SMOKE FAIL"; exit 1; }
