"""pravaha install / uninstall: the only code that touches the system.

install creates exactly one folder (default /usr/local/libexec/pravaha):
    pravaha-reader   GPU L2 reader   cap_perfmon (or cap_sys_admin if the driver needs it and you ask)
    pravaha-pmcg     SMMU reader     cap_perfmon
    MANIFEST         JSON: files, sha256, sizes, capability, version, date
Everything is root owned and not writable by users. Nothing else on the system changes.
uninstall removes exactly the files listed in MANIFEST (after checking their sha256),
then the folder, and confirms nothing is left.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import shlex
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

from . import __version__

DEFAULT_PREFIX = "/usr/local/libexec/pravaha"
HERE = Path(__file__).parent


def _say(s):
    print(f"pravaha: {s}", flush=True)


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _cuda_paths(cuda):
    t = Path(cuda) / "targets" / "sbsa-linux"
    return (t / "include", t / "lib") if t.exists() else (Path(cuda) / "include", Path(cuda) / "lib64")


def build(out_dir: str, cuda: str | None = None) -> dict:
    """Compile both readers from the source shipped in the package. Returns {name: path}."""
    cuda = cuda or os.environ.get("CUDA_HOME", "/usr/local/cuda")
    inc, lib = _cuda_paths(cuda)
    gxx, cc = shutil.which("g++"), shutil.which("cc") or shutil.which("gcc")
    if not gxx or not cc:
        raise RuntimeError("a C and C++ compiler is needed: install gcc and g++, then run pravaha install again")
    reader = os.path.join(out_dir, "pravaha-reader")
    subprocess.run([gxx, "-O2", "-o", reader, str(HERE / "reader.cpp"), f"-I{inc}", f"-L{lib}",
                    f"-L{cuda}/extras/CUPTI/lib64", "-lcupti", "-lcuda", f"-Wl,-rpath,{lib}"], check=True)
    pmcg = os.path.join(out_dir, "pravaha-pmcg")
    subprocess.run([cc, "-O2", "-o", pmcg, str(HERE / "smmu" / "pravaha_pmcg.c")], check=True)
    return {"pravaha-reader": reader, "pravaha-pmcg": pmcg}


def _check_prefix(prefix: str) -> tuple[str, list[str]]:
    """The folder pravaha creates must sit under root owned folders that users cannot write.
    Returns the prefix and the folders install will create (recorded so uninstall removes them)."""
    prefix = os.path.abspath(prefix)
    create, d = [], prefix
    while not os.path.exists(d):
        create.append(d)
        d = os.path.dirname(d)
    for x in [d] + ([prefix] if os.path.exists(prefix) else []):
        st = os.stat(x)
        if st.st_uid != 0 or st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise RuntimeError(f"{x} must be owned by root and not writable by others: refusing to install "
                               "a reader with a capability there")
    return prefix, list(reversed(create))


def install(prefix: str = DEFAULT_PREFIX, cap: str = "cap_perfmon", cuda: str | None = None) -> int:
    prefix, created = _check_prefix(prefix)
    caps = {"pravaha-reader": cap, "pravaha-pmcg": "cap_perfmon"}
    with tempfile.TemporaryDirectory() as d:
        _say("building the readers (C and C++)")
        built = build(d, cuda)
        manifest = {"pravaha": __version__, "date": datetime.datetime.now().isoformat(timespec="seconds"),
                    "prefix": prefix, "created_dirs": created, "files": {}}
        for name, p in built.items():
            manifest["files"][name] = {"sha256": _sha256(p), "size": os.path.getsize(p), "capability": caps[name]}
        mpath = os.path.join(d, "MANIFEST")
        with open(mpath, "w") as fh:
            json.dump(manifest, fh, indent=1)
        q = shlex.quote
        script = " && ".join([
            f"install -d -o root -g root -m 0755 {q(prefix)}",
            *(f"install -o root -g root -m 0755 {q(p)} {q(os.path.join(prefix, n))}" for n, p in built.items()),
            *(f"setcap {caps[n]}=ep {q(os.path.join(prefix, n))}" for n in built),
            f"install -o root -g root -m 0644 {q(mpath)} {q(os.path.join(prefix, 'MANIFEST'))}",
        ])
        _say(f"installing into {prefix} (asks for your password once)")
        rc = subprocess.call(["sudo", "sh", "-c", script])
        if rc:
            return rc
    reader = os.path.join(prefix, "pravaha-reader")
    r = subprocess.run([reader, "--seconds", "0.3"], capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or [""])[-1]
    if last.startswith("E ok") or last.startswith("E busy"):
        _say(f"installed: {prefix} (pravaha-reader {cap}, pravaha-pmcg cap_perfmon). Nothing else was changed.")
        _say("next: pravaha status, then pravaha verify")
        return 0
    _say("installed, but the reader cannot read the GPU counters with " + cap + ":")
    _say("  " + (r.stderr.strip() or last))
    if cap == "cap_perfmon":
        _say("this driver may need the broader capability. To use it: pravaha install --cap cap_sys_admin")
    return 1


def uninstall(prefix: str = DEFAULT_PREFIX) -> int:
    prefix = os.path.abspath(prefix)
    mpath = os.path.join(prefix, "MANIFEST")
    if not os.path.exists(mpath):
        if os.path.exists(prefix):
            _say(f"{prefix} has no MANIFEST; not removing anything. Remove it by hand if it is pravaha's: "
                 f"sudo rm -r {prefix}")
            return 1
        _say("nothing installed")
        return 0
    manifest = json.load(open(mpath))
    remove, keep = [], []
    for name, info in manifest.get("files", {}).items():
        p = os.path.abspath(os.path.join(prefix, name))
        if os.path.dirname(p) != prefix:
            keep.append(f"{p} (outside {prefix}, not removed)")
        elif os.path.exists(p) and _sha256(p) != info.get("sha256"):
            keep.append(f"{p} (changed since install, not removed)")
        elif os.path.exists(p):
            remove.append(p)
    q = shlex.quote
    created = [d for d in reversed(manifest.get("created_dirs", [])) if d != prefix and prefix.startswith(d + "/")]
    dirs = [prefix] + created        # the install folder, then any parents install created, deepest first
    script = " && ".join([*(f"rm -f {q(p)}" for p in remove), f"rm -f {q(mpath)}",
                          *(f"(rmdir {q(d)} 2>/dev/null || true)" for d in dirs)])
    _say("removing pravaha's readers (asks for your password once)")
    rc = subprocess.call(["sudo", "sh", "-c", script])
    left = [p for p in remove + [mpath, prefix] if os.path.exists(p)] + keep
    left += [d for d in dirs if os.path.exists(d) and d != prefix and not os.listdir(d)]
    if not left and rc == 0:
        _say("pravaha removed: nothing left on this system. To remove the Python package too: pip uninstall pravaha-mem")
        return 0
    _say("could not remove: " + ", ".join(left))
    return 1
