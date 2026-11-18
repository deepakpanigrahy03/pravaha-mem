"""The small counter reader `pravaha-pmcg`, used when this user may not open perf directly.

Author: Deepak Panigrahy
Part of pravaha-mem (GB10 GPU memory traffic measurement). License: MIT.

Reading the SMMU copy counter needs kernel.perf_event_paranoid <= 0 or the capability
CAP_PERFMON. When neither applies to the user, an administrator can install the reader
`pravaha-pmcg` once with that one capability (docs/MANUAL.md, "Running without sudo").
pravaha then uses it automatically: every user can measure, also from unattended scripts,
and no system setting changes.

The reader is found at $PRAVAHA_PMCG, else as `pravaha-pmcg` on the PATH.
Its C source ships inside this package (pravaha_pmcg.c); `pravaha helper build` compiles it.
"""

import os
import shutil
import subprocess

NAME = "pravaha-pmcg"
SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pravaha_pmcg.c")


class HelperError(OSError):
    """The reader is missing, lacks its capability, or answered something unexpected."""


def find():
    """Full path of the reader, or None."""
    p = os.environ.get("PRAVAHA_PMCG")
    if p:
        return p if os.access(p, os.X_OK) else None
    return shutil.which(NAME)


def capability(path):
    """'cap_perfmon' if the file carries it, 'none' if not, None if this cannot be checked."""
    getcap = shutil.which("getcap") or "/usr/sbin/getcap"
    try:
        out = subprocess.run([getcap, path], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return "cap_perfmon" if "cap_perfmon" in out else "none"


class HelperCounter:
    """Same start/stop/read/close as perf.Counter, done by the reader process."""

    def __init__(self, unit, path=None):
        path = path or find()
        if not path:
            raise HelperError(f"{NAME} not found")
        self.unit = unit
        self.path = path
        self.proc = subprocess.Popen([path, unit], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, bufsize=1)
        first = self.proc.stdout.readline().strip()
        if not first.startswith("ready"):
            self.close()
            raise HelperError(f"{NAME}: {first or 'no answer'}")

    def _ask(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()
        ans = self.proc.stdout.readline().strip()
        if not ans or ans.startswith("error"):
            raise HelperError(f"{NAME}: {ans or 'no answer'}")
        return ans

    def start(self):
        self._ask("s")

    def stop(self):
        self._ask("p")

    def read(self):
        return int(self._ask("r"))

    def close(self):
        p = getattr(self, "proc", None)
        if p and p.poll() is None:
            try:
                p.stdin.write("q\n")
                p.stdin.flush()
                p.wait(timeout=2)
            except (OSError, subprocess.SubprocessError):
                p.kill()
        self.proc = None


def build(out_dir="."):
    """Compile the reader from the source in this package; return the new file's path."""
    cc = shutil.which("cc") or shutil.which("gcc")
    if not cc:
        raise HelperError("no C compiler found (install gcc)")
    out = os.path.join(os.path.abspath(out_dir), NAME)
    subprocess.run([cc, "-O2", "-Wall", "-Wextra", "-o", out, SOURCE], check=True)
    return out
