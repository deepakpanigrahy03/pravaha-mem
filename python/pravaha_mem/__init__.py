"""pravaha: GPU memory traffic meter for NVIDIA GB10 / DGX Spark.

Author: Deepak Panigrahy
Part of pravaha (GB10 GPU memory-traffic measurement). License: MIT.

"Pravaha" is Sanskrit for flow. The GB10's GPU reaches its memory through the
CPU die, and NVIDIA's tools report none of that traffic on this chip.
pravaha reads two counters that do see it:

  copy path    the GPU's SMMU counter (Linux perf), live, SYSTEM WIDE
               -> measure(), Measurement, watch()
  kernel path  Nsight Compute's L2 system-memory events, in a separate pass
               -> kernels()

Quick start (after `pravaha setup` once):

    import pravaha_mem as pravaha
    with pravaha.measure() as m:
        run_my_code()
    print(m.summary())

Without a calibration pravaha reports transactions, never bytes.
"""

__version__ = "1.0.0"      # defined before the imports: submodules read it

from .calib import Calibration, CalibrationMissing, Estimate  # noqa: E402,F401
from .discover import DiscoveryError, candidates, find_gpus  # noqa: E402,F401
from .measure import Measurement, default_unit, measure, watch  # noqa: E402,F401
from .copycounter import CopyCounter  # noqa: E402,F401  (0.2 name)
from .ncu import KernelTraffic, kernel_traffic, kernels  # noqa: E402,F401
from .perf import Counter, PerfError, RawSample  # noqa: E402,F401
from .status import show  # noqa: E402,F401
from .c2c_sampler import c2c, run as c2c_run, build as c2c_build  # noqa: E402,F401
