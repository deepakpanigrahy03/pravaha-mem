"""Compatibility names from pravaha 0.1 and 0.2.

Author: Deepak Panigrahy
Part of pravaha (GB10 GPU memory-traffic measurement). License: MIT.

`CopyCounter` is now `pravaha.measure()` / `pravaha.Measurement`. This module
keeps old scripts working; new code should use pravaha.measure.
"""

from .measure import Measurement as CopyCounter  # noqa: F401
from .measure import default_unit  # noqa: F401
