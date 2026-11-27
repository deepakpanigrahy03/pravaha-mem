"""pravaha: live GPU memory traffic over NVLink C2C on the NVIDIA GB10.

    import pravaha_mem as pv
    with pv.observe() as m:
        m.mark("load");   load_model()
        m.mark("decode"); generate()
    print(m.summary())

Values are device wide (all processes on the GPU). Command line: `pravaha --help`.
Part of pravaha-mem. MIT License. Copyright (c) 2026 MANTRA Labs.
"""

__version__ = "1.2.0"      # defined before the imports: submodules read it; equals pyproject version

from .observer import observe, run  # noqa: E402,F401
from .status import status  # noqa: E402,F401
from .verify import verify  # noqa: E402,F401
