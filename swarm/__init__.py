"""swarm-kit: don't orchestrate. Fan out, judge, converge.

A local-first swarm framework for AI agents: dumb fan-out of worker
functions, one-pass smart review, and a convergence loop with budget,
stall detection, and benching.
"""

from .worker import Context, Result, Task, WorkerFn
from .report import SwarmReport

__version__ = "0.1.0"

__all__ = [
    "Task",
    "Context",
    "Result",
    "WorkerFn",
    "SwarmReport",
]

# Sibling modules are built in parallel by other builders; import them
# defensively so the package still imports if one is missing. In the test
# suite they are always present (real or faked), so every name below is
# exported there.
try:
    from .review import Verdict, review

    __all__ += ["Verdict", "review"]
except ImportError:  # pragma: no cover
    pass

try:
    from .harness import Swarm

    __all__ += ["Swarm"]
except ImportError:  # pragma: no cover
    pass

try:
    from .memory import Memory

    __all__ += ["Memory"]
except ImportError:  # pragma: no cover
    pass

try:
    from .ledger import Treasury

    __all__ += ["Treasury"]
except ImportError:  # pragma: no cover
    pass

try:
    from .watchdog import StallWatcher, SwarmWatchdog

    __all__ += ["StallWatcher", "SwarmWatchdog"]
except ImportError:  # pragma: no cover
    pass

try:
    from .identity import WorkerIdentity

    __all__ += ["WorkerIdentity"]
except ImportError:  # pragma: no cover
    pass

try:
    from .policy import NeedsApproval, Policy

    __all__ += ["Policy", "NeedsApproval"]
except ImportError:  # pragma: no cover
    pass
