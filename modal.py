"""Compatibility shim for `modal run modal.py`.

The actual Modal app lives in `workers/worker_modal.py`, but the CLI invocation
in the repo root expects a top-level `modal.py`. This module loads the worker
app and re-exports its Modal symbols.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent
_WORKER_MODAL = _ROOT / "workers" / "worker_modal.py"

# Avoid shadowing the installed `modal` package when the worker module imports it.
_root_path = str(_ROOT)
if _root_path in sys.path:
    sys.path.remove(_root_path)

globals().update(runpy.run_path(_WORKER_MODAL))
