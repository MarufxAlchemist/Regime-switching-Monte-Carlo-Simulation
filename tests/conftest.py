"""
pytest configuration — ensures project root is on sys.path.

This makes ``import models`` and ``import main`` work regardless of
whether pytest is invoked from the project root or a subdirectory,
and regardless of whether the ``pythonpath`` key in pyproject.toml
is supported by the installed pytest version.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Insert the project root (parent of tests/) at the front of sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
