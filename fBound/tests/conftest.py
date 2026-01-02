from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root (fBound/) is on sys.path for legacy shims like causal_bound.py.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
