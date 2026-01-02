from __future__ import annotations

from pathlib import Path
import sys as _sys

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

from fbound.utils import data_generating as _mod

for _name, _value in _mod.__dict__.items():
    if _name.startswith("__"):
        continue
    globals()[_name] = _value

__all__ = [name for name in _mod.__dict__ if not name.startswith("__")]
