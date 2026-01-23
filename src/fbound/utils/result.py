from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

@dataclass(frozen=True)
class BoundResult:
    """
    Container for bound outputs and validity masks.

    Arrays are expected to be broadcastable to the same shape.
    """
    upper: np.ndarray
    lower: np.ndarray
    valid_up: np.ndarray
    valid_lo: np.ndarray
    valid_interval: np.ndarray
    diagnostics: Optional[Dict[str, Any]] = None
