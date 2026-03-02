"""itbound package wrapper for fbound."""

from fbound.estimators.causal_bound import DebiasedCausalBoundEstimator, compute_causal_bounds
from fbound.utils.data_generating import generate_data
from .api import fit
from .claims import compute_claims
from .report import BoundsReport
from .standard import DEFAULT_DIVERGENCES, StandardRunResult, run_standard_bounds

__all__ = [
    "DebiasedCausalBoundEstimator",
    "compute_causal_bounds",
    "generate_data",
    "fit",
    "compute_claims",
    "BoundsReport",
    "DEFAULT_DIVERGENCES",
    "StandardRunResult",
    "run_standard_bounds",
]

__version__ = "0.1.0"
