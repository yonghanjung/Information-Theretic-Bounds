"""itbound package wrapper for fbound."""

from fbound.estimators.causal_bound import DebiasedCausalBoundEstimator, compute_causal_bounds
from fbound.utils.data_generating import generate_data

__all__ = [
    "DebiasedCausalBoundEstimator",
    "compute_causal_bounds",
    "generate_data",
]

__version__ = "0.1.0"
