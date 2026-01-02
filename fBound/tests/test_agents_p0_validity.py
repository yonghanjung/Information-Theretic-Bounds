import numpy as np
import pytest
import torch

from causal_bound import compute_causal_bounds
from data_generating import generate_data
from result import BoundResult

def _run_tiny_bound():
    """
    Run the smallest deterministic bound computation you can.

    IMPORTANT:
    - This helper is allowed to adapt to your current API.
    - Do NOT modify estimator logic; only call it.
    - Keep runtime < 10s.
    """
    data = generate_data(n=60, d=3, seed=123, structural_type="linear")

    def _phi_identity(y: torch.Tensor) -> torch.Tensor:
        return y

    dual_net_config = {
        "hidden_sizes": (16, 16),
        "activation": "relu",
        "dropout": 0.0,
        "h_clip": 10.0,
        "device": "cpu",
    }
    fit_config = {
        "n_folds": 2,
        "num_epochs": 2,
        "batch_size": 32,
        "lr": 5e-3,
        "weight_decay": 0.0,
        "max_grad_norm": 5.0,
        "eps_propensity": 1e-3,
        "deterministic_torch": True,
        "train_m_on_fold": True,
        "propensity_config": {
            "C": 1.0,
            "max_iter": 200,
            "penalty": "l2",
            "solver": "lbfgs",
            "n_jobs": 1,
        },
        "m_config": {
            "alpha": 1.0,
        },
        "verbose": False,
        "log_every": 1,
    }

    df = compute_causal_bounds(
        Y=data["Y"],
        A=data["A"],
        X=data["X"],
        divergence="KL",
        phi=_phi_identity,
        propensity_model="logistic",
        m_model="linear",
        dual_net_config=dual_net_config,
        fit_config=fit_config,
        seed=123,
        GroundTruth=None,
    )

    upper = df["upper"].to_numpy(dtype=np.float32)
    lower = df["lower"].to_numpy(dtype=np.float32)
    valid_up = df["valid_up"].to_numpy(dtype=bool)
    valid_lo = df["valid_lo"].to_numpy(dtype=bool)
    valid_interval = df["valid_interval"].to_numpy(dtype=bool)

    return BoundResult(
        upper=upper,
        lower=lower,
        valid_up=valid_up,
        valid_lo=valid_lo,
        valid_interval=valid_interval,
        diagnostics=None,
    )

def test_agents_requires_separate_valid_up_valid_lo():
    """
    AGENTS P0: must track validity separately for upper and lower computations.
    """
    result = _run_tiny_bound()

    assert hasattr(result, "valid_up"), "P0: missing valid_up"
    assert hasattr(result, "valid_lo"), "P0: missing valid_lo"

    vu = np.asarray(result.valid_up)
    vl = np.asarray(result.valid_lo)

    assert vu.dtype == bool and vl.dtype == bool, "valid_* must be boolean arrays"
    assert vu.shape == vl.shape, "valid_up and valid_lo must have same shape"

def test_agents_interval_validity_rule_applied_after_bounds():
    """
    AGENTS P0: valid_interval = valid_up & valid_lo & finite(lower) & finite(upper) & (lower<=upper)
    """
    result = _run_tiny_bound()

    assert hasattr(result, "upper"), "missing upper bound output"
    assert hasattr(result, "lower"), "missing lower bound output"
    assert hasattr(result, "valid_up"), "missing valid_up"
    assert hasattr(result, "valid_lo"), "missing valid_lo"
    assert hasattr(result, "valid_interval"), "P0: missing valid_interval"

    upper = np.asarray(result.upper, dtype=float)
    lower = np.asarray(result.lower, dtype=float)
    vu = np.asarray(result.valid_up, dtype=bool)
    vl = np.asarray(result.valid_lo, dtype=bool)
    vi = np.asarray(result.valid_interval, dtype=bool)

    expected = vu & vl & np.isfinite(lower) & np.isfinite(upper) & (lower <= upper)
    assert vi.shape == expected.shape
    assert np.array_equal(vi, expected), "valid_interval does not match AGENTS P0 definition"

def test_agents_invalid_intervals_are_nan():
    """
    AGENTS P0: when valid_interval is False, interval endpoints should be NaN (blanked).
    """
    result = _run_tiny_bound()

    upper = np.asarray(result.upper, dtype=float)
    lower = np.asarray(result.lower, dtype=float)
    vi = np.asarray(result.valid_interval, dtype=bool)

    # Only check where invalid
    idx = np.where(~vi)
    if idx[0].size == 0:
        pytest.skip("No invalid points in tiny run; cannot verify blanking.")
    assert np.all(np.isnan(upper[idx])), "upper should be NaN where invalid"
    assert np.all(np.isnan(lower[idx])), "lower should be NaN where invalid"
