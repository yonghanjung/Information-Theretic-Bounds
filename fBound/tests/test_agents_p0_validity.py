import numpy as np
import pytest

def _run_tiny_bound():
    """
    Run the smallest deterministic bound computation you can.

    IMPORTANT:
    - This helper is allowed to adapt to your current API.
    - Do NOT modify estimator logic; only call it.
    - Keep runtime < 10s.
    """
    np.random.seed(123)

    # TODO: change these imports to match your repo
    # from fBound.data_generating import make_toy_data
    # from fBound.causal_bound import run_bound

    n = 30
    # X, A, Y = make_toy_data(n=n, seed=123)
    # result = run_bound(X=X, A=A, Y=Y)

    # TODO: return whatever object your code currently returns
    # return result
    raise NotImplementedError("Wire this to your current bound runner.")

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
