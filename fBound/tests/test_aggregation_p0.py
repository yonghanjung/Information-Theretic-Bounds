import numpy as np

from causal_bound import _apply_interval_validity, aggregate_endpointwise


def test_apply_interval_validity_blanks_inverted():
    lower_raw = np.array([0.0, 2.0, np.nan], dtype=float)
    upper_raw = np.array([1.0, 1.0, 3.0], dtype=float)
    valid_up = np.array([True, True, True], dtype=bool)
    valid_lo = np.array([True, True, True], dtype=bool)

    lower, upper, valid_interval, inverted = _apply_interval_validity(
        lower_raw=lower_raw,
        upper_raw=upper_raw,
        valid_up=valid_up,
        valid_lo=valid_lo,
    )

    assert inverted.tolist() == [False, True, False]
    assert valid_interval.tolist() == [True, False, False]
    assert np.isnan(lower[1]) and np.isnan(upper[1])
    assert np.isnan(lower[2]) and np.isnan(upper[2])


def test_aggregate_endpointwise_filters_and_diagnostics():
    lower_mat = np.array(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [2.0, 4.0],
        ],
        dtype=float,
    )
    upper_mat = np.array(
        [
            [1.0, np.nan],
            [np.nan, 1.0],
            [3.0, 3.0],
        ],
        dtype=float,
    )
    valid_up = np.array(
        [
            [True, False],
            [False, True],
            [True, True],
        ],
        dtype=bool,
    )
    valid_lo = np.array(
        [
            [True, True],
            [True, True],
            [False, True],
        ],
        dtype=bool,
    )

    agg = aggregate_endpointwise(
        lower_mat=lower_mat,
        upper_mat=upper_mat,
        valid_up=valid_up,
        valid_lo=valid_lo,
    )

    assert np.allclose(agg["upper"][0], 1.0)
    assert np.allclose(agg["lower"][0], 1.0)

    assert np.isnan(agg["upper"][1])
    assert np.isnan(agg["lower"][1])

    assert agg["n_eff_up"].tolist() == [2, 0]
    assert agg["n_eff_lo"].tolist() == [2, 1]
    assert agg["invalid_up"].tolist() == [1, 1]
    assert agg["invalid_lo"].tolist() == [1, 0]
    assert agg["nonfinite_upper"].tolist() == [1, 1]
    assert agg["nonfinite_lower"].tolist() == [0, 0]
    assert agg["inverted_filtered"].tolist() == [0, 2]
