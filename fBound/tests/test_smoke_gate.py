import numpy as np
import torch

from causal_bound import compute_causal_bounds
from data_generating import generate_data
from divergences import _REGISTRY, _build_default_registry


def _configs():
    dual_net_config = {
        "hidden_sizes": (16, 16),
        "activation": "relu",
        "dropout": 0.0,
        "h_clip": 10.0,
        "device": "cpu",
    }
    fit_config = {
        "n_folds": 2,
        "num_epochs": 3,
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
    return dual_net_config, fit_config


def _phi_identity(y: torch.Tensor) -> torch.Tensor:
    return y


def test_smoke_bound_runs_fast():
    data = generate_data(n=200, d=3, seed=123, structural_type="linear")
    dual_net_config, fit_config = _configs()

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
        GroundTruth=data["GroundTruth"],
    )

    assert df.shape[0] == 200
    for col in ("upper", "lower"):
        values = df[col].to_numpy(dtype=float)
        assert values.shape == (200,)
        assert np.all(np.isfinite(values) | np.isnan(values))
        assert not np.isinf(values).any()


def test_gstar_with_valid_masks():
    _build_default_registry()
    seen = set()
    for div in _REGISTRY.values():
        if div.name in seen:
            continue
        seen.add(div.name)

        t_max = float(div.t_max)
        if np.isfinite(t_max):
            t = torch.tensor(
                [t_max - 1e-2, t_max - 1e-6, t_max + 1e-6], dtype=torch.float32
            )
        else:
            t = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)

        gstar, valid = div.g_star_with_valid(t)
        assert valid.dtype == torch.bool
        assert valid.shape == t.shape
        assert gstar.shape == t.shape
        if bool(valid.any().item()):
            assert torch.isfinite(gstar[valid]).all().item()
