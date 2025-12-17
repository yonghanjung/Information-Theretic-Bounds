"""
Minimal sanity tests for the refactored estimator.

Run:
    python test_sanity.py
"""
from __future__ import annotations

import os
import sys
import unittest

# THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# if THIS_DIR not in sys.path:
#     sys.path.insert(0, THIS_DIR)

from utils import apply_macos_thread_safety_knobs

apply_macos_thread_safety_knobs(enable=False)

import numpy as np
import torch

from causal_bound import compute_causal_bounds
from data_generating import generate_data
from divergences import get_divergence


def phi_identity(y: torch.Tensor) -> torch.Tensor:
    """Identity observable: phi(y)=y."""
    return y


class TestSanity(unittest.TestCase):
    def _configs(self):
        """Small, fast configs to keep tests quick."""
        dual_net_config = {
            "hidden_sizes": (32, 32),
            "activation": "relu",
            "dropout": 0.0,
            "h_clip": 20.0,
            "device": "cpu",
        }
        fit_config = {
            "n_folds": 2,
            "num_epochs": 8,
            "batch_size": 64,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "max_grad_norm": 10.0,
            "eps_propensity": 1e-3,
            "deterministic_torch": True,
            "train_m_on_fold": True,
            "propensity_config": {
                "C": 1.0,
                "max_iter": 2000,
                "penalty": "l2",
                "solver": "lbfgs",
                "n_jobs": 1,
            },
            "m_config": {
                "n_estimators": 80,
                "max_depth": None,
                "min_samples_leaf": 5,
                "min_samples_split": 10,
                "n_jobs": 1,
            },
            "verbose": False,
            "log_every": 10,
        }
        return dual_net_config, fit_config

    def test_shape_validation(self):
        data = generate_data(n=100, d=4, seed=0)
        X = data["X"]
        A = data["A"]
        Y = data["Y"]
        dual_net_config, fit_config = self._configs()

        with self.assertRaises(ValueError):
            compute_causal_bounds(
                Y=Y[:-1],
                A=A,
                X=X,
                divergence="KL",
                phi=phi_identity,
                propensity_model="logistic",
                m_model="random_forest",
                dual_net_config=dual_net_config,
                fit_config=fit_config,
                seed=0,
                GroundTruth=data["GroundTruth"],
            )

    def test_divergence_penalty_finite(self):
        """g* should return finite values even when t is near/above the boundary."""
        div = get_divergence("KL")
        t = torch.tensor([0.1, 0.0, -1.0])
        val = div.g_star(t)
        self.assertTrue(torch.isfinite(val).all().item())

    def test_reproducibility_same_seed(self):
        """Same seed implies identical bounds (determinism knobs on)."""
        seed = 7
        data = generate_data(n=200, d=4, seed=seed)
        X = data["X"]
        A = data["A"]
        Y = data["Y"]
        dual_net_config, fit_config = self._configs()

        df1 = compute_causal_bounds(
            Y=Y,
            A=A,
            X=X,
            divergence="TV",
            phi=phi_identity,
            propensity_model="logistic",
            m_model="random_forest",
            dual_net_config=dual_net_config,
            fit_config=fit_config,
            seed=seed,
            GroundTruth=data["GroundTruth"],
        )
        df2 = compute_causal_bounds(
            Y=Y,
            A=A,
            X=X,
            divergence="TV",
            phi=phi_identity,
            propensity_model="logistic",
            m_model="random_forest",
            dual_net_config=dual_net_config,
            fit_config=fit_config,
            seed=seed,
            GroundTruth=data["GroundTruth"],
        )

        self.assertTrue(np.allclose(df1["upper"].to_numpy(), df2["upper"].to_numpy(), atol=1e-6))
        self.assertTrue(np.allclose(df1["lower"].to_numpy(), df2["lower"].to_numpy(), atol=1e-6))

    def test_bound_ordering(self):
        """Lower bound should not exceed upper (within tiny tolerance)."""
        seed = 11
        data = generate_data(n=200, d=4, seed=seed)
        X = data["X"]
        A = data["A"]
        Y = data["Y"]
        dual_net_config, fit_config = self._configs()

        df = compute_causal_bounds(
            Y=Y,
            A=A,
            X=X,
            divergence="KL",
            phi=phi_identity,
            propensity_model="logistic",
            m_model="random_forest",
            dual_net_config=dual_net_config,
            fit_config=fit_config,
            seed=seed,
            GroundTruth=data["GroundTruth"],
        )
        self.assertTrue(np.isfinite(df["upper"]).all())
        self.assertTrue(np.isfinite(df["lower"]).all())
        self.assertTrue(((df["lower"] <= df["upper"] + 1e-6).to_numpy()).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
