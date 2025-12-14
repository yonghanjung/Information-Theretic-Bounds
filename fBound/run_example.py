"""
End-to-end example.

Runs:
- data generation with known GroundTruth
- bound estimation for KL and TV
- reports mean width and empirical coverage

Run
---
From the project directory:
    python run_example.py
"""
from __future__ import annotations

import os
import sys

# THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# if THIS_DIR not in sys.path:
#     sys.path.insert(0, THIS_DIR)

from utils import apply_macos_thread_safety_knobs

apply_macos_thread_safety_knobs(enable=False)

import numpy as np
import torch

from causal_bound import compute_causal_bounds
from data_generating import generate_data

def phi_identity(y: torch.Tensor) -> torch.Tensor:
    return y

if __name__ == "__main__":
    seed = 123
    n = 5000
    d = 10
    div = "Chi2" # KL, TV, Hellinger, Chi2, JS

    data = generate_data(n=n, d=d, seed=seed, structural_type="nonlinear")
    X = data["X"]
    A = data["A"]
    Y = data["Y"]
    GroundTruth = data["GroundTruth"]

    dual_net_config = {
        "hidden_sizes": (64, 64),
        "activation": "relu",
        "dropout": 0.0,
        "h_clip": 20.0,
        "device": "cpu",
    }
    
    propensity_model = "xgboost"
    m_model = "xgboost"

    fit_config = {
        "n_folds": 3,
        "num_epochs": 100,
        "batch_size": 256,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "max_grad_norm": 10.0,
        "eps_propensity": 1e-3,
        "deterministic_torch": True,
        "train_m_on_fold": True,
        "propensity_config": {
            "n_estimators": 300,
            "max_depth": 10,
            "learning_rate": 0.01,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "min_child_weight": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_jobs": 1,
            "verbosity": 0,
        },
            # regression head for Z
        "m_config": {
            "n_estimators": 400,
            "max_depth": 10,
            "learning_rate": 0.01,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "min_child_weight": 1.0,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "n_jobs": 1,
            "verbosity": 0,
        },
        "verbose": False,
        "log_every": 10,
    }

    df = compute_causal_bounds(
        Y=Y,
        A=A,
        X=X,
        divergence=div,
        phi=phi_identity,
        propensity_model=propensity_model,
        m_model=m_model,
        dual_net_config=dual_net_config,
        fit_config=fit_config,
        seed=seed,
        GroundTruth=GroundTruth,
    )

    assert np.isfinite(df["upper"]).all(), "upper has non-finite values"
    assert np.isfinite(df["lower"]).all(), "lower has non-finite values"
    # assert (df["lower"] <= df["upper"] + 1e-6).all(), "lower > upper detected"
    assert (
        (df["ehat1_oof"] >= fit_config["eps_propensity"] - 1e-8)
        & (df["ehat1_oof"] <= 1 - fit_config["eps_propensity"] + 1e-8)
    ).all(), "propensity outside clipping range"

    width = float(df["width"].mean())
    truth = df["truth_do1"].to_numpy()
    cover = float(np.mean((truth >= df["lower"].to_numpy()) & (truth <= df["upper"].to_numpy())))
    print(f"divergence={div:>2} | mean width={width:.4f} | coverage={cover:.3f}")

    # print("Done.")
