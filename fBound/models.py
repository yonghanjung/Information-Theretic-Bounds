"""
Model factories.

- Propensity models: sklearn-style classifiers with predict_proba.
- Pseudo-outcome regressors: sklearn-style regressors with fit/predict.

Dual networks (h_beta, u_gamma) are implemented as PyTorch MLPs and are used
only inside the causal bound estimator.

Engineering note on "no silent defaults"
---------------------------------------
For string-based factories, this module *requires* that model hyperparameters are
provided explicitly in `config`. Pass values even if you want library defaults.
"""
from __future__ import annotations

from typing import Any, Dict, Sequence, Union, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ClassifierLike(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Any: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class RegressorLike(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Any: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


def _validate_classifier(est: Any) -> None:
    if not hasattr(est, "fit") or not hasattr(est, "predict_proba"):
        raise TypeError(
            "Propensity model must implement fit(X,y) and predict_proba(X)->(n,2)."
        )


def _validate_regressor(est: Any) -> None:
    if not hasattr(est, "fit") or not hasattr(est, "predict"):
        raise TypeError("Regressor must implement fit(X,y) and predict(X)->(n,).")


def _require_keys(config: Dict[str, Any], keys: Sequence[str], ctx: str) -> None:
    missing = [k for k in keys if k not in config]
    if missing:
        raise KeyError(
            f"Missing required hyperparameters for {ctx}: {missing}. "
            f"Provide them explicitly in the config dict."
        )


def make_classifier(
    name_or_obj: Union[str, Any],
    config: Dict[str, Any],
    seed: int,
) -> Any:
    """
    Factory for propensity score estimators.

    Supported names
    ---------------
    - "logistic"
    - "xgboost" (optional dependency; lazy import)

    If `name_or_obj` is not a string, it is treated as an instantiated sklearn-like
    estimator, validated, and returned.
    """
    if config is None:
        raise ValueError("config must be provided (use {} only if you pass a prebuilt estimator).")
    if not isinstance(seed, int):
        raise TypeError("seed must be an int.")

    if not isinstance(name_or_obj, str):
        _validate_classifier(name_or_obj)
        return name_or_obj

    name = name_or_obj.strip().lower()
    if name == "logistic":
        _require_keys(config, ["C", "max_iter", "penalty", "solver", "n_jobs"], "logistic propensity")

        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        C = float(config["C"])
        max_iter = int(config["max_iter"])
        penalty = str(config["penalty"])
        solver = str(config["solver"])
        n_jobs = int(config["n_jobs"])

        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                (
                    "logreg",
                    LogisticRegression(
                        C=C,
                        max_iter=max_iter,
                        penalty=penalty,
                        solver=solver,
                        n_jobs=n_jobs,
                        random_state=seed,
                    ),
                ),
            ]
        )
        _validate_classifier(clf)
        return clf

    if name == "xgboost":
        try:
            import xgboost as xgb
        except Exception as e:
            raise ImportError(
                "xgboost is not installed. Install it or use another propensity_model."
            ) from e

        required = [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "reg_lambda",
            "min_child_weight",
            "objective",
            "eval_metric",
            "n_jobs",
            "verbosity",
        ]
        _require_keys(config, required, "xgboost propensity")

        params = dict(config)
        params.setdefault("random_state", seed)

        clf = xgb.XGBClassifier(**params)
        _validate_classifier(clf)
        return clf

    raise KeyError(f"Unknown classifier name '{name_or_obj}'. Supported: logistic, xgboost.")


def make_regressor(
    name_or_obj: Union[str, Any],
    config: Dict[str, Any],
    seed: int,
) -> Any:
    """
    Factory for pseudo-outcome regression models m^k.

    Supported names
    ---------------
    - "random_forest"
    - "linear"
    - "xgboost" (optional dependency; lazy import)

    If `name_or_obj` is not a string, it is treated as an instantiated sklearn-like
    estimator, validated, and returned.
    """
    if config is None:
        raise ValueError("config must be provided (use {} only if you pass a prebuilt estimator).")
    if not isinstance(seed, int):
        raise TypeError("seed must be an int.")

    if not isinstance(name_or_obj, str):
        _validate_regressor(name_or_obj)
        return name_or_obj

    name = name_or_obj.strip().lower()
    if name == "random_forest":
        _require_keys(
            config,
            ["n_estimators", "max_depth", "min_samples_leaf", "min_samples_split", "n_jobs"],
            "random_forest regressor",
        )
        from sklearn.ensemble import RandomForestRegressor

        n_estimators = int(config["n_estimators"])
        max_depth = config["max_depth"]
        min_samples_leaf = int(config["min_samples_leaf"])
        min_samples_split = int(config["min_samples_split"])
        n_jobs = int(config["n_jobs"])

        reg = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            random_state=seed,
            n_jobs=n_jobs,
        )
        _validate_regressor(reg)
        return reg

    if name == "linear":
        _require_keys(config, ["alpha"], "linear (ridge) regressor")

        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        alpha = float(config["alpha"])

        reg = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("ridge", Ridge(alpha=alpha, random_state=seed)),
            ]
        )
        _validate_regressor(reg)
        return reg

    if name == "xgboost":
        try:
            import xgboost as xgb
        except Exception as e:
            raise ImportError(
                "xgboost is not installed. Install it or use another m_model."
            ) from e

        required = [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "reg_lambda",
            "min_child_weight",
            "objective",
            "n_jobs",
            "verbosity",
        ]
        _require_keys(config, required, "xgboost regressor")

        params = dict(config)
        params.setdefault("random_state", seed)

        reg = xgb.XGBRegressor(**params)
        _validate_regressor(reg)
        return reg

    raise KeyError(
        f"Unknown regressor name '{name_or_obj}'. Supported: random_forest, linear, xgboost."
    )


# ---- Torch dual networks (h_beta, u_gamma) ------------------------------------

import torch
import torch.nn as nn


def _get_activation(name: str) -> nn.Module:
    name_l = name.strip().lower()
    if name_l == "relu":
        return nn.ReLU()
    if name_l == "tanh":
        return nn.Tanh()
    if name_l == "gelu":
        return nn.GELU()
    if name_l == "elu":
        return nn.ELU()
    raise KeyError(f"Unknown activation '{name}'. Supported: relu, tanh, gelu, elu.")


class TorchMLP(nn.Module):
    """
    Simple MLP producing a scalar output.

    Conventions
    -----------
    - Input: concatenated [A, X] where A is a single scalar feature (0/1).
    - Output: shape (n,) float tensor.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int],
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes must be non-empty.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0,1).")

        layers: list[nn.Module] = []
        in_dim = input_dim
        act = _get_activation(activation)
        for h in hidden_sizes:
            if h <= 0:
                raise ValueError("hidden_sizes must contain positive integers.")
            layers.append(nn.Linear(in_dim, h))
            layers.append(act)
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.squeeze(-1)
