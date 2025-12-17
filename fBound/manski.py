"""
Empirical Manski bounds (bounded outcomes) using propensity + outcome regression.

Two entry points:
- empirical_manski_bounds: plug-in Manski bounds given known L,U and supplied models.
- empirical_extrema_manski_bounds: heuristic using empirical min/max of Y for (L,U).
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from models import make_classifier, make_regressor


def _predict_proba_class1(model: Any, X: np.ndarray) -> np.ndarray:
    """Return P(A=1|X) respecting sklearn's class ordering."""
    proba = model.predict_proba(X)
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError(f"predict_proba must return shape (n,2+). Got {proba.shape}")
    col = 1
    classes = getattr(model, "classes_", None)
    if classes is not None:
        classes = np.asarray(classes)
        if 1 in classes:
            col = int(np.where(classes == 1)[0][0])
    return proba[:, col]


def empirical_manski_bounds(
    Y: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    L: float,
    U: float,
    a: int,
    propensity_model: Any,
    propensity_config: Dict[str, Any],
    outcome_model: Any,
    outcome_config: Dict[str, Any],
    seed: int,
    eps_propensity: float = 1e-3,
) -> Dict[str, Any]:
    """
    Empirical Manski bounds for E[Y(a)|X] under bounded outcomes.

    - Fit propensity on full data.
    - Fit outcome regression on the treated subset (A=a).
    - Bound the unobserved counterfactual by L/U.
    """
    Y_arr = np.asarray(Y, dtype=np.float64).reshape(-1)
    A_arr = np.asarray(A).reshape(-1)
    X_arr = np.asarray(X)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    n = len(Y_arr)
    if len(A_arr) != n or len(X_arr) != n:
        raise ValueError("Y, A, X must have the same number of rows.")
    if a not in (0, 1):
        raise ValueError("Target treatment a must be 0 or 1.")

    prop_model = make_classifier(
        name_or_obj=propensity_model,
        config=propensity_config,
        seed=seed,
    )
    prop_model.fit(X_arr.astype(np.float64, copy=False), A_arr)
    # ehat is P(A=1|X); flip for a=0 below.
    ehat = _predict_proba_class1(prop_model, X_arr.astype(np.float64, copy=False))
    ehat = np.clip(ehat, eps_propensity, 1.0 - eps_propensity)
    if a == 0:
        ehat = 1.0 - ehat

    mask_a = A_arr == a
    if not np.any(mask_a):
        raise ValueError(f"No samples with A={a} to fit outcome regression.")

    # Outcome regression only on observed arm a, then evaluated on all X.
    outcome_reg = make_regressor(
        name_or_obj=outcome_model,
        config=outcome_config,
        seed=seed,
    )
    outcome_reg.fit(X_arr[mask_a].astype(np.float64, copy=False), Y_arr[mask_a])
    muhat = outcome_reg.predict(X_arr.astype(np.float64, copy=False))
    muhat = np.clip(muhat, L, U)

    # Manski lower/upper via worst-case counterfactual fill-in.
    lower = muhat * ehat + L * (1.0 - ehat)
    upper = muhat * ehat + U * (1.0 - ehat)

    return {
        "mu_lower": float(np.mean(lower)),
        "mu_upper": float(np.mean(upper)),
        "lower": lower.astype(np.float32),
        "upper": upper.astype(np.float32),
        "ehat": ehat.astype(np.float32),
        "muhat": muhat.astype(np.float32),
        "L": float(L),
        "U": float(U),
    }


def empirical_extrema_manski_bounds(
    Y: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    a: int,
    propensity_model: Any,
    propensity_config: Dict[str, Any],
    outcome_model: Any,
    outcome_config: Dict[str, Any],
    seed: int,
    eps_propensity: float = 1e-3,
) -> Dict[str, Any]:
    """Heuristic: set L,U to empirical min/max of Y then run Manski bounds."""
    Y_arr = np.asarray(Y, dtype=np.float64).reshape(-1)
    Lhat = float(np.min(Y_arr))
    Uhat = float(np.max(Y_arr))
    return empirical_manski_bounds(
        Y=Y_arr,
        A=A,
        X=X,
        L=Lhat,
        U=Uhat,
        a=a,
        propensity_model=propensity_model,
        propensity_config=propensity_config,
        outcome_model=outcome_model,
        outcome_config=outcome_config,
        seed=seed,
        eps_propensity=eps_propensity,
    )
