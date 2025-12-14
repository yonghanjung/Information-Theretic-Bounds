"""
Simulation data generators for the CLeaR debiased causal bound estimator.

Provides:
- Observational data (X, A, Y) with unmeasured confounding via latent U.
- Interventional data under do(A=0) and do(A=1).
- Exact GroundTruth(a, X)=E[Y | do(A=a), X] (vectorized).

Design choice (for exact GroundTruth):
- We generate U independent of X, and make Y depend on U linearly with mean-zero U.
  Then E[Y|do(A=a),X]=mu(X)+tau(X)*a is analytic.
"""
from __future__ import annotations

from typing import Dict
import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def generate_data(
    n: int,
    d: int,
    seed: int,
    structural_type: str = "nonlinear",
) -> Dict[str, object]:
    if n <= 0:
        raise ValueError("n must be positive.")
    if d <= 0:
        raise ValueError("d must be positive.")
    if structural_type not in {"linear", "nonlinear"}:
        raise ValueError("structural_type must be 'linear' or 'nonlinear'.")

    rng = np.random.default_rng(seed)

    d_cont = max(1, d // 2)
    d_bin = d - d_cont

    X_cont = rng.normal(0.0, 1.0, size=(n, d_cont))
    if d_bin > 0:
        logits_bin = 0.5 * X_cont[:, 0] + 0.25 * (X_cont[:, 0] ** 2) - 0.1
        probs_bin = _sigmoid(logits_bin)[:, None]  # broadcast across binary dims
        X_bin = rng.binomial(1, probs_bin, size=(n, d_bin)).astype(np.float32)
        X = np.concatenate([X_cont, X_bin], axis=1).astype(np.float32)
    else:
        X = X_cont.astype(np.float32)

    U = rng.normal(0.0, 1.0, size=(n,)).astype(np.float32)

    w = rng.normal(0.0, 0.6, size=(d,)).astype(np.float32)
    logits = X @ w + 0.8 * U
    if structural_type == "nonlinear":
        logits = logits + 0.5 * np.sin(X[:, 0]) - 0.25 * (X[:, 0] ** 2)

    p = _sigmoid(logits)
    A = rng.binomial(1, p, size=(n,)).astype(np.int64)

    def mu_x(X_in: np.ndarray) -> np.ndarray:
        if structural_type == "linear":
            b = np.linspace(0.2, -0.2, num=X_in.shape[1], dtype=np.float32)
            return 0.5 + X_in @ b
        base = 0.5 + 0.8 * np.tanh(X_in[:, 0])
        if X_in.shape[1] > 1:
            base = base + 0.25 * (X_in[:, 1] ** 2)
        if X_in.shape[1] > 2:
            base = base - 0.15 * np.sin(X_in[:, 2])
        return base

    def tau_x(X_in: np.ndarray) -> np.ndarray:
        if structural_type == "linear":
            return 0.7 + 0.1 * X_in[:, 0]
        return 0.7 + 0.2 * np.sin(X_in[:, 0]) + 0.1 * X_in[:, 0]

    noise = rng.normal(0.0, 1.0, size=(n,)).astype(np.float32)
    Y = (mu_x(X) + tau_x(X) * A.astype(np.float32) + 0.7 * U + noise).astype(np.float32)

    def sample_do(a: int) -> np.ndarray:
        a_arr = np.full((n,), a, dtype=np.float32)
        eps = rng.normal(0.0, 1.0, size=(n,)).astype(np.float32)
        return (mu_x(X) + tau_x(X) * a_arr + 0.7 * U + eps).astype(np.float32)

    Y_do0 = sample_do(0)
    Y_do1 = sample_do(1)

    def GroundTruth(a: int, X_query: np.ndarray) -> np.ndarray:
        if a not in (0, 1):
            raise ValueError("a must be 0 or 1.")
        Xq = np.asarray(X_query, dtype=np.float32)
        if Xq.ndim != 2 or Xq.shape[1] != d:
            raise ValueError(f"X_query must have shape (m,{d}). Got {Xq.shape}.")
        return (mu_x(Xq) + tau_x(Xq) * float(a)).astype(np.float32)

    return {
        "X": X,
        "A": A,
        "Y": Y,
        "X_do0": X.copy(),
        "A_do0": np.zeros_like(A),
        "Y_do0": Y_do0,
        "X_do1": X.copy(),
        "A_do1": np.ones_like(A),
        "Y_do1": Y_do1,
        "GroundTruth": GroundTruth,
    }
