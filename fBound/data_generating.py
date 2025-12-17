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
    """Stable-ish logistic helper used for treatment assignment."""
    return 1.0 / (1.0 + np.exp(-z))


def _enforce_margin(p: np.ndarray, eps: float = 0.05) -> np.ndarray:
    """Map probabilities p to eps + (1-2eps)*p to enforce overlap."""
    return eps + (1.0 - 2.0 * eps) * p


def generate_data(
    n: int,
    d: int,
    seed: int,
    structural_type: str = "nonlinear",
    x_range: float = 2.0,
) -> Dict[str, object]:
    if n <= 0:
        raise ValueError("n must be positive.")
    if d <= 0:
        raise ValueError("d must be positive.")
    if structural_type not in {"linear", "nonlinear", "simpson", "cyclic", "cyclic2"}:
        raise ValueError("structural_type must be 'linear', 'nonlinear', 'simpson', 'cyclic', or 'cyclic2'.")

    rng = np.random.default_rng(seed)

    if structural_type == "cyclic":
        # Confounded DGP with sin(eX) signal and latent U (positivity enforced).
        X = rng.uniform(-x_range, x_range, size=(n, d)).astype(np.float32)
        U = rng.normal(0.0, 1.0, size=(n,)).astype(np.float32)
        eps = rng.normal(0.0, 1.0, size=(n,)).astype(np.float32)

        # Hyperparameters for the DGP.
        beta = np.ones((d,), dtype=np.float32) * (0.5 / float(x_range))
        B = np.linspace(0.8, -0.2, num=d, dtype=np.float32)
        alpha = 0.0
        gamma = 0.5
        delta = 1.0
        sig_y = 1.0
        eps = eps * np.float32(sig_y)

        # Scale (1/d) dot(beta, X) into [0,1].
        lin_x = (X @ beta) / float(d)
        lin_x = np.clip(lin_x, 0.0, 1.0)
        dot_main = lin_x * float(d)  # lies in [0, d]

        # Scale gamma * U into [0,1] by clipping.
        gamma_u = np.clip(gamma * U, 0.0, 1.0)

        eX = _sigmoid(dot_main)
        p = _enforce_margin(_sigmoid(alpha + dot_main + gamma_u))
        A = rng.binomial(1, p, size=(n,)).astype(np.int64)

        Y = (5.0 * (2.0 * A.astype(np.float32) - 1.0) * np.sin(eX) + (X @ B) / float(d) + delta * U + eps).astype(np.float32)

        def GroundTruth(a: int, X_query: np.ndarray) -> np.ndarray:
            if a not in (0, 1):
                raise ValueError("a must be 0 or 1.")
            Xq = np.asarray(X_query, dtype=np.float32)
            if Xq.ndim != 2 or Xq.shape[1] != d:
                raise ValueError(f"X_query must have shape (m,{d}). Got {Xq.shape}.")
            lin_x_q = (Xq @ beta) / float(d)
            lin_x_q = np.clip(lin_x_q, 0.0, 1.0)
            dot_q = lin_x_q * float(d)
            eX_q = _sigmoid(dot_q)
            return (5.0 * (2.0 * float(a) - 1.0) * np.sin(eX_q) + (Xq @ B) / float(d)).astype(np.float32)

        def sample_do(a: int) -> np.ndarray:
            a_arr = np.full((n,), a, dtype=np.float32)
            eps_do = rng.normal(0.0, sig_y, size=(n,)).astype(np.float32)
            return (5.0 * (2.0 * a_arr - 1.0) * np.sin(eX) + (X @ B) / float(d) + delta * U + eps_do).astype(np.float32)

        return {
            "X": X,
            "A": A,
            "Y": Y,
            "X_do0": X.copy(),
            "A_do0": np.zeros_like(A),
            "Y_do0": sample_do(0),
            "X_do1": X.copy(),
            "A_do1": np.ones_like(A),
            "Y_do1": sample_do(1),
            "GroundTruth": GroundTruth,
        }

    if structural_type == "cyclic2":
        # New cyclic2 DGP per specification: bounded linear/confounder components and sinusoidal signal.
        X = rng.uniform(-np.pi, np.pi, size=(n, d)).astype(np.float32)
        U = rng.normal(0.0, 1.0, size=(n,)).astype(np.float32)

        beta = np.ones((d,), dtype=np.float32)  # can be tuned; defaults to all ones
        alpha = 0.0
        theta = 1.0
        eta = 1.0
        gamma_s = 1.0
        sig_y = 1.0

        L = _sigmoid((X @ beta) / float(d))
        G = _sigmoid(gamma_s * U)
        p = _enforce_margin(_sigmoid(alpha + theta * np.sin(X[:, 0]) + eta * G))
        A = rng.binomial(1, p, size=(n,)).astype(np.int64)

        eps = rng.normal(0.0, sig_y, size=(n,)).astype(np.float32)
        Y = (5.0 * (2.0 * A.astype(np.float32) - 1.0) * np.sin(X[:, 0]) + L + G + eps).astype(np.float32)

        def GroundTruth(a: int, X_query: np.ndarray) -> np.ndarray:
            if a not in (0, 1):
                raise ValueError("a must be 0 or 1.")
            Xq = np.asarray(X_query, dtype=np.float32)
            if Xq.ndim != 2 or Xq.shape[1] != d:
                raise ValueError(f"X_query must have shape (m,{d}). Got {Xq.shape}.")
            Lq = _sigmoid((Xq @ beta) / float(d))
            return (5.0 * (2.0 * float(a) - 1.0) * np.sin(Xq[:, 0]) + Lq + np.mean(G)).astype(np.float32)

        def sample_do(a: int) -> np.ndarray:
            a_arr = np.full((n,), a, dtype=np.float32)
            eps_do = rng.normal(0.0, sig_y, size=(n,)).astype(np.float32)
            return (5.0 * (2.0 * a_arr - 1.0) * np.sin(X[:, 0]) + L + G + eps_do).astype(np.float32)

        return {
            "X": X,
            "A": A,
            "Y": Y,
            "X_do0": X.copy(),
            "A_do0": np.zeros_like(A),
            "Y_do0": sample_do(0),
            "X_do1": X.copy(),
            "A_do1": np.ones_like(A),
            "Y_do1": sample_do(1),
            "GroundTruth": GroundTruth,
        }

    if structural_type == "simpson":
        # Simpson's paradox toy: A is strongly confounded by latent U that also drives Y.
        # Observational E[Y|A=a] is far from interventional E[Y|do(a)] for both a.
        rho = 0.7  # correlation strength between X0 and U
        beta_u = 1.0
        tau = -2.0  # true causal effect (negative)
        # NEW: make outcome strongly depend on propensity score (p) to vary across overlap.
        # This injects heterogeneity aligned with p(X).
        def prop_effect(p):
            return 1.0 * (p - 0.5)

        # Features: first coord correlated with U; remaining coords standard normal.
        U = rng.normal(0.0, 1.0, size=(n,)).astype(np.float32)
        X = rng.normal(0.0, 1.0, size=(n, d)).astype(np.float32)
        X[:, 0] = (rho * U + rng.normal(0.0, 1.0, size=(n,))).astype(np.float32)

        logits = 2.5 * U + 0.5 * X[:, 0] - 0.2
        p = _enforce_margin(_sigmoid(logits))
        A = rng.binomial(1, p, size=(n,)).astype(np.int64)

        def mu_x(X_in: np.ndarray) -> np.ndarray:
            base = 0.3 + 0.4 * np.tanh(X_in[:, 0])
            if X_in.shape[1] > 1:
                base = base + 0.2 * X_in[:, 1]
            return base.astype(np.float32)

        noise = rng.normal(0.0, 1.0, size=(n,)).astype(np.float32)
        Y = (
            mu_x(X)
            + tau * A.astype(np.float32)
            + beta_u * U
            + prop_effect(p.astype(np.float32))
            + noise
        ).astype(np.float32)

        def sample_do(a: int) -> np.ndarray:
            a_arr = np.full((n,), a, dtype=np.float32)
            eps = rng.normal(0.0, 1.0, size=(n,)).astype(np.float32)
            return (mu_x(X) + tau * a_arr + beta_u * U + prop_effect(p.astype(np.float32)) + eps).astype(np.float32)

        # Analytic E[U | X]: only X0 is correlated with U.
        coef_u_given_x0 = rho / (rho ** 2 + 1.0)

        def GroundTruth(a: int, X_query: np.ndarray) -> np.ndarray:
            if a not in (0, 1):
                raise ValueError("a must be 0 or 1.")
            Xq = np.asarray(X_query, dtype=np.float32)
            if Xq.ndim != 2 or Xq.shape[1] != d:
                raise ValueError(f"X_query must have shape (m,{d}). Got {Xq.shape}.")
            est_u = coef_u_given_x0 * Xq[:, 0]
            # Use the same propensity-driven shift as in the DGP. Recompute p under observed X,U? We approximate with p from data (depends on X,U).
            logits_q = 2.5 * est_u + 0.5 * Xq[:, 0] - 0.2
            p_q = _enforce_margin(_sigmoid(logits_q))
            return (mu_x(Xq) + tau * float(a) + beta_u * est_u + prop_effect(p_q)).astype(np.float32)

        return {
            "X": X,
            "A": A,
            "Y": Y,
            "X_do0": X.copy(),
            "A_do0": np.zeros_like(A),
            "Y_do0": sample_do(0),
            "X_do1": X.copy(),
            "A_do1": np.ones_like(A),
            "Y_do1": sample_do(1),
            "GroundTruth": GroundTruth,
        }

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

    p = _enforce_margin(_sigmoid(logits))
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

    def prop_effect(p_arr: np.ndarray) -> np.ndarray:
        # NEW: amplify outcome heterogeneity along propensity for nonlinear setting.
        return 0.8 * (p_arr - 0.5)

    def tau_x(X_in: np.ndarray, p_arr: np.ndarray | None = None) -> np.ndarray:
        if structural_type == "linear":
            return 0.7 + 0.1 * X_in[:, 0]
        base = 0.7 + 0.2 * np.sin(X_in[:, 0]) + 0.1 * X_in[:, 0]
        if p_arr is not None:
            base = base + prop_effect(p_arr)
        return base

    noise = rng.normal(0.0, 1.0, size=(n,)).astype(np.float32)
    Y = (mu_x(X) + tau_x(X, p) * A.astype(np.float32) + 0.7 * U + noise).astype(np.float32)

    def sample_do(a: int) -> np.ndarray:
        a_arr = np.full((n,), a, dtype=np.float32)
        eps = rng.normal(0.0, 1.0, size=(n,)).astype(np.float32)
        return (mu_x(X) + tau_x(X, p) * a_arr + 0.7 * U + eps).astype(np.float32)

    Y_do0 = sample_do(0)
    Y_do1 = sample_do(1)

    def GroundTruth(a: int, X_query: np.ndarray) -> np.ndarray:
        if a not in (0, 1):
            raise ValueError("a must be 0 or 1.")
        Xq = np.asarray(X_query, dtype=np.float32)
        if Xq.ndim != 2 or Xq.shape[1] != d:
            raise ValueError(f"X_query must have shape (m,{d}). Got {Xq.shape}.")
        # Approximate p for X_query using the same structural logits (without U): use w and nonlinear terms from above.
        logits_q = Xq @ w + 0.0  # omit U contribution; this keeps directionality
        if structural_type == "nonlinear":
            logits_q = logits_q + 0.5 * np.sin(Xq[:, 0]) - 0.25 * (Xq[:, 0] ** 2)
        p_q = _enforce_margin(_sigmoid(logits_q))
        return (mu_x(Xq) + tau_x(Xq, p_q) * float(a)).astype(np.float32)

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
