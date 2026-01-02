"""
Core estimator implementation for information-theoretic causal bounds.

Implements:
- Debiased loss (Eq. (25)-(26))
- Debiased estimator pipeline (Eq. (27)-(28), Step 1-6)

This module provides both:
- a functional API: compute_causal_bounds(...)
- a class API: DebiasedCausalBoundEstimator
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..utils.divergences import FDivergence, FDivergenceLike, get_divergence
from ..utils.models import TorchMLP, make_classifier, make_regressor
from ..utils.utils import check_shapes, make_kfold_splits, set_global_seed
from ..utils.result import BoundResult

PhiFn = Callable[[torch.Tensor], torch.Tensor]


def _require_keys(d: Dict[str, Any], required: Sequence[str], ctx: str) -> None:
    """Validate that a config dict contains all required keys."""
    missing = [k for k in required if k not in d]
    if missing:
        raise KeyError(f"Missing required keys in {ctx}: {missing}. Provided keys: {sorted(d.keys())}")


@dataclass(frozen=True)
class DualNetConfig:
    hidden_sizes: Tuple[int, ...]
    activation: str
    dropout: float
    h_clip: float
    device: str
    lambda_min: float = 1e-2

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DualNetConfig":
        _require_keys(d, ["hidden_sizes", "activation", "dropout", "h_clip", "device"], "dual_net_config")
        hs = tuple(int(x) for x in d["hidden_sizes"])
        return DualNetConfig(
            hidden_sizes=hs,
            activation=str(d["activation"]),
            dropout=float(d["dropout"]),
            h_clip=float(d["h_clip"]),
            device=str(d["device"]),
            lambda_min=float(d.get("lambda_min", 1e-2)),
        )


@dataclass(frozen=True)
class FitConfig:
    n_folds: int
    num_epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    max_grad_norm: Optional[float]
    eps_propensity: float
    deterministic_torch: bool
    train_m_on_fold: bool
    propensity_config: Dict[str, Any]
    m_config: Dict[str, Any]
    verbose: bool
    log_every: int
    domain_penalty_weight: float = 1e4

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FitConfig":
        _require_keys(
            d,
            [
                "n_folds",
                "num_epochs",
                "batch_size",
                "lr",
                "weight_decay",
                "max_grad_norm",
                "eps_propensity",
                "deterministic_torch",
                "train_m_on_fold",
                "propensity_config",
                "m_config",
                "verbose",
                "log_every",
            ],
            "fit_config",
        )
        max_grad_norm = d["max_grad_norm"]
        if max_grad_norm is not None:
            max_grad_norm = float(max_grad_norm)
        return FitConfig(
            n_folds=int(d["n_folds"]),
            num_epochs=int(d["num_epochs"]),
            batch_size=int(d["batch_size"]),
            lr=float(d["lr"]),
            weight_decay=float(d["weight_decay"]),
            max_grad_norm=max_grad_norm,
            eps_propensity=float(d["eps_propensity"]),
            deterministic_torch=bool(d["deterministic_torch"]),
            train_m_on_fold=bool(d["train_m_on_fold"]),
            propensity_config=dict(d["propensity_config"]),
            m_config=dict(d["m_config"]),
            verbose=bool(d["verbose"]),
            log_every=int(d["log_every"]),
            domain_penalty_weight=float(d.get("domain_penalty_weight", 1e4)),
        )


def _concat_ax(A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Concatenate scalar treatment A with covariates X into a single design matrix."""
    if A.ndim != 1:
        raise ValueError(f"A must be 1D torch tensor. Got shape {tuple(A.shape)}")
    if X.ndim != 2:
        raise ValueError(f"X must be 2D torch tensor. Got shape {tuple(X.shape)}")
    return torch.cat([A.reshape(-1, 1), X], dim=1)


def _predict_proba_class1(model: Any, X: np.ndarray) -> np.ndarray:
    """Return P(A=1|X) while respecting sklearn's class ordering."""
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


class _ConstantPropensityModel:
    """Fallback model that predicts a constant propensity when a fold has a single class."""

    def __init__(self, p: float) -> None:
        self.p = float(p)
        self.classes_ = np.asarray([0, 1], dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = int(np.asarray(X).shape[0])
        p1 = np.full((n,), self.p, dtype=np.float64)
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)


# ---- Propensity prefit cache (speed) -----------------------------------------

def prefit_propensity_cache(
    X: np.ndarray,
    A: np.ndarray,
    propensity_model: Union[str, Any],
    propensity_config: Dict[str, Any],
    n_folds: int,
    seed: int,
    eps_propensity: float,
) -> Dict[str, Any]:
    """
    Pre-fit cross-fitted propensity models and OOF predictions.

    This is a pure speed helper: in many experiments we re-fit the same propensity
    model repeatedly (e.g., for phi and -phi, across divergences, and for ablations).
    Since e(X)=P(A=1|X) depends only on (X,A), we can compute it once and reuse it.

    Returns a dict that can be passed to:
        DebiasedCausalBoundEstimator.fit(..., propensity_cache=cache)

    Notes
    -----
    - Uses the same split logic and per-fold seeds as DebiasedCausalBoundEstimator.fit.
    - Clips ehat into [eps_propensity, 1-eps_propensity] to match the estimator.
    """
    X_arr = np.asarray(X)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2D (n,d). Got shape {X_arr.shape}.")
    n = int(X_arr.shape[0])

    # Validate A is binary and convert dtypes consistently.
    Y_dummy = np.zeros((n,), dtype=np.float32)
    _, Ac, Xc = check_shapes(Y=Y_dummy, A=A, X=X_arr)

    if n_folds < 2:
        raise ValueError("n_folds must be >= 2.")
    if not (0.0 < eps_propensity < 0.5):
        raise ValueError("eps_propensity must be in (0, 0.5).")

    splits = make_kfold_splits(n=n, n_splits=int(n_folds), seed=int(seed), shuffle=True)

    fold_id = np.empty(n, dtype=int)
    for k, (_, fold_idx) in enumerate(splits):
        fold_id[fold_idx] = k

    models: list[Any] = []
    e1_oof = np.empty(n, dtype=np.float32)

    for k, (train_idx, fold_idx) in enumerate(splits):
        A_train = Ac[train_idx]
        classes = np.unique(A_train)
        if classes.size < 2:
            prop_model = _ConstantPropensityModel(float(classes[0]))
        else:
            prop_model = make_classifier(
                propensity_model,
                config=propensity_config,
                seed=int(seed) + 10_000 + k,
            )
            prop_model.fit(Xc[train_idx].astype(np.float64, copy=False), A_train)

        e1_fold = _predict_proba_class1(prop_model, Xc[fold_idx].astype(np.float64, copy=False))
        e1_fold = np.clip(e1_fold, eps_propensity, 1.0 - eps_propensity).astype(np.float32)
        e1_oof[fold_idx] = e1_fold
        models.append(prop_model)

    return {
        "splits": splits,
        "fold_id": fold_id,
        "e1_oof": e1_oof,
        "models": models,
        "seed": int(seed),
        "n_folds": int(n_folds),
        "eps_propensity": float(eps_propensity),
    }


class DebiasedCausalBoundEstimator:
    """
    Implements the paper's Step 1-6 estimator for a fixed phi and divergence.

    This class returns the *upper* bound for E[phi(Y) | do(A=a), X=x].
    The lower bound can be computed via the "sign flip" identity:
        lower_phi(a,x) = - upper_{-phi}(a,x)
    """

    def __init__(
        self,
        divergence: Union[str, FDivergenceLike],
        phi: PhiFn,
        propensity_model: Union[str, Any],
        m_model: Union[str, Any],
        dual_net_config: Dict[str, Any],
        fit_config: Dict[str, Any],
        seed: int,
    ) -> None:
        self.divergence: FDivergence = get_divergence(divergence)
        self.phi: PhiFn = phi
        self.propensity_model_spec = propensity_model
        self.m_model_spec = m_model
        self.dual_net_cfg = DualNetConfig.from_dict(dual_net_config)
        self.fit_cfg = FitConfig.from_dict(fit_config)
        self.seed = int(seed)

        if self.fit_cfg.n_folds < 2:
            raise ValueError("fit_config['n_folds'] must be >= 2.")
        if self.fit_cfg.batch_size <= 0:
            raise ValueError("fit_config['batch_size'] must be positive.")
        if self.fit_cfg.num_epochs <= 0:
            raise ValueError("fit_config['num_epochs'] must be positive.")
        if self.fit_cfg.lr <= 0:
            raise ValueError("fit_config['lr'] must be positive.")
        if not (0.0 < self.fit_cfg.eps_propensity < 0.5):
            raise ValueError("eps_propensity must be in (0, 0.5).")

        self._fitted: bool = False

        self.propensity_models_: list[Any] = []
        self.h_nets_: list[nn.Module] = []
        self.u_nets_: list[nn.Module] = []
        self.m_models_: list[Any] = []

        self.splits_: list[tuple[np.ndarray, np.ndarray]] = []
        self.fold_id_: Optional[np.ndarray] = None
        self.e1_hat_oof_: Optional[np.ndarray] = None
        self.final_dual_loss_: list[float] = []

        self.X_: Optional[np.ndarray] = None
        self.A_: Optional[np.ndarray] = None
        self.Y_: Optional[np.ndarray] = None


    def fit(
        self,
        X: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray,
        propensity_cache: Optional[Dict[str, Any]] = None,
        e_train_true: Optional[np.ndarray] = None,
    ) -> "DebiasedCausalBoundEstimator":
        """
        Fit the estimator on observational data.

        Parameters
        ----------
        propensity_cache:
            Optional cache returned by `prefit_propensity_cache(...)`. If provided,
            this skips re-fitting propensity models and reuses the cross-fitted
            propensity predictions. This can dramatically speed up experiments that
            fit many estimators on the same (X,A) with different phi/divergence variants.
        """
        Yc, Ac, Xc = check_shapes(Y=Y, A=A, X=X)
        n, _ = Xc.shape
        self.X_ = Xc
        self.A_ = Ac
        self.Y_ = Yc

        set_global_seed(self.seed, deterministic_torch=self.fit_cfg.deterministic_torch)

        e_true = None
        if e_train_true is not None:
            e_true = np.asarray(e_train_true, dtype=np.float32).reshape(-1)
            if e_true.shape[0] != n:
                raise ValueError(f"e_train_true must have length n={n}. Got {e_true.shape[0]}.")
            e_true = np.clip(e_true, self.fit_cfg.eps_propensity, 1.0 - self.fit_cfg.eps_propensity)

        # Cross-fitting splits / propensity models (can be cached).
        if propensity_cache is None and e_true is None:
            self.splits_ = make_kfold_splits(n=n, n_splits=self.fit_cfg.n_folds, seed=self.seed, shuffle=True)
            prop_models: Optional[list[Any]] = None
            e1_oof = np.empty(n, dtype=np.float32)
        elif e_true is None:
            splits = propensity_cache.get("splits", None)
            models = propensity_cache.get("models", None)
            e1_oof_in = propensity_cache.get("e1_oof", None)

            if splits is None or models is None or e1_oof_in is None:
                raise KeyError("propensity_cache must contain keys: 'splits', 'models', 'e1_oof'.")

            if len(splits) != self.fit_cfg.n_folds:
                raise ValueError(
                    f"propensity_cache has {len(splits)} folds but fit_config requires {self.fit_cfg.n_folds}."
                )
            if len(models) != len(splits):
                raise ValueError("propensity_cache['models'] length must match number of splits.")

            e1_oof = np.asarray(e1_oof_in, dtype=np.float32).reshape(-1)
            if e1_oof.shape[0] != n:
                raise ValueError(f"propensity_cache['e1_oof'] must have length n={n}. Got {e1_oof.shape[0]}.")

            # Clip to the estimator's eps (safety in case cache used a different eps).
            e1_oof = np.clip(e1_oof, self.fit_cfg.eps_propensity, 1.0 - self.fit_cfg.eps_propensity).astype(np.float32)

            self.splits_ = splits
            prop_models = list(models)
        else:
            self.splits_ = make_kfold_splits(n=n, n_splits=self.fit_cfg.n_folds, seed=self.seed, shuffle=True)
            prop_models = [None] * len(self.splits_)
            e1_oof = e_true

        # Fold alignment
        fold_id = np.empty(n, dtype=int)
        for k, (_, fold_idx) in enumerate(self.splits_):
            fold_id[fold_idx] = k
        self.fold_id_ = fold_id

        # Reset fitted components (keep propensity models if cached).
        if propensity_cache is None:
            self.propensity_models_.clear()
            if prop_models is not None:
                self.propensity_models_ = prop_models
        else:
            self.propensity_models_ = prop_models if prop_models is not None else []

        self.h_nets_.clear()
        self.u_nets_.clear()
        self.m_models_.clear()
        self.final_dual_loss_.clear()

        # Cross-fitting loop: fit nuisance models on train folds, evaluate dual loss on held-out fold.
        for k, (train_idx, fold_idx) in enumerate(self.splits_):
            if propensity_cache is None and e_true is None:
                A_train = Ac[train_idx]
                classes = np.unique(A_train)
                if classes.size < 2:
                    prop_model = _ConstantPropensityModel(float(classes[0]))
                else:
                    prop_model = make_classifier(
                        self.propensity_model_spec,
                        config=self.fit_cfg.propensity_config,
                        seed=self.seed + 10_000 + k,
                    )
                    prop_model.fit(Xc[train_idx].astype(np.float64, copy=False), A_train)

                e1_fold = _predict_proba_class1(prop_model, Xc[fold_idx].astype(np.float64, copy=False))
                e1_fold = np.clip(e1_fold, self.fit_cfg.eps_propensity, 1.0 - self.fit_cfg.eps_propensity).astype(
                    np.float32
                )
                e1_oof[fold_idx] = e1_fold
                self.propensity_models_.append(prop_model)
            elif e_true is None:
                # Cache mode: propensity models + e1_oof are provided.
                _ = self.propensity_models_[k]  # used later at prediction time
                e1_fold = e1_oof[fold_idx]
            else:
                e1_fold = e1_oof[fold_idx]

            h_net, u_net, last_loss = self._fit_dual_nets_on_fold(
                X_fold=Xc[fold_idx],
                A_fold=Ac[fold_idx],
                Y_fold=Yc[fold_idx],
                e1_fold=e1_fold,
                fold_seed=self.seed + 20_000 + k,
            )

            if self.fit_cfg.train_m_on_fold:
                X_m = Xc[fold_idx]
                A_m = Ac[fold_idx]
                Y_m = Yc[fold_idx]
            else:
                X_m = Xc[train_idx]
                A_m = Ac[train_idx]
                Y_m = Yc[train_idx]

            Z_m = self._compute_Z(
                X=X_m,
                A=A_m,
                Y=Y_m,
                h_net=h_net,
                u_net=u_net,
            )
            AX_m = np.concatenate([A_m.reshape(-1, 1).astype(np.float32), X_m.astype(np.float32)], axis=1)

            m_reg = make_regressor(
                self.m_model_spec,
                config=self.fit_cfg.m_config,
                seed=self.seed + 30_000 + k,
            )
            m_reg.fit(AX_m.astype(np.float64, copy=False), Z_m.astype(np.float64, copy=False))

            self.h_nets_.append(h_net)
            self.u_nets_.append(u_net)
            self.m_models_.append(m_reg)
            self.final_dual_loss_.append(float(last_loss))

            if self.fit_cfg.verbose:
                print(
                    f"[fold {k+1}/{self.fit_cfg.n_folds}] last_dual_loss={last_loss:.6f} "
                    f"(div={self.divergence.name})"
                )

        self.e1_hat_oof_ = e1_oof
        self._fitted = True
        return self

    def _fit_dual_nets_on_fold(
        self,
        X_fold: np.ndarray,
        A_fold: np.ndarray,
                Y_fold: np.ndarray,
                e1_fold: np.ndarray,
                fold_seed: int,
    ) -> tuple[nn.Module, nn.Module, float]:
        """Train dual networks (h, u) on one held-out fold."""
        if X_fold.ndim != 2:
            raise ValueError("X_fold must be 2D.")
        n_fold, d = X_fold.shape
        if n_fold == 0:
            raise ValueError("Empty fold.")

        device = torch.device(self.dual_net_cfg.device)
        input_dim = d + 1

        h_net = TorchMLP(
            input_dim=input_dim,
            hidden_sizes=self.dual_net_cfg.hidden_sizes,
            activation=self.dual_net_cfg.activation,
            dropout=self.dual_net_cfg.dropout,
        ).to(device)

        u_net = TorchMLP(
            input_dim=input_dim,
            hidden_sizes=self.dual_net_cfg.hidden_sizes,
            activation=self.dual_net_cfg.activation,
            dropout=self.dual_net_cfg.dropout,
        ).to(device)

        params = list(h_net.parameters()) + list(u_net.parameters())
        opt = torch.optim.Adam(params, lr=self.fit_cfg.lr, weight_decay=self.fit_cfg.weight_decay)

        X_t = torch.tensor(X_fold, dtype=torch.float32, device=device)
        A_t = torch.tensor(A_fold.astype(np.float32), dtype=torch.float32, device=device)
        Y_t = torch.tensor(Y_fold, dtype=torch.float32, device=device)
        e1_t = torch.tensor(e1_fold, dtype=torch.float32, device=device)
        e0_t = 1.0 - e1_t

        rng = np.random.default_rng(fold_seed)

        last_loss = float("nan")
        for epoch in range(self.fit_cfg.num_epochs):
            perm = rng.permutation(n_fold)
            for start in range(0, n_fold, self.fit_cfg.batch_size):
                idx = perm[start : start + self.fit_cfg.batch_size]
                idx_t = torch.tensor(idx, dtype=torch.int64, device=device)

                loss = self._debiased_loss_batch(
                    X=X_t.index_select(0, idx_t),
                    A=A_t.index_select(0, idx_t),
                    Y=Y_t.index_select(0, idx_t),
                    e1=e1_t.index_select(0, idx_t),
                    e0=e0_t.index_select(0, idx_t),
                    h_net=h_net,
                    u_net=u_net,
                )

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.fit_cfg.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(params, max_norm=self.fit_cfg.max_grad_norm)
                opt.step()
                last_loss = float(loss.detach().cpu().item())

            if self.fit_cfg.verbose and (epoch + 1) % max(1, self.fit_cfg.log_every) == 0:
                print(f"  epoch {epoch+1}/{self.fit_cfg.num_epochs} loss={last_loss:.6f}")

        h_net_cpu = h_net.to("cpu").eval()
        u_net_cpu = u_net.to("cpu").eval()
        return h_net_cpu, u_net_cpu, float(last_loss)

    def _debiased_loss_batch(
        self,
        X: torch.Tensor,
        A: torch.Tensor,
        Y: torch.Tensor,
        e1: torch.Tensor,
        e0: torch.Tensor,
        h_net: nn.Module,
        u_net: nn.Module,
    ) -> torch.Tensor:
        """Compute the debiased dual loss for one minibatch."""
        if X.ndim != 2:
            raise ValueError("X must be 2D in batch.")
        if A.ndim != 1 or Y.ndim != 1:
            raise ValueError("A and Y must be 1D in batch.")
        if not (X.shape[0] == A.shape[0] == Y.shape[0] == e1.shape[0] == e0.shape[0]):
            raise ValueError("Batch tensors must have the same first dimension.")

        ax = _concat_ax(A, X)
        u_ax = u_net(ax)

        # Debiasing correction terms at A=0 and A=1 (Eq. 26).
        zeros = torch.zeros_like(A)
        ones = torch.ones_like(A)

        ax0 = _concat_ax(zeros, X)
        ax1 = _concat_ax(ones, X)

        h0 = torch.clamp(h_net(ax0), min=-self.dual_net_cfg.h_clip, max=self.dual_net_cfg.h_clip)
        h1 = torch.clamp(h_net(ax1), min=-self.dual_net_cfg.h_clip, max=self.dual_net_cfg.h_clip)

        h_ax = torch.where(A >= 0.5, h1, h0)
        lam_ax = torch.exp(h_ax).clamp(min=self.dual_net_cfg.lambda_min)

        phi_y = self.phi(Y)
        t = (phi_y - u_ax) / lam_ax
        g_star_val = self.divergence.g_star(t)
        domain_pen = self.fit_cfg.domain_penalty_weight * self.divergence.domain_violation(t).pow(2)

        eA = torch.where(A >= 0.5, e1, e0)
        eta = self.divergence.B_torch(eA)

        main = lam_ax * (eta + g_star_val) + u_ax + domain_pen

        lam0 = torch.exp(h0).clamp(min=self.dual_net_cfg.lambda_min)
        lam1 = torch.exp(h1).clamp(min=self.dual_net_cfg.lambda_min)

        I0 = 1.0 - A
        I1 = A

        # Eq. (23) in the paper: derivative evaluated at observed A
        # eta_primeA = self.divergence.dB_torch(eA)
        # corr = eta_primeA * (e0 * lam0 * (I0 - e0) + e1 * lam1 * (I1 - e1))
        
        # Eq. (23) in the paper: derivative evaluated at observed a
        eta_prime0 = self.divergence.dB_torch(e0)
        eta_prime1 = self.divergence.dB_torch(e1)
        corr = eta_prime0 * (e0 * lam0 * (I0 - e0)) + eta_prime1 * (e1 * lam1 * (I1 - e1))

        loss = (main + corr).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss encountered. divergence={self.divergence.name}. "
                f"Try increasing eps_propensity, lowering lr, or adjusting penalty config in divergences."
            )
        return loss

    @torch.no_grad()
    def _compute_Z(
        self,
        X: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray,
        h_net: nn.Module,
        u_net: nn.Module,
    ) -> np.ndarray:
        """Compute pseudo-outcome Z for regression head m^k on a given fold."""
        Xc = np.asarray(X, dtype=np.float32)
        Ac = np.asarray(A, dtype=np.float32).reshape(-1)
        Yc = np.asarray(Y, dtype=np.float32).reshape(-1)

        X_t = torch.tensor(Xc, dtype=torch.float32)
        A_t = torch.tensor(Ac, dtype=torch.float32)
        Y_t = torch.tensor(Yc, dtype=torch.float32)

        ax = _concat_ax(A_t, X_t)
        h = h_net(ax)
        h = torch.clamp(h, min=-self.dual_net_cfg.h_clip, max=self.dual_net_cfg.h_clip)
        lam = torch.exp(h).clamp(min=self.dual_net_cfg.lambda_min)
        u = u_net(ax)

        t = (self.phi(Y_t) - u) / lam
        Z = self.divergence.g_star(t)
        return Z.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def debug_g_star_values(
        self,
        X: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray,
        fold: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Diagnostic helper: compute g*(t) and a validity mask for a specific fold.

        Returns
        -------
        (g_star_values, valid_mask) as NumPy arrays (float32, bool).
        """
        if not self._fitted:
            raise RuntimeError("Estimator is not fitted. Call fit(X,A,Y) first.")
        if fold < 0 or fold >= len(self.h_nets_):
            raise IndexError(f"fold index out of range: {fold}")

        Xc = np.asarray(X, dtype=np.float32)
        Ac = np.asarray(A, dtype=np.float32).reshape(-1)
        Yc = np.asarray(Y, dtype=np.float32).reshape(-1)

        X_t = torch.tensor(Xc, dtype=torch.float32)
        A_t = torch.tensor(Ac, dtype=torch.float32)
        Y_t = torch.tensor(Yc, dtype=torch.float32)

        h_net = self.h_nets_[fold]
        u_net = self.u_nets_[fold]

        ax = _concat_ax(A_t, X_t)
        h = torch.clamp(h_net(ax), min=-self.dual_net_cfg.h_clip, max=self.dual_net_cfg.h_clip)
        lam = torch.exp(h).clamp(min=self.dual_net_cfg.lambda_min)
        u = u_net(ax)

        t = (self.phi(Y_t) - u) / lam
        g_val, valid_mask = self.divergence.g_star_with_valid(t)
        return g_val.cpu().numpy().astype(np.float32), valid_mask.cpu().numpy()


    def predict_bound(self, a: int, X: np.ndarray, e_eval: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict the (upper) bound for E[phi(Y) | do(A=a), X=x] at query covariates X."""
        if not self._fitted:
            raise RuntimeError("Estimator is not fitted. Call fit(X,A,Y) first.")
        if a not in (0, 1):
            raise ValueError("a must be 0 or 1.")
        if self.X_ is None:
            raise RuntimeError("Internal error: missing fitted X_.")
        Xq32 = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        if Xq32.ndim != 2:
            raise ValueError(f"X must be 2D. Got {Xq32.shape}.")
        if Xq32.shape[1] != self.X_.shape[1]:
            raise ValueError(
                f"X has wrong number of features: expected {self.X_.shape[1]}, got {Xq32.shape[1]}."
            )

        K = len(self.propensity_models_)
        if K == 0:
            raise RuntimeError("No fitted folds found.")

        # Shared NumPy views used by sklearn/xgboost models.
        Xq64 = Xq32.astype(np.float64, copy=False)

        # Shared design for regression head m^k: [a, X].
        n_q = int(Xq32.shape[0])
        AX32 = np.concatenate([np.full((n_q, 1), float(a), dtype=np.float32), Xq32], axis=1)
        AX64 = AX32.astype(np.float64, copy=False)

        # Shared Torch design for h/u nets: concat([a], X).
        X_t = torch.from_numpy(Xq32)  # shares memory with Xq32 (CPU)
        A_t = torch.full((n_q,), float(a), dtype=torch.float32)
        ax_t = torch.cat([A_t.reshape(-1, 1), X_t], dim=1)

        e1_eval = None
        if e_eval is not None:
            e1_eval = np.asarray(e_eval, dtype=np.float64).reshape(-1)
            if e1_eval.ndim != 1 or e1_eval.shape[0] != n_q:
                raise ValueError(f"e_eval must be 1D of length {n_q}, got shape {e1_eval.shape}.")
            e1_eval = np.clip(e1_eval, self.fit_cfg.eps_propensity, 1.0 - self.fit_cfg.eps_propensity)

        preds = np.zeros((n_q,), dtype=np.float64)
        for k in range(K):
            preds += self._predict_fold_precomputed(
                k=k,
                a=a,
                Xq64=Xq64,
                ax_t=ax_t,
                AX64=AX64,
                e1_eval=e1_eval,
            )
        preds /= float(K)
        return preds.astype(np.float32)

    def _predict_fold_precomputed(
        self,
        k: int,
        a: int,
        Xq64: np.ndarray,
        ax_t: torch.Tensor,
        AX64: np.ndarray,
        e1_eval: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Per-fold bound contribution theta_k(a, x), using shared precomputed designs."""
        prop = self.propensity_models_[k]
        h_net = self.h_nets_[k]
        u_net = self.u_nets_[k]
        m_reg = self.m_models_[k]

        if e1_eval is None:
            e1 = _predict_proba_class1(prop, Xq64)
            e1 = np.clip(e1, self.fit_cfg.eps_propensity, 1.0 - self.fit_cfg.eps_propensity)
        else:
            e1 = e1_eval
        eA = e1 if a == 1 else (1.0 - e1)

        eta = self.divergence.B_numpy(eA.astype(np.float64, copy=False)).astype(np.float64, copy=False)

        with torch.no_grad():
            h = h_net(ax_t)
            h = torch.clamp(h, min=-self.dual_net_cfg.h_clip, max=self.dual_net_cfg.h_clip)
            lam = torch.exp(h).clamp(min=self.dual_net_cfg.lambda_min).cpu().numpy().astype(np.float64, copy=False)
            u = u_net(ax_t).cpu().numpy().astype(np.float64, copy=False)

        m = m_reg.predict(AX64).astype(np.float64, copy=False)

        theta = lam * (eta + m) + u
        return theta

    def _predict_fold(self, k: int, a: int, Xq: np.ndarray) -> np.ndarray:
        """Backward-compatible helper: build precomputations and delegate."""
        Xq32 = np.ascontiguousarray(np.asarray(Xq, dtype=np.float32))
        Xq64 = Xq32.astype(np.float64, copy=False)
        n_q = int(Xq32.shape[0])
        AX32 = np.concatenate([np.full((n_q, 1), float(a), dtype=np.float32), Xq32], axis=1)
        AX64 = AX32.astype(np.float64, copy=False)

        X_t = torch.from_numpy(Xq32)
        A_t = torch.full((n_q,), float(a), dtype=torch.float32)
        ax_t = torch.cat([A_t.reshape(-1, 1), X_t], dim=1)

        return self._predict_fold_precomputed(k=k, a=a, Xq64=Xq64, ax_t=ax_t, AX64=AX64)

    def predict_bound_for_observed_X(self, a: int = 1) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Estimator is not fitted. Call fit(X,A,Y) first.")
        if self.X_ is None or self.A_ is None or self.Y_ is None:
            raise RuntimeError("Internal error: missing fitted data.")
        if self.fold_id_ is None or self.e1_hat_oof_ is None:
            raise RuntimeError("Internal error: missing cross-fitting diagnostics.")

        upper = self.predict_bound(a=a, X=self.X_)
        df = pd.DataFrame(
            {
                "i": np.arange(self.X_.shape[0], dtype=int),
                "fold": self.fold_id_.astype(int),
                "ehat1_oof": self.e1_hat_oof_.astype(np.float32),
                "upper": upper.astype(np.float32),
            }
        )
        loss_by_fold = {k: self.final_dual_loss_[k] for k in range(len(self.final_dual_loss_))}
        df["dual_loss_fold"] = df["fold"].map(loss_by_fold).astype(np.float32)
        df["divergence"] = self.divergence.name
        return df


def compute_causal_bounds(
    Y: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    divergence: Union[str, FDivergenceLike],
    phi: PhiFn,
    propensity_model: Union[str, Any],
    m_model: Union[str, Any],
    dual_net_config: Dict[str, Any],
    fit_config: Dict[str, Any],
    seed: int,
    GroundTruth: Optional[Callable[[int, np.ndarray], np.ndarray]] = None,
    propensity_cache: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper that runs the two-pass (phi and -phi) estimator.

    Returns
    -------
    pd.DataFrame
        Columns: i, lower, truth_do1, upper, width, ehat1_oof, dual_loss_fold, divergence

    Speed notes
    -----------
    This wrapper *automatically* caches the cross-fitted propensity model when
    `propensity_cache` is not provided. This avoids redundant propensity fits for:
      - the sign-flip run (-phi), and
      - multi-divergence aggregations (e.g., divergence="combined").
    """
    Yc, Ac, Xc = check_shapes(Y=Y, A=A, X=X)
    n = int(Xc.shape[0])

    truth = np.full((n,), np.nan, dtype=np.float32)
    if GroundTruth is not None:
        try:
            truth = np.asarray(GroundTruth(1, Xc), dtype=np.float32).reshape(-1)
        except Exception as e:
            raise TypeError(
                "GroundTruth must be callable like GroundTruth(a:int, X:(n,d))->(n,)."
            ) from e

    fit_cfg = FitConfig.from_dict(fit_config)
    prop_cache = propensity_cache
    if prop_cache is None:
        prop_cache = prefit_propensity_cache(
            X=Xc,
            A=Ac,
            propensity_model=propensity_model,
            propensity_config=fit_cfg.propensity_config,
            n_folds=fit_cfg.n_folds,
            seed=int(seed),
            eps_propensity=fit_cfg.eps_propensity,
        )

    def _compute_for_div(div_name: Union[str, FDivergenceLike]) -> pd.DataFrame:
        """Run the standard two-pass (phi and -phi) pipeline for a single divergence."""
        est = DebiasedCausalBoundEstimator(
            divergence=div_name,
            phi=phi,
            propensity_model=propensity_model,
            m_model=m_model,
            dual_net_config=dual_net_config,
            fit_config=fit_config,
            seed=seed,
        ).fit(Xc, Ac, Yc, propensity_cache=prop_cache)

        df_upper = est.predict_bound_for_observed_X(a=1)

        def phi_neg(y: torch.Tensor) -> torch.Tensor:
            return -phi(y)

        est_neg = DebiasedCausalBoundEstimator(
            divergence=div_name,
            phi=phi_neg,
            propensity_model=propensity_model,
            m_model=m_model,
            dual_net_config=dual_net_config,
            fit_config=fit_config,
            seed=seed,
        ).fit(Xc, Ac, Yc, propensity_cache=prop_cache)

        df_upper_neg = est_neg.predict_bound_for_observed_X(a=1)

        df_upper = df_upper.sort_values("i").reset_index(drop=True)
        df_upper_neg = df_upper_neg.sort_values("i").reset_index(drop=True)

        if not np.array_equal(df_upper["i"].values, df_upper_neg["i"].values):
            raise RuntimeError("Internal alignment error between phi and -phi runs.")

        lower = -df_upper_neg["upper"].to_numpy(dtype=np.float32)

        out = df_upper.copy()
        out["truth_do1"] = truth
        out["lower"] = lower
        out["width"] = (out["upper"] - out["lower"]).astype(np.float32)

        cols = [
            "i",
            "lower",
            "truth_do1",
            "upper",
            "width",
            "ehat1_oof",
            "dual_loss_fold",
            "divergence",
        ]
        return out[cols]

    def _stack_divergence_runs(div_names: Sequence[str]) -> list[pd.DataFrame]:
        div_dfs = [_compute_for_div(div_name) for div_name in div_names]
        i_ref = div_dfs[0]["i"].to_numpy()
        for df_other in div_dfs[1:]:
            if not np.array_equal(i_ref, df_other["i"].to_numpy()):
                raise RuntimeError("Alignment error when combining divergences.")
        return div_dfs

    def _combined_robust(div_names: Sequence[str], c: int = 3) -> pd.DataFrame:
        """
        Robust c-wise intersection aggregation (Version B).

        For each observation, search subsets of size t (t=c, c-1, ..., 2) with
        non-empty intersection; pick the subset with smallest intersection width
        (ties broken by sum of widths, then lexicographic), and return [L_hat, U_hat].
        If infeasible, fallback to the hull over valid intervals.
        """
        div_dfs = _stack_divergence_runs(div_names)
        base = div_dfs[0]
        upper_stack = np.vstack([df["upper"].to_numpy(dtype=np.float32) for df in div_dfs])
        lower_stack = np.vstack([df["lower"].to_numpy(dtype=np.float32) for df in div_dfs])
        n_obs = upper_stack.shape[1]

        lower_out = np.full(n_obs, np.nan, dtype=np.float32)
        upper_out = np.full(n_obs, np.nan, dtype=np.float32)

        for i in range(n_obs):
            lowers = lower_stack[:, i]
            uppers = upper_stack[:, i]
            valid = np.isfinite(lowers) & np.isfinite(uppers) & (lowers <= uppers)
            if not np.any(valid):
                finite = np.isfinite(lowers) & np.isfinite(uppers)
                if np.any(finite):
                    lo_f = float(np.min(lowers[finite]))
                    hi_f = float(np.max(uppers[finite]))
                    if lo_f > hi_f:
                        mid = 0.5 * (lo_f + hi_f)
                        lo_f = mid
                        hi_f = mid
                    lower_out[i] = np.float32(lo_f)
                    upper_out[i] = np.float32(hi_f)
                else:
                    lower_out[i] = np.float32(0.0)
                    upper_out[i] = np.float32(0.0)
                continue

            lowers_v = lowers[valid]
            uppers_v = uppers[valid]

            if len(lowers_v) < 2:
                lower_out[i] = np.float32(np.min(lowers_v))
                upper_out[i] = np.float32(np.max(uppers_v))
                continue

            best_subset: Optional[tuple[int, ...]] = None
            best_width = np.inf
            best_sum = np.inf
            best_L = np.nan
            best_U = np.nan

            max_t = min(max(c, 2), len(lowers_v))
            for t in range(max_t, 1, -1):
                found = False
                for combo in combinations(range(len(lowers_v)), t):
                    L_int = float(np.max(lowers_v[list(combo)]))
                    U_int = float(np.min(uppers_v[list(combo)]))
                    width = U_int - L_int
                    if width < 0:
                        continue
                    sum_widths = float(np.sum(uppers_v[list(combo)] - lowers_v[list(combo)]))
                    if (
                        (width < best_width)
                        or (np.isclose(width, best_width) and sum_widths < best_sum)
                        or (
                            np.isclose(width, best_width)
                            and np.isclose(sum_widths, best_sum)
                            and (best_subset is None or combo < best_subset)
                        )
                    ):
                        best_subset = combo
                        best_width = width
                        best_sum = sum_widths
                        best_L = L_int
                        best_U = U_int
                    found = True
                if found and best_subset is not None:
                    break

            if best_subset is not None:
                lower_out[i] = np.float32(best_L)
                upper_out[i] = np.float32(best_U)
            else:
                lower_out[i] = np.float32(np.min(lowers_v))
                upper_out[i] = np.float32(np.max(uppers_v))

        combined = base.copy()
        combined["upper"] = upper_out
        combined["lower"] = lower_out
        combined["width"] = (combined["upper"] - combined["lower"]).astype(np.float32)
        combined["divergence"] = "combined"
        return combined

    if isinstance(divergence, str):
        div_key = divergence.strip().lower()
        base_divs = ["KL", "Chi2", "JS", "TV", "Hellinger"]
        if div_key == "combined":
            return _combined_robust(base_divs, c=3)

    return _compute_for_div(divergence)


def _fit_def6_upper_only(
    Y: np.ndarray,
    A: np.ndarray,
    divergence: Union[str, FDivergenceLike],
    phi: PhiFn,
    *,
    e0_hat: float,
    e1_hat: float,
    seed: int = 0,
    num_epochs: int = 3000,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    max_grad_norm: Optional[float] = 10.0,
    h_clip: float = 20.0,
    device: str = "cpu",
    verbose: bool = False,
    log_every: int = 200,
) -> tuple[Dict[int, float], Dict[str, Any], Dict[str, Any]]:
    div = get_divergence(divergence)
    set_global_seed(seed, deterministic_torch=False)

    Y_arr = np.asarray(Y, dtype=np.float32).reshape(-1)
    A_arr = np.asarray(A, dtype=np.int64).reshape(-1)

    Y_t = torch.as_tensor(Y_arr, dtype=torch.float32, device=device)
    A_t = torch.as_tensor(A_arr, dtype=torch.int64, device=device)

    h = torch.nn.Parameter(torch.zeros(2, device=device))
    u = torch.nn.Parameter(torch.zeros(2, device=device))

    optimizer = torch.optim.Adam([h, u], lr=lr, weight_decay=weight_decay)

    eta0_t = div.B_torch(torch.tensor(float(e0_hat), dtype=torch.float32, device=device))
    eta1_t = div.B_torch(torch.tensor(float(e1_hat), dtype=torch.float32, device=device))
    etaA = torch.where(A_t == 1, eta1_t, eta0_t)

    phiY = phi(Y_t)

    last_loss = None
    for epoch in range(int(num_epochs)):
        optimizer.zero_grad(set_to_none=True)

        hA = h[A_t]
        uA = u[A_t]
        hA_clamped = torch.clamp(hA, min=-h_clip, max=h_clip)
        lamA = torch.exp(hA_clamped)

        t = (phiY - uA) / lamA
        gstar = div.g_star(t)
        loss = torch.mean(lamA * (etaA + gstar) + uA)

        if not torch.isfinite(loss).item():
            raise FloatingPointError(
                "Non-finite loss in Definition 6 optimization. "
                "Try reducing lr or increasing h_clip/eps_propensity."
            )

        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_([h, u], max_grad_norm)
        optimizer.step()

        last_loss = float(loss.detach().cpu().item())
        if verbose and log_every > 0:
            if epoch % log_every == 0 or epoch == int(num_epochs) - 1:
                print(f"[def6] epoch={epoch} loss={last_loss:.6f}")

    with torch.no_grad():
        h_clamped = torch.clamp(h, min=-h_clip, max=h_clip)
        lam_hat = torch.exp(h_clamped)
        u_hat = u.detach()

        t = (phiY - u_hat[A_t]) / lam_hat[A_t]
        if hasattr(div, "g_star_with_valid"):
            gstar, valid_mask = div.g_star_with_valid(t)
            gstar_valid_frac = float(valid_mask.float().mean().item())
        else:
            gstar = div.g_star(t)
            gstar_valid_frac = float("nan")

        mask0 = A_t == 0
        mask1 = A_t == 1
        m_hat0 = torch.mean(gstar[mask0])
        m_hat1 = torch.mean(gstar[mask1])

        mu0 = lam_hat[0] * (eta0_t + m_hat0) + u_hat[0]
        mu1 = lam_hat[1] * (eta1_t + m_hat1) + u_hat[1]

    mu_upper = {0: float(mu0.item()), 1: float(mu1.item())}
    diagnostics = {
        "final_loss": last_loss,
        "epochs": int(num_epochs),
        "gstar_valid_frac": gstar_valid_frac,
    }
    fitted_params = {
        "h": h.detach().cpu().numpy(),
        "u": u.detach().cpu().numpy(),
        "lambda": lam_hat.detach().cpu().numpy(),
    }
    return mu_upper, diagnostics, fitted_params


def compute_marginal_bounds_def6(
    Y: np.ndarray,
    A: np.ndarray,
    divergence: Union[str, FDivergenceLike],
    phi: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    *,
    seed: int = 0,
    num_epochs: int = 3000,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    max_grad_norm: Optional[float] = 10.0,
    eps_propensity: float = 1e-6,
    h_clip: float = 20.0,
    device: str = "cpu",
    verbose: bool = False,
    log_every: int = 200,
) -> Dict[str, Any]:
    """
    Implements Definition 6 (X=∅) estimator for marginal causal bounds; lower bound via sign flip.
    """
    if not (0.0 < eps_propensity < 0.5):
        raise ValueError("eps_propensity must be in (0, 0.5).")

    A_arr = np.asarray(A).reshape(-1)
    Y_arr = np.asarray(Y).reshape(-1)

    if A_arr.shape[0] != Y_arr.shape[0]:
        raise ValueError(
            f"Shape mismatch: len(Y)={Y_arr.shape[0]}, len(A)={A_arr.shape[0]}."
        )

    unique = set(np.unique(A_arr).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"A must be binary in {{0,1}}. Found unique values: {sorted(unique)}")

    finite_mask = np.isfinite(Y_arr)
    if not np.all(finite_mask):
        bad = int(Y_arr.shape[0] - np.count_nonzero(finite_mask))
        raise ValueError(f"Y must be finite. Found {bad} non-finite values.")

    n = int(Y_arr.shape[0])
    n1 = int(np.sum(A_arr == 1))
    n0 = int(np.sum(A_arr == 0))
    if n0 == 0 or n1 == 0:
        raise ValueError("Definition 6 requires both treatment arms present in the data.")

    e1_hat = float(n1) / float(n)
    e0_hat = float(n0) / float(n)
    e1_hat = float(np.clip(e1_hat, eps_propensity, 1.0 - eps_propensity))
    e0_hat = float(np.clip(e0_hat, eps_propensity, 1.0 - eps_propensity))

    if phi is None:
        def _phi_identity(y: torch.Tensor) -> torch.Tensor:
            return y
        phi_fn = _phi_identity
    else:
        phi_fn = phi

    mu_upper_pos, diag_pos, params_pos = _fit_def6_upper_only(
        Y=Y_arr,
        A=A_arr,
        divergence=divergence,
        phi=phi_fn,
        e0_hat=e0_hat,
        e1_hat=e1_hat,
        seed=seed,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        h_clip=h_clip,
        device=device,
        verbose=verbose,
        log_every=log_every,
    )

    def _phi_neg(y: torch.Tensor) -> torch.Tensor:
        return -phi_fn(y)

    mu_upper_neg, diag_neg, params_neg = _fit_def6_upper_only(
        Y=Y_arr,
        A=A_arr,
        divergence=divergence,
        phi=_phi_neg,
        e0_hat=e0_hat,
        e1_hat=e1_hat,
        seed=seed,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        h_clip=h_clip,
        device=device,
        verbose=verbose,
        log_every=log_every,
    )

    mu0_upper = mu_upper_pos[0]
    mu1_upper = mu_upper_pos[1]
    mu0_lower = -mu_upper_neg[0]
    mu1_lower = -mu_upper_neg[1]

    div = get_divergence(divergence)

    return {
        "mu0_lower": mu0_lower,
        "mu0_upper": mu0_upper,
        "mu1_lower": mu1_lower,
        "mu1_upper": mu1_upper,
        "e0_hat": e0_hat,
        "e1_hat": e1_hat,
        "divergence": div.name,
        "gstar_valid_frac_pos": diag_pos.get("gstar_valid_frac"),
        "gstar_valid_frac_neg": diag_neg.get("gstar_valid_frac"),
        "opt": {
            "final_loss_pos": diag_pos.get("final_loss"),
            "final_loss_neg": diag_neg.get("final_loss"),
            "epochs": diag_pos.get("epochs"),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "max_grad_norm": max_grad_norm,
            "h_clip": float(h_clip),
            "device": str(device),
            "params_pos": params_pos,
            "params_neg": params_neg,
        },
    }


def compute_ate_bounds_def6(
    Y: np.ndarray,
    A: np.ndarray,
    divergence: Union[str, FDivergenceLike],
    *,
    seed: int = 0,
    num_epochs: int = 3000,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    max_grad_norm: Optional[float] = 10.0,
    eps_propensity: float = 1e-6,
    h_clip: float = 20.0,
    device: str = "cpu",
    verbose: bool = False,
    log_every: int = 200,
) -> Dict[str, Any]:
    """
    Implements Definition 6 (X=∅) estimator for marginal causal bounds; lower bound via sign flip.
    """
    out = compute_marginal_bounds_def6(
        Y=Y,
        A=A,
        divergence=divergence,
        phi=None,
        seed=seed,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        eps_propensity=eps_propensity,
        h_clip=h_clip,
        device=device,
        verbose=verbose,
        log_every=log_every,
    )

    ate_lower = out["mu1_lower"] - out["mu0_upper"]
    ate_upper = out["mu1_upper"] - out["mu0_lower"]

    ate_order_fixed = False
    if ate_lower > ate_upper:
        mid = 0.5 * (ate_lower + ate_upper)
        ate_lower = mid
        ate_upper = mid
        ate_order_fixed = True

    out["ate_lower"] = ate_lower
    out["ate_upper"] = ate_upper
    out["ate_width"] = ate_upper - ate_lower
    out["ate_order_fixed"] = ate_order_fixed
    return out


def run_bound_with_result(*args, **kwargs) -> BoundResult:
    """
    Temporary wrapper to expose a stable API required by AGENTS P0 tests.

    IMPORTANT:
    - This wrapper should call the existing implementation.
    - It should not change training, optimization, or math.
    """
    # TODO: Replace this call with your current entrypoint.
    # Example:
    # out = run_bound(*args, **kwargs)
    out = run_bound(*args, **kwargs)  # adjust if your function name differs

    # TODO: Map your existing outputs to upper/lower arrays.
    # If you currently have only an interval (lower, upper) with a single validity mask,
    # temporarily map it as follows:
    upper = np.asarray(out["upper"], dtype=float) if isinstance(out, dict) else np.asarray(out.upper, dtype=float)
    lower = np.asarray(out["lower"], dtype=float) if isinstance(out, dict) else np.asarray(out.lower, dtype=float)

    # TEMPORARY mapping (will be fixed in Step 8):
    # If your current code has a single mask, map it to both valid_up/valid_lo for now.
    if isinstance(out, dict) and "valid_mask" in out:
        valid_mask = np.asarray(out["valid_mask"], dtype=bool)
    elif hasattr(out, "valid_mask"):
        valid_mask = np.asarray(out.valid_mask, dtype=bool)
    else:
        # If no mask exists, treat finite endpoints + ordering as valid (temporary)
        valid_mask = np.isfinite(lower) & np.isfinite(upper) & (lower <= upper)

    valid_up = valid_mask.copy()
    valid_lo = valid_mask.copy()

    valid_interval = valid_up & valid_lo & np.isfinite(lower) & np.isfinite(upper) & (lower <= upper)

    # blank invalid endpoints
    lower2 = lower.copy()
    upper2 = upper.copy()
    lower2[~valid_interval] = np.nan
    upper2[~valid_interval] = np.nan

    return BoundResult(
        upper=upper2,
        lower=lower2,
        valid_up=valid_up,
        valid_lo=valid_lo,
        valid_interval=valid_interval,
        diagnostics=getattr(out, "diagnostics", None) if not isinstance(out, dict) else out.get("diagnostics", None),
    )
