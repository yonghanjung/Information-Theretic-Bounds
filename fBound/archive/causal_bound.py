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

from divergences import FDivergence, FDivergenceLike, get_divergence
from models import TorchMLP, make_classifier, make_regressor
from utils import check_shapes, make_kfold_splits, set_global_seed


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
        prop_model = make_classifier(
            propensity_model,
            config=propensity_config,
            seed=int(seed) + 10_000 + k,
        )
        prop_model.fit(Xc[train_idx].astype(np.float64, copy=False), Ac[train_idx])

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

        # Cross-fitting splits / propensity models (can be cached).
        if propensity_cache is None:
            self.splits_ = make_kfold_splits(n=n, n_splits=self.fit_cfg.n_folds, seed=self.seed, shuffle=True)
            prop_models: Optional[list[Any]] = None
            e1_oof = np.empty(n, dtype=np.float32)
        else:
            splits = propensity_cache.get("splits", None)
            models = propensity_cache.get("models", None)
            e1_oof_in = propensity_cache.get("e1_oof", None)
            fold_id_in = propensity_cache.get("fold_id", None)

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

        # Fold alignment
        if propensity_cache is not None and fold_id_in is not None:
            fold_id_arr = np.asarray(fold_id_in, dtype=int).reshape(-1)
            if fold_id_arr.shape[0] != n:
                raise ValueError(f"propensity_cache['fold_id'] length mismatch: expected {n}, got {fold_id_arr.shape[0]}.")
            self.fold_id_ = fold_id_arr
        else:
            fold_id = np.empty(n, dtype=int)
            for k, (_, fold_idx) in enumerate(self.splits_):
                fold_id[fold_idx] = k
            self.fold_id_ = fold_id

        # Reset fitted components (keep propensity models if cached).
        if propensity_cache is None:
            self.propensity_models_.clear()
        else:
            self.propensity_models_ = prop_models if prop_models is not None else []

        self.h_nets_.clear()
        self.u_nets_.clear()
        self.m_models_.clear()
        self.final_dual_loss_.clear()

        # Cross-fitting loop: fit nuisance models on train folds, evaluate dual loss on held-out fold.
        for k, (train_idx, fold_idx) in enumerate(self.splits_):
            if propensity_cache is None:
                prop_model = make_classifier(
                    self.propensity_model_spec,
                    config=self.fit_cfg.propensity_config,
                    seed=self.seed + 10_000 + k,
                )
                prop_model.fit(Xc[train_idx].astype(np.float64, copy=False), Ac[train_idx])

                e1_fold = _predict_proba_class1(prop_model, Xc[fold_idx].astype(np.float64, copy=False))
                e1_fold = np.clip(e1_fold, self.fit_cfg.eps_propensity, 1.0 - self.fit_cfg.eps_propensity).astype(
                    np.float32
                )
                e1_oof[fold_idx] = e1_fold
                self.propensity_models_.append(prop_model)
            else:
                # Cache mode: propensity models + e1_oof are provided.
                _ = self.propensity_models_[k]  # used later at prediction time
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
        h_ax = h_net(ax)
        u_ax = u_net(ax)
        h_ax = torch.clamp(h_ax, min=-self.dual_net_cfg.h_clip, max=self.dual_net_cfg.h_clip)
        lam_ax = torch.exp(h_ax)

        phi_y = self.phi(Y)
        t = (phi_y - u_ax) / lam_ax
        g_star_val = self.divergence.g_star(t)

        eA = torch.where(A >= 0.5, e1, e0)
        eta = self.divergence.B_torch(eA)

        main = lam_ax * (eta + g_star_val) + u_ax

        # Debiasing correction terms at A=0 and A=1 (Eq. 26).
        zeros = torch.zeros_like(A)
        ones = torch.ones_like(A)

        ax0 = _concat_ax(zeros, X)
        ax1 = _concat_ax(ones, X)

        h0 = torch.clamp(h_net(ax0), min=-self.dual_net_cfg.h_clip, max=self.dual_net_cfg.h_clip)
        h1 = torch.clamp(h_net(ax1), min=-self.dual_net_cfg.h_clip, max=self.dual_net_cfg.h_clip)

        lam0 = torch.exp(h0)
        lam1 = torch.exp(h1)

        eta_prime0 = self.divergence.dB_torch(e0)
        eta_prime1 = self.divergence.dB_torch(e1)

        I0 = 1.0 - A
        I1 = A

        corr = e0 * lam0 * eta_prime0 * (I0 - e0) + e1 * lam1 * eta_prime1 * (I1 - e1)

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
        lam = torch.exp(h)
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
        lam = torch.exp(h)
        u = u_net(ax)

        t = (self.phi(Y_t) - u) / lam
        g_val, valid_mask = self.divergence.g_star_with_valid(t)
        return g_val.cpu().numpy().astype(np.float32), valid_mask.cpu().numpy()

    def predict_bound(self, a: int, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Estimator is not fitted. Call fit(X,A,Y) first.")
        if a not in (0, 1):
            raise ValueError("a must be 0 or 1.")
        Xq = np.asarray(X, dtype=np.float32)
        if Xq.ndim != 2:
            raise ValueError(f"X must be 2D. Got {Xq.shape}.")
        if self.X_ is None:
            raise RuntimeError("Internal error: missing fitted X_.")
        if Xq.shape[1] != self.X_.shape[1]:
            raise ValueError(
                f"X has wrong number of features: expected {self.X_.shape[1]}, got {Xq.shape[1]}."
            )

        K = len(self.propensity_models_)
        if K == 0:
            raise RuntimeError("No fitted folds found.")

        # Precompute shared tensors/arrays once to avoid per-fold allocations.
        Xq64 = Xq.astype(np.float64, copy=False)
        AX_shared = np.concatenate(
            [np.full((Xq.shape[0], 1), float(a), dtype=np.float32), Xq.astype(np.float32)],
            axis=1,
        ).astype(np.float64, copy=False)
        X_t_shared = torch.tensor(Xq.astype(np.float32), dtype=torch.float32)
        A_t_shared = torch.full((Xq.shape[0],), float(a), dtype=torch.float32)
        ax_t_shared = _concat_ax(A_t_shared, X_t_shared)

        preds = np.zeros((Xq.shape[0],), dtype=np.float64)
        for k in range(K):
            preds += self._predict_fold(
                k=k,
                a=a,
                Xq=Xq64,
                AX=AX_shared,
                ax_t=ax_t_shared,
            )
        preds /= float(K)
        return preds.astype(np.float32)

    def _predict_fold(self, k: int, a: int, Xq: np.ndarray, AX: np.ndarray, ax_t: torch.Tensor) -> np.ndarray:
        """Per-fold bound contribution theta_k(a, x)."""
        prop = self.propensity_models_[k]
        h_net = self.h_nets_[k]
        u_net = self.u_nets_[k]
        m_reg = self.m_models_[k]

        e1 = _predict_proba_class1(prop, Xq)
        e1 = np.clip(e1, self.fit_cfg.eps_propensity, 1.0 - self.fit_cfg.eps_propensity)
        eA = e1 if a == 1 else (1.0 - e1)

        eta = self.divergence.B_numpy(eA.astype(np.float64)).astype(np.float64)

        with torch.no_grad():
            h = h_net(ax_t)
            h = torch.clamp(h, min=-self.dual_net_cfg.h_clip, max=self.dual_net_cfg.h_clip)
            lam = torch.exp(h).cpu().numpy().astype(np.float64)
            u = u_net(ax_t).cpu().numpy().astype(np.float64)

        m = m_reg.predict(AX.astype(np.float64, copy=False)).astype(np.float64)

        theta = lam * (eta + m) + u
        return theta

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
) -> pd.DataFrame:
    """
    Convenience wrapper that runs the two-pass (phi and -phi) estimator.

    Returns a DataFrame with lower/upper bounds for E[phi(Y)|do(A=1), X].
    """
    Yc, Ac, Xc = check_shapes(Y=Y, A=A, X=X)

    truth = np.full((Xc.shape[0],), np.nan, dtype=np.float32)
    if GroundTruth is not None:
        try:
            truth = np.asarray(GroundTruth(1, Xc), dtype=np.float32).reshape(-1)
        except Exception as e:
            raise TypeError(
                "GroundTruth must be callable like GroundTruth(a:int, X:(n,d))->(n,)."
            ) from e

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
        ).fit(Xc, Ac, Yc)

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
        ).fit(Xc, Ac, Yc)

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
            # "fold",
            "ehat1_oof",
            "dual_loss_fold",
            "divergence",
        ]
        out = out[cols]
        return out

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
