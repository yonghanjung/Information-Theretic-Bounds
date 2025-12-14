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
    if A.ndim != 1:
        raise ValueError(f"A must be 1D torch tensor. Got shape {tuple(A.shape)}")
    if X.ndim != 2:
        raise ValueError(f"X must be 2D torch tensor. Got shape {tuple(X.shape)}")
    return torch.cat([A.reshape(-1, 1), X], dim=1)


def _predict_proba_class1(model: Any, X: np.ndarray) -> np.ndarray:
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

    def fit(self, X: np.ndarray, A: np.ndarray, Y: np.ndarray) -> "DebiasedCausalBoundEstimator":
        Yc, Ac, Xc = check_shapes(Y=Y, A=A, X=X)
        n, _ = Xc.shape
        self.X_ = Xc
        self.A_ = Ac
        self.Y_ = Yc

        set_global_seed(self.seed, deterministic_torch=self.fit_cfg.deterministic_torch)

        self.splits_ = make_kfold_splits(n=n, n_splits=self.fit_cfg.n_folds, seed=self.seed, shuffle=True)

        fold_id = np.empty(n, dtype=int)
        for k, (_, fold_idx) in enumerate(self.splits_):
            fold_id[fold_idx] = k
        self.fold_id_ = fold_id

        e1_oof = np.empty(n, dtype=np.float32)

        self.propensity_models_.clear()
        self.h_nets_.clear()
        self.u_nets_.clear()
        self.m_models_.clear()
        self.final_dual_loss_.clear()

        for k, (train_idx, fold_idx) in enumerate(self.splits_):
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

            self.propensity_models_.append(prop_model)
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

        preds = np.zeros((Xq.shape[0],), dtype=np.float64)
        for k in range(K):
            preds += self._predict_fold(k=k, a=a, Xq=Xq)
        preds /= float(K)
        return preds.astype(np.float32)

    def _predict_fold(self, k: int, a: int, Xq: np.ndarray) -> np.ndarray:
        prop = self.propensity_models_[k]
        h_net = self.h_nets_[k]
        u_net = self.u_nets_[k]
        m_reg = self.m_models_[k]

        e1 = _predict_proba_class1(prop, Xq.astype(np.float64, copy=False))
        e1 = np.clip(e1, self.fit_cfg.eps_propensity, 1.0 - self.fit_cfg.eps_propensity)
        eA = e1 if a == 1 else (1.0 - e1)

        eta = self.divergence.B_numpy(eA.astype(np.float64)).astype(np.float64)

        X_t = torch.tensor(Xq.astype(np.float32), dtype=torch.float32)
        A_t = torch.full((Xq.shape[0],), float(a), dtype=torch.float32)
        ax = _concat_ax(A_t, X_t)

        with torch.no_grad():
            h = h_net(ax)
            h = torch.clamp(h, min=-self.dual_net_cfg.h_clip, max=self.dual_net_cfg.h_clip)
            lam = torch.exp(h).cpu().numpy().astype(np.float64)
            u = u_net(ax).cpu().numpy().astype(np.float64)

        AX = np.concatenate([np.full((Xq.shape[0], 1), float(a), dtype=np.float32), Xq.astype(np.float32)], axis=1)
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
    Yc, Ac, Xc = check_shapes(Y=Y, A=A, X=X)

    est = DebiasedCausalBoundEstimator(
        divergence=divergence,
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
        divergence=divergence,
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

    truth = np.full((Xc.shape[0],), np.nan, dtype=np.float32)
    if GroundTruth is not None:
        try:
            truth = np.asarray(GroundTruth(1, Xc), dtype=np.float32).reshape(-1)
        except Exception as e:
            raise TypeError(
                "GroundTruth must be callable like GroundTruth(a:int, X:(n,d))->(n,)."
            ) from e

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
