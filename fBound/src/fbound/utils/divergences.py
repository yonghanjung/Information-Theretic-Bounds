"""
f-divergence components used by the debiased causal bound estimator.

This module implements, for each supported divergence:
- Radius B_f(e) and derivative dB_f(e) w.r.t. e (NumPy + Torch),
- Convex conjugate g^*(t) (Torch), used in the dual objective.

The estimator relies on:
- Eq. (5): B_f(e) = e f(1/e) + (1-e) f(0)
- Eq. (25)-(26): uses dB_f(e) in the debiasing correction term.
- Step 5: pseudo-outcome uses g^*.

Numerical stability:
- g^*(t) is implemented with a large finite penalty outside its domain, instead of +inf.
  This preserves differentiability and training stability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Protocol, runtime_checkable, Union

import numpy as np
import torch


@runtime_checkable
class FDivergenceLike(Protocol):
    name: str
    notes: str
    domain: str

    def B_torch(self, e: torch.Tensor) -> torch.Tensor: ...
    def dB_torch(self, e: torch.Tensor) -> torch.Tensor: ...
    def B_numpy(self, e: np.ndarray) -> np.ndarray: ...
    def dB_numpy(self, e: np.ndarray) -> np.ndarray: ...
    def g_star(self, t: torch.Tensor) -> torch.Tensor: ...
    # Implementations may optionally provide g_star_with_valid(t)->(val,mask).


@dataclass(frozen=True)
class PenaltyConfig:
    boundary_eps: float
    penalty_value: float
    penalty_growth: float


@dataclass(frozen=True)
class FDivergence(FDivergenceLike):
    name: str
    notes: str
    domain: str

    _B_torch: Callable[[torch.Tensor], torch.Tensor]
    _dB_torch: Callable[[torch.Tensor], torch.Tensor]
    _B_numpy: Callable[[np.ndarray], np.ndarray]
    _dB_numpy: Callable[[np.ndarray], np.ndarray]
    _g_star: Callable[[torch.Tensor], torch.Tensor]
    _valid_mask: Callable[[torch.Tensor], torch.Tensor]
    t_max: float = float("inf")
    domain_penalty_scale: float = 1.0
    lambda_min_override: Optional[float] = None

    def B_torch(self, e: torch.Tensor) -> torch.Tensor:
        return self._B_torch(e)

    def dB_torch(self, e: torch.Tensor) -> torch.Tensor:
        return self._dB_torch(e)

    def B_numpy(self, e: np.ndarray) -> np.ndarray:
        return self._B_numpy(e)

    def dB_numpy(self, e: np.ndarray) -> np.ndarray:
        return self._dB_numpy(e)

    def g_star(self, t: torch.Tensor) -> torch.Tensor:
        return self._g_star(t)

    def g_star_with_valid(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return both g*(t) and a boolean mask indicating whether t lies in the
        valid domain (per-divergence threshold).
        """
        t = _ensure_tensor(t)
        return self._g_star(t), self._valid_mask(t)

    def domain_violation(self, t: torch.Tensor) -> torch.Tensor:
        """Return nonnegative violation magnitude for t above the valid domain."""
        t = _ensure_tensor(t)
        return torch.relu(t - self.t_max)


def _ensure_tensor(x: torch.Tensor) -> torch.Tensor:
    """Fail fast if callers accidentally pass NumPy or Python numbers."""
    if not isinstance(x, torch.Tensor):
        raise TypeError("Expected a torch.Tensor.")
    return x


def _penalty_like(t: torch.Tensor, cfg: PenaltyConfig) -> torch.Tensor:
    """Smoothly penalize values outside the valid domain of g*."""
    return cfg.penalty_value + cfg.penalty_growth * (t**2)


def _safe_log(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Numerically safe log with a floor to avoid -inf."""
    return torch.log(torch.clamp(x, min=eps))


def _kl_divergence(cfg: PenaltyConfig, eps_e: float) -> FDivergence:
    def B_t(e: torch.Tensor) -> torch.Tensor:
        e = torch.clamp(e, min=eps_e, max=1.0 - eps_e)
        return -torch.log(e)

    def dB_t(e: torch.Tensor) -> torch.Tensor:
        e = torch.clamp(e, min=eps_e, max=1.0 - eps_e)
        return -1.0 / e

    def B_n(e: np.ndarray) -> np.ndarray:
        e = np.clip(e, eps_e, 1.0 - eps_e)
        return -np.log(e)

    def dB_n(e: np.ndarray) -> np.ndarray:
        e = np.clip(e, eps_e, 1.0 - eps_e)
        return -1.0 / e

    def g_star(t: torch.Tensor) -> torch.Tensor:
        t = _ensure_tensor(t)
        valid = t < -cfg.boundary_eps
        safe = -1.0 - _safe_log(-t, eps=cfg.boundary_eps)
        return torch.where(valid, safe, _penalty_like(t, cfg))

    def valid_mask(t: torch.Tensor) -> torch.Tensor:
        t = _ensure_tensor(t)
        return t < -cfg.boundary_eps

    return FDivergence(
        name="KL",
        notes="KL divergence with f(t)=t log t and g*(t) domain t<0.",
        domain="t < 0",
        _B_torch=B_t,
        _dB_torch=dB_t,
        _B_numpy=B_n,
        _dB_numpy=dB_n,
        _g_star=g_star,
        _valid_mask=valid_mask,
        t_max=-cfg.boundary_eps,
    )


def _hellinger_divergence(cfg: PenaltyConfig, eps_e: float) -> FDivergence:
    def B_t(e: torch.Tensor) -> torch.Tensor:
        e = torch.clamp(e, min=eps_e, max=1.0 - eps_e)
        b1 = 1.0 - torch.sqrt(e)
        b2 = -0.5 * torch.log(e)
        return torch.minimum(b1, b2)

    def dB_t(e: torch.Tensor) -> torch.Tensor:
        e = torch.clamp(e, min=eps_e, max=1.0 - eps_e)
        b1 = 1.0 - torch.sqrt(e)
        b2 = -0.5 * torch.log(e)
        d1 = -0.5 / torch.sqrt(e)
        d2 = -0.5 / e
        return torch.where(b1 <= b2, d1, d2)

    def B_n(e: np.ndarray) -> np.ndarray:
        e = np.clip(e, eps_e, 1.0 - eps_e)
        b1 = 1.0 - np.sqrt(e)
        b2 = -0.5 * np.log(e)
        return np.minimum(b1, b2)

    def dB_n(e: np.ndarray) -> np.ndarray:
        e = np.clip(e, eps_e, 1.0 - eps_e)
        b1 = 1.0 - np.sqrt(e)
        b2 = -0.5 * np.log(e)
        d1 = -0.5 / np.sqrt(e)
        d2 = -0.5 / e
        return np.where(b1 <= b2, d1, d2)

    def g_star(t: torch.Tensor) -> torch.Tensor:
        t = _ensure_tensor(t)
        bound = 0.5
        valid = t < (bound - cfg.boundary_eps)
        denom = 1.0 - 2.0 * t
        val = t / torch.clamp(denom, min=cfg.boundary_eps)
        return torch.where(valid, val, _penalty_like(t, cfg))

    def valid_mask(t: torch.Tensor) -> torch.Tensor:
        t = _ensure_tensor(t)
        bound = 0.5
        return t < (bound - cfg.boundary_eps)

    return FDivergence(
        name="Hellinger",
        notes="Hellinger divergence with B(e)=min(1-sqrt(e), -0.5*log e). g*(t) domain t<1/2.",
        domain="t < 1/2",
        _B_torch=B_t,
        _dB_torch=dB_t,
        _B_numpy=B_n,
        _dB_numpy=dB_n,
        _g_star=g_star,
        _valid_mask=valid_mask,
        t_max=0.5 - cfg.boundary_eps,
    )


def _chi2_divergence(cfg: PenaltyConfig, eps_e: float) -> FDivergence:
    def B_t(e: torch.Tensor) -> torch.Tensor:
        e = torch.clamp(e, min=eps_e, max=1.0 - eps_e)
        return (1.0 - e) / (2.0 * e)

    def dB_t(e: torch.Tensor) -> torch.Tensor:
        e = torch.clamp(e, min=eps_e, max=1.0 - eps_e)
        return -1.0 / (2.0 * (e**2))

    def B_n(e: np.ndarray) -> np.ndarray:
        e = np.clip(e, eps_e, 1.0 - eps_e)
        return (1.0 - e) / (2.0 * e)

    def dB_n(e: np.ndarray) -> np.ndarray:
        e = np.clip(e, eps_e, 1.0 - eps_e)
        return -1.0 / (2.0 * (e**2))

    def g_star(t: torch.Tensor) -> torch.Tensor:
        t = _ensure_tensor(t)
        # Pearson chi-square g*:
        # g*(t) = 1 - sqrt(1 - 2t) for t < 1/2; penalty otherwise.
        bound = 0.5
        valid = t < (bound - cfg.boundary_eps)
        rad = torch.clamp(1.0 - 2.0 * t, min=cfg.boundary_eps)
        val = 1.0 - torch.sqrt(rad)
        return torch.where(valid, val, _penalty_like(t, cfg))

    def valid_mask(t: torch.Tensor) -> torch.Tensor:
        t = _ensure_tensor(t)
        return t < (0.5 - cfg.boundary_eps)

    return FDivergence(
        name="Chi2",
        notes="Pearson chi-square divergence with B(e)=(1-e)/(2e). g*(t)=1-sqrt(1-2t) for t<1/2.",
        domain="t < 1/2",
        _B_torch=B_t,
        _dB_torch=dB_t,
        _B_numpy=B_n,
        _dB_numpy=dB_n,
        _g_star=g_star,
        _valid_mask=valid_mask,
        t_max=0.5 - cfg.boundary_eps,
        domain_penalty_scale=1.0,
        lambda_min_override=None,
    )


def _tv_divergence(cfg: PenaltyConfig, eps_e: float, scaled: bool) -> FDivergence:
    """
    Total variation (TV) divergence.

    Two conventions are used in the literature:

    1) "standard TV" (default here, consistent with the original fBound.py):
       f(t) = |t-1|/2  =>  B_TV(e) = 1 - e and g*(t) has thresholds at ±1/2.

    2) "paper-unscaled TV":
       f(t) = |t-1|    =>  B_TV(e) = 2(1-e) and g*(t) thresholds at ±1.

    This module exposes both:
      - "TV"           -> standard TV (scaled=True)
      - "TV_unscaled"  -> paper-unscaled (scaled=False)
    """
    if scaled:
        def B_t(e: torch.Tensor) -> torch.Tensor:
            e = torch.clamp(e, min=eps_e, max=1.0 - eps_e)
            b1 = torch.sqrt(1.0 - e)
            b2 = torch.sqrt(-0.5 * torch.log(e))
            return torch.minimum(b1, b2)

        def dB_t(e: torch.Tensor) -> torch.Tensor:
            e = torch.clamp(e, min=eps_e, max=1.0 - eps_e)
            b1 = torch.sqrt(1.0 - e)
            b2 = torch.sqrt(-0.5 * torch.log(e))
            d1 = -0.5 / b1
            d2 = -0.25 / (e * b2)
            return torch.where(b1 <= b2, d1, d2)

        def B_n(e: np.ndarray) -> np.ndarray:
            e = np.clip(e, eps_e, 1.0 - eps_e)
            b1 = np.sqrt(1.0 - e)
            b2 = np.sqrt(-0.5 * np.log(e))
            return np.minimum(b1, b2)

        def dB_n(e: np.ndarray) -> np.ndarray:
            e = np.clip(e, eps_e, 1.0 - eps_e)
            b1 = np.sqrt(1.0 - e)
            b2 = np.sqrt(-0.5 * np.log(e))
            d1 = -0.5 / b1
            d2 = -0.25 / (e * b2)
            return np.where(b1 <= b2, d1, d2)

        thr = 0.5
        c = 0.5

        def g_star(t: torch.Tensor) -> torch.Tensor:
            t = _ensure_tensor(t)
            left = torch.full_like(t, -c)
            mid = t
            val = torch.where(t <= -thr, left, mid)
            valid = t < (thr - cfg.boundary_eps)
            return torch.where(valid, val, _penalty_like(t, cfg))

        def valid_mask(t: torch.Tensor) -> torch.Tensor:
            t = _ensure_tensor(t)
            return t < (thr - cfg.boundary_eps)

        return FDivergence(
            name="TV",
            notes="Standard TV: f(t)=|t-1|/2, B(e)=min(sqrt(1-e), sqrt(-0.5 log e)), g* thresholds ±1/2.",
            domain="t <= 1/2 (finite); penalty for t>1/2.",
            _B_torch=B_t,
            _dB_torch=dB_t,
            _B_numpy=B_n,
            _dB_numpy=dB_n,
            _g_star=g_star,
            _valid_mask=valid_mask,
            t_max=thr - cfg.boundary_eps,
        )

    def B_t(e: torch.Tensor) -> torch.Tensor:
        e = torch.clamp(e, min=eps_e, max=1.0 - eps_e)
        return 2.0 * (1.0 - e)

    def dB_t(e: torch.Tensor) -> torch.Tensor:
        _ = e
        return torch.full_like(e, -2.0)

    def B_n(e: np.ndarray) -> np.ndarray:
        e = np.clip(e, eps_e, 1.0 - eps_e)
        return 2.0 * (1.0 - e)

    def dB_n(e: np.ndarray) -> np.ndarray:
        return -2.0 * np.ones_like(e, dtype=float)

    thr = 1.0
    c = 1.0

    def g_star(t: torch.Tensor) -> torch.Tensor:
        t = _ensure_tensor(t)
        left = torch.full_like(t, -c)
        mid = t
        val = torch.where(t <= -thr, left, mid)
        valid = t < (thr - cfg.boundary_eps)
        return torch.where(valid, val, _penalty_like(t, cfg))

    def valid_mask(t: torch.Tensor) -> torch.Tensor:
        t = _ensure_tensor(t)
        return t < (thr - cfg.boundary_eps)

    return FDivergence(
        name="TV_unscaled",
        notes="Unscaled TV (paper): f(t)=|t-1|, B(e)=2(1-e), g* thresholds ±1.",
        domain="t <= 1 (finite); penalty for t>1.",
        _B_torch=B_t,
        _dB_torch=dB_t,
        _B_numpy=B_n,
        _dB_numpy=dB_n,
        _g_star=g_star,
        _valid_mask=valid_mask,
        t_max=thr - cfg.boundary_eps,
    )


def _js_divergence(cfg: PenaltyConfig, eps_e: float) -> FDivergence:
    """
    Jensen–Shannon (JS) divergence.

    g*_JS(t) = -0.5 * log(2 - exp(2t)) for t < 0.5 * log 2; penalty otherwise.
    Radius:
      B(e) = 0.5 * [e log e - (1+e) log(1+e)] + log 2.
    """
    log2 = float(np.log(2.0))
    half_log2 = 0.5 * log2

    def B_t(e: torch.Tensor) -> torch.Tensor:
        e = torch.clamp(e, min=eps_e, max=1.0 - eps_e)
        return 0.5 * (e * torch.log(e) - (1.0 + e) * torch.log(1.0 + e)) + log2

    def dB_t(e: torch.Tensor) -> torch.Tensor:
        e = torch.clamp(e, min=eps_e, max=1.0 - eps_e)
        return 0.5 * (torch.log(e) - torch.log(1.0 + e))

    def B_n(e: np.ndarray) -> np.ndarray:
        e = np.clip(e, eps_e, 1.0 - eps_e)
        return 0.5 * (e * np.log(e) - (1.0 + e) * np.log(1.0 + e)) + log2

    def dB_n(e: np.ndarray) -> np.ndarray:
        e = np.clip(e, eps_e, 1.0 - eps_e)
        return 0.5 * (np.log(e) - np.log(1.0 + e))

    def g_star(t: torch.Tensor) -> torch.Tensor:
        t = _ensure_tensor(t)
        valid = t < (half_log2 - cfg.boundary_eps)
        exp_2t = torch.exp(torch.clamp(2.0 * t, max=2.0 * (half_log2 - cfg.boundary_eps)))
        denom = 2.0 - exp_2t
        safe = -0.5 * torch.log(torch.clamp(denom, min=cfg.boundary_eps))
        return torch.where(valid, safe, _penalty_like(t, cfg))

    def valid_mask(t: torch.Tensor) -> torch.Tensor:
        t = _ensure_tensor(t)
        return t < (half_log2 - cfg.boundary_eps)

    return FDivergence(
        name="JS",
        notes="JS divergence with g*(t)=-0.5*log(2-exp(2t)) domain t<0.5*log 2.",
        domain="t < 0.5*log 2",
        _B_torch=B_t,
        _dB_torch=dB_t,
        _B_numpy=B_n,
        _dB_numpy=dB_n,
        _g_star=g_star,
        _valid_mask=valid_mask,
        t_max=half_log2 - cfg.boundary_eps,
    )


_REGISTRY: Dict[str, FDivergence] = {}


def _build_default_registry() -> None:
    if _REGISTRY:
        return
    penalty = PenaltyConfig(boundary_eps=1e-6, penalty_value=1e6, penalty_growth=1e2)
    eps_e = 1e-6

    for div in (
        _kl_divergence(penalty, eps_e),
        _hellinger_divergence(penalty, eps_e),
        _chi2_divergence(penalty, eps_e),
        _tv_divergence(penalty, eps_e, scaled=True),
        _tv_divergence(penalty, eps_e, scaled=False),
        _js_divergence(penalty, eps_e),
    ):
        _REGISTRY[div.name] = div

    _REGISTRY["TOTAL_VARIATION"] = _REGISTRY["TV"]
    _REGISTRY["TV_PAPER"] = _REGISTRY["TV_unscaled"]


def register_divergence(name: str, divergence: FDivergenceLike) -> None:
    """Add a custom divergence to the registry (validates interface at runtime)."""
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string.")
    if not isinstance(divergence, FDivergenceLike):
        raise TypeError(
            "divergence must implement FDivergenceLike (B, dB, g_star and metadata)."
        )

    def default_mask(t: torch.Tensor) -> torch.Tensor:
        t = _ensure_tensor(t)
        return torch.ones_like(t, dtype=torch.bool)

    # Prefer a custom validity mask if the divergence exposes g_star_with_valid.
    if hasattr(divergence, "g_star_with_valid"):
        try:
            _, mask_sample = divergence.g_star_with_valid(torch.tensor([0.0]))
            _ = mask_sample  # ensure it works

            def custom_mask(t: torch.Tensor) -> torch.Tensor:
                _, m = divergence.g_star_with_valid(t)
                return m

            valid_mask = custom_mask
        except Exception:
            valid_mask = default_mask
    else:
        valid_mask = default_mask

    _REGISTRY[name] = FDivergence(
        name=divergence.name,
        notes=divergence.notes,
        domain=divergence.domain,
        _B_torch=divergence.B_torch,
        _dB_torch=divergence.dB_torch,
        _B_numpy=divergence.B_numpy,
        _dB_numpy=divergence.dB_numpy,
        _g_star=divergence.g_star,
        _valid_mask=valid_mask,
        t_max=float(getattr(divergence, "t_max", float("inf"))),
        domain_penalty_scale=float(getattr(divergence, "domain_penalty_scale", 1.0)),
        lambda_min_override=getattr(divergence, "lambda_min_override", None),
    )


def get_divergence(divergence: Union[str, FDivergenceLike]) -> FDivergence:
    """Retrieve a registered divergence (case-insensitive), or wrap a custom one."""
    _build_default_registry()

    if isinstance(divergence, str):
        key = divergence.strip()
        if not key:
            raise ValueError("divergence name must be non-empty.")
        key_upper = key.upper()
        if key in _REGISTRY:
            return _REGISTRY[key]
        if key_upper in _REGISTRY:
            return _REGISTRY[key_upper]
        alias = {
            "CHI": "Chi2",
            "CHI2": "Chi2",
            "HELLINGER": "Hellinger",
            "KL": "KL",
            "JS": "JS",
            "TV": "TV",
            "TV_UNSCALED": "TV_unscaled",
        }.get(key_upper)
        if alias and alias in _REGISTRY:
            return _REGISTRY[alias]
        raise KeyError(
            f"Unknown divergence '{divergence}'. Available: {sorted(_REGISTRY.keys())}"
        )

    if not isinstance(divergence, FDivergenceLike):
        raise TypeError("Custom divergence must satisfy FDivergenceLike protocol.")

    def default_mask(t: torch.Tensor) -> torch.Tensor:
        t = _ensure_tensor(t)
        return torch.ones_like(t, dtype=torch.bool)

    if hasattr(divergence, "g_star_with_valid"):
        try:
            def custom_mask(t: torch.Tensor) -> torch.Tensor:
                _, m = divergence.g_star_with_valid(t)
                return m
            valid_mask = custom_mask
        except Exception:
            valid_mask = default_mask
    else:
        valid_mask = default_mask

    return FDivergence(
        name=divergence.name,
        notes=divergence.notes,
        domain=divergence.domain,
        _B_torch=divergence.B_torch,
        _dB_torch=divergence.dB_torch,
        _B_numpy=divergence.B_numpy,
        _dB_numpy=divergence.dB_numpy,
        _g_star=divergence.g_star,
        _valid_mask=valid_mask,
        t_max=float(getattr(divergence, "t_max", float("inf"))),
    )
