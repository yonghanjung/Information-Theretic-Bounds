import os
# ---------------------------------------------------------------------
# macOS safety knobs (must be set BEFORE numpy/sklearn/torch imports)
# ---------------------------------------------------------------------
# These reduce BLAS/OpenMP instability and oversubscription on some macOS
# Apple-silicon + Accelerate/SME builds that can emit spurious matmul warnings
# and (in worst cases) segfault.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# NOTE: xgboost is unused in this script. Importing it on macOS can pull in
# another OpenMP runtime and increase chances of low-level crashes.
# If/when you actually need it, import it inside that code path.
# from xgboost import XGBClassifier, XGBRegressor

import torch
import torch.nn as nn
import torch.optim as optim

# Keep torch from spawning many threads (helps stability on some macOS builds)
torch.set_num_threads(1)


# ---------------------------------------------------------------------
# Helper modules
# ---------------------------------------------------------------------

class MLP(nn.Module):
    """
    Simple MLP for h_beta(a, x) and u_gamma(a, x).
    Input is concatenated [A, X].
    """
    def __init__(self, input_dim, hidden_sizes=(64, 64), activation=nn.ReLU()):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(d, h))
            layers.append(activation)
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (n, input_dim)
        return self.net(x).squeeze(-1)  # (n,)


# ---------------------------------------------------------------------
# f-divergence ingredients (Corollary 3 and 8)
#
# We implement the following divergences:
#   - KL
#   - Hellinger
#   - Chi-square
#   - Total variation (TV)  [implemented as standard TV distance]
#   - Jensen-Shannon (JS)
#
# Each divergence provides:
#   - B_f(e)    : radius function (Theorem 2 / Corollary 3)
#   - B'_f(e)   : derivative wrt e (Definition 12 debiasing term)
#   - g*_f(t)   : convex conjugate (Corollary 8)
#
# Notes on TV:
#   The paper writes f(t)=|t-1| but then uses the bound 1-e (which corresponds
#   to standard TV = (1/2)||P-Q||_1). Here we implement standard TV with
#   f(t)=|t-1|/2 so that B_TV(e)=1-e and g*_TV has threshold ±1/2.
# ---------------------------------------------------------------------

def B_kl_torch(e):
    e = torch.clamp(e, 1e-6, 1 - 1e-6)
    return -torch.log(e)


def dB_kl_torch(e):
    e = torch.clamp(e, 1e-6, 1 - 1e-6)
    return -1.0 / e


def B_kl_numpy(e):
    e = np.clip(e, 1e-6, 1 - 1e-6)
    return -np.log(e)


def g_star_kl_torch(t):
    """
    Convex conjugate g*_KL(t) = -1 - log(-t) for t < 0, +∞ otherwise.
    We implement +∞ as a large penalty to keep gradients finite.
    """
    eps = 1e-6
    # mask for valid region t < -eps
    mask = (t < -eps)
    val_valid = -1.0 - torch.log(-t.clamp_max(-eps))
    val_invalid = 1e6 + 1e3 * t.pow(2)  # huge penalty when t >= -eps
    return torch.where(mask, val_valid, val_invalid)


# ----------------------------- Hellinger -----------------------------

def B_hellinger_torch(e):
    """B_H(e) = 1 - sqrt(e)."""
    e = torch.clamp(e, 1e-6, 1 - 1e-6)
    return 1.0 - torch.sqrt(e)


def dB_hellinger_torch(e):
    e = torch.clamp(e, 1e-6, 1 - 1e-6)
    return -0.5 / torch.sqrt(e)


def B_hellinger_numpy(e):
    e = np.clip(e, 1e-6, 1 - 1e-6)
    return 1.0 - np.sqrt(e)


def g_star_hellinger_torch(t):
    """g*_H(t) = t/(1-2t) for t<1/2, +∞ otherwise."""
    eps = 1e-6
    thresh = 0.5 - eps
    mask = t < thresh
    denom = (1.0 - 2.0 * t).clamp_min(eps)
    val_valid = t / denom
    val_invalid = 1e6 + 1e3 * (t - thresh).pow(2)
    return torch.where(mask, val_valid, val_invalid)


# ----------------------------- Chi-square ----------------------------

def B_chi2_torch(e):
    """B_{chi^2}(e) = (1-e)/(2e)."""
    e = torch.clamp(e, 1e-6, 1 - 1e-6)
    return (1.0 - e) / (2.0 * e)


def dB_chi2_torch(e):
    e = torch.clamp(e, 1e-6, 1 - 1e-6)
    # (1-e)/(2e) = 1/(2e) - 1/2 => derivative = -1/(2e^2)
    return -0.5 / (e * e)


def B_chi2_numpy(e):
    e = np.clip(e, 1e-6, 1 - 1e-6)
    return (1.0 - e) / (2.0 * e)


def g_star_chi2_torch(t):
    """g*_{chi^2}(t) = -(1+sqrt(1-2t)) for t<=1/2, +∞ otherwise."""
    eps = 1e-6
    thresh = 0.5 - eps
    mask = t <= thresh
    # Keep the sqrt argument in a safe range
    inside = (1.0 - 2.0 * t).clamp_min(eps)
    val_valid = -(1.0 + torch.sqrt(inside))
    val_invalid = 1e6 + 1e3 * (t - thresh).pow(2)
    return torch.where(mask, val_valid, val_invalid)


# --------------------------- Total Variation -------------------------

def B_tv_torch(e):
    """B_TV(e) = 1 - e (standard TV distance)."""
    e = torch.clamp(e, 1e-6, 1 - 1e-6)
    return 1.0 - e


def dB_tv_torch(e):
    # derivative of 1-e is -1
    return -torch.ones_like(e)


def B_tv_numpy(e):
    e = np.clip(e, 1e-6, 1 - 1e-6)
    return 1.0 - e


def g_star_tv_torch(t):
    """g*_TV(t) for standard TV: g(s)=|1-s|/2.

    g*(t) = -1/2                  if t <= -1/2
          = t                     if -1/2 < t <= 1/2
          = +∞                    if t > 1/2
    """
    c = 0.5
    eps = 1e-6
    upper = c - eps
    # region t > 1/2 => invalid
    invalid = t > upper
    val = torch.where(t <= -c, -c * torch.ones_like(t), t)
    val = torch.where(invalid, 1e6 + 1e3 * (t - upper).pow(2), val)
    return val


# -------------------------- Jensen-Shannon ---------------------------

def B_js_torch(e):
    """B_JS(e) = log(4 * e^e / (1+e)^(1+e))."""
    e = torch.clamp(e, 1e-6, 1 - 1e-6)
    log4 = torch.tensor(np.log(4.0), device=e.device, dtype=e.dtype)
    return log4 + e * torch.log(e) - (1.0 + e) * torch.log(1.0 + e)


def dB_js_torch(e):
    e = torch.clamp(e, 1e-6, 1 - 1e-6)
    return torch.log(e) - torch.log(1.0 + e)


def B_js_numpy(e):
    e = np.clip(e, 1e-6, 1 - 1e-6)
    return np.log(4.0) + e * np.log(e) - (1.0 + e) * np.log(1.0 + e)


def g_star_js_torch(t):
    """g*_JS(t) = log( 2 / (4 - exp(t)) ) for t < log 4, +∞ otherwise."""
    eps = 1e-6
    log4 = float(np.log(4.0))
    thresh = log4 - eps
    mask = t < thresh
    # denom = 4 - exp(t)
    denom = (4.0 - torch.exp(t.clamp_max(thresh))).clamp_min(eps)
    log2 = torch.tensor(np.log(2.0), device=t.device, dtype=t.dtype)
    val_valid = log2 - torch.log(denom)
    val_invalid = 1e6 + 1e3 * (t - thresh).pow(2)
    return torch.where(mask, val_valid, val_invalid)


def get_f_divergence(name: str):
    """Return (B_torch, dB_torch, B_numpy, g_star_torch) for the requested divergence."""
    key = name.strip().lower()
    if key in {"kl", "kullback-leibler", "kullback", "kld"}:
        return B_kl_torch, dB_kl_torch, B_kl_numpy, g_star_kl_torch
    if key in {"hellinger", "h"}:
        return B_hellinger_torch, dB_hellinger_torch, B_hellinger_numpy, g_star_hellinger_torch
    if key in {"chi2", "chisq", "chi-square", "chi_square", "chi"}:
        return B_chi2_torch, dB_chi2_torch, B_chi2_numpy, g_star_chi2_torch
    if key in {"tv", "total_variation", "total-variation"}:
        return B_tv_torch, dB_tv_torch, B_tv_numpy, g_star_tv_torch
    if key in {"js", "jensen-shannon", "jensen_shannon"}:
        return B_js_torch, dB_js_torch, B_js_numpy, g_star_js_torch
    raise ValueError(
        f"Unknown divergence '{name}'. Choose from: kl, hellinger, chi2, tv, js."
    )


# ---------------------------------------------------------------------
# Debiased estimator (Definition 14)
# ---------------------------------------------------------------------

class DebiasedCausalBoundEstimator:
    """
    Implementation of Definition 14 (debiased causal bound estimator)
    for a user-selected f-divergence and a functional φ(Y).

    - h_beta(a,x), u_gamma(a,x) are neural nets (PyTorch).
    - Propensity model e1(x) = P(A=1 | X=x) is any sklearn-style classifier
      with fit(X, A) and predict_proba(X).
    - m(a,x) is any regressor with fit(X, y) and predict(X) (e.g., RF, XGBoost).
    """

    def __init__(
        self,
        K=2,
        divergence: str = "kl",
        h_hidden=(64, 64),
        u_hidden=(64, 64),
        device=None,
        propensity_model_factory=None,
        m_model_factory=None,
        # φ is a torch function; default φ(y)=y (mean functional)
        phi_torch=None,
    ):
        self.K = K
        self.h_hidden = list(h_hidden)
        self.u_hidden = list(u_hidden)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # default propensity: logistic regression
        if propensity_model_factory is None:
            self.propensity_model_factory = lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000,
                    # liblinear avoids the NumPy/BLAS matmul path used by lbfgs
                    # that can trigger spurious matmul warnings (and sometimes
                    # segfaults) on some macOS Apple-silicon builds.
                    solver="liblinear",
                    C=0.05,  # stronger regularization for stability on separable data
                    penalty="l2",
                    n_jobs=1,
                ),
            )
        else:
            self.propensity_model_factory = propensity_model_factory

        # default m-model: random forest
        if m_model_factory is None:
            self.m_model_factory = lambda: RandomForestRegressor(
                n_estimators=200, min_samples_leaf=5, random_state=0
            )
        else:
            self.m_model_factory = m_model_factory

        # φ(Y): default is identity (mean)
        if phi_torch is None:
            self.phi_torch = lambda y: y
        else:
            self.phi_torch = phi_torch

        # f-divergence pieces (Corollary 3 and 8)
        self.divergence = divergence
        self.B_f, self.dB_f, self.B_f_np, self.g_star = get_f_divergence(divergence)

        # fitted objects (populated by fit)
        self.kfold_ = None
        self.propensity_models_ = []
        self.h_nets_ = []
        self.u_nets_ = []
        self.m_models_ = []

    # ---------------------- internal helpers --------------------------

    def _build_net(self, input_dim, hidden_sizes):
        return MLP(input_dim, hidden_sizes)

    def _batch_loss(self, X_batch, A_batch, Y_batch, e1_batch, e0_batch, h_net, u_net):
        """
        One minibatch of the debiased risk ℓ_db(V; (β,γ), e)
        as in Definition 12 + 14. :contentReference[oaicite:1]{index=1}
        """
        # Shapes:
        #   X_batch:  (B, d)
        #   A_batch:  (B, 1) with values 0 or 1
        #   Y_batch:  (B,)
        #   e1_batch: (B,) = P(A=1|X)
        #   e0_batch: (B,) = P(A=0|X)

        # Feature input for h_beta(A,X), u_gamma(A,X)
        AX = torch.cat([A_batch, X_batch], dim=1)  # (B, d+1)
        h_AX = h_net(AX)                           # (B,)
        u_AX = u_net(AX)                           # (B,)
        # exp() can overflow; clamp h to keep training numerically stable
        lam_AX = torch.exp(torch.clamp(h_AX, -20.0, 20.0))  # λ(A,X) > 0

        # φ(Y)
        phi_Y = self.phi_torch(Y_batch)            # (B,)

        # g*( (φ(Y) - u(A,X)) / λ(A,X) )
        t = (phi_Y - u_AX) / lam_AX
        g_val = self.g_star(t)                     # (B,)

        # η_f(A,X) = B_f(e_A(X))
        A_flat = A_batch.view(-1)                  # (B,)
        e_A = torch.where(A_flat > 0.5, e1_batch, e0_batch)
        eta_A = self.B_f(e_A)                      # (B,)

        # Main term: exp(h(A,X)) * ( η_f(A,X) + g* ) + u(A,X)
        loss_main = lam_AX * (eta_A + g_val) + u_AX

        # Debiasing term (Definition 12, eq. (26)): sum over a∈{0,1}
        # sum_a e_a(X) * exp(h(a,X)) * η'_f(e_a(X)) * [1(A=a) - e_a(X)]
        #
        # Here we evaluate h(a,x) for a=0 and a=1 using the same nets.

        A0 = torch.zeros_like(A_batch)
        A1 = torch.ones_like(A_batch)
        AX0 = torch.cat([A0, X_batch], dim=1)
        AX1 = torch.cat([A1, X_batch], dim=1)

        h0 = h_net(AX0)
        h1 = h_net(AX1)
        lam0 = torch.exp(torch.clamp(h0, -20.0, 20.0))
        lam1 = torch.exp(torch.clamp(h1, -20.0, 20.0))

        eta_prime0 = self.dB_f(e0_batch)
        eta_prime1 = self.dB_f(e1_batch)

        A_is1 = (A_flat > 0.5).float()
        A_is0 = 1.0 - A_is1

        sum_term = (
            e0_batch * lam0 * eta_prime0 * (A_is0 - e0_batch)
            + e1_batch * lam1 * eta_prime1 * (A_is1 - e1_batch)
        )

        loss = loss_main + sum_term
        return loss.mean()

    @torch.no_grad()
    def _compute_Z(self, X_batch, A_batch, Y_batch, h_net, u_net):
        """
        Z_i = g*( (φ(Y_i) - u(A_i,X_i)) / λ(A_i,X_i) ) for Step 5 in Definition 14.
        """
        AX = torch.cat([A_batch, X_batch], dim=1)
        h_AX = h_net(AX)
        u_AX = u_net(AX)
        lam_AX = torch.exp(torch.clamp(h_AX, -20.0, 20.0))
        phi_Y = self.phi_torch(Y_batch)
        t = (phi_Y - u_AX) / lam_AX
        g_val = self.g_star(t)
        return g_val.cpu().numpy()

    # -------------------------- public API ----------------------------

    def fit(
        self,
        X,
        A,
        Y,
        num_epochs=200,
        batch_size=256,
        lr=1e-3,
        random_state=0,
    ):
        """
        Fit debiased estimator with K-fold cross-fitting (Definition 14).
        """
        X = np.asarray(X, dtype=np.float32)
        A = np.asarray(A, dtype=np.int64)
        Y = np.asarray(Y, dtype=np.float32)

        n, d = X.shape
        self.kfold_ = KFold(
            n_splits=self.K, shuffle=True, random_state=random_state
        )

        self.propensity_models_ = []
        self.h_nets_ = []
        self.u_nets_ = []
        self.m_models_ = []

        for k, (idx_train, idx_valid) in enumerate(self.kfold_.split(X)):
            print(f"[Fold {k+1}/{self.K}]")

            # -------- Step 2: propensity model ê_k on D^{-k} --------
            prop_model = self.propensity_model_factory()
            # sklearn solvers are generally more stable in float64
            prop_model.fit(X[idx_train].astype(np.float64, copy=False), A[idx_train])
            self.propensity_models_.append(prop_model)

            e1_all = prop_model.predict_proba(X.astype(np.float64, copy=False))[:, 1]
            e1_all = np.clip(e1_all, 1e-6, 1.0 - 1e-6)
            e0_all = 1.0 - e1_all

            # Restrict to the validation fold D_k (used to train h,u)
            X_k = torch.tensor(X[idx_valid], dtype=torch.float32, device=self.device)
            A_k = torch.tensor(A[idx_valid], dtype=torch.float32, device=self.device).view(-1, 1)
            Y_k = torch.tensor(Y[idx_valid], dtype=torch.float32, device=self.device)

            e1_k = torch.tensor(e1_all[idx_valid], dtype=torch.float32, device=self.device)
            e0_k = torch.tensor(e0_all[idx_valid], dtype=torch.float32, device=self.device)

            n_k = X_k.shape[0]
            batch_size_k = min(batch_size, n_k)

            # -------- Step 3: solve min_{β,γ} ∑ ℓ_db on D_k --------
            input_dim = d + 1  # A plus X
            h_net = self._build_net(input_dim, self.h_hidden).to(self.device)
            u_net = self._build_net(input_dim, self.u_hidden).to(self.device)

            params = list(h_net.parameters()) + list(u_net.parameters())
            optimizer = optim.Adam(params, lr=lr)

            for epoch in range(num_epochs):
                perm = torch.randperm(n_k, device=self.device)
                for start in range(0, n_k, batch_size_k):
                    idx = perm[start:start + batch_size_k]
                    loss = self._batch_loss(
                        X_batch=X_k[idx],
                        A_batch=A_k[idx],
                        Y_batch=Y_k[idx],
                        e1_batch=e1_k[idx],
                        e0_batch=e0_k[idx],
                        h_net=h_net,
                        u_net=u_net,
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if (epoch + 1) % max(1, num_epochs // 5) == 0:
                    print(f"  Epoch {epoch+1:4d}, loss={loss.item():.4f}")

            # Move trained nets to CPU for later use
            h_net_cpu = h_net.to("cpu")
            u_net_cpu = u_net.to("cpu")
            self.h_nets_.append(h_net_cpu)
            self.u_nets_.append(u_net_cpu)

            # -------- Step 5: construct Z_i on D_k and regress Z on (A,X) --------
            with torch.no_grad():
                Z_k = self._compute_Z(
                    X_batch=X_k.to("cpu"),
                    A_batch=A_k.to("cpu"),
                    Y_batch=Y_k.to("cpu"),
                    h_net=h_net_cpu,
                    u_net=u_net_cpu,
                )  # shape (n_k,)

            AX_valid = np.concatenate(
                [A[idx_valid].reshape(-1, 1).astype(np.float32), X[idx_valid]],
                axis=1,
            )
            m_model = self.m_model_factory()
            m_model.fit(AX_valid, Z_k)
            self.m_models_.append(m_model)

        return self

    def predict(self, a, x):
        """
        Compute the debiased upper bound θ̂_φ(a,x) for given
        treatment a and covariates x using the fitted cross-fitted models.

        Arguments:
            a: array-like of shape (n_eval,) with values {0,1}
            x: array-like of shape (n_eval, d)

        Returns:
            θ̂_φ(a,x): np.ndarray of shape (n_eval,)
        """
        if len(self.h_nets_) == 0:
            raise RuntimeError("Call fit(...) before predict(...).")

        a = np.asarray(a, dtype=np.float32).reshape(-1, 1)
        x = np.asarray(x, dtype=np.float32)
        assert x.shape[0] == a.shape[0]
        n_eval, d = x.shape

        theta_per_fold = np.zeros((self.K, n_eval), dtype=np.float32)

        for k in range(self.K):
            prop_model = self.propensity_models_[k]
            h_net = self.h_nets_[k]
            u_net = self.u_nets_[k]
            m_model = self.m_models_[k]

            # ê_a(x)
            e1 = prop_model.predict_proba(x.astype(np.float64, copy=False))[:, 1]
            e1 = np.clip(e1, 1e-6, 1.0 - 1e-6)
            e0 = 1.0 - e1
            eA = np.where(a[:, 0] > 0.5, e1, e0)
            eta = self.B_f_np(eA)  # η̂_f(a,x) = B_f(ê_a(x))

            # h, u at (a,x)
            AX = np.concatenate([a, x], axis=1)
            AX_t = torch.tensor(AX, dtype=torch.float32)
            with torch.no_grad():
                h_ax = h_net(AX_t).cpu().numpy().reshape(-1)
                u_ax = u_net(AX_t).cpu().numpy().reshape(-1)
            # Match the training-time clamp to avoid overflow in exp.
            h_ax = np.clip(h_ax, -20.0, 20.0)
            lam_ax = np.exp(h_ax)

            # m̂_k(a,x)
            m_pred = m_model.predict(AX).reshape(-1)

            theta_per_fold[k, :] = lam_ax * (eta + m_pred) + u_ax

        return theta_per_fold.mean(axis=0)


# ---------------------------------------------------------------------
# Toy SCM: data generation and ground truth
# ---------------------------------------------------------------------

def simulate_toy(
    n,
    gamma=0.8,
    alpha0=0.0,
    alpha_x=1.0,
    alpha_u=1.0,
    beta0=0.0,
    beta_a=1.0,
    beta_x=1.0,
    beta_u=1.0,
    sigma_eps=1.0,
    random_state=0,
):
    """
    Linear-Gaussian SCM with unmeasured confounder U:
        U ~ N(0,1)
        X | U ~ N(gamma * U, 1)
        A | X,U ~ Bernoulli( sigmoid(alpha0 + alpha_x X + alpha_u U) )
        Y | X,A,U = beta0 + beta_a A + beta_x X + beta_u U + eps, eps ~ N(0, sigma_eps^2)
    Returns:
        X: (n,1), A: (n,), Y: (n,), U: (n,)
    """
    rng = np.random.RandomState(random_state)

    U = rng.normal(loc=0.0, scale=1.0, size=n)
    X = rng.normal(loc=gamma * U, scale=1.0, size=n)

    lin = alpha0 + alpha_x * X + alpha_u * U
    p = 1.0 / (1.0 + np.exp(-lin))
    A = rng.binomial(1, p, size=n)

    eps = rng.normal(loc=0.0, scale=sigma_eps, size=n)
    Y = beta0 + beta_a * A + beta_x * X + beta_u * U + eps

    return X.reshape(-1, 1).astype(np.float32), A.astype(np.int64), Y.astype(np.float32), U


def true_interventional_mean(a, x, beta0=0.0, beta_a=1.0, beta_x=1.0):
    """
    For the SCM above, E[Y | do(A=a), X=x] = beta0 + beta_a * a + beta_x * x.
    """
    return beta0 + beta_a * a + beta_x * x


# ---------------------------------------------------------------------
# Running the toy experiment
# ---------------------------------------------------------------------

def run_toy_experiment():
    # 1. Simulate observational data with latent confounding
    n = 5000
    X, A, Y, U = simulate_toy(n, random_state=42)

    # Choose f-divergence: one of {"kl", "hellinger", "chi2", "tv", "js"}
    divergence = "chi2"

    # 2. Instantiate debiased estimators (Definition 14, φ(y)=y)
    est_upper = DebiasedCausalBoundEstimator(
        K=2,
        divergence=divergence,
        h_hidden=(64, 64),
        u_hidden=(64, 64),
    )
    est_lower = DebiasedCausalBoundEstimator(
        K=2,
        divergence=divergence,
        h_hidden=(64, 64),
        u_hidden=(64, 64),
    )

    # 3a. Fit the upper-bound estimator on (X, A, Y)
    est_upper.fit(
        X=X,
        A=A,
        Y=Y,
        num_epochs=200,   # increase for better convergence
        batch_size=512,
        lr=1e-3,
        random_state=42,
    )

    # 3b. Fit the lower-bound estimator by negating Y, then flip sign at predict time
    est_lower.fit(
        X=X,
        A=A,
        Y=-Y,
        num_epochs=200,
        batch_size=512,
        lr=1e-3,
        random_state=42,
    )

    # 4. Evaluate at a few (a, x) pairs and compare to true interventional means
    beta0, beta_a, beta_x = 0.0, 1.0, 1.0  # must match simulate_toy

    eval_points = [(-1.0, 0), (0.0, 0), (1.0, 0), (-1.0, 1), (0.0, 1), (1.0, 1)]
    print("\n=== Toy experiment: true interventional means vs. estimated bounds ===")
    for x0, a0 in eval_points:
        theta_hat_upper = est_upper.predict(
            a=np.array([a0], dtype=np.float32),
            x=np.array([[x0]], dtype=np.float32),
        )[0]
        theta_hat_lower = -est_lower.predict(
            a=np.array([a0], dtype=np.float32),
            x=np.array([[x0]], dtype=np.float32),
        )[0]
        q_true = true_interventional_mean(a0, x0, beta0=beta0, beta_a=beta_a, beta_x=beta_x)

        print(
            f"a={a0}, x={x0: .2f}  |  "
            f"true E[Y|do(A={a0}),X={x0:.2f}] = {q_true: .4f}   "
            f"lower bound θ̌ = {theta_hat_lower: .4f}   "
            f"upper bound θ̂ = {theta_hat_upper: .4f}   "
            f"bound width = {theta_hat_upper - theta_hat_lower: .4f}   "
        )


if __name__ == "__main__":
    run_toy_experiment()
