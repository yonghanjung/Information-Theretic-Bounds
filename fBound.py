import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

import torch
import torch.nn as nn
import torch.optim as optim


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
# KL f-divergence ingredients (Corollary 3 and 8)
#   B_f(e)  = - log e
#   B'_f(e) = - 1 / e
#   g*_KL(t) = -1 - log(-t) for t < 0, +∞ otherwise (we penalize invalid t)
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


# ---------------------------------------------------------------------
# Debiased estimator (Definition 14 for KL, φ(y) = y)
# ---------------------------------------------------------------------

class DebiasedCausalBoundEstimator:
    """
    Implementation of Definition 14 (debiased causal bound estimator)
    for one f-divergence (KL) and one functional φ(Y)=Y.

    - h_beta(a,x), u_gamma(a,x) are neural nets (PyTorch).
    - Propensity model e1(x) = P(A=1 | X=x) is any sklearn-style classifier
      with fit(X, A) and predict_proba(X).
    - m(a,x) is any regressor with fit(X, y) and predict(X) (e.g., RF, XGBoost).
    """

    def __init__(
        self,
        K=2,
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
            self.propensity_model_factory = lambda: LogisticRegression(
                max_iter=1000, solver="lbfgs"
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

        # f-divergence pieces for KL
        self.B_f = B_kl_torch
        self.dB_f = dB_kl_torch
        self.B_f_np = B_kl_numpy
        self.g_star = g_star_kl_torch

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
        lam_AX = torch.exp(h_AX)                   # λ(A,X) > 0

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
        lam0 = torch.exp(h0)
        lam1 = torch.exp(h1)

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
        lam_AX = torch.exp(h_AX)
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
            prop_model.fit(X[idx_train], A[idx_train])
            self.propensity_models_.append(prop_model)

            e1_all = prop_model.predict_proba(X)[:, 1]
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
            e1 = prop_model.predict_proba(x)[:, 1]
            e0 = 1.0 - e1
            eA = np.where(a[:, 0] > 0.5, e1, e0)
            eta = self.B_f_np(eA)  # η̂_f(a,x) = B_f(ê_a(x))

            # h, u at (a,x)
            AX = np.concatenate([a, x], axis=1)
            AX_t = torch.tensor(AX, dtype=torch.float32)
            with torch.no_grad():
                h_ax = h_net(AX_t).cpu().numpy().reshape(-1)
                u_ax = u_net(AX_t).cpu().numpy().reshape(-1)
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

    # 2. Instantiate debiased estimator (Definition 14, KL, φ(y)=y)
    est = DebiasedCausalBoundEstimator(
        K=2,
        h_hidden=(64, 64),
        u_hidden=(64, 64),
    )

    # 3. Fit the estimator
    est.fit(
        X=X,
        A=A,
        Y=Y,
        num_epochs=200,   # increase for better convergence
        batch_size=512,
        lr=1e-3,
        random_state=42,
    )

    # 4. Evaluate at a few (a, x) pairs and compare to true interventional means
    beta0, beta_a, beta_x = 0.0, 1.0, 1.0  # must match simulate_toy

    eval_points = [(-1.0, 0), (0.0, 0), (1.0, 0), (-1.0, 1), (0.0, 1), (1.0, 1)]
    print("\n=== Toy experiment: true interventional means vs. estimated upper bounds ===")
    for x0, a0 in eval_points:
        theta_hat = est.predict(
            a=np.array([a0], dtype=np.float32),
            x=np.array([[x0]], dtype=np.float32),
        )[0]
        q_true = true_interventional_mean(a0, x0, beta0=beta0, beta_a=beta_a, beta_x=beta_x)

        print(
            f"a={a0}, x={x0: .2f}  |  "
            f"true E[Y|do(A={a0}),X={x0:.2f}] = {q_true: .4f}   "
            f"upper bound θ̂ = {theta_hat: .4f}"
        )


if __name__ == "__main__":
    run_toy_experiment()
