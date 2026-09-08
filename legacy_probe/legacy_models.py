"""Verbatim copy of the pre-rewrite `src/models.py`, recovered with
`git show 28b7615^:src/models.py`. Untouched — do not edit; this file exists so
the legacy probe never imports from the live `src/` tree.

Provenance: commit 28b7615 ("Complete experimental layer: E0-E6, P13") deleted it.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import BayesianRidge


# --------------------------------------------------------------------- #
# Neural network
# --------------------------------------------------------------------- #
class MLP(nn.Module):
    """Feed-forward network with optional dropout; outputs mean and log-variance."""
    def __init__(self, in_dim=1, hidden=64, dropout=0.0):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.mean_head = nn.Linear(hidden, 1)
        self.logvar_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.backbone(x)
        mu = self.mean_head(h)
        log_var = self.logvar_head(h).clamp(-20, 2)
        return mu, log_var


def make_mlp(in_dim, dropout=0.0):
    """Factory used by load_nn — must reproduce the architecture used at training time."""
    return MLP(in_dim=in_dim, dropout=dropout)


def train_mlp(X, y, dropout=0.0, epochs=1000, lr=0.01, seed=42):
    """Train an MLP with Gaussian NLL loss; stores aleatoric noise_std_ on the model."""
    torch.manual_seed(seed)
    model = MLP(in_dim=X.shape[1], dropout=dropout)
    opt = optim.Adam(model.parameters(), lr=lr)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        mu, log_var = model(X_t)
        loss = F.gaussian_nll_loss(mu, y_t, torch.exp(log_var))
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        mu, _ = model(X_t)
    model.noise_std_ = float((y_t.squeeze() - mu.squeeze()).std())
    return model


def mlp_point_predict(model, X):
    """Deterministic point prediction; returns (mean, aleatoric_variance)"""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        mu, log_var = model(X_t)
    return mu.numpy().ravel(), torch.exp(log_var).numpy().ravel()


# --------------------------------------------------------------------- #
# Gaussian Process
# --------------------------------------------------------------------- #
def make_gp(random_state=42):
    """Factory for a GP regressor."""
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    return GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        normalize_y=True,
        random_state=random_state,
    )
