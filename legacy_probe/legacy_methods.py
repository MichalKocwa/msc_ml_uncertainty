# ============================================================================
# RECONSTRUCTION of the old src/methods.py
#
# WARNING: this file was NEVER committed to git. It is decompiled by hand from
# the stale bytecode cache src/__pycache__/methods.cpython-313.pyc (mtime
# 2026-05-11 11:35). Structure, constants, default arguments and control flow
# are read straight off the disassembly and are faithful; only local variable
# spelling, comments and blank lines are guesswork. Line numbers below match
# co_firstlineno of the original file.
# ============================================================================
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from legacy_models import train_mlp, mlp_point_predict     # line 8 (was: from src.models import ...)


class _VILayer(nn.Module):                               # line 11
    def __init__(self, in_f, out_f, prior_std=1.0):      # line 12
        super().__init__()
        self.prior_std = prior_std
        self.w_mu = nn.Parameter(torch.randn(out_f, in_f) * math.sqrt(2.0 / in_f))
        self.w_ls = nn.Parameter(torch.full((out_f, in_f), math.log(0.1)))
        self.b_mu = nn.Parameter(torch.zeros(out_f))
        self.b_ls = nn.Parameter(torch.full((out_f,), math.log(0.1)))

    def forward(self, x):                                # line 21 — local reparam trick
        mean = F.linear(x, self.w_mu, self.b_mu)
        var = F.linear(x.pow(2), self.w_ls.exp().pow(2), self.b_ls.exp().pow(2))
        return mean + var.clamp(min=1e-8).sqrt() * torch.randn_like(mean)

    def kl(self):                                        # line 26
        p2 = self.prior_std ** 2

        def _kl(mu, ls):                                 # line 28
            s2 = ls.exp().pow(2)
            return 0.5 * ((mu.pow(2) + s2) / p2 - 1 - torch.log(s2 / p2)).sum()

        return _kl(self.w_mu, self.w_ls) + _kl(self.b_mu, self.b_ls)


class _VariationalMLP(nn.Module):                        # line 33
    def __init__(self, in_dim, hidden=64):               # line 34
        super().__init__()
        self.l1 = _VILayer(in_dim, hidden)
        self.l2 = _VILayer(hidden, hidden)
        self.l3 = _VILayer(hidden, 1)
        self.logvar_head = nn.Linear(hidden, 1)          # deterministic noise head

    def forward(self, x):                                # line 41
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        mean = self.l3(h2)
        log_var = self.logvar_head(h2).clamp(-20, 2)
        return mean, log_var

    def kl(self):                                        # line 48
        return self.l1.kl() + self.l2.kl() + self.l3.kl()


def mc_dropout_predict(model, X, n_samples=200):         # line 52
    X_t = torch.tensor(X, dtype=torch.float32)
    model.train()                                        # dropout stays ON
    mus = []
    with torch.no_grad():
        for _ in range(n_samples):
            mu, _ = model(X_t)
            mus.append(mu.numpy().ravel())
    mus = np.stack(mus, axis=0)
    epistemic = mus.var(axis=0)
    noise = getattr(model, "noise_std_", 0.0) ** 2       # residual std from train_mlp
    return mus.mean(axis=0), np.sqrt(epistemic + noise)


def train_ensemble(X, y, n_models=5, dropout=0.0, epochs=1000, lr=0.01):   # line 66
    return [train_mlp(X, y, dropout=dropout, epochs=epochs, lr=lr, seed=s)
            for s in range(n_models)]   # member i gets seed=i


def ensemble_predict(models, X):                         # line 71
    means, vars_ = zip(*[mlp_point_predict(m, X) for m in models])
    means = np.stack(means, axis=0)
    vars_ = np.stack(vars_, axis=0)
    total_var = means.var(axis=0) + vars_.mean(axis=0)   # epistemic + aleatoric
    return means.mean(axis=0), np.sqrt(total_var)


def train_vi(X, y, epochs=1000, lr=0.01, n_samples=5, seed=42):   # line 79
    torch.manual_seed(seed)
    model = _VariationalMLP(in_dim=X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    N = len(X_t)

    n_vi = sum(p.numel() for p in model.parameters()) // 2
    kl_target = 1.0 / (N * n_vi)
    half = epochs // 2

    for epoch in range(epochs):
        if epoch < half:
            kl_w = 0.0                                   # KL warm-up: off for first half
        else:
            kl_w = kl_target * (epoch - half) / max(half - 1, 1)

        optimizer.zero_grad()
        nll = 0
        for _ in range(n_samples):
            mean, log_var = model(X_t)
            nll += F.gaussian_nll_loss(
                mean.squeeze(), y_t.squeeze(),
                torch.exp(log_var).squeeze().clamp(min=1e-6))
        loss = nll / n_samples + kl_w * model.kl()
        loss.backward()
        optimizer.step()

    return model


def vi_predict(model, X, n_samples=200):                 # line 110
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        means, log_vars = [], []
        for _ in range(n_samples):
            mean, log_var = model(X_t)
            means.append(mean.squeeze())
            log_vars.append(log_var.squeeze())
        means = torch.stack(means)
        log_vars = torch.stack(log_vars)
    epistemic = means.std(0).numpy()
    aleatoric = torch.exp(log_vars).mean(0).sqrt().numpy()
    return means.mean(0).numpy(), np.sqrt(epistemic ** 2 + aleatoric ** 2)


def _laplace_fit_predict(model, X_train, y_train, X_eval,          # line 126
                         subset_of_weights, hessian_structure, batch_size=256):
    from laplace import Laplace
    from torch.utils.data import DataLoader, TensorDataset

    class _MeanWrapper(nn.Module):                       # line 131
        def __init__(self, backbone, mean_head):
            super().__init__()
            self.backbone = backbone
            self.mean_head = mean_head

        @property
        def last_layer(self):
            return self.mean_head

        def forward(self, x):
            return self.mean_head(self.backbone(x))

    wrapped = _MeanWrapper(model.backbone, model.mean_head)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size)

    sigma_noise = getattr(model, "noise_std_", 1.0)
    la = Laplace(wrapped, likelihood="regression",
                 subset_of_weights=subset_of_weights,
                 hessian_structure=hessian_structure,
                 sigma_noise=sigma_noise)
    la.fit(loader)
    la.optimize_prior_precision(method="marglik")

    X_eval_t = torch.tensor(X_eval, dtype=torch.float32)
    mean, f_var = la(X_eval_t)
    noise_var = float(la.sigma_noise.detach()) ** 2
    total_std = (f_var.detach() + noise_var).clamp(min=0).sqrt()
    return mean.detach().numpy().ravel(), total_std.numpy().ravel()


def laplace_ll_predict(model, X_train, y_train, X_eval):           # line 165
    return _laplace_fit_predict(model, X_train, y_train, X_eval,
                                "last_layer", "full", len(X_train))


def laplace_all_predict(model, X_train, y_train, X_eval):          # line 170
    return _laplace_fit_predict(model, X_train, y_train, X_eval,
                                "all", "diag", 256)
