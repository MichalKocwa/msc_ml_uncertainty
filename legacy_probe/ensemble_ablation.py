"""Why does the LEGACY deep ensemble fan out where the CURRENT one does not?

The current `ml_TESTYYY` E1 run gives the ensemble an MPIW extrapolation /
in-range ratio of 1.08 -- the band is essentially as narrow two units outside
the training range as it is inside it, and PICP@95 there is 0.62. The legacy
ensemble on the same data visibly fans out. Five things differ at once
(activation, depth/width, prior, epochs, M), so this script changes them ONE AT
A TIME and reports what each is worth.

Everything runs on `ml_TESTYYY`'s own seed-0 E1 data under its own protocol:
fit on the standardised arrays, un-standardise before scoring, metrics against
the noisy `y_test` with z = 1.96, split by that project's own region masks.

The training loop is the LEGACY one throughout (Adam lr=0.01, full batch,
Gaussian NLL) so that optimiser and batching are not additional confounds; the
one exception is the explicit prior penalty, which is switched on as its own
ablation step using the current project's coefficient 1/(2*gamma^2*N).

Not a result. A diagnostic, run once to attribute a difference.

    C:/Users/michal/Desktop/ml_uncertainty_msc/.venv/Scripts/python.exe ensemble_ablation.py
"""
import sys
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, ".")

import testyyy_data

Z95 = 1.959963984540054


class FlexibleMLP(nn.Module):
    """The legacy `MLP` with activation, depth and the noise model as knobs.

    At `activation="relu", depth=2, hidden=64, homoscedastic=False` this is
    exactly `legacy_models.MLP`. The other settings walk it towards the current
    `HomoscedasticMLP` (`tanh`, `depth=1`, `hidden=50`, one global log-variance).
    """

    def __init__(self, in_dim=1, hidden=64, dropout=0.0, activation="relu",
                 depth=2, homoscedastic=False):
        super().__init__()
        act = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation]
        layers, d_in = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d_in, hidden), act(), nn.Dropout(dropout)]
            d_in = hidden
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden, 1)
        self.homoscedastic = homoscedastic
        if homoscedastic:
            self.log_sigma2 = nn.Parameter(torch.zeros(()))
        else:
            self.logvar_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.backbone(x)
        mu = self.mean_head(h)
        if self.homoscedastic:
            log_var = self.log_sigma2.clamp(-20, 2).expand_as(mu)
        else:
            log_var = self.logvar_head(h).clamp(-20, 2)
        return mu, log_var


def train_one(X, y, seed, hidden, activation, depth, homoscedastic,
              epochs, lr=1e-2, prior_gamma=None):
    """The legacy `train_mlp` loop, with an optional explicit prior penalty."""
    torch.manual_seed(seed)
    model = FlexibleMLP(in_dim=X.shape[1], hidden=hidden, activation=activation,
                        depth=depth, homoscedastic=homoscedastic)
    opt = optim.Adam(model.parameters(), lr=lr)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    coefficient = None if prior_gamma is None else 1.0 / (2.0 * prior_gamma ** 2 * len(X_t))

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        mu, log_var = model(X_t)
        loss = F.gaussian_nll_loss(mu, y_t, torch.exp(log_var))
        if coefficient is not None:
            loss = loss + coefficient * sum(p.pow(2).sum() for p in model.parameters())
        loss.backward()
        opt.step()

    model.eval()
    return model


def predict_one(model, X):
    model.eval()
    with torch.no_grad():
        mu, log_var = model(torch.tensor(X, dtype=torch.float32))
    return mu.numpy().ravel(), torch.exp(log_var).numpy().ravel()


@dataclass
class Variant:
    label: str
    hidden: int = 64
    activation: str = "relu"
    depth: int = 2
    homoscedastic: bool = False
    epochs: int = 1000
    M: int = 10
    prior_gamma: float = None


LADDER = [
    Variant("legacy as-is (relu, 2x64, no prior, 1000ep, M=10)"),
    Variant("+ tanh", activation="tanh"),
    Variant("+ tanh, 1 hidden layer", activation="tanh", depth=1),
    Variant("+ tanh, 1x50  [current architecture]", activation="tanh", depth=1, hidden=50),
    Variant("+ 4000 epochs", activation="tanh", depth=1, hidden=50, epochs=4000),
    Variant("+ prior gamma=1", activation="tanh", depth=1, hidden=50, epochs=4000, prior_gamma=1.0),
    Variant("+ M=5", activation="tanh", depth=1, hidden=50, epochs=4000, prior_gamma=1.0, M=5),
    Variant("+ homoscedastic  [= current method]", activation="tanh", depth=1, hidden=50,
            epochs=4000, prior_gamma=1.0, M=5, homoscedastic=True),
    # control: the one knob at a time, from the legacy end
    Variant("control: legacy but M=5", M=5),
    Variant("control: legacy but prior gamma=1", prior_gamma=1.0),
    Variant("control: legacy but 4000 epochs", epochs=4000),
]


def run(variant, data):
    std = data.standardiser
    X_fit, y_fit = data.x_train_std, data.y_train_std
    X_pred = data.x_test_std

    means, variances = [], []
    for member in range(variant.M):
        model = train_one(X_fit, y_fit, seed=member, hidden=variant.hidden,
                          activation=variant.activation, depth=variant.depth,
                          homoscedastic=variant.homoscedastic, epochs=variant.epochs,
                          prior_gamma=variant.prior_gamma)
        m, v = predict_one(model, X_pred)
        means.append(m)
        variances.append(v)

    means = np.stack(means, axis=0)
    variances = np.stack(variances, axis=0)

    # Legacy estimator, and the current one: var_epistemic = Var(member means),
    # var_aleatoric = mean(member variances). Identical formula in both codebases.
    var_epistemic = means.var(axis=0)
    var_aleatoric = variances.mean(axis=0)

    mean_orig = std.unstandardise_mean(means.mean(axis=0))
    var_epi_orig = std.unstandardise_var(var_epistemic)
    var_ale_orig = std.unstandardise_var(var_aleatoric)
    sd_total = np.sqrt(var_epi_orig + var_ale_orig)

    out = {}
    for region in ("in_range", "extrapolation"):
        mask = data.regions[region]
        y = data.y_test[mask]
        half = Z95 * sd_total[mask]
        out[f"mpiw_{region}"] = float(np.mean(2.0 * half))
        out[f"picp_{region}"] = float(np.mean((y >= mean_orig[mask] - half)
                                              & (y <= mean_orig[mask] + half)))
        out[f"epi_{region}"] = float(np.mean(np.sqrt(var_epi_orig[mask])))
    out["ratio"] = out["mpiw_extrapolation"] / out["mpiw_in_range"]
    out["epi_ratio"] = out["epi_extrapolation"] / max(out["epi_in_range"], 1e-12)
    return out


def main():
    data = testyyy_data.make_toy_data(seed=0)
    testyyy_data.restore_float32_default()

    print("ml_TESTYYY E1 data, seed 0 | legacy training loop | metrics in original y units\n")
    header = f"{'variant':52s} {'MPIW in':>8s} {'MPIW ext':>9s} {'ratio':>7s} " \
             f"{'PICP in':>8s} {'PICP ext':>9s} {'epi ratio':>10s}"
    print(header)
    print("-" * len(header))
    for variant in LADDER:
        r = run(variant, data)
        print(f"{variant.label:52s} {r['mpiw_in_range']:8.3f} {r['mpiw_extrapolation']:9.3f} "
              f"{r['ratio']:7.2f} {r['picp_in_range']:8.3f} {r['picp_extrapolation']:9.3f} "
              f"{r['epi_ratio']:10.2f}")

    print("\nReference, current methods, 20 seeds (results/e1_toy_metrics.csv):")
    print(f"{'ensemble (current)':52s} {0.400:8.3f} {0.432:9.3f} {1.08:7.2f} "
          f"{0.950:8.3f} {0.622:9.3f} {'-':>10s}")


if __name__ == "__main__":
    main()
