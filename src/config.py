"""Prior parametrisation mapping.

BBB, Laplace and HMC must all use the same prior N(0, gamma^2 I). Each
library expects that prior expressed through a different parameter, and the
deterministic baseline / MC dropout must use a matching regularisation for
the comparison to be meaningful. `prior_parametrisations` is the single
place where `gamma` is translated into every parametrisation actually
consumed by library code, so that they never drift apart.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PriorParametrisation:
    gamma: float                      # standard deviation of N(0, gamma^2 I), the reference quantity
    prior_sigma: float                # bayesian-torch: LinearReparameterization(prior_sigma=...)
    prior_precision: float            # laplace-torch: Laplace(prior_precision=...)
    prior_penalty_coefficient: float  # RECOMMENDED for MAP baseline / MC dropout: add
                                       # `prior_penalty_coefficient * sum(theta**2)` to the
                                       # (mean-reduced) loss directly, with optimizer weight_decay=0
    weight_decay: float                # FALLBACK ONLY, not recommended — see prior_penalty_coefficient's
                                        # docstring note below for why
    numpyro_scale: float               # numpyro: dist.Normal(0, scale=...), for an HMC reference run


def prior_parametrisations(gamma: float, n: int) -> PriorParametrisation:
    """Translate a single `gamma` into every parametrisation library code consumes.

    `n` is the training set size: both `prior_penalty_coefficient` and
    `weight_decay` fold in a 1/N factor because the data loss is
    mean-reduced over N, and the prior term must be on the same footing.

    `prior_penalty_coefficient = 1/(2*gamma^2*N)` is the coefficient of
    `sum(theta**2)` in the loss that corresponds exactly to MAP under
    N(0, gamma^2 I). Use it directly in the loss (`loss = data_loss +
    prior_penalty_coefficient * sum(p.pow(2).sum() for p in params)`) with
    the optimizer's own `weight_decay=0`. This is the recommended path for
    the MAP baseline and MC dropout: it makes the prior optimizer-agnostic,
    which matters because `torch.optim`'s built-in `weight_decay` is only
    equivalent to this penalty for plain SGD. For Adam, `weight_decay=wd`
    adds `wd*theta` to the gradient *before* the adaptive per-parameter
    scaling by `1/sqrt(v_hat)`, so the effective prior strength ends up
    parameter- and time-dependent — not the fixed, isotropic N(0, gamma^2 I) the
    prior specifies. AdamW decouples the decay from that scaling
    (`p -= lr*wd*p`, independent of the gradient), which is closer but still
    not an exact MAP correspondence the way an explicit loss term is for any
    optimizer. This also matches Yarin Gal's reference implementation
    (Keras `regularizers.l2`, added directly to the loss).

    `weight_decay = 1/(gamma^2*N)` is kept as a fallback/reference value —
    e.g. for a quick SGD sanity check — but is **not recommended** for
    training any of the five methods, for the Adam-vs-SGD reason above.
    Note it is exactly `2 * prior_penalty_coefficient`: `torch.optim`
    applies `weight_decay` as `wd*theta` added straight to the gradient,
    while differentiating `prior_penalty_coefficient * theta**2` introduces
    a factor of 2 — so matching the two conventions cancels the 1/2 in the
    penalty. Worth a unit test asserting exactly this factor-of-two
    relationship between the two fields: getting it wrong changes the
    effective prior strength by 2x with no visible error anywhere.
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    return PriorParametrisation(
        gamma=gamma,
        prior_sigma=gamma,
        prior_precision=1.0 / gamma ** 2,
        prior_penalty_coefficient=1.0 / (2.0 * gamma ** 2 * n),
        weight_decay=1.0 / (gamma ** 2 * n),
        numpyro_scale=gamma,
    )


def layerwise_prior_sigma(is_bias: bool, fan_in: int, omega: float) -> float:
    """Foong, Li, Hernandez-Lobato & Turner 2019's layer-scaled prior
    (arXiv 1906.11537), as used in their 1D experiments: `N(0, 1)` on every
    bias, `N(0, omega^2/fan_in)` on every weight
    matrix, `fan_in` read off that matrix's own shape (`nn.Linear.weight` is
    `(out_features, in_features)`, so `fan_in = weight.shape[1]`) — no
    architecture-specific constant, works for any width/depth of the shared
    1-hidden-layer backbone. This is a *tested alternative* to the flat
    `N(0, gamma^2 I)` prior, not a replacement for it: the flat `gamma`
    remains the default for every method, and this scaling exists only for
    the Foong-setup variant comparison. It is still the same prior shape
    (independent zero-mean Gaussian per weight), only its per-parameter
    scale differs, and it still applies identically to every method by
    construction — so it does not break the "same prior everywhere"
    requirement.
    """
    if omega <= 0:
        raise ValueError(f"omega must be positive, got {omega}")
    if fan_in <= 0:
        raise ValueError(f"fan_in must be positive, got {fan_in}")
    return 1.0 if is_bias else omega / (fan_in ** 0.5)


def layerwise_prior_sigmas(named_parameters, omega: float) -> dict:
    """`{parameter_name: sigma}` for every `(name, param)` pair from e.g.
    `HomoscedasticMLP.mlp.named_parameters()`, via `layerwise_prior_sigma`.
    `fan_in` for a bias parameter is unused by that function's `is_bias`
    branch — `param.shape[0]` is passed only to satisfy its `fan_in > 0`
    validation, not because it means anything for a bias.
    """
    sigmas = {}
    for name, param in named_parameters:
        is_bias = name.endswith("bias")
        fan_in = param.shape[0] if is_bias else param.shape[1]
        sigmas[name] = layerwise_prior_sigma(is_bias=is_bias, fan_in=fan_in, omega=omega)
    return sigmas


def layerwise_penalty_coefficients(named_parameters, omega: float, n: int) -> dict:
    """`{parameter_name: 1/(2*sigma^2*n)}` — the layer-scaled analogue of
    `prior_penalty_coefficient`, for the explicit-loss-penalty methods
    (map/mcd/ensemble).
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    return {name: 1.0 / (2.0 * sigma ** 2 * n) for name, sigma in layerwise_prior_sigmas(named_parameters, omega).items()}


def layerwise_prior_precisions(named_parameters, omega: float) -> dict:
    """`{parameter_name: 1/sigma^2}` — the layer-scaled analogue of
    `prior_precision` (laplace-torch).
    """
    return {name: 1.0 / sigma ** 2 for name, sigma in layerwise_prior_sigmas(named_parameters, omega).items()}
