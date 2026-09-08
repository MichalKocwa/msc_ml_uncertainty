"""Prior parametrisation: the relationships between the fields must hold exactly.

The factor-of-two check is the one `src/config.py`'s own docstring asks for by
name: `weight_decay` is applied by `torch.optim` as `wd*theta` added straight to
the gradient, while differentiating `prior_penalty_coefficient * theta**2`
introduces a factor of 2. Getting the correspondence wrong changes the effective
prior strength by 2x with no visible error anywhere - no exception, no warning,
just a different posterior in every method at once.
"""
import pytest

from src.config import (
    layerwise_penalty_coefficients, layerwise_prior_precisions, layerwise_prior_sigma,
    prior_parametrisations,
)


@pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("n", [1, 250, 1000])
def test_weight_decay_is_twice_the_penalty_coefficient(gamma, n):
    p = prior_parametrisations(gamma, n)
    assert p.weight_decay == pytest.approx(2.0 * p.prior_penalty_coefficient)


@pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
def test_every_parametrisation_encodes_the_same_prior(gamma):
    p = prior_parametrisations(gamma, n=250)
    assert p.prior_sigma == pytest.approx(gamma)
    assert p.numpyro_scale == pytest.approx(gamma)
    assert p.prior_precision == pytest.approx(1.0 / gamma ** 2)
    assert p.prior_penalty_coefficient == pytest.approx(1.0 / (2.0 * gamma ** 2 * 250))


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_gamma_and_n_are_rejected(bad):
    with pytest.raises(ValueError):
        prior_parametrisations(bad, n=250)
    with pytest.raises(ValueError):
        prior_parametrisations(1.0, n=int(bad))


def test_layerwise_prior_scales_weights_by_fan_in_and_leaves_biases_at_one():
    assert layerwise_prior_sigma(is_bias=True, fan_in=50, omega=3.0) == 1.0
    assert layerwise_prior_sigma(is_bias=False, fan_in=4, omega=3.0) == pytest.approx(1.5)


def test_layerwise_helpers_agree_with_the_scalar_relationships():
    named = [("linear1.weight", _Shape((50, 1))), ("linear1.bias", _Shape((50,)))]
    omega, n = 2.0, 250
    sigmas = {name: layerwise_prior_sigma(name.endswith("bias"),
                                          p.shape[0] if name.endswith("bias") else p.shape[1], omega)
              for name, p in named}
    precisions = layerwise_prior_precisions(named, omega)
    penalties = layerwise_penalty_coefficients(named, omega, n)
    for name, sigma in sigmas.items():
        assert precisions[name] == pytest.approx(1.0 / sigma ** 2)
        assert penalties[name] == pytest.approx(1.0 / (2.0 * sigma ** 2 * n))


class _Shape:
    """Minimal stand-in for an `nn.Parameter`: the layerwise helpers read only
    `.shape`, so a tensor (and a torch import) is not needed here.
    """

    def __init__(self, shape):
        self.shape = shape
