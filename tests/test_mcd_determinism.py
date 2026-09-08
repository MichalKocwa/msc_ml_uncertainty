"""MC dropout reproducibility, and the batch dependence that DEC-007 exists for.

Two separate properties, both from `docs/experiment_notes.md`:

1. `MCDropoutMethod.predict` reseeds from the fit seed, so a whole call is
   reproducible - including after a CACHE HIT, which skips training and so does
   not consume RNG state the way a real training run would. `mcd.py`'s own
   docstring asks for exactly this regression test.
2. N-006: a fixed point's `mcd` prediction still moves with the SIZE of the
   array it was predicted in, because `F.dropout` draws `n * hidden` variates in
   one go. That is not a bug to fix (per-point masks would be a different, much
   slower method), but it is the reason E1 predicts once per method per seed on
   a single array (DEC-007). The test below pins the behaviour so that anyone
   who changes it is forced to revisit that decision rather than silently
   invalidating its rationale.

Epochs are cut to keep these fast; nothing here reports a number.
"""
import numpy as np
import pytest

from src.data import make_toy_data
from src.methods import METHODS

EPOCHS = 5
T = 4


@pytest.fixture(scope="module")
def data():
    return make_toy_data(seed=0)


def _fit_mcd(data, use_cache):
    method = METHODS["mcd"](epochs=EPOCHS, T=T)
    return method.fit(data.x_train_std, data.y_train_std, seed=0, use_cache=use_cache)


def test_predict_is_reproducible_across_calls_on_the_same_model(data):
    method = _fit_mcd(data, use_cache=False)
    first = method.predict(data.x_test_std)
    second = method.predict(data.x_test_std)
    assert np.array_equal(first.mean, second.mean)
    assert np.array_equal(first.var_epistemic, second.var_epistemic)


def test_cache_hit_and_real_retrain_give_identical_predictions(data):
    """The check `mcd.py`'s docstring names: fit twice with the same seed, once
    through the cache and once not, and assert the predictions match.
    """
    uncached = _fit_mcd(data, use_cache=False).predict(data.x_test_std)
    _fit_mcd(data, use_cache=True)                     # populates the cache
    cached = _fit_mcd(data, use_cache=True).predict(data.x_test_std)
    assert np.allclose(uncached.mean, cached.mean)
    assert np.allclose(uncached.var_epistemic, cached.var_epistemic)


def test_mcd_predictions_depend_on_the_evaluation_batch_size(data):
    """N-006, pinned. If this ever fails, DEC-007's rationale has changed."""
    method = _fit_mcd(data, use_cache=False)
    probe = np.array([[-2.0], [3.0]])
    probe_std = data.standardiser.standardise_x(probe)

    small = method.predict(np.vstack([probe_std, data.x_test_std[:50]]))
    large = method.predict(np.vstack([probe_std, data.x_test_std]))
    assert not np.allclose(small.var_epistemic[:2], large.var_epistemic[:2])


def test_bbb_predictions_are_invariant_to_the_evaluation_batch_size(data):
    """The contrast that makes N-006 a dropout property rather than a pipeline
    one: `LinearReparameterization` samples weight matrices whose shape does not
    depend on the batch, so BBB is exactly invariant.
    """
    method = METHODS["bbb"](epochs=EPOCHS, T=T).fit(
        data.x_train_std, data.y_train_std, seed=0, use_cache=False)
    probe_std = data.standardiser.standardise_x(np.array([[-2.0], [3.0]]))

    small = method.predict(np.vstack([probe_std, data.x_test_std[:50]]))
    large = method.predict(np.vstack([probe_std, data.x_test_std]))
    assert np.allclose(small.var_epistemic[:2], large.var_epistemic[:2])
