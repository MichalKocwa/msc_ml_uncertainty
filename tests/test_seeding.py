"""`set_seed` must pin dtype and thread count, not just the RNGs.

Both are load-bearing for "every result reproducible from the seed" (CLAUDE.md):
float64 because `set_seed` runs before model construction and layers with no
explicit dtype inherit the global default, and one thread because the intra-op
thread count changes the order of float64 reductions, which an optimiser then
amplifies over tens of thousands of steps.
"""
import numpy as np
import pytest
import torch

from src.seeding import DEFAULT_TORCH_THREADS, TORCH_THREADS_ENV, set_seed


def test_set_seed_pins_float64_as_the_default_dtype():
    torch.set_default_dtype(torch.float32)
    try:
        set_seed(0)
        assert torch.get_default_dtype() == torch.float64
        # a layer built with no explicit dtype must come out float64
        assert torch.nn.Linear(2, 2).weight.dtype == torch.float64
    finally:
        set_seed(0)


def test_set_seed_pins_the_thread_count(monkeypatch):
    monkeypatch.delenv(TORCH_THREADS_ENV, raising=False)
    set_seed(0)
    assert torch.get_num_threads() == DEFAULT_TORCH_THREADS == 1


def test_thread_count_env_override_is_honoured(monkeypatch):
    """The escape hatch exists to MEASURE the effect; a normal run leaves it unset."""
    monkeypatch.setenv(TORCH_THREADS_ENV, "2")
    try:
        set_seed(0)
        assert torch.get_num_threads() == 2
    finally:
        monkeypatch.delenv(TORCH_THREADS_ENV, raising=False)
        set_seed(0)


@pytest.mark.parametrize("draw", [
    lambda: torch.randn(5).numpy(),
    lambda: np.random.rand(5),
])
def test_same_seed_reproduces_every_rng_stream(draw):
    set_seed(7)
    first = draw()
    set_seed(7)
    assert np.array_equal(first, draw())
