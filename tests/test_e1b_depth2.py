"""E1b plumbing: depth 2 really is built, and a saved prediction reproduces its rows.

`experiments/e1b_toy_depth2.py` differs from E1 in one thing (depth) and adds
one thing (saved predictions). Both are checked here at `--quick` epochs on
two cheap methods; no number here is a result.
"""
import numpy as np
import pytest

from experiments.e1b_toy_depth2 import ARCH, ARCH_ID, build_method, run_seed
from src.data import REGIONS, region_masks
from src.experiment_io import load_predictions, save_predictions
from src.metrics import compute_metrics


@pytest.fixture(scope="module")
def run():
    return run_seed(seed=0, method_names=["map", "ensemble", "gp"], quick=True, use_cache=False)


def test_network_methods_are_depth_two_and_gp_is_untouched():
    assert ARCH["depth"] == 2 and ARCH_ID == "d2_h50_tanh"
    assert build_method("map", quick=True).depth == 2
    assert build_method("bbb", quick=True).depth == 2
    assert not hasattr(build_method("gp", quick=True), "depth")


def test_parameter_count_is_the_two_layer_network(run):
    rows, _, _ = run
    p = {r["method"]: r["n_parameters"] for r in rows}
    # 1->50, 50->50, 50->1 weights and biases, plus one global log_sigma2
    assert p["map"] == (1 * 50 + 50) + (50 * 50 + 50) + (50 + 1) + 1 == 2702
    assert p["ensemble"] == 2 * 2702   # M=2 under --quick
    assert {r["arch"] for r in rows if r["method"] == "map"} == {ARCH_ID}
    assert {r["arch"] for r in rows if r["method"] == "gp"} == {""}


def test_saved_predictions_round_trip_and_reproduce_the_rows(run, tmp_path):
    rows, panels, data = run
    arrays = dict(x_test=data.x_test_flat, y_test=data.y_test, x_train=data.x_train.ravel(), y_train=data.y_train)
    path = tmp_path / "seed0.npz"
    save_predictions(path, arrays, panels, meta=dict(arch=ARCH_ID, seed=0, quick=True))
    loaded_arrays, loaded_panels, meta = load_predictions(path)

    assert meta["arch"] == ARCH_ID and meta["seed"] == 0 and meta["quick"] is True
    assert set(loaded_panels) == {"map", "ensemble", "gp"}
    assert loaded_panels["ensemble"]["samples"].shape == (2, data.x_test.shape[0])
    assert "samples" not in loaded_panels["gp"]   # None is not stored

    masks = region_masks(loaded_arrays["x_test"])
    for r in rows:
        p = loaded_panels[r["method"]]
        m = masks[r["region"]]
        values = compute_metrics(loaded_arrays["y_test"][m], p["mean"][m], p["var_aleatoric"][m], p["var_epistemic"][m])
        for key, value in values.items():
            assert np.isclose(r[key], value), (r["method"], r["region"], key)
    assert set(r["region"] for r in rows) == set(REGIONS)
