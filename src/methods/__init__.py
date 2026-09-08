"""Registry of uncertainty-estimation methods, keyed by CLI name.

All six are implemented against the `UncertaintyMethod` protocol in `base.py`.
"""
from typing import Callable, Dict

from .base import Prediction, UncertaintyMethod
from .bbb import BBBMethod
from .ensemble import DeepEnsembleMethod
from .gp import GPMethod
from .laplace import LaplaceMethod
from .map import MAPMethod
from .mcd import MCDropoutMethod

METHODS: Dict[str, Callable[..., UncertaintyMethod]] = {
    "map": MAPMethod,
    "mcd": MCDropoutMethod,
    "ensemble": DeepEnsembleMethod,
    "gp": GPMethod,
    "laplace": LaplaceMethod,
    "bbb": BBBMethod,
}

__all__ = ["Prediction", "UncertaintyMethod", "METHODS"]
