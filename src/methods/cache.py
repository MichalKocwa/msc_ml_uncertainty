"""Model cache.

Most exploration questions after a method is trained turn out to be about
PREDICTION, not training — axis ranges, eval-grid resolution, metric
breakdowns, how a figure looks — yet every one needed a full retrain before
this. `cache/{method}_{hash}.pt`, where `hash` covers the method name, every
fit-relevant hyperparameter, and a fingerprint of `(X, y, seed)` — so a
cache hit is only possible for a byte-identical training call, never a
near-miss. `fit()` checks the cache before training; each `fit()` takes a
`use_cache: bool = True` parameter, and each experiment script exposes
`--no-cache` to force a real retrain (e.g. to verify determinism, or after
a code change that isn't reflected in the config dict below).

Not a general-purpose caching framework — deliberately minimal: two
functions (`load`, `save`) plus a path-naming scheme. The state saved is
whatever each training function chooses to save (typically
`model.state_dict()` plus a few scalars), not a serialised Python object
graph — safer across code changes than pickling arbitrary objects, and
avoids depending on third-party classes (e.g. `laplace-torch`'s `Laplace`)
being pickle-safe.
"""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = REPO_ROOT / "cache"


def _fingerprint(method_name: str, config: Dict[str, Any], X: np.ndarray, y: np.ndarray, seed: int) -> str:
    """`config` must be JSON-serialisable (floats/ints/strings/bools/None/
    dicts of those — e.g. `layerwise_prior_omega`'s per-parameter coefficient
    dict is fine as-is). `default=str` on anything else so an accidental
    non-serialisable value changes the hash rather than crashing training.
    """
    hasher = hashlib.sha256()
    hasher.update(method_name.encode())
    hasher.update(json.dumps(config, sort_keys=True, default=str).encode())
    hasher.update(np.ascontiguousarray(X).tobytes())
    hasher.update(np.ascontiguousarray(y).tobytes())
    hasher.update(str(seed).encode())
    return hasher.hexdigest()[:24]


def cache_path(method_name: str, config: Dict[str, Any], X: np.ndarray, y: np.ndarray, seed: int) -> Path:
    key = _fingerprint(method_name, config, X, y, seed)
    return CACHE_DIR / f"{method_name}_{key}.pt"


def load(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return torch.load(path, weights_only=False)


def save(path: Path, state: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
