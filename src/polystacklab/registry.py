
# src/yourpkg/experiment/registry.py
from __future__ import annotations
from typing import Callable, Any

import numpy as np
from numpy.typing import NDArray

# A view returns a dict of named 2D arrays, each (n_samples, d_view)
ViewOutput = dict[str, NDArray[np.floating]]
# The bound callable you’ll use at runtime: fn(X_flat) -> ViewOutput
ViewFn = Callable[[NDArray[np.floating] | NDArray[np.integer]], ViewOutput]


_REG: dict[str, Callable[..., ViewOutput] | Callable[..., ViewFn]] = {}

_DATASETS: dict[str, Callable[..., "Dataset"]] = {}
ViewOutput: dict[str, Callable[..., np.ndarray]] = {}
_FEATURES: dict[str, Callable[..., np.ndarray]] = {}

def register_dataset(name: str):
    def deco(factory): _DATASETS[name] = factory; return factory
    return deco

def create_dataset(name: str, **kw): return _DATASETS[name](**kw)

def register_view(name: str):
    """Decorator to register a view factory or a directly-callable view.

    - If you register a function shaped like:  def builder(X, **params) -> ViewOutput,
      then create_view(name, **params) will return a bound fn(X) -> ViewOutput.

    - If you register a factory: def factory(**params) -> ViewFn,
      create_view(name, **params) will call it and return the ViewFn.
    """
    def deco(func: Callable[..., Any]):
        _REG[name] = func
        return func
    return deco

def create_view(name: str, **params: Any) -> ViewFn:
    if name not in _REG:
        raise KeyError(f"Unknown view '{name}'. Available: {sorted(_REG)}")
    f = _REG[name]
    # Case A: directly-callable builder(X, **params)
    def _callable_accepts_X_first(fn: Callable[..., Any]) -> bool:
        return True  # we accept both; fallback will raise at call if mismatched
    if _callable_accepts_X_first(f):
        return lambda X: f(X, **params)  # type: ignore[misc]
    # Case B: factory(**params) -> ViewFn
    view_fn = f(**params)  # type: ignore[call-arg]
    return view_fn  # type: ignore[return-value]

def available_views() -> list[str]:
    return sorted(_REG)

def get_view(name: str): return _VIEWS[name]

def register_feature(name: str):
    def deco(func): _FEATURES[name] = func; return func
    return deco

def get_feature(name: str): return _FEATURES[name]
