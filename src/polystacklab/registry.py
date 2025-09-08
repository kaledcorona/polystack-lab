
# src/yourpkg/experiment/registry.py
from __future__ import annotations
from typing import Callable, Any, Callable, ParamSpec, TypeVar, overload, Concatenate, cast
import inspect

import numpy as np
from numpy.typing import NDArray

# ---------- Public types ----------
XArr = NDArray[np.floating] | NDArray[np.integer]
# A view returns a dict of named 2D arrays, each (n_samples, d_view)
ViewOutput = dict[str, NDArray[np.floating]]
# The bound callable you’ll use at runtime: fn(X_flat) -> ViewOutput
ViewFn = Callable[[XArr], ViewOutput]

P = ParamSpec("P")

# A builder consumes X plus extra bound params; a factory produces a ViewFn.
Builder = Callable[Concatenate[XArr, P], ViewOutput]
Factory = Callable[P, ViewFn]

# ---------- Registries ----------
_REG: Dict[str, Callable[..., Any]] = {}
_DATASETS: Dict[str, Callable[..., "Dataset"]] = {}
_FEATURES: Dict[str, Callable[..., np.ndarray]] = {}

# ---------- Dataset API (unchanged) ----------
def register_dataset(name: str):
    def deco(factory): _DATASETS[name] = factory; return factory
    return deco

def create_dataset(name: str, **kw): return _DATASETS[name](**kw)

# ---------- View API (typed) ----------
@overload
def register_view(name: str) -> Callable[[Builder[P]], Builder[P]]: ...
@overload
def register_view(name: str) -> Callable[[Factory[P]], Factory[P]]: ...

def register_view(name: str):
    """Register a view builder or factory under `name` (type-preserving).

    - Builder form:  def f(X: XArr, **params) -> ViewOutput
      create_view(name, **params) returns a ViewFn that calls f(X, **params).

    - Factory form:  def f(**params) -> ViewFn
      create_view(name, **params) returns the ViewFn from the factory.
    """
    def deco(func: Callable[..., Any]):
        _REG[name] = func
        return func  # type preserved by overloads
    return deco

def create_view(name: str, **params: Any) -> ViewFn:
    """Instantiate a view callable `ViewFn = (X) -> ViewOutput`."""
    if name not in _REG:
        raise KeyError(f"Unknown view '{name}'. Available: {sorted(_REG)}")
    f = _REG[name]

    # Try factory(**params) -> ViewFn first.
    try:
        maybe_fn = f(**params)  # type: ignore[misc]
        if callable(maybe_fn):
            # Best-effort sanity check: expect one positional (X) or only kwargs.
            sig = inspect.signature(maybe_fn)
            return cast(ViewFn, maybe_fn)
    except TypeError:
        pass  # Not a factory with these params; treat as builder.

    # Fallback: builder(X, **params) -> ViewOutput
    def bound(X: XArr) -> ViewOutput:
        return cast(Builder[Any], f)(X, **params)  # type: ignore[misc]
    return bound

def available_views() -> list[str]:
    return sorted(_REG)

def get_view(name: str) -> Callable[..., Any]:
    return _REG[name]  # fix: `_VIEWS` does not exist in the file

# ---------- Features API (unchanged) ----------
def register_feature(name: str):
    def deco(func): _FEATURES[name] = func; return func
    return deco

def get_feature(name: str): 
    return _FEATURES[name]

