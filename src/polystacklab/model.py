from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np
from numpy.typing import NDArray, ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier

#from polystack import Polystack  # your package
# If your meta-features registry is exposed:
# from polystack.meta_features import create as create_meta


# ---------------------------------------------------------------------
# Estimator construction (safe, config-friendly)
# ---------------------------------------------------------------------

def _import_object(dotted: str) -> Any:
    mod, name = dotted.rsplit(".", 1)
    return getattr(import_module(mod), name)

def _make_estimator(spec: Any) -> BaseEstimator:
    """
    Accepts:
      - a ready sklearn estimator instance
      - {"class": "sklearn.ensemble.RandomForestClassifier", "params": {...}}
      - legacy string "sklearn.linear_model.LogisticRegression(max_iter=1000)"
        (trusted configs only; avoid in shared code)
    """
    if isinstance(spec, BaseEstimator):
        return spec
    if isinstance(spec, dict) and "class" in spec:
        cls = _import_object(spec["class"])
        return cls(**dict(spec.get("params", {})))
    if isinstance(spec, str):
        # Very limited eval for legacy strings; safe in your own experiments.
        return eval(spec, {"__builtins__": {}}, {"sklearn": import_module("sklearn")})
    raise TypeError(f"Unsupported estimator spec: {spec!r}")


def _resolve_base_estimators(
    views: list[str],
    models_cfg: dict[str, Any] | None,
    *,
    random_state: int | None,
    n_jobs: int | None,
) -> dict[str, BaseEstimator]:
    """Build a base estimator per view with sensible defaults."""
    # Default RF if nothing is provided
    default_est: BaseEstimator = RandomForestClassifier(
        n_estimators=100, random_state=random_state, n_jobs=n_jobs
    )

    if not models_cfg:
        return {v: clone(default_est) for v in views}

    base_cfg = models_cfg.get("base", {})
    # explicit default (overrides our RF)
    if base_cfg.get("default") is not None:
        default_est = _make_estimator(base_cfg["default"])

    by_view_cfg: dict[str, Any] = base_cfg.get("by_view", {}) or {}

    ests: dict[str, BaseEstimator] = {}
    for v in views:
        if v in by_view_cfg:
            ests[v] = _make_estimator(by_view_cfg[v])
        else:
            ests[v] = clone(default_est)
    return ests


def _resolve_final_estimator(
    models_cfg: dict[str, Any] | None,
    *,
    random_state: int | None,
    n_jobs: int | None,
) -> BaseEstimator | None:
    """Pick one final estimator. Return None to bypass (single-view case)."""
    if not models_cfg:
        # sensible default final
        return RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=n_jobs)

    final_cfg = models_cfg.get("final", {})
    if "choices" in final_cfg:
        sel = final_cfg.get("select", 0)
        if sel == "all":
            # caller should loop externally; here pick the first as default
            spec = final_cfg["choices"][0]
        else:
            spec = final_cfg["choices"][int(sel)]
        return _make_estimator(spec)

    # direct single spec or None
    if final_cfg:
        return _make_estimator(final_cfg)
    return None


# ---------------------------------------------------------------------
# Training / inference API
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class FitPredictResult:
    y_pred: NDArray[np.int64]                   # predictions on X_test (or X_train if no X_test)
    y_proba: NDArray[np.floating] | None        # class probabilities if available
    model: Polystack                            # fitted model
    views: list[str]                            # view order used


def fit_predict_polystack(
    X_train: dict[str, ArrayLike],
    y_train: ArrayLike,
    *,
    X_test: dict[str, ArrayLike] | None = None,
    models: dict[str, Any] | None = None,
    meta: str | Any = "concat_proba",
    cv: int | str | Any | None = 5,
    random_state: int | None = 123,
    n_jobs: int | None = None,
) -> FitPredictResult:
    """
    Train a stacking model with Polystack and predict.

    - Builds per-view base estimators (default RF) and a final estimator.
    - If only ONE view is present, the final estimator is bypassed (plain fit/predict of the base).
    - `meta` can be the name of your meta-features plugin (e.g., "concat_proba")
      or an already constructed MetaFeatures instance.

    Returns predictions on `X_test` if provided, else on `X_train`.
    """
    if not isinstance(X_train, dict) or not X_train:
        raise TypeError("X_train must be a non-empty dict[str, array-like].")

    views = list(X_train.keys())
    # Validate consistent keys between train/test
    if X_test is not None:
        if not isinstance(X_test, dict) or set(X_test.keys()) != set(views):
            raise ValueError("X_test must be a dict with the same view keys as X_train.")

    # Build estimators
    base_ests = _resolve_base_estimators(views, models, random_state=random_state, n_jobs=n_jobs)
    final_est = None if len(views) == 1 else _resolve_final_estimator(models, random_state=random_state, n_jobs=n_jobs)

    # If you want to accept plugin instances too:
    meta_arg = meta  # if isinstance(meta, str): meta_arg = create_meta(meta)  # (optional) resolve here

    # Fit
    clf = Polystack(
        estimators=base_ests,
        final_estimator=final_est,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        meta=meta_arg,
    ).fit(X_train, y_train)

    # Predict (default to test when provided)
    X_for_pred = X_test if X_test is not None else X_train
    y_hat = clf.predict(X_for_pred)  # type: ignore[assignment]
    y_proba = None
    # Try to get class probabilities if supported
    try:
        y_proba = clf.predict_proba(X_for_pred)  # type: ignore[assignment]
    except Exception:
        y_proba = None

    return FitPredictResult(
        y_pred=np.asarray(y_hat),
        y_proba=None if y_proba is None else np.asarray(y_proba),
        model=clf,
        views=views,
    )


# ---------------------------------------------------------------------
# Back-compat shim (keeps your old function name usable)
# ---------------------------------------------------------------------

def train_multiview_stacking(
    X_train: dict[str, np.ndarray],
    X_test: dict[str, np.ndarray],
    y_train: np.ndarray,
    y_test: np.ndarray | None = None,
    *,
    models: dict[str, Any] | None = None,
    cv: int | str | Any | None = 5,
    meta: str | Any = "concat_proba",
    random_state: int = 123,
    n_jobs: int | None = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Legacy wrapper around `fit_predict_polystack` to minimize code changes.
    Returns (y_pred, y_true) where y_true is `y_test` if given, else `y_train`.
    """
    res = fit_predict_polystack(
        X_train, y_train,
        X_test=X_test,
        models=models,
        meta=meta,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    y_true = y_test if y_test is not None else y_train
    return (res.y_pred, np.asarray(y_true))
