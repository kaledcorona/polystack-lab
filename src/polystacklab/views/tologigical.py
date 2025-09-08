from __future__ import annotations

from typing import Any, Callable, Literal
import math
import numpy as np
from numpy.typing import NDArray

# scikit-learn
from sklearn.pipeline import make_pipeline, make_union, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

# giotto-tda
from gtda.images import (
    Binarizer, HeightFiltration, RadialFiltration,
    DensityFiltration, DilationFiltration,
)
from gtda.homology import CubicalPersistence, VietorisRipsPersistence
from gtda.diagrams import Scaler, Amplitude, PersistenceEntropy
from gtda.time_series import TakensEmbedding

# (optional) hook into your view registry
try:
    from .images import as_image  # your general reshaper
except Exception:
    as_image = None  # type: ignore

try:
    from ..registry import register_view, ViewOutput  # your view registry
except Exception:
    def register_view(_name: str):  # no-op fallback
        def deco(f): return f
        return deco
    ViewOutput = dict[str, NDArray[np.floating]]  # type: ignore


# ---------------------------------------------------------------------
# Small TDA pipeline registry (name -> factory(**params) -> sklearn tfm)
# ---------------------------------------------------------------------

_TDA_REGISTRY: dict[str, Callable[..., BaseEstimator]] = {}

def register_tda(name: str):
    def deco(fn: Callable[..., BaseEstimator]):
        _TDA_REGISTRY[name] = fn
        return fn
    return deco

def create_tda(name: str, **params: Any) -> BaseEstimator:
    if name not in _TDA_REGISTRY:
        raise KeyError(f"Unknown TDA pipeline '{name}'. Available: {sorted(_TDA_REGISTRY)}")
    return _TDA_REGISTRY[name](**params)

def available_tda() -> list[str]:
    return sorted(_TDA_REGISTRY)


# ---------------------------------------------------------------------
# Shared helpers (metrics, directions, centers)
# ---------------------------------------------------------------------

def amplitude_metrics_default() -> list[dict[str, Any]]:
    """Reasonable default set of amplitude vectorizations (scalar/vector)."""
    return [
        {"metric": "bottleneck", "metric_params": {}},
        {"metric": "wasserstein", "metric_params": {"p": 1}},
        {"metric": "wasserstein", "metric_params": {"p": 2}},
        {"metric": "landscape",   "metric_params": {"p": 2}},
        {"metric": "betti",       "metric_params": {"p": 2}},
        {"metric": "heat",        "metric_params": {"sigma": 1.5}},
    ]

def _height_directions(dim: int = 2) -> NDArray[np.int8]:
    """8 compass directions in 2D; extendable to 3D if you wish."""
    if dim != 2:
        raise ValueError("Only 2D height directions are provided here.")
    dirs = [[dx, dy] for dx in [-1, 0, 1] for dy in [-1, 0, 1] if (dx, dy) != (0, 0)]
    return np.array(dirs, dtype=np.int8)

def _radial_centers_grid(H: int, W: int, k: int = 3) -> NDArray[np.float32]:
    """k×k grid of centers at ~25%, 50%, 75% in each axis (generalizes MNIST 28x28 positions)."""
    xs = [int(round((i / (k + 1)) * (H + W) / 2.0)) for i in range(1, k + 1)]  # not used; keep simple
    # better: directly in pixel coords
    rs = [int(round((i / (k + 1)) * (H + 1))) for i in range(1, k + 1)]
    cs = [int(round((j / (k + 1)) * (W + 1))) for j in range(1, k + 1)]
    centers = np.array([[r, c] for r in rs for c in cs], dtype=np.float32)
    return centers


# ---------------------------------------------------------------------
# Feature unions (entropy + amplitudes)
# ---------------------------------------------------------------------

def _diagram_features_union(metrics: list[dict[str, Any]] | None, n_jobs: int | None) -> FeatureUnion:
    if metrics is None:
        metrics = amplitude_metrics_default()
    entropy = PersistenceEntropy(n_jobs=n_jobs)
    amps = [Amplitude(**m, n_jobs=n_jobs) for m in metrics]
    return make_union(*([entropy] + amps), n_jobs=n_jobs)


# ---------------------------------------------------------------------
# 1) Image pipelines: filtrations -> CubicalPersistence -> diagram features
# ---------------------------------------------------------------------

def _image_filtrations(
    H: int,
    W: int,
    *,
    include: list[str] = ["height", "radial", "density", "dilation"],
    directions: NDArray[np.int8] | None = None,
    centers: NDArray[np.float32] | None = None,
    density_radii: tuple[int, ...] = (2, 4, 6),
    n_jobs: int | None = None,
) -> list[Any]:
    fltrs: list[Any] = []
    if "height" in include:
        if directions is None:
            directions = _height_directions(2)
        for dvec in directions:
            fltrs.append(HeightFiltration(direction=dvec, n_jobs=n_jobs))  # docs: direction kwarg :contentReference[oaicite:1]{index=1}
    if "radial" in include:
        if centers is None:
            centers = _radial_centers_grid(H, W, k=3)
        for c in centers:
            fltrs.append(RadialFiltration(center=c, n_jobs=n_jobs))        # docs: center kwarg :contentReference[oaicite:2]{index=2}
    if "density" in include:
        for r in density_radii:
            fltrs.append(DensityFiltration(radius=r, n_jobs=n_jobs))       # radius kwarg :contentReference[oaicite:3]{index=3}
    if "dilation" in include:
        fltrs.append(DilationFiltration(n_jobs=n_jobs))                    # n_iterations optional :contentReference[oaicite:4]{index=4}
    return fltrs

def _image_pipeline_union(
    H: int,
    W: int,
    *,
    binarize_threshold: float = 0.4,
    filtrations: list[Any],
    n_jobs: int | None = None,
) -> FeatureUnion:
    steps_per = []
    for f in filtrations:
        steps_per.append(
            make_pipeline(
                Binarizer(threshold=binarize_threshold, n_jobs=n_jobs),     # :contentReference[oaicite:5]{index=5}
                f,
                CubicalPersistence(n_jobs=n_jobs),                          # :contentReference[oaicite:6]{index=6}
                Scaler(),                                                   # scale diagrams before features :contentReference[oaicite:7]{index=7}
            )
        )
    return make_union(*steps_per, n_jobs=n_jobs)

@register_tda("tda_image_basic")
def build_tda_image_basic(
    *,
    image_shape: tuple[int, int],
    binarize_threshold: float = 0.4,
    include_filtrations: list[str] = ["height", "radial", "density", "dilation"],
    density_radii: tuple[int, ...] = (2, 4, 6),
    directions: NDArray[np.int8] | None = None,
    centers: NDArray[np.float32] | None = None,
    metrics: list[dict[str, Any]] | None = None,
    n_jobs: int | None = None,
) -> BaseEstimator:
    """Images -> (Filtration -> CubicalPersistence -> Scaler)* -> (Entropy + Amplitudes)."""
    H, W = image_shape
    fltrs = _image_filtrations(
        H, W, include=include_filtrations, directions=directions,
        centers=centers, density_radii=density_radii, n_jobs=n_jobs
    )
    diag_union = _image_pipeline_union(
        H, W, binarize_threshold=binarize_threshold, filtrations=fltrs, n_jobs=n_jobs
    )
    feats = _diagram_features_union(metrics, n_jobs)
    return make_pipeline(diag_union, feats)


# ---------------------------------------------------------------------
# 2) Time-series / signals: TakensEmbedding -> VR persistence -> diagram features
# ---------------------------------------------------------------------

@register_tda("tda_signal_vr")
def build_tda_signal_vr(
    *,
    time_delay: int = 1,
    dimension: int = 3,
    stride: int = 1,
    flatten: bool = True,
    vr_homology_dims: tuple[int, ...] = (0, 1),
    vr_metric: str = "euclidean",
    vr_max_edge_length: float | None = None,
    metrics: list[dict[str, Any]] | None = None,
    n_jobs: int | None = None,
) -> BaseEstimator:
    """Signals (n, T) or list -> Takens -> VietorisRips -> Scaler -> (Entropy + Amplitudes)."""
    takens = TakensEmbedding(time_delay=time_delay, dimension=dimension, stride=stride, flatten=flatten)  # :contentReference[oaicite:8]{index=8}
    vr = VietorisRipsPersistence(
        homology_dimensions=vr_homology_dims, metric=vr_metric,
        max_edge_length=vr_max_edge_length, n_jobs=n_jobs
    )  # :contentReference[oaicite:9]{index=9}
    feats = _diagram_features_union(metrics, n_jobs)
    return make_pipeline(takens, vr, Scaler(), feats)


# ---------------------------------------------------------------------
# 3) Point clouds: VR persistence -> diagram features
# ---------------------------------------------------------------------

@register_tda("tda_pointcloud_vr")
def build_tda_pointcloud_vr(
    *,
    vr_homology_dims: tuple[int, ...] = (0, 1, 2),
    vr_metric: str = "euclidean",
    vr_max_edge_length: float | None = None,
    metrics: list[dict[str, Any]] | None = None,
    n_jobs: int | None = None,
) -> BaseEstimator:
    """Point clouds: (n_clouds, n_points, d) or list -> VR -> Scaler -> (Entropy + Amplitudes)."""
    vr = VietorisRipsPersistence(
        homology_dimensions=vr_homology_dims, metric=vr_metric,
        max_edge_length=vr_max_edge_length, n_jobs=n_jobs
    )  # :contentReference[oaicite:10]{index=10}
    feats = _diagram_features_union(metrics, n_jobs)
    return make_pipeline(vr, Scaler(), feats)


# ---------------------------------------------------------------------
# Back-compat convenience (your original entrypoint name)
# ---------------------------------------------------------------------

def build_tda_pipeline(num_jobs: int = 3) -> FeatureUnion:
    """Backwards-compatible builder for MNIST-style images (28×28)."""
    pipe = build_tda_image_basic(
        image_shape=(28, 28),
        binarize_threshold=0.4,
        include_filtrations=["height", "radial", "density", "dilation"],
        density_radii=(2, 4, 6),
        metrics=amplitude_metrics_default(),
        n_jobs=num_jobs,
    )
    # return the final union stage for parity with older code
    # (make_pipeline returns Pipeline; last step is FeatureUnion)
    return pipe[-1] if isinstance(pipe, make_pipeline().__class__) else pipe  # type: ignore


# ---------------------------------------------------------------------
# Optional: register as VIEWS so you can call from YAML via view registry
# ---------------------------------------------------------------------

@register_view("tda")  # images -> 'tda' view matrix
def view_tda_image(
    X: NDArray[np.floating] | NDArray[np.integer],
    *,
    image_shape: tuple[int, int] | None = None,
    n_jobs: int | None = None,
    binarize_threshold: float = 0.4,
    include_filtrations: list[str] = ["height", "radial", "density", "dilation"],
    density_radii: tuple[int, ...] = (2, 4, 6),
    metrics: list[dict[str, Any]] | None = None,
) -> ViewOutput:
    """Return {'tda': feature_matrix} from images (flattened or H×W)."""
    if as_image is None and (image_shape is None or X.ndim == 2):
        raise RuntimeError("as_image helper not available; import images.as_image or pass already shaped images.")
    if X.ndim == 2:
        if image_shape is None:
            raise ValueError("Pass image_shape=(H,W) when X is flattened.")
        imgs = as_image(X, height=image_shape[0], width=image_shape[1], mode="pad")  # reuse your helper
    elif X.ndim == 3:
        imgs = X
        H, W = imgs.shape[1], imgs.shape[2]
        image_shape = (H, W)
    else:
        raise ValueError(f"Expected (n,d) or (n,H,W) images; got shape {X.shape}.")

    tda = build_tda_image_basic(
        image_shape=image_shape, binarize_threshold=binarize_threshold,
        include_filtrations=include_filtrations, density_radii=density_radii,
        metrics=metrics, n_jobs=n_jobs,
    )
    feats = tda.fit_transform(imgs)
    return {"tda": np.asarray(feats, dtype=float)}

@register_view("tda_signal")
def view_tda_signal(
    X: NDArray[np.floating] | list[NDArray[np.floating]],
    *,
    n_jobs: int | None = None,
    time_delay: int = 1,
    dimension: int = 3,
    stride: int = 1,
    metrics: list[dict[str, Any]] | None = None,
) -> ViewOutput:
    """Return {'tda_signal': feature_matrix} from (n, T) or list of 1D arrays."""
    tda = build_tda_signal_vr(
        time_delay=time_delay, dimension=dimension, stride=stride,
        metrics=metrics, n_jobs=n_jobs,
    )
    feats = tda.fit_transform(X)
    return {"tda_signal": np.asarray(feats, dtype=float)}

@register_view("tda_pointcloud")
def view_tda_pointcloud(
    X: NDArray[np.floating] | list[NDArray[np.floating]],
    *,
    n_jobs: int | None = None,
    vr_homology_dims: tuple[int, ...] = (0, 1, 2),
    metrics: list[dict[str, Any]] | None = None,
) -> ViewOutput:
    """Return {'tda_pointcloud': feature_matrix} from point clouds (n_clouds, n_points, d) or list."""
    tda = build_tda_pointcloud_vr(
        vr_homology_dims=vr_homology_dims, metrics=metrics, n_jobs=n_jobs,
    )
    feats = tda.fit_transform(X)
    return {"tda_pointcloud": np.asarray(feats, dtype=float)}
