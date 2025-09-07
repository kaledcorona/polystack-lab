from __future__ import annotations

from typing import Any, Callable, Literal
import math
import numpy as np
from numpy.typing import NDArray

# If you have the general image reshaper in images.py
try:
    from .images import as_image
except Exception:
    as_image = None  # fallback if not available

ArrayF = NDArray[np.floating] | NDArray[np.integer]
RNG = np.random.Generator


# -----------------------------------------------------------------------------
# Registry (noise name -> factory(**defaults) -> callable(X, rng, **overrides))
# -----------------------------------------------------------------------------

_NOISE_REGISTRY: dict[str, Callable[..., Callable[..., ArrayF]]] = {}

def register_noise(name: str):
    """Decorator to register a noise factory under a string key."""
    def deco(factory: Callable[..., Callable[..., ArrayF]]):
        _NOISE_REGISTRY[name] = factory
        return factory
    return deco

def create_noise(name: str, **defaults: Any) -> Callable[[ArrayF], ArrayF]:
    """Create a noise function with pre-bound defaults.

    Returns a callable:  fn(X, *, rng: RNG | None = None, **overrides) -> X_noisy
    """
    if name not in _NOISE_REGISTRY:
        raise KeyError(f"Unknown noise '{name}'. Available: {sorted(_NOISE_REGISTRY)}")
    factory = _NOISE_REGISTRY[name]
    base = factory(**defaults)

    def _fn(X: ArrayF, *, rng: RNG | None = None, **overrides: Any) -> ArrayF:
        if rng is None:
            rng = np.random.default_rng()
        return base(X, rng=rng, **overrides)

    return _fn

def available_noises() -> list[str]:
    return sorted(_NOISE_REGISTRY)


# -----------------------------------------------------------------------------
# Common helpers
# -----------------------------------------------------------------------------

def _value_range_for(X: ArrayF, value_range: tuple[float, float] | None) -> tuple[float, float]:
    if value_range is not None:
        return float(value_range[0]), float(value_range[1])
    if np.issubdtype(X.dtype, np.integer):
        # Assume 8-bit images/signals if integer (safe default)
        info = np.iinfo(X.dtype)
        return float(info.min), float(info.max)
    # Float: assume 0..1 by default; fall back to data min/max if outside
    lo, hi = 0.0, 1.0
    xmin, xmax = float(np.min(X)), float(np.max(X))
    if xmin < lo - 1e-9 or xmax > hi + 1e-9:
        lo, hi = xmin, xmax
    return lo, hi

def _per_sample_rngs(rng: RNG, n: int) -> list[RNG]:
    seeds = rng.integers(0, 2**32 - 1, size=n, dtype=np.uint32)
    return [np.random.default_rng(int(s)) for s in seeds]


# -----------------------------------------------------------------------------
# IMAGE NOISES (accept 2-D (H,W), 3-D (N,H,W), or 2-D flattened (N,D) with image_shape)
# -----------------------------------------------------------------------------

def _normalize_images(
    X: ArrayF,
    *,
    image_shape: tuple[int, int] | None,
    reshape_mode: Literal["strict", "pad", "crop"] = "strict",
) -> tuple[np.ndarray, tuple[int, int], bool]:
    """Return imgs as float array (N,H,W), its shape, and a flag 'flattened_input'."""
    X_arr = np.asarray(X)
    flattened_input = False

    if X_arr.ndim == 3:
        # (N, H, W)
        n, H, W = X_arr.shape
        imgs = X_arr.astype(float, copy=False)
        return imgs, (H, W), flattened_input

    if X_arr.ndim == 2:
        if image_shape is None:
            # try auto-square if (N, D)
            n, d = X_arr.shape
            m = int(math.isqrt(d))
            if m * m != d and reshape_mode == "strict":
                raise ValueError("image_shape required for flattened arrays when mode='strict'")
            H = W = m if m * m == d else math.ceil(math.sqrt(d))
        else:
            H, W = int(image_shape[0]), int(image_shape[1])

        # use as_image helper if available, otherwise manual reshape/pad/crop
        if as_image is not None:
            imgs = as_image(X_arr, height=H, width=W, mode=reshape_mode)
        else:
            need = H * W
            n, d = X_arr.shape
            Xw = X_arr
            if need != d:
                if reshape_mode == "strict":
                    raise ValueError(f"d={d} != H*W={need} and mode='strict'")
                if d < need:
                    pad = np.pad(Xw, ((0, 0), (0, need - d)))
                    Xw = pad
                else:
                    Xw = Xw[:, :need]
            imgs = Xw.reshape(n, H, W)
        flattened_input = True
        return imgs.astype(float, copy=False), (H, W), flattened_input

    raise ValueError(f"Images must be (H,W), (N,H,W), or (N,D). Got shape {X_arr.shape!r}.")


@register_noise("image/salt_pepper")
def _factory_image_salt_pepper(
    *,
    p: float = 0.05,
    value_range: tuple[float, float] | None = None,
) -> Callable[..., ArrayF]:
    """Salt & pepper for images. 'p' is total corruption probability."""
    def _apply(X: ArrayF, *, rng: RNG, image_shape: tuple[int, int] | None = None,
               reshape_mode: Literal["strict","pad","crop"] = "strict") -> ArrayF:
        imgs, _, flattened = _normalize_images(X, image_shape=image_shape, reshape_mode=reshape_mode)
        lo, hi = _value_range_for(imgs, value_range)
        # Draw a single mask for batch (vectorized)
        mask = rng.random(size=imgs.shape)
        imgs = imgs.copy()
        imgs[mask < (p / 2)] = lo
        imgs[mask > (1 - p / 2)] = hi
        if flattened:
            n = imgs.shape[0]
            return imgs.reshape(n, -1).astype(X.dtype, copy=False)
        return imgs.astype(X.dtype, copy=False)
    return _apply


@register_noise("image/gaussian")
def _factory_image_gaussian(
    *,
    sigma: float = 0.05,
    value_range: tuple[float, float] | None = None,
    clip: bool = True,
) -> Callable[..., ArrayF]:
    """Additive Gaussian: X' = X + N(0, sigma*(hi-lo))."""
    def _apply(X: ArrayF, *, rng: RNG, image_shape: tuple[int,int] | None = None,
               reshape_mode: Literal["strict","pad","crop"] = "strict") -> ArrayF:
        imgs, _, flattened = _normalize_images(X, image_shape=image_shape, reshape_mode=reshape_mode)
        lo, hi = _value_range_for(imgs, value_range)
        scale = sigma * (hi - lo)
        noise = rng.normal(loc=0.0, scale=scale, size=imgs.shape)
        out = imgs + noise
        if clip:
            out = np.clip(out, lo, hi)
        if flattened:
            return out.reshape(out.shape[0], -1).astype(X.dtype, copy=False)
        return out.astype(X.dtype, copy=False)
    return _apply


@register_noise("image/speckle")
def _factory_image_speckle(
    *,
    sigma: float = 0.05,
    value_range: tuple[float, float] | None = None,
    clip: bool = True,
) -> Callable[..., ArrayF]:
    """Multiplicative noise: X' = X + X * N(0, sigma)."""
    def _apply(X: ArrayF, *, rng: RNG, image_shape: tuple[int,int] | None = None,
               reshape_mode: Literal["strict","pad","crop"] = "strict") -> ArrayF:
        imgs, _, flattened = _normalize_images(X, image_shape=image_shape, reshape_mode=reshape_mode)
        lo, hi = _value_range_for(imgs, value_range)
        noise = rng.normal(loc=0.0, scale=sigma, size=imgs.shape)
        out = imgs + imgs * noise
        if clip:
            out = np.clip(out, lo, hi)
        if flattened:
            return out.reshape(out.shape[0], -1).astype(X.dtype, copy=False)
        return out.astype(X.dtype, copy=False)
    return _apply


@register_noise("image/poisson")
def _factory_image_poisson(
    *,
    scale: float = 1.0,
    value_range: tuple[float, float] | None = None,
    clip: bool = True,
) -> Callable[..., ArrayF]:
    """Poisson noise by scaling intensities to counts, sampling, and rescaling."""
    def _apply(X: ArrayF, *, rng: RNG, image_shape: tuple[int,int] | None = None,
               reshape_mode: Literal["strict","pad","crop"] = "strict") -> ArrayF:
        imgs, _, flattened = _normalize_images(X, image_shape=image_shape, reshape_mode=reshape_mode)
        lo, hi = _value_range_for(imgs, value_range)
        span = max(hi - lo, 1e-12)
        # Shift to [0, span] for counts
        s = (imgs - lo) / span
        out = rng.poisson(lam=np.maximum(s * scale, 0.0))
        out = (out.astype(float) / max(scale, 1e-12)) * span + lo
        if clip:
            out = np.clip(out, lo, hi)
        if flattened:
            return out.reshape(out.shape[0], -1).astype(X.dtype, copy=False)
        return out.astype(X.dtype, copy=False)
    return _apply


@register_noise("image/dropout")
def _factory_image_dropout(
    *,
    rate: float = 0.1,
    value: float | None = 0.0,
) -> Callable[..., ArrayF]:
    """Randomly drop a fraction of pixels to a constant value (default 0)."""
    def _apply(X: ArrayF, *, rng: RNG, image_shape: tuple[int,int] | None = None,
               reshape_mode: Literal["strict","pad","crop"] = "strict") -> ArrayF:
        imgs, _, flattened = _normalize_images(X, image_shape=image_shape, reshape_mode=reshape_mode)
        mask = rng.random(size=imgs.shape) < rate
        out = imgs.copy()
        if value is None:
            # if None, keep existing; (useful placeholder)
            pass
        else:
            out[mask] = value
        if flattened:
            return out.reshape(out.shape[0], -1).astype(X.dtype, copy=False)
        return out.astype(X.dtype, copy=False)
    return _apply


# -----------------------------------------------------------------------------
# SIGNAL NOISES (accept (T,), (N,T))
# -----------------------------------------------------------------------------

def _normalize_signals(X: ArrayF) -> tuple[np.ndarray, bool]:
    """Return (N,T) float array and flag if original was (T,)."""
    X_arr = np.asarray(X)
    if X_arr.ndim == 1:
        return X_arr[None, :].astype(float, copy=False), True
    if X_arr.ndim == 2:
        return X_arr.astype(float, copy=False), False
    raise ValueError(f"Signals must be (T,) or (N,T). Got {X_arr.shape!r}.")

@register_noise("signal/gaussian")
def _factory_signal_gaussian(*, sigma: float = 0.05, clip: tuple[float,float] | None = None) -> Callable[..., ArrayF]:
    def _apply(X: ArrayF, *, rng: RNG) -> ArrayF:
        S, was_1d = _normalize_signals(X)
        noise = rng.normal(0.0, sigma, size=S.shape)
        out = S + noise
        if clip is not None:
            out = np.clip(out, clip[0], clip[1])
        return out[0] if was_1d else out
    return _apply

@register_noise("signal/impulse")
def _factory_signal_impulse(*, p: float = 0.01, lo: float = -1.0, hi: float = 1.0) -> Callable[..., ArrayF]:
    """Impulse 'spikes' with prob p per sample."""
    def _apply(X: ArrayF, *, rng: RNG) -> ArrayF:
        S, was_1d = _normalize_signals(X)
        mask = rng.random(size=S.shape) < p
        spikes = rng.uniform(lo, hi, size=S.shape)
        out = S.copy()
        out[mask] = spikes[mask]
        return out[0] if was_1d else out
    return _apply


# -----------------------------------------------------------------------------
# Convenience front-ends
# -----------------------------------------------------------------------------

def apply_noise(
    X: ArrayF,
    name: str,
    *,
    rng: RNG | None = None,
    image_shape: tuple[int, int] | None = None,
    **params: Any,
) -> ArrayF:
    """Apply a registered noise by name. Works for images or signals.

    - Images: pass flattened (N,D) + image_shape, or (N,H,W)/(H,W)
    - Signals: pass (T,) or (N,T)
    """
    fn = create_noise(name, **params)
    return fn(X, rng=rng, image_shape=image_shape)  # 'image_shape' ignored by signal noises


# -----------------------------------------------------------------------------
# Back-compat shims (your old API names)
# -----------------------------------------------------------------------------

def add_salt_and_pepper_noise(image: np.ndarray, p: float, random_seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    fn = create_noise("image/salt_pepper", p=p)
    return fn(image, rng=rng)

def add_salt_and_pepper_noise_batch(images: np.ndarray, p: float, random_seed: int, max_workers: int | None = None) -> np.ndarray:
    # vectorized; max_workers kept only for signature compatibility
    rng = np.random.default_rng(random_seed)
    fn = create_noise("image/salt_pepper", p=p)
    return fn(images, rng=rng)

# Legacy geometric noises (kept for now; consider removing later)
def add_lines_noise_to_images(*args, **kwargs):  # deprecated
    raise NotImplementedError("Deprecated: geometric 'lines' noise is no longer supported.")
def add_circles_noise_to_images(*args, **kwargs):  # deprecated
    raise NotImplementedError("Deprecated: geometric 'circles' noise is no longer supported.")


# -----------------------------------------------------------------------------
# Dataset helpers (flattened MNIST-style arrays)
# -----------------------------------------------------------------------------

def contaminate_flat_images(
    X_flat: np.ndarray,
    *,
    name: str,
    image_shape: tuple[int, int],
    rng: RNG | None = None,
    **params: Any,
) -> np.ndarray:
    """Apply an image noise to flattened (N,D) arrays and return flattened (N,D)."""
    noisy = apply_noise(X_flat, name, rng=rng, image_shape=image_shape, **params)
    if noisy.ndim == 3:
        n = noisy.shape[0]
        return noisy.reshape(n, -1)
    return noisy

def add_noise_to_data(
    X_chunk: np.ndarray,
    noise_type: str,
    quantity: float,
    random_state: int,
    image_shape: tuple[int, int] | None = None,
):
    """Compatibility wrapper for your previous call sites.

    - noise_type: one of {"salt_pepper","gaussian","speckle","poisson","dropout"}
    - quantity: mapped to the primary parameter (p / sigma / scale / rate)
    """
    name_map = {
        "salt_pepper": "image/salt_pepper",
        "gaussian": "image/gaussian",
        "speckle": "image/speckle",
        "poisson": "image/poisson",
        "dropout": "image/dropout",
        # legacy passthroughs (will raise NotImplementedError):
        "lines": None,
        "circles": None,
    }
    key = name_map.get(noise_type)
    if key is None:
        raise ValueError(f"Unknown or deprecated noise type: {noise_type}")

    rng = np.random.default_rng(int(random_state))
    params: dict[str, Any]
    if key.endswith("salt_pepper"):
        params = {"p": float(quantity)
    elif key.endswith("gaussian"):
        params = {"sigma": float(quantity)}
    elif key.endswith("speckle"):
        params = {"sigma": float(quantity)}
    elif key.endswith("poisson"):
        params = {"scale": float(quantity)}
    elif key.endswith("dropout"):
        params = {"rate": float(quantity)}
    else:
        params = {}

    return contaminate_flat_images(
        X_chunk, name=key, image_shape=image_shape or (28, 28), rng=rng, **params
    )
