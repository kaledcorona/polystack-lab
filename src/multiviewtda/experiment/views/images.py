from __future__ import annotations

from typing import Literal
import math
import numpy as np
from numpy.typing import NDArray

from ..registry import register_view, ViewOutput

# ----------------------------
# Core reshaping utilities
# ----------------------------

def as_image(
    X_flat: NDArray[np.floating] | NDArray[np.integer],
    *,
    height: int | None = None,
    width: int | None = None,
    mode: Literal["strict", "pad", "crop"] = "strict",
    pad_value: float = 0.0,
) -> NDArray[np.floating]:
    """
    Convert a flat design matrix (n, d) to images (n, H, W).

    - If height/width are given, reshape to (H, W).
    - If not given, auto-infer a square: H = W = ceil(sqrt(d)).
      * strict: require d to be a perfect square
      * pad: pad to H*W with pad_value
      * crop: crop d down to H*W

    Returns float array without copying when possible.
    """
    X = np.asarray(X_flat)
    if X.ndim != 2:
        raise ValueError(f"X must be (n, d), got {X.shape}")
    n, d = X.shape

    if height is None or width is None:
        m = int(math.isqrt(d))
        if m * m != d:
            if mode == "strict":
                raise ValueError(f"n_features={d} is not a perfect square; pass height/width or use mode='pad'/'crop'.")
            m = math.ceil(math.sqrt(d))
        height = width = m

    H, W = int(height), int(width)
    need = H * W
    if need == d:
        return X.reshape(n, H, W).astype(float, copy=False)

    if mode == "strict":
        raise ValueError(f"d={d} does not match H*W={need} and mode='strict'.")

    if mode == "pad" and d < need:
        pad = np.pad(X, ((0, 0), (0, need - d)), mode="constant", constant_values=pad_value)
        return pad.reshape(n, H, W).astype(float, copy=False)

    if mode == "crop" and d >= need:
        return X[:, :need].reshape(n, H, W).astype(float, copy=False)

    if mode == "pad" and d > need:
        # pad cannot reduce; fallback to crop to keep shape valid
        return X[:, :need].reshape(n, H, W).astype(float, copy=False)

    if mode == "crop" and d < need:
        # crop cannot expand; fallback to pad to keep shape valid
        pad = np.pad(X, ((0, 0), (0, need - d)), mode="constant", constant_values=pad_value)
        return pad.reshape(n, H, W).astype(float, copy=False)

    # should not reach
    raise RuntimeError("Unhandled reshape case")


# ----------------------------
# Grid splitting (k×k)
# ----------------------------

def split_grid(
    imgs: NDArray[np.floating],
    *,
    rows: int,
    cols: int,
    mode: Literal["strict", "pad", "crop"] = "strict",
    pad_value: float = 0.0,
) -> dict[str, NDArray[np.floating]]:
    """
    Split (n, H, W) into a rows×cols grid of patches (flattened per patch).

    - strict: require H % rows == 0 and W % cols == 0
    - pad: pad images at the bottom/right to the next multiples
    - crop: crop images at the bottom/right to the nearest multiples

    Returns a dict of (n, patch_dim) arrays with row-major names:
      r1c1, r1c2, ..., r{rows}c{cols}.

    For the special case rows=cols=2, we also alias:
      top_left, top_right, bottom_left, bottom_right.
    """
    if imgs.ndim != 3:
        raise ValueError(f"imgs must be (n, H, W), got {imgs.shape}")
    n, H, W = imgs.shape

    # Compute new H', W' that are multiples if needed
    Hmul = (H // rows) * rows
    Wmul = (W // cols) * cols
    if H % rows != 0 or W % cols != 0:
        if mode == "strict":
            raise ValueError(f"(H, W)=({H}, {W}) not divisible by (rows, cols)=({rows}, {cols}).")
        if mode == "crop":
            H, W = Hmul, Wmul
            imgs = imgs[:, :H, :W]
        elif mode == "pad":
            Hpad = ((H + rows - 1) // rows) * rows
            Wpad = ((W + cols - 1) // cols) * cols
            padH, padW = Hpad - H, Wpad - W
            imgs = np.pad(imgs, ((0, 0), (0, padH), (0, padW)), mode="constant", constant_values=pad_value)
            H, W = Hpad, Wpad

    # Patch size
    ph, pw = H // rows, W // cols

    # Strided / reshape trick: reshape into (n, rows, ph, cols, pw) then swap axes
    patches = imgs.reshape(n, rows, ph, cols, pw)
    # flatten per patch: (n, rows, cols, ph*pw)
    patches = patches.swapaxes(2, 3).reshape(n, rows, cols, ph * pw)

    out: dict[str, NDArray[np.floating]] = {}
    for r in range(rows):
        for c in range(cols):
            name = f"r{r+1}c{c+1}"
            out[name] = patches[:, r, c]

    # Friendly aliases for 2×2
    if rows == 2 and cols == 2:
        out["top_left"] = out["r1c1"]
        out["top_right"] = out["r1c2"]
        out["bottom_left"] = out["r2c1"]
        out["bottom_right"] = out["r2c2"]

    return out


# ----------------------------
# High-level builder
# ----------------------------

def build_image_views(
    X_flat: NDArray[np.floating] | NDArray[np.integer],
    *,
    image_shape: tuple[int, int] | None = None,  # (H, W); if None, auto square
    grid: tuple[int, int] | None = (2, 2),       # None to skip grid
    include_raw: bool = True,
    reshape_mode: Literal["strict", "pad", "crop"] = "strict",
    grid_mode: Literal["strict", "pad", "crop"] = "strict",
    pad_value: float = 0.0,
) -> dict[str, NDArray[np.floating]]:
    """
    Convenience: reshape flats -> images -> optional k×k grid -> flattened views.

    Returns a dict mapping view names to (n, d_view) arrays:
      - "raw" if include_raw=True (original flattened vectors, possibly padded/cropped)
      - grid names: r1c1... or 2×2 aliases (top_left, ...)

    Examples
    --------
    >>> # MNIST (n, 784)
    >>> views = build_image_views(X, grid=(2, 2), include_raw=True)
    >>> views.keys()
    dict_keys(['raw', 'r1c1', 'r1c2', 'r2c1', 'r2c2', 'top_left', 'top_right', 'bottom_left', 'bottom_right'])
    """
    if image_shape:
        imgs = as_image(X_flat, height=image_shape[0], width=image_shape[1], mode=reshape_mode, pad_value=pad_value)
    else:
        imgs = as_image(X_flat, mode=reshape_mode, pad_value=pad_value)

    out: dict[str, NDArray[np.floating]] = {}
    if include_raw:
        # If we had to pad/crop to get imgs, use that flattened as "raw" to keep shapes consistent.
        out["raw"] = imgs.reshape(imgs.shape[0], -1)

    if grid is not None:
        r, c = grid
        grid_dict = split_grid(imgs, rows=r, cols=c, mode=grid_mode, pad_value=pad_value)
        out.update(grid_dict)

    return out


####################
# Registered views
####################

@register_view("raw")
def view_raw(
    X_flat: NDArray[np.floating] | NDArray[np.integer],
    *,
    image_shape: tuple[int, int] | None = None,
    reshape_mode: Literal["strict", "pad", "crop"] = "strict",
) -> ViewOutput:
    if image_shape is None and reshape_mode == "strict":
        # if perfectly square, keep as-is; else raise
        m = int(math.isqrt(X_flat.shape[1]))
        if m * m != X_flat.shape[1]:
            raise ValueError("raw view needs square features or pass image_shape when mode='strict'")
        return {"raw": X_flat.astype(float, copy=False)}
    imgs = as_image(X_flat, height=image_shape[0] if image_shape else None,
                    width=image_shape[1] if image_shape else None,
                    mode=reshape_mode)
    return {"raw": imgs.reshape(imgs.shape[0], -1)}

@register_view("quadrants")
def view_quadrants(
    X_flat: NDArray[np.floating] | NDArray[np.integer],
    *,
    image_shape: tuple[int, int] | None = None,
    reshape_mode: Literal["strict", "pad", "crop"] = "strict",
    grid_mode: Literal["strict", "pad", "crop"] = "strict",
    prefix: str = "",
) -> ViewOutput:
    imgs = as_image(X_flat, height=image_shape[0] if image_shape else None,
                    width=image_shape[1] if image_shape else None,
                    mode=reshape_mode)
    return split_grid(imgs, rows=2, cols=2, mode=grid_mode, prefix=prefix)

@register_view("grid")
def view_grid(
    X_flat: NDArray[np.floating] | NDArray[np.integer],
    *,
    rows: int,
    cols: int,
    image_shape: tuple[int, int] | None = None,
    reshape_mode: Literal["strict", "pad", "crop"] = "strict",
    grid_mode: Literal["strict", "pad", "crop"] = "strict",
    prefix: str = "",
    include_raw: bool = False,
) -> ViewOutput:
    imgs = as_image(X_flat, height=image_shape[0] if image_shape else None,
                    width=image_shape[1] if image_shape else None,
                    mode=reshape_mode)
    out = split_grid(imgs, rows=rows, cols=cols, mode=grid_mode, prefix=prefix)
    if include_raw:
        out[f"{prefix}raw"] = imgs.reshape(imgs.shape[0], -1)
    return out
