from __future__ import annotations

from typing import Literal
import math
import numpy as np
from numpy.typing import NDArray

from polystacklab.registry import register_view, ViewOutput

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

    if mode == "pad":
        if d > need:
            raise ValueError(f"pad mode cannot reduce: d={d} > H*W={need}. Use mode='crop'.")
        pad_width = need - d
        padded = np.pad(X, ((0, 0), (0, pad_width)), mode="constant", constant_values=pad_value)
        return padded.reshape(n, H, W).astype(np.float64, copy=False)

    if mode == "crop":
        if d < need:
            raise ValueError(f"crop mode cannot expand: d={d} < H*W={need}. Use mode='pad'.")
        cropped = X[:, :need]
        return cropped.reshape(n, H, W).astype(np.float64, copy=False)

    raise ValueError(f"Unknown mode: {mode!r}.")

def split_grid(
    imgs: NDArray[np.floating],
    *,
    rows: int,
    cols: int,
    mode: Literal["strict", "pad", "crop"] = "strict",
    pad_value: float = 0.0,
) -> dict[str, NDArray[np.floating]]:
    """
    Split a batch of images into a `rows × cols` grid of flattened patches.

    Each image `(H, W)` is divided into equal patches of size `(ph, pw)` so that
    the batch becomes a dictionary of arrays `(n, ph * pw)`, named in row-major
    order: `r1c1, r1c2, ..., r{rows}c{cols}`. Optionally provides friendly
    aliases for the `2×2` case.

    Handling of non-divisible dimensions is controlled by `mode`:

    * `"strict"`: require `H % rows == 0` and `W % cols == 0`.
    * `"pad"`: pad on bottom/right to the next multiples with `pad_value`.
    * `"crop"`: crop from bottom/right to the previous multiples.

    Args:
      imgs: Array of shape `(n, H, W)` with a floating dtype.
      rows: Number of grid rows; must be positive.
      cols: Number of grid columns; must be positive.
      mode: How to reconcile `H, W` with `(rows, cols)`; see above.
      pad_value: Constant used for padding when `mode == "pad"`.

    Returns:
      A dict mapping patch names (e.g., `"r2c3"`) to arrays of shape `(n, ph * pw)`.
      When `rows == cols == 2`, includes aliases: `"top_left"`, `"top_right"`,
      `"bottom_left"`, `"bottom_right"`.

    Raises:
      ValueError: If `imgs` is not 3D `(n, H, W)`.
      ValueError: If `rows` or `cols` is not positive.
      ValueError: If `mode == "strict"` and dimensions are not divisible.
      ValueError: If `mode == "crop"` and `rows > H` or `cols > W` (would yield empty patches).

    Examples:
      >>> X = np.arange(2*4*6, dtype=float).reshape(2, 4, 6)
      >>> out = split_grid(X, rows=2, cols=3, mode="strict")
      >>> out["r1c1"].shape
      (2, 4)  # since ph=2, pw=2 → 4

      >>> X = np.ones((1, 5, 5))
      >>> split_grid(X, rows=2, cols=2, mode="crop")["bottom_right"].shape
      (1, 4)  # cropped to (4, 4) → ph=pw=2
    """
    if imgs.ndim != 3:
        raise ValueError(f"`imgs` must be (n, H, W); got shape {imgs.shape!r}.")
    if rows <= 0 or cols <= 0:
        raise ValueError(f"`rows` and `cols` must be positive; got rows={rows}, cols={cols}.")

    n, H, W = (int(imgs.shape[0]), int(imgs.shape[1]), int(imgs.shape[2]))

    # Determine target H', W' consistent with mode.
    if mode == "strict":
        if (H % rows) != 0 or (W % cols) != 0:
            raise ValueError(
                f"(H, W)=({H}, {W}) not divisible by (rows, cols)=({rows}, {cols}) in strict mode."
            )
        H_t, W_t = H, W
        imgs_t = imgs  # no change
    elif mode == "crop":
        if rows > H or cols > W:
            raise ValueError(
                f"crop mode cannot expand: rows <= H and cols <= W required; got rows={rows}, H={H}, cols={cols}, W={W}."
            )
        H_t = (H // rows) * rows
        W_t = (W // cols) * cols
        imgs_t = imgs[:, :H_t, :W_t]
    elif mode == "pad":
        H_t = ((H + rows - 1) // rows) * rows
        W_t = ((W + cols - 1) // cols) * cols
        pad_h, pad_w = H_t - H, W_t - W
        if pad_h or pad_w:
            imgs_t = np.pad(
                imgs,
                pad_width=((0, 0), (0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=pad_value,
            )
        else:
            imgs_t = imgs
    else:
        # Defensive; should be unreachable with typing.
        raise ValueError(f"Unknown mode: {mode!r}.")

    ph, pw = H_t // rows, W_t // cols  # guaranteed integers by construction
    # Reshape to (n, rows, ph, cols, pw) then move to (n, rows, cols, ph*pw).
    patches = imgs_t.reshape(n, rows, ph, cols, pw).swapaxes(2, 3).reshape(n, rows, cols, ph * pw)

    out: dict[str, NDArray[np.floating]] = {}
    for r in range(rows):
        for c in range(cols):
            out[f"r{r+1}c{c+1}"] = patches[:, r, c]

    if rows == 2 and cols == 2:
        out["top_left"] = out["r1c1"]
        out["top_right"] = out["r1c2"]
        out["bottom_left"] = out["r2c1"]
        out["bottom_right"] = out["r2c2"]

    return out

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
    Build named flattened views from flat feature rows via image/grid transforms.

    The pipeline is:
      1) Reshape each flat row `(d,)` to an image `(H, W)` via `as_image`, with
         optional padding/cropping controlled by `reshape_mode`.
      2) Optionally include a `"raw"` view by flattening the *post-reshape* images
         back to `(H*W,)` (keeps shapes consistent when step 1 changed length).
      3) Optionally split images into a `rows × cols` grid via `split_grid`,
         returning row-major patch names (`"r1c1"`, ..., `"r{rows}c{cols}"`).
         For `2×2`, convenient aliases (`"top_left"`, ...) are included.

    Args:
      X_flat: Array of shape `(n, d)` with float or integer dtype.
      image_shape: Optional `(H, W)`. If `None`, `as_image` infers a square
        shape. See `reshape_mode`.
      grid: Optional `(rows, cols)` for grid splitting. If `None`, grid step is
        skipped.
      include_raw: If `True`, include `"raw"` view formed by flattening the
        images produced by step 1.
      reshape_mode: Reconciliation policy for `d` vs `H*W` in `as_image`:
        `"strict"`, `"pad"`, or `"crop"`.
      grid_mode: Reconciliation policy for `(H, W)` vs `(rows, cols)` in
        `split_grid`: `"strict"`, `"pad"`, or `"crop"`.
      pad_value: Constant used when padding in either step.

    Returns:
      A dict mapping view names to arrays of shape `(n, d_view)` with floating
      dtype. Dict insertion order follows creation order (`"raw"` first if present).

    Raises:
      ValueError: If `X_flat` is not `(n, d)`.
      ValueError: If `image_shape` is malformed or non-positive.
      ValueError: If `reshape_mode` or `grid_mode` reject the input sizes (e.g.,
        non-divisible in `"strict"`), as enforced by `as_image`/`split_grid`.

    Examples:
      >>> # MNIST (n, 784), 2×2 grid with raw included
      >>> views = build_image_views(X, grid=(2, 2), include_raw=True)
      >>> sorted(views.keys())[:5]
      ['bottom_left', 'bottom_right', 'r1c1', 'r1c2', 'r2c1']
    """
    X = np.asarray(X_flat)
    if X.ndim != 2:
        raise ValueError(f"`X_flat` must be 2D (n, d); got shape {X.shape!r}.")

    if image_shape is not None:
        if (
            not isinstance(image_shape, tuple)
            or len(image_shape) != 2
            or image_shape[0] <= 0
            or image_shape[1] <= 0
        ):
            raise ValueError(
                f"`image_shape` must be a positive (H, W) tuple; got {image_shape!r}."
            )
        H, W = int(image_shape[0]), int(image_shape[1])
        imgs = as_image(
            X, height=H, width=W, mode=reshape_mode, pad_value=pad_value
        )
    else:
        imgs = as_image(X, mode=reshape_mode, pad_value=pad_value)

    out: dict[str, NDArray[np.floating]] = {}

    if include_raw:
        # Use the post-reshape images so that any pad/crop is reflected consistently.
        n = int(imgs.shape[0])
        out["raw"] = imgs.reshape(n, -1).astype(np.float64, copy=False)

    if grid is not None:
        rows, cols = grid
        grid_views = split_grid(
            imgs, rows=int(rows), cols=int(cols), mode=grid_mode, pad_value=pad_value
        )
        # `split_grid` already returns float arrays with row-major names.
        out.update(grid_views)

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
    """
    Return a single `"raw"` flattened view, optionally via image reshape.

    Behavior:
      * If `image_shape is None` and `reshape_mode == "strict"`, the feature
        length `d` must be a perfect square; otherwise an error is raised and
        no implicit padding/cropping occurs.
      * In all other cases, rows are reshaped to images via `as_image` (respecting
        `reshape_mode`), then flattened back to `(H*W,)` to keep dimensions
        consistent with any pad/crop that occurred.

    Args:
      X_flat: Array of shape `(n, d)` with float or integer dtype.
      image_shape: Optional `(H, W)`. If `None`, `as_image` may infer a square.
      reshape_mode: Policy for reconciling `d` with `H*W` in `as_image`:
        `"strict"`, `"pad"`, or `"crop"`.

    Returns:
      A dict with a single key `"raw"` mapping to a float array of shape
      `(n, d_view)`.

    Raises:
      ValueError: If `X_flat` is not 2D `(n, d)`.
      ValueError: If `image_shape` is malformed or non-positive.
      ValueError: If `reshape_mode == "strict"` and `d` is not a perfect square
        when `image_shape is None`.
    """
    X = np.asarray(X_flat)
    if X.ndim != 2:
        raise ValueError(f"`X_flat` must be 2D (n, d); got shape {X.shape!r}.")

    if image_shape is None and reshape_mode == "strict":
        # Keep input as-is if the feature length is a perfect square.
        d = int(X.shape[1])
        m = math.isqrt(d)
        if m * m != d:
            raise ValueError(
                "raw view needs square features or pass image_shape when mode='strict'."
            )
        return {"raw": X.astype(np.float64, copy=False)}

    # Otherwise, reshape via as_image then flatten to preserve any pad/crop.
    if image_shape is not None:
        H, W = image_shape
        if H <= 0 or W <= 0:
            raise ValueError(f"`image_shape` must be positive; got {(H, W)!r}.")
        imgs = as_image(X, height=int(H), width=int(W), mode=reshape_mode)
    else:
        imgs = as_image(X, mode=reshape_mode)

    n = int(imgs.shape[0])
    return {"raw": imgs.reshape(n, -1).astype(np.float64, copy=False)}

@register_view("quadrants")
def view_quadrants(
    X_flat: NDArray[np.floating] | NDArray[np.integer],
    *,
    image_shape: tuple[int, int] | None = None,
    reshape_mode: Literal["strict", "pad", "crop"] = "strict",
    grid_mode: Literal["strict", "pad", "crop"] = "strict",
    prefix: str = "",
) -> ViewOutput:
    """
    Return a `2×2` grid view with quadrant aliases.

    Pipeline:
      1) Reshape `X_flat` to images via `as_image` (using `reshape_mode`).
      2) Split into a `rows=2, cols=2` grid via `split_grid` (using `grid_mode`).
      3) Prefix all keys with `prefix` (if provided).

    Args:
      X_flat: Array of shape `(n, d)` with float or integer dtype.
      image_shape: Optional `(H, W)` for `as_image`. If `None`, it may infer.
      reshape_mode: Policy for `as_image`: `"strict"`, `"pad"`, or `"crop"`.
      grid_mode: Policy for `split_grid`: `"strict"`, `"pad"`, or `"crop"`.
      prefix: Optional string to prepend to view names (e.g., `"aug1_"`).

    Returns:
      A dict with keys `r1c1, r1c2, r2c1, r2c2` and the aliases
      `top_left, top_right, bottom_left, bottom_right`, each mapping to
      `(n, ph*pw)` float arrays. Keys are prefixed when `prefix` is non-empty.

    Raises:
      ValueError: As surfaced by `as_image`/`split_grid` for invalid sizes/modes.
    """
    if image_shape is not None:
      H, W = image_shape
      if H <= 0 or W <= 0:
          raise ValueError(f"`image_shape` must be positive; got {(H, W)!r}.")
      imgs = as_image(X_flat, height=int(H), width=int(W), mode=reshape_mode)
    else:
      imgs = as_image(X_flat, mode=reshape_mode)

    grid_dict = split_grid(imgs, rows=2, cols=2, mode=grid_mode)

    if not prefix:
        return grid_dict

    # Apply prefix without copying arrays.
    return {f"{prefix}{k}": v for k, v in grid_dict.items()}

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
    """
    Return a generic `rows × cols` grid view, optionally including `"raw"`.

    Pipeline:
      1) Reshape `X_flat` to images via `as_image`.
      2) Split into a grid via `split_grid`.
      3) Optionally add a `"raw"` view formed by flattening the post-reshape images.
      4) Prefix all keys with `prefix` (if provided).

    Args:
      X_flat: Array of shape `(n, d)` with float or integer dtype.
      rows: Grid rows; must be positive.
      cols: Grid cols; must be positive.
      image_shape: Optional `(H, W)` for `as_image`. If `None`, it may infer.
      reshape_mode: Policy for `as_image`: `"strict"`, `"pad"`, or `"crop"`.
      grid_mode: Policy for `split_grid`: `"strict"`, `"pad"`, or `"crop"`.
      prefix: Optional string to prepend to view names (e.g., `"aug1_"`).
      include_raw: If `True`, add a prefixed raw view named `f"{prefix}raw"`.

    Returns:
      A dict mapping row-major patch names (and optional `"raw"`) to
      `(n, d_view)` float arrays.

    Raises:
      ValueError: If `rows`/`cols` are non-positive.
      ValueError: As surfaced by `as_image`/`split_grid` for invalid sizes/modes.
    """
    if rows <= 0 or cols <= 0:
        raise ValueError(f"`rows` and `cols` must be positive; got rows={rows}, cols={cols}.")

    if image_shape is not None:
        H, W = image_shape
        if H <= 0 or W <= 0:
            raise ValueError(f"`image_shape` must be positive; got {(H, W)!r}.")
        imgs = as_image(X_flat, height=int(H), width=int(W), mode=reshape_mode)
    else:
        imgs = as_image(X_flat, mode=reshape_mode)

    grid_views = split_grid(imgs, rows=int(rows), cols=int(cols), mode=grid_mode)
    out: ViewOutput = {}

    if include_raw:
        n = int(imgs.shape[0])
        out[f"{prefix}raw"] = imgs.reshape(n, -1).astype(np.float64, copy=False)

    # Merge grid views, applying prefix if requested.
    if prefix:
        out.update({f"{prefix}{k}": v for k, v in grid_views.items()})
    else:
        out.update(grid_views)

    return out

