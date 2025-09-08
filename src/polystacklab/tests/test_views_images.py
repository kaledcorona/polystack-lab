import numpy as np
import pytest

from polystacklab.views.images import as_image, split_grid, view_grid, view_quadrants, view_raw, build_image_views

def test_exact_reshape():
    X = np.arange(12).reshape(3, 4)
    Y = as_image(X, height=2, width=2, mode="strict")
    assert Y.shape == (3, 2, 2)
    assert np.shares_memory(Y, X) is False or Y.dtype == np.float64  # dtype cast may copy

def test_auto_square_pad():
    X = np.arange(5).reshape(1, 5)
    Y = as_image(X, mode="pad")
    assert Y.shape == (1, 3, 3)
    assert np.allclose(Y.ravel()[:5], np.arange(5))
    assert np.allclose(Y.ravel()[5:], 0.0)

def test_crop():
    X = np.arange(10).reshape(1, 10)
    Y = as_image(X, height=3, width=3, mode="crop")
    assert Y.shape == (1, 3, 3)
    assert np.allclose(Y.ravel(), np.arange(9))

def test_errors():
    X = np.zeros((2, 5))
    with pytest.raises(ValueError):
        as_image(X, height=2, width=None)  # one missing
    with pytest.raises(ValueError):
        as_image(X, height=2, width=3, mode="strict")  # mismatch
    with pytest.raises(ValueError):
        as_image(X, height=2, width=3, mode="pad")  # d>H*W not allowed in pad
    with pytest.raises(ValueError):
        as_image(X, height=4, width=4, mode="crop")  # d<H*W not allowed in crop

def test_strict_exact():
    X = np.arange(2*4*6, dtype=float).reshape(2, 4, 6)
    out = split_grid(X, rows=2, cols=3, mode="strict")
    assert set(out.keys()) == {f"r{i}c{j}" for i in (1,2) for j in (1,2,3)}
    ph, pw = 2, 2
    assert all(v.shape == (2, ph*pw) for v in out.values())

def test_crop_reduce():
    X = np.arange(1*5*5, dtype=float).reshape(1, 5, 5)
    out = split_grid(X, rows=2, cols=2, mode="crop")
    assert out["r1c1"].shape == (1, 4)  # ph=pw=2 → 4 elements
    # first patch preserves top-left 2x2
    assert np.array_equal(out["r1c1"][0].reshape(2,2), X[0, :2, :2])

def test_pad_expand():
    X = np.zeros((3, 3, 4), dtype=float)
    out = split_grid(X, rows=2, cols=3, mode="pad", pad_value=1.0)
    # padded to (4, 6) → ph=2, pw=2
    assert all(v.shape == (3, 4) for v in out.values())
    # bottom-right pixel of padded area should be pad_value
    padded_img = np.pad(X, ((0,0),(0,1),(0,2)), mode="constant", constant_values=1.0)
    check = split_grid(padded_img, rows=2, cols=3, mode="strict")
    for k in out:
        assert np.array_equal(out[k], check[k])

def test_errors():
    X = np.zeros((2, 5, 5), dtype=float)
    with pytest.raises(ValueError):
        split_grid(X, rows=0, cols=2)
    with pytest.raises(ValueError):
        split_grid(X, rows=3, cols=3, mode="strict")  # not divisible
    with pytest.raises(ValueError):
        split_grid(X, rows=6, cols=2, mode="crop")    # crop cannot expand

def test_includes_raw_and_grid_shapes():
    X = np.arange(2*10, dtype=float).reshape(2, 10)  # will auto-square to 4x4 (pad)
    views = build_image_views(X, grid=(2, 2), include_raw=True, reshape_mode="pad")
    assert "raw" in views
    assert all(v.shape[0] == 2 for v in views.values())  # batch preserved
    # raw reflects padded size 16
    assert views["raw"].shape == (2, 16)
    # 2x2 grid over 4x4 -> each patch is 2x2=4
    for key in ("r1c1", "r1c2", "r2c1", "r2c2"):
        assert views[key].shape == (2, 4)

def test_no_grid_only_raw():
    X = np.arange(6, dtype=float).reshape(1, 6)
    views = build_image_views(X, grid=None, include_raw=True, image_shape=(1, 6))
    assert set(views.keys()) == {"raw"}
    assert views["raw"].shape == (1, 6)

def test_grid_modes_passed_through():
    X = np.zeros((1, 15), dtype=float)  # auto-square -> 4x4 via pad
    # grid strict over 4x4 with rows=3, cols=2 is divisible -> ok
    views = build_image_views(X, grid=(3, 2), reshape_mode="pad", grid_mode="crop")
    assert all(v.shape == (1, (4//3)*(4//2)) for v in views.values() if v is not views.get("raw"))

def test_errors_surface():
    X = np.zeros((2, 7))
    with pytest.raises(ValueError):
        build_image_views(X.reshape(14), image_shape=(2, 7))  # not 2D
    with pytest.raises(ValueError):
        build_image_views(X, image_shape=(0, 7))  # invalid shape

def test_view_raw_square_strict_passes_through():
    X = np.arange(2*16, dtype=float).reshape(2, 16)
    out = view_raw(X, reshape_mode="strict")
    assert set(out) == {"raw"}
    assert out["raw"].shape == (2, 16)

def test_view_raw_infers_and_flattens():
    X = np.arange(10, dtype=float).reshape(1, 10)
    out = view_raw(X, reshape_mode="pad")  # auto 4x4
    assert out["raw"].shape == (1, 16)

def test_view_quadrants_prefix_and_aliases():
    X = np.zeros((1, 16))
    out = view_quadrants(X, reshape_mode="pad", grid_mode="strict", prefix="q_")
    for k in ("q_r1c1","q_r1c2","q_r2c1","q_r2c2","q_top_left","q_top_right","q_bottom_left","q_bottom_right"):
        assert k in out

def test_view_grid_with_raw():
    X = np.arange(2*12, dtype=float).reshape(2, 12)
    out = view_grid(X, rows=3, cols=2, reshape_mode="pad", grid_mode="crop", include_raw=True, prefix="g_")
    assert "g_raw" in out
    assert all(v.shape[0] == 2 for v in out.values())

def test_view_grid_validates_rows_cols():
    X = np.zeros((1, 9))
    with pytest.raises(ValueError):
        view_grid(X, rows=0, cols=2)
