import itertools
from typing import Iterator

import numpy as np
import pandas as pd
import pytest

from polystacklab.datasets.mnist import MNIST


# --------- Fixtures ---------

@pytest.fixture
def tiny_df() -> pd.DataFrame:
    # 4 pixel columns to keep tests small (loader doesn't require 784)
    return pd.DataFrame(
        {
            "label": [0, 1, 9],
            "p0": [0, 255, 128],
            "p1": [10, 20, 30],
            "p2": [255, 0, 64],
            "p3": [42, 42, 42],
        }
    )


@pytest.fixture
def ten_rows_df() -> pd.DataFrame:
    n = 10
    data = {"label": list(range(n))}
    for i in range(4):
        data[f"p{i}"] = list(range(n))
    return pd.DataFrame(data)


# --------- Unit tests: _process_frame ---------

def test_process_frame_happy_path(tiny_df: pd.DataFrame) -> None:
    ds = MNIST()
    X, y = ds._process_frame(tiny_df)

    assert X.shape == (3, 4)
    assert y.shape == (3,)
    assert X.dtype == np.float32
    assert y.dtype == np.int32
    assert float(X.min()) >= 0.0 and float(X.max()) <= 1.0
    # Check a couple exact normalizations
    # 0/255 -> 0.0, 255/255 -> 1.0
    assert X[0, 0] == np.float32(0.0)
    assert X[0, 2] == np.float32(1.0)


def test_process_frame_missing_label_raises(tiny_df: pd.DataFrame) -> None:
    ds = MNIST()
    df = tiny_df.drop(columns=["label"])
    with pytest.raises(ValueError):
        ds._process_frame(df)


def test_process_frame_subsample_deterministic(ten_rows_df: pd.DataFrame) -> None:
    ds = MNIST(subsample=0.5, random_state=123)
    X1, y1 = ds._process_frame(ten_rows_df)
    X2, y2 = ds._process_frame(ten_rows_df)
    # Exactly floor(10 * 0.5) == 5 rows in pandas.sample with frac=0.5
    assert y1.shape == (5,)
    # Deterministic selection with fixed seed
    assert np.array_equal(y1, y2)
    assert np.all((X1 >= 0.0) & (X1 <= 1.0))


# --------- System tests: load() with real CSV ---------

def test_load_fullfile_no_chunk(tmp_path, tiny_df: pd.DataFrame) -> None:
    csv = tmp_path / "mnist.csv"
    tiny_df.to_csv(csv, index=False)

    ds = MNIST(path=csv)
    X, y = ds.load()

    assert X.shape == (3, 4)
    assert y.shape == (3,)
    assert X.dtype == np.float32 and y.dtype == np.int32
    assert float(X.min()) >= 0.0 and float(X.max()) <= 1.0


def test_load_streaming_chunks(tmp_path, ten_rows_df: pd.DataFrame) -> None:
    csv = tmp_path / "mnist.csv"
    ten_rows_df.to_csv(csv, index=False)

    ds = MNIST(path=csv, chunk_size=3)  # chunk sizes: 3,3,3,1
    it = ds.load()
    assert hasattr(it, "__iter__")

    total = 0
    seen_shapes = set()
    for Xc, yc in it:
        total += yc.shape[0]
        seen_shapes.add(Xc.shape[1])
        assert Xc.dtype == np.float32 and yc.dtype == np.int32
        assert float(Xc.min()) >= 0.0 and float(Xc.max()) <= 1.0

    assert total == 10
    # feature dimension is number of non-label columns (4 here)
    assert seen_shapes == {4}


def test_load_missing_file_raises() -> None:
    ds = MNIST(path="__does_not_exist__.csv")
    with pytest.raises(FileNotFoundError):
        ds.load()


def test_load_missing_label_raises(tmp_path) -> None:
    df = pd.DataFrame({"p0": [0, 1], "p1": [2, 3], "p2": [4, 5], "p3": [6, 7]})
    csv = tmp_path / "mnist.csv"
    df.to_csv(csv, index=False)

    ds = MNIST(path=csv)
    with pytest.raises(ValueError):
        ds.load()


# --------- System tests: simulate file with monkeypatch (no disk IO) ---------

def test_load_simulated_fullfile(monkeypatch, tiny_df: pd.DataFrame) -> None:
    # Pretend the file exists
    from pathlib import Path as _Path

    def fake_exists(self) -> bool:  # type: ignore[override]
        return True

    monkeypatch.setattr(_Path, "exists", fake_exists, raising=False)

    # Pretend pandas.read_csv returns our df (non-chunked path)
    monkeypatch.setattr(pd, "read_csv", lambda path: tiny_df, raising=True)

    ds = MNIST(path="dummy.csv")  # no actual file
    X, y = ds.load()
    assert X.shape == (3, 4) and y.shape == (3,)


def test_load_simulated_chunks(monkeypatch, ten_rows_df: pd.DataFrame) -> None:
    # Fake existence check
    from pathlib import Path as _Path

    def fake_exists(self) -> bool:  # type: ignore[override]
        return True

    monkeypatch.setattr(_Path, "exists", fake_exists, raising=False)

    # Create fake chunk iterator (3,3,3,1 rows)
    chunks = [ten_rows_df.iloc[i : i + 3] for i in range(0, len(ten_rows_df), 3)]

    def fake_read_csv(path, chunksize=None):
        if chunksize is None:
            return ten_rows_df
        # Return any iterable of DataFrames; the loader 'for chunk in reader' will work
        return iter(chunks)

    monkeypatch.setattr(pd, "read_csv", fake_read_csv, raising=True)

    ds = MNIST(path="dummy.csv", chunk_size=3)
    it = ds.load()
    total = 0
    for Xc, yc in it:
        total += yc.shape[0]
        assert Xc.shape[1] == 4
    assert total == 10
