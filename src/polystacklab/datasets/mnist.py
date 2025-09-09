"""
MNIST dataset loader with optional chunking and subsampling.

This module exposes a single entry point, :class:`MNIST`, that reads a CSV file
containing a `label` column and pixel columns (e.g., 784 columns for 28×28).
Pixels are normalized to the range [0, 1] (float32) and labels are int32.

Typical CSV schema
------------------
- Required column: ``label`` (digits 0–9)
- Feature columns: pixel values per row (0–255), any column names are accepted.
  The loader infers feature count from all non-``label`` columns.

Quickstart
----------
Load the whole file into memory:

    >>> from polystacklab.datasets.mnist import MNIST
    >>> ds = MNIST(path="mnist.csv")
    >>> X, y = ds.load()
    >>> X.shape  # (n_rows, n_features)
    (60000, 784)
    >>> X.dtype, y.dtype
    (dtype('float32'), dtype('int32'))
    >>> float(X.min()) >= 0.0 and float(X.max()) <= 1.0
    True

Streaming / chunked loading (large files)
-----------------------------------------
Set ``chunk_size`` to stream batches instead of loading everything at once:

    >>> ds = MNIST(path="mnist.csv", chunk_size=2048)
    >>> for Xc, yc in ds.load():
    ...     # train on the chunk
    ...     pass

Subsampling (for quick experiments)
-----------------------------------
Apply a deterministic random subsample per (full file or chunk):

    >>> ds = MNIST(path="mnist.csv", subsample=0.1, random_state=123)
    >>> X_small, y_small = ds.load()

Registry integration (optional)
-------------------------------
If your project uses a registry with ``@register_dataset("mnist")``, the class is
registered under the name ``"mnist"`` and can be constructed via your factory
(e.g., ``create_dataset("mnist", path=...)``), then call ``.load()`` as above.

API at a glance
---------------
class MNIST:
    - Attributes
        * name: "mnist"
        * task: "classification"
        * path: str | pathlib.Path
        * chunk_size: int | None
        * subsample: float | None
        * random_state: int | None
    - Methods
        * load(split: str = "train") -> (X: float32[n,d], y: int32[n]) | iterator[(Xc, yc)]

Returns
-------
- ``X``: numpy.ndarray (float32), shape ``(n, d)``, values in ``[0, 1]``.
- ``y``: numpy.ndarray (int32), shape ``(n,)``.

Raises
------
- ``FileNotFoundError``: when ``path`` does not exist.
- ``ValueError``: when the input frame/chunk lacks a ``'label'`` column.

Notes
-----
- The loader treats every non-``label`` column as a pixel feature.
- For performance, dtypes are cast without unnecessary copies when possible.
- Feature count is not enforced to 784; the code works with any number of pixels.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


import pandas as pd
import numpy as np
from numpy.typing import NDArray

from polystacklab.registry import register_dataset

Float32Array = NDArray[np.float32]
Int32Array = NDArray[np.int32]
Batch = tuple[Float32Array, Int32Array]
LoadResult = Batch | Iterator[Batch]

@register_dataset("mnist")
@dataclass(slots=True)
class MNIST:
    """
    MNIST dataset loader with optional chunking and subsampling.

    This loader reads a CSV with a `label` column and 784 pixel columns.
    Pixels are scaled to `[0, 1]` as `float32`; labels are `int32`.

    Attributes:
      name: Dataset name (constant).
      task: Task kind (e.g., "classification").
      path: Path to a CSV file (or directory if you handle `split` externally).
      chunk_size: If provided, stream data in chunks of this many rows.
      subsample: If provided in `(0, 1)`, sample this fraction of rows.
      random_state: Random seed for subsampling.
    """
    # Class metadata
    name: str = "mnist"
    task: str = "classification"

    # Configuration
    path: str | Path = ""
    chunk_size: int | None = None
    subsample: float | None = None
    random_state: int | None = None

    # -- Public API
    def load(self, split: str = "train") -> LoadResult:
        """Load MNIST data.

        Args:
          split: Split identifier (e.g., "train", "test"). Ignored if `path` points
            directly to a single CSV.

        Returns:
          Either:
            * `(X, y)` where `X: float32 (n, d)` in `[0, 1]`, `y: int32 (n,)`, or
            * An iterator yielding `(X_chunk, y_chunk)` batches if `chunk_size` is set.

        Raises:
          FileNotFoundError: If `path` does not exist.
          ValueError: If the input frame/chunk lacks a `'label'` column.
        """
        csv_path = Path(self.path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Path does not exist: {csv_path}")

        if self.chunk_size is None:
            df = pd.read_csv(csv_path)
            return self._process_frame(df)
        else:
            reader = pd.read_csv(csv_path, chunksize=self.chunk_size)

            def gen() -> Iterator[Batch]:
                for chunk in reader:
                    yield self._process_frame(chunk)

            return gen()

    # ---- Helper 

    def _process_frame(self, df: pd.DataFrame) -> Batch:
        """Subsample, extract features/labels, and normalize a DataFrame.

        Args:
          df: DataFrame containing `label` and pixel columns.

        Returns:
          `(X, y)` where `X: float32 (n, d)` in `[0, 1]`, `y: int32 (n,)`.

        Raises:
          ValueError: If `'label'` is missing.
        """
        if self.subsample is not None and 0.0 < self.subsample < 1.0:
            df = df.sample(frac=self.subsample, random_state=self.random_state)

        if "label" not in df.columns:
            raise ValueError("Expected a 'label' column in the CSV.")

        y: Int32Array = df["label"].to_numpy(dtype=np.int32, copy=False)
        X_raw = df.drop(columns=["label"]).to_numpy(dtype=np.float32, copy=False)

        X: Float32Array = (X_raw / np.float32(255.0)).astype(np.float32, copy=False)
        return X, y
