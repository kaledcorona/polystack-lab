from __future__ import annotations
from typing import Protocol, Literal
import numpy as np

Split = Literal["train", "test"]
Task = Literal["classification", "regression"]

# Dataset Protocol + common utils

class Dataset(Protocol):
    name: str
    task: Task

    def load(self, split: Split = "train") -> tuple[np.ndarray, np.ndarray]: ...
