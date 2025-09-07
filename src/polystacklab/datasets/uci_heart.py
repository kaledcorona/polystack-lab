from __future__ import annotations
import numpy as np, pandas as pd
from .registry import register
from .base import Dataset

@register("uci_heart")
class UCIHeart:
    name, task = "uci_heart", "classification"
    def __init__(self, csv_path: str = "data/heart.csv"): self.csv_path = csv_path
    def load(self, split="train"):
        df = pd.read_csv(self.csv_path).dropna()
        y = df.pop("target").to_numpy()
        X = df.to_numpy(dtype=float)
        return {"tabular": X}, y
