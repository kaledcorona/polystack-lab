# Mnist adapter

from __future__ import annotations
import numpy as np
from skimage.feature import hog
from .registry import register
from .base import Dataset, Task

@register("mnist")
class MNIST(Dataset):
    name, task = "mnist", "classification"

    def __init__(self, path: str | None = None):  # path optional if you fetch via sklearn/torch
        self.path = path

    def load(self, split="train"):
        # load X_img: (n, 28, 28), y: (n,)
        # ... (use your existing loader)
        X_img, y = load_mnist_images_and_labels(self.path, split)
        n = X_img.shape[0]
        view1 = X_img.reshape(n, -1).astype(np.float32) / 255.0
        view2 = np.stack([hog(x, pixels_per_cell=(4,4), cells_per_block=(1,1), feature_vector=True)
                          for x in X_img], axis=0).astype(np.float32)
        return {"pixels": view1, "hog": view2}, y.astype(int)
