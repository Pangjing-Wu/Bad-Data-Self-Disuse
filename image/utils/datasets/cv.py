import os

import numpy as np
import torchvision
from PIL import Image


class Caltech101(torchvision.datasets.Caltech101):

    def __init__(self, root: str, train=True, transform=None, target_transform=None, download=False, seed=0) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform, download=download)
        self.classes = self.categories
        self.index   = list()
        self.y       = list()
        for (i, c) in enumerate(self.categories):
            n = len(os.listdir(os.path.join(self.root, "101_ObjectCategories", c)))
            self.index.extend(range(1, n + 1))
            self.y.extend(n * [i])
        np.random.seed(seed)
        data_index = np.random.choice(len(self.index), int(0.8 * len(self.index)), replace=False)
        data_index = data_index if train else np.array([i for i in range(len(self.index)) if i not in data_index])
        self.index = np.array(self.index)[data_index].tolist()
        self.y     = np.array(self.y)[data_index].tolist()

    def __getitem__(self, index: int):
        img = Image.open(
            os.path.join(self.root, "101_ObjectCategories", self.categories[self.y[index]], f"image_{self.index[index]:04d}.jpg",)
        ).convert('RGB') # some pictures are grayscale.
        target = self.y[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class Caltech256(torchvision.datasets.Caltech256):

    def __init__(self, root: str, train=True, transform=None, target_transform=None, download=False, seed=0) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform, download=download)
        self.classes = self.categories
        self.index   = list()
        self.y       = list()
        for (i, c) in enumerate(self.categories):
            n = len([item for item in os.listdir(os.path.join(self.root, "256_ObjectCategories", c)) if item.endswith(".jpg")])
            self.index.extend(range(1, n + 1))
            self.y.extend(n * [i])
        np.random.seed(seed)
        data_index = np.random.choice(len(self.index), int(0.8 * len(self.index)), replace=False)
        data_index = data_index if train else np.array([i for i in range(len(self.index)) if i not in data_index])
        self.index = np.array(self.index)[data_index].tolist()
        self.y     = np.array(self.y)[data_index].tolist()

    def __getitem__(self, index):
        img = Image.open(
            os.path.join(self.root, "256_ObjectCategories", self.categories[self.y[index]], f"{self.y[index] + 1:03d}_{self.index[index]:04d}.jpg")
        ).convert('RGB') # some pictures are grayscale.
        target = self.y[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target