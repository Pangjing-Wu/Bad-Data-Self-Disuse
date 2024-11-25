import numpy as np
from sklearn.neighbors import NearestNeighbors

from ..base import EmbeddingScore


class kNNConsistence(EmbeddingScore):

    def __init__(self, k: int, dist: str = 'euclidean') -> None:
        self.dist = dist
        self.k = k

    def fit(self, x: np.ndarray, y: np.ndarray):
        nearest_neighbors = NearestNeighbors(n_neighbors=self.k+1, metric=self.dist)
        nearest_neighbors.fit(x)
        ret = list()
        for x_, y_ in zip(x, y):
            neighbors = nearest_neighbors.kneighbors(x_[np.newaxis,], return_distance=False)[0, 1:]
            neighbor_labels = y[neighbors].tolist()
            ret.append(neighbor_labels.count(y_) / len(neighbor_labels))
        self.score_ = np.array(ret)
        return self