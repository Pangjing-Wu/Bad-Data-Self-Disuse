import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

from ..base import EmbeddingScore


class Centroid(EmbeddingScore):

    def __init__(self, dist: str, n_classes: int) -> None:
        super().__init__(dist=dist)
        self.n_classes = n_classes
        self.dist_func = self.__set_dist_func(dist)
    
    def fit(self, x: np.ndarray, y: np.ndarray):
        self.score_    = np.zeros(x.shape[0])
        self.centroids = self.__calculate_centroid(x, y)
        self.radiuses  = self.__calculate_radius(x, y)
        for y_ in range(self.n_classes):
            self.score_[y==y_] = self.dist_func(x[y==y_], self.centroids[np.newaxis, y_])[:,0] / self.radiuses[y_]
        return self

    def __calculate_centroid(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.stack([np.mean(x[y==y_], axis=0) for y_ in range(self.n_classes)], axis=0)
    
    def __calculate_radius(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.stack([np.mean(self.dist_func(x[y==y_], self.centroids[np.newaxis, y_])) for y_ in range(self.n_classes)])
        
    def __set_dist_func(self, dist):
        if dist == 'euclidean':
            return euclidean_distances
        elif dist == 'cosine':
            return cosine_distances
        else:
            raise ValueError(f"unknown distance function '{dist}'")