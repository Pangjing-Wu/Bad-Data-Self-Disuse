import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances

from ..base import EmbeddingScore


class Centroid(EmbeddingScore):

    def __init__(self, n_classes: int) -> None:
        self.n_classes = n_classes
        self.dist_func = cosine_distances
        self.cluster   = KMeans(n_clusters=n_classes, n_init="auto")
    
    def fit(self, x: np.ndarray):
        y = self.cluster.fit(x).labels_
        self.score_    = np.zeros(x.shape[0])
        self.centroids = self.cluster.cluster_centers_
        for y_ in range(self.n_classes):
            self.score_[y==y_] = self.dist_func(x[y==y_], self.centroids[np.newaxis, y_])[:,0]
        return self