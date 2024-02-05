import torch
from sklearn.decomposition import PCA as SklearnPCA


__all__ = ['PCA']


class PCA(object):

    def __init__(self, n_components=2, *args, **kwargs):
        self.pca = SklearnPCA(n_components, *args, **kwargs)
        self.n_components = n_components

    def fit(self, x:torch.Tensor):
        x = x.numpy()
        self.pca.fit(x)
        return self

    def transform(self, x:torch.Tensor) -> torch.Tensor:
        x = x.numpy()
        return torch.Tensor(self.pca.transform(x))
    
    def fit_transform(self, x:torch.Tensor) -> torch.Tensor:
        x = x.numpy()
        return torch.Tensor(self.pca.fit_transform(x))