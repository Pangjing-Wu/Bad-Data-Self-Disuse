from collections import namedtuple


hyperparameters = [
    'n_classes',
    'k',
    'dist',
    'pca_dim',
    'repeat',
]

Configs = namedtuple('Consistency_Arguments', hyperparameters)


adults = Configs(
    n_classes=2,
    k=50,
    dist='euclidean',
    pca_dim=False,
    repeat=10
)

diabetes130 = Configs(
    n_classes=2,
    k=50,
    dist='euclidean',
    pca_dim=False,
    repeat=10
)
