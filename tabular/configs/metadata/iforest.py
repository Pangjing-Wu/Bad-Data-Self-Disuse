from collections import namedtuple


hyperparameters = [
    'n_classes',
    'iforest_kwargs',
    'pca_dim',
    'repeat',
]

Configs = namedtuple('IForest_Arguments', hyperparameters)


adults = Configs(
    n_classes=2,
    iforest_kwargs=dict(
        n_estimators=100, 
        max_samples='auto',
        n_jobs=30
    ),
    pca_dim=None,
    repeat=10
)

diabetes130 = Configs(
    n_classes=2,
     iforest_kwargs=dict(
        n_estimators=100, 
        max_samples='auto',
        n_jobs=30
    ),
    pca_dim=None,
    repeat=10
)