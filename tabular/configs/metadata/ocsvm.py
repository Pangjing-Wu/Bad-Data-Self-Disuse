from collections import namedtuple


hyperparameters = [
    'n_classes',
    'ocsvm_kwargs',
    'pca_dim',
    'repeat',
]

Configs = namedtuple('OCSVM_Arguments', hyperparameters)


adults = Configs(
    n_classes=2,
    ocsvm_kwargs=dict(kernel='linear'),
    pca_dim=None,
    repeat=10
)

diabetes130 = Configs(
    n_classes=2,
    ocsvm_kwargs=dict(kernel='linear'),
    pca_dim=None,
    repeat=10
)