from collections import namedtuple


hyperparameters = [
    'n_classes',
    'repeat',
]

Configs = namedtuple('Centroid_Arguments', hyperparameters)


adults = Configs(
    n_classes=2,
    repeat=10
)


diabetes130 = Configs(
    n_classes=2,
    repeat=10
)