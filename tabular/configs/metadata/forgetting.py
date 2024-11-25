from collections import namedtuple


hyperparameters = [
    'n_classes',
    'model_kwargs',
    'bs',
    'epoch',
    'lr',
    'min_lr',
    'repeat',
]

Configs = namedtuple('Forgetting_Arguments', hyperparameters)


adults = Configs(
    n_classes=2,
    model_kwargs=dict(
        input_dim=14,
        embedding_dim=24,
        n_blocks=1, 
        d_block=256, 
        dropout=0.1
        ),
    bs=2048,
    epoch=20,
    lr=1e-2,
    min_lr=1e-3,
    repeat=10
)

diabetes130 = Configs(
    n_classes=2,
    model_kwargs=dict(
        input_dim=47,
        embedding_dim=4,
        n_blocks=1, 
        d_block=256, 
        dropout=0.1
        ),
    bs=2048,
    epoch=20,
    lr=1e-2,
    min_lr=1e-3,
    repeat=10
)