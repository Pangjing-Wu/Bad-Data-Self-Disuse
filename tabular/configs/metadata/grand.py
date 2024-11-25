from collections import namedtuple


hyperparameters = [
    'n_classes',
    'order',
    'model_kwargs',
    'bs',
    'epoch',
    'lr',
    'min_lr',
    'repeat',
    'fc_name'
]

Configs = namedtuple('GraNd_Arguments', hyperparameters)


adults = Configs(
    n_classes=2,
    order=2,
    model_kwargs=dict(
        input_dim=14,
        embedding_dim=4,
        n_blocks=1, 
        d_block=256, 
        dropout=0.1
        ),
    bs=512,
    epoch=10,
    lr=1e-2,
    min_lr=1e-3,
    repeat=10,
    fc_name='fc'
)

diabetes130 = Configs(
    n_classes=2,
    order=2,
    model_kwargs=dict(
        input_dim=47,
        embedding_dim=4,
        n_blocks=1, 
        d_block=256, 
        dropout=0.1
        ),
    bs=512,
    epoch=0,
    lr=1e-2,
    min_lr=1e-3,
    repeat=10,
    fc_name='fc'
)