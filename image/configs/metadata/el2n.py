from collections import namedtuple


hyperparameters = [
    'n_classes',
    'order',
    'model',
    'model_kwargs',
    'bs',
    'epoch',
    'lr',
    'min_lr',
    'repeat',
]

Configs = namedtuple('Forgetting_Arguments', hyperparameters)


cifar10 = Configs(
    n_classes=10,
    order=2,
    model='resnet18',
    model_kwargs=dict(
        out_channel=64,
        kernel=3,
        stride=1,
        padding=1,
        maxpool=False
        ),
    bs=512,
    epoch=10,
    lr=1e-2,
    min_lr=1e-3,
    repeat=10
)

cifar100 = Configs(
    n_classes=100,
    order=2,
    model='resnet18',
    model_kwargs=dict(
        out_channel=64,
        kernel=3,
        stride=1,
        padding=1,
        maxpool=False
        ),
    bs=512,
    epoch=10,
    lr=1e-2,
    min_lr=1e-3,
    repeat=10
)

caltech101 = Configs(
    n_classes=101,
    order=2,
    model='resnet18',
    model_kwargs=dict(
        out_channel=64,
        kernel=7,
        stride=2,
        padding=2,
        maxpool=True
        ),
    bs=128,
    epoch=10,
    lr=1e-2,
    min_lr=1e-3,
    repeat=10
)

caltech256 = Configs(
    n_classes=257,
    order=2,
    model='resnet18',
    model_kwargs=dict(
        out_channel=64,
        kernel=7,
        stride=2,
        padding=2,
        maxpool=True
        ),
    bs=128,
    epoch=10,
    lr=1e-2,
    min_lr=1e-3,
    repeat=10
)