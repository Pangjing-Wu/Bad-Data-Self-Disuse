from collections import namedtuple


hyperparameters = [
    'n_classes',
    'k',
    'dist',
    'pca_dim',
    'embed_method',
    'embed_epoch',
    'embed_date',
    'backbone',
    'backbone_kwargs',
    'bs',
    'repeat',
]

Configs = namedtuple('Consistency_Arguments', hyperparameters)


cifar10 = Configs(
    n_classes=10,
    k=50,
    dist='euclidean',
    pca_dim=64,
    embed_method='mixed-bt',
    embed_epoch=None,
    embed_date=None,
    backbone='resnet50',
    backbone_kwargs=dict(
        out_channel=64,
        kernel=3,
        stride=1,
        padding=1,
        maxpool=False
    ),
    bs=1000,
    repeat=10
)

cifar100 = Configs(
    n_classes=100,
    k=50,
    dist='euclidean',
    pca_dim=128,
    embed_method='mixed-bt',
    embed_epoch=None,
    embed_date=None,
    backbone='resnet50',
    backbone_kwargs=dict(
        out_channel=64,
        kernel=3,
        stride=1,
        padding=1,
        maxpool=False
    ),
    bs=1000,
    repeat=10
)

caltech101 = Configs(
    n_classes=101,
    k=50,
    dist='euclidean',
    pca_dim=128,
    embed_method='pre-trained',
    embed_epoch=None,
    embed_date=None,
    backbone='imagenet1k-resnet50',
    backbone_kwargs=dict(
        out_channel=64,
        kernel=7,
        stride=2,
        padding=2,
        maxpool=True
    ),
    bs=1000,
    repeat=10
)

caltech256 = Configs(
    n_classes=257,
    k=50,
    dist='euclidean',
    pca_dim=128,
    embed_method='pre-trained',
    embed_epoch=None,
    embed_date=None,
    backbone='imagenet1k-resnet50',
    backbone_kwargs=dict(
        out_channel=64,
        kernel=7,
        stride=2,
        padding=2,
        maxpool=True
    ),
    bs=1000,
    repeat=10
)