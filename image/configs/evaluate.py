from collections import namedtuple


hyperparameters = [
    'dataset',
    'n_shard',
    'metadata',
    'eval_mode',
    'n_eval_per_iter',
    'max_iter',
    'n_jobs',
    'budget',
    'significance',
    'net',
    'net_kwargs',
    'train_kwargs',
    'graph_ssl_kwargs',
    'seed',
    'ema_coef',
    'warmup_num'
]


Configs = namedtuple('Evaluation_Argument', hyperparameters)


cifar10 = Configs(
    dataset='cifar10',
    n_shard=7,
    metadata='./results/metadata/cifar10/metadata.csv',
    eval_mode='same',
    n_eval_per_iter=200,
    max_iter=1000,
    n_jobs=10,
    budget=0.8,
    significance=0.1,
    net='mobilenetv3',
    net_kwargs=dict(n_classes=10, stride=1),
    train_kwargs=dict(epoch=5, lr=1e-2, bs=512, min_lr=5e-2),
    graph_ssl_kwargs=dict(kernel='knn', n_neighbors=50, alpha=0.01, max_iter=100),
    seed=0,
    ema_coef=0.9,
    warmup_num=500
)

cifar100 = Configs(
    dataset='cifar100',
    n_shard=7,
    metadata='./results/metadata/cifar100/metadata.csv',
    eval_mode='same',
    n_eval_per_iter=200,
    max_iter=1000,
    n_jobs=10,
    budget=0.8,
    significance=0.1,
    net='mobilenetv3',
    net_kwargs=dict(n_classes=100, stride=1),
    train_kwargs=dict(epoch=5, lr=1e-2, bs=512, min_lr=5e-2),
    graph_ssl_kwargs=dict(kernel='knn', n_neighbors=50, alpha=0.01, max_iter=100),
    seed=0,
    ema_coef=0.9,
    warmup_num=500
)

caltech101 = Configs(
    dataset='caltech101',
    n_shard=7,
    metadata='./results/metadata/caltech101/metadata.csv',
    eval_mode='same',
    n_eval_per_iter=200,
    max_iter=1000,
    n_jobs=10,
    budget=0.8,
    significance=0.1,
    net='mobilenetv3',
    net_kwargs=dict(n_classes=101),
    train_kwargs=dict(epoch=5, lr=1e-2, bs=512, min_lr=5e-2),
    graph_ssl_kwargs=dict(kernel='knn', n_neighbors=50, alpha=0.01, max_iter=100),
    seed=0,
    ema_coef=0.9,
    warmup_num=500
)

caltech256 = Configs(
    dataset='caltech256',
    n_shard=7,
    metadata='./results/metadata/caltech256/metadata.csv',
    eval_mode='same',
    n_eval_per_iter=200,
    max_iter=1000,
    n_jobs=10,
    budget=0.8,
    significance=0.1,
    net='mobilenetv3',
    net_kwargs=dict(n_classes=256),
    train_kwargs=dict(epoch=5, lr=1e-2, bs=512, min_lr=5e-2),
    graph_ssl_kwargs=dict(kernel='knn', n_neighbors=50, alpha=0.01, max_iter=100),
    seed=0,
    ema_coef=0.9,
    warmup_num=500
)