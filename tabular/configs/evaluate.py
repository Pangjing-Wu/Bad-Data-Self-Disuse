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
    'graph_ssl_kwargs',
    'seed',
    'warmup_num'
]


Configs = namedtuple('Evaluation_Argument', hyperparameters)


adults = Configs(
    dataset='adults',
    n_shard=10,
    metadata='./results/metadata/adults/metadata.csv',
    eval_mode='same',
    n_eval_per_iter=200,
    max_iter=1000,
    n_jobs=3,
    budget=0.8,
    significance=0.1,
    graph_ssl_kwargs=dict(kernel='knn', n_neighbors=50, alpha=0.01, max_iter=100),
    seed=0,
    warmup_num=500
)

diabetes130 = Configs(
    dataset='diabetes130',
    n_shard=10,
    metadata='./results/metadata/diabetes130/metadata.csv',
    eval_mode='same',
    n_eval_per_iter=200,
    max_iter=1000,
    n_jobs=3,
    budget=0.8,
    significance=0.1,
    graph_ssl_kwargs=dict(kernel='knn', n_neighbors=50, alpha=0.01, max_iter=100),
    seed=0,
    warmup_num=500
)