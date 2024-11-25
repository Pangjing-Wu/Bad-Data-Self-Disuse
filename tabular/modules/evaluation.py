# NOTE: do not change anything in this file when you code are running!!!
from typing import Tuple

import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import accuracy_score

from .shard import DataShard



def benchmark_evaluation(
    shard_index: int,
    shards: DataShard, 
    mode: str
    ) -> Tuple[int, dict]:
    # set model and dataset.
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
    

    train_X, train_y = zip(*shards.exclude(shard_index))
    valid_X, valid_y = zip(*shards[shard_index])
    train_X, train_y = torch.stack(train_X, dim=0), torch.stack(train_y, dim=-1)
    valid_X, valid_y = torch.stack(valid_X, dim=0), torch.stack(valid_y, dim=-1)
    classes = valid_y.unique(sorted=True).cpu().numpy()
    
    # set initial utility.
    if mode == 'all':
        acc = 0
    else:
        acc = np.zeros(len(classes))
        
    # evaluate performance.
    train_X, train_y = train_X.numpy(), train_y.numpy()
    valid_X, valid_y = valid_X.numpy(), valid_y.numpy()
    model.fit(train_X, train_y)
    y_pred = model.predict(valid_X)
    if mode == 'all':
        acc = accuracy_score(valid_y, y_pred)
    elif mode == 'same':
        for i in classes:
            acc[i] = accuracy_score(valid_y[valid_y==i], y_pred[valid_y==i])
    elif mode == 'diff':
        for i in classes:
            acc[i] = accuracy_score(valid_y[valid_y!=i], y_pred[valid_y!=i])
    return (shard_index, acc)


def sample_evaluation(
    eval_index: int,
    shards: DataShard, 
    mode: str,
    ) -> Tuple[int, dict]:
    # obtain dataset and sample feature.
    sample_label = shards.dataset[eval_index][-1].item()
    sample_shard = shards.get_shard_by_sample_index(eval_index)
    # set initial loss value.
    acc  = list()
    # k-fold validation.
    print(f'{sample_label = }')
    for i in range(shards.k):
        if i == sample_shard:
            continue
        acc.append(0)
        # set model and dataset.
        train_X, train_y = zip(*shards.exclude(i).remove(eval_index))
        valid_X, valid_y = zip(*shards[i])
        train_X, train_y = torch.stack(train_X, dim=0), torch.stack(train_y, dim=-1)
        valid_X, valid_y = torch.stack(valid_X, dim=0), torch.stack(valid_y, dim=-1)
        train_X, train_y = train_X.numpy(), train_y.numpy()
        valid_X, valid_y = valid_X.numpy(), valid_y.numpy()
        model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
        model.fit(train_X, train_y)
        y_pred = model.predict(valid_X)
        if mode == 'all':
            acc[-1] = accuracy_score(valid_y, y_pred)
        elif mode == 'same':
            acc[-1] = accuracy_score(valid_y[valid_y==sample_label], y_pred[valid_y==sample_label])
        elif mode == 'diff':
            acc[-1] = accuracy_score(valid_y[valid_y!=sample_label], y_pred[valid_y!=sample_label])
        else:
            raise ValueError
    return (eval_index, acc)