# NOTE: do not change anything in this file when you code are running!!!
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import KFoldDataset
from .nn import set_seed
from .nn.classifier import train


def baseline_evaluation(
    val_index: int, 
    dataset: KFoldDataset, 
    model_fn: nn.Module, 
    epoch: int, 
    lr: float, 
    bs: int, 
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module, 
    device: str, 
    alpha: float,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> Tuple[int, dict]:
    """
    Returns:
        loss: list. shape = [each class loss + total loss]
    """
    
    # set seed.
    set_seed(val_index)
    
    # set model and dataset.
    model = model_fn().to(device)
    optimizer_  = optimizer(model.parameters(), lr=lr)
    scheduler_  = scheduler(optimizer_) if scheduler else None
    trainloader = DataLoader(dataset.exclude(val_index), batch_size=bs, shuffle=True)
    validloader = DataLoader(dataset[val_index], batch_size=bs, shuffle=False)
    labels  = torch.cat([y for _, y in validloader]).to(device)
    classes = labels.unique(sorted=True).cpu().numpy()
    
    # set initial value.
    all_loss  = 0
    same_loss = np.zeros(len(classes))
    diff_loss = np.zeros(len(classes))
    
    # evaluate performance.
    for _ in range(epoch):
        train(model=model, dataloader=trainloader, optimizer=optimizer_, 
              criterion=criterion, schedule=scheduler_)
        # test model.
        model.eval()
        logistic = list()
        with torch.no_grad():
            for x, _ in validloader:
                x = x.to(device)
                logistic.append(F.softmax(model(x), dim=-1))
            logistic = torch.cat(logistic)
            # calculate total loss.
            loss_    = criterion(logistic, labels).item()
            all_loss = alpha * loss_ + (1 - alpha) * all_loss
            # calculate loss of each class.
            for i in classes:
                loss_ = criterion(logistic[labels == i], labels[labels == i]).item()
                same_loss[i] = alpha * loss_ + (1 - alpha) * same_loss[i]
            # calculate loss of different class.
            for i in classes:
                loss_ = criterion(logistic[labels != i], labels[labels != i]).item()
                diff_loss[i] = alpha * loss_ + (1 - alpha) * diff_loss[i]    
    return (val_index, dict(all=all_loss, same=same_loss, diff=diff_loss))


def sample_evaluation(
    sample_index: int, 
    dataset: KFoldDataset, 
    model_fn: nn.Module, 
    epoch: int, 
    lr: float, 
    bs: int, 
    optimizer: torch.optim.Optimizer, 
    criterion: nn.Module, 
    device: str, 
    alpha: float,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> Tuple[int, dict]:
    """
    Returns:
        ret (bool): `True` indicates bad data, and `False` indicates good data.
        loss (list): shape = [('fold-{i}-all', 'fold-{i}-same', 'fold-{i}-diff') * k-fold]
    """
    
    # set seed.
    set_seed(sample_index)
    
    # obtain dataset and sample feature.
    sample_label = dataset.dataset[sample_index][-1]
    sample_fold  = dataset.get_fold_by_sample_index(sample_index)
    
    # set initial value.
    all_loss  = np.zeros(dataset.k)
    same_loss = np.zeros(dataset.k)
    diff_loss = np.zeros(dataset.k)
    
    # k-fold validation.
    for i in range(dataset.k):
        if i == sample_fold:
            all_loss[i]  = -99.
            same_loss[i] = -99.
            diff_loss[i] = -99.
            continue
        
        # set model and dataset.
        model = model_fn().to(device)
        optimizer_  = optimizer(model.parameters(), lr=lr)
        scheduler_  = scheduler(optimizer_) if scheduler else None
        trainloader = DataLoader(dataset.exclude(i).remove(sample_index), batch_size=bs, shuffle=True)
        validloader = DataLoader(dataset[i], batch_size=bs, shuffle=False)
        labels = torch.cat([y for _, y in validloader]).to(device)

        # evaluate performance.
        for _ in range(epoch):
            train(model=model, dataloader=trainloader, optimizer=optimizer_, 
                  criterion=criterion, schedule=scheduler_)
            # test model.
            model.eval()
            logistic = list()
            with torch.no_grad():
                for x, _ in validloader:
                    x = x.to(device)
                    logistic.append(F.softmax(model(x), dim=-1))
                logistic = torch.cat(logistic)
                # calculate total loss.
                loss_ = criterion(logistic, labels).item()
                all_loss[i] = alpha * loss_ + (1 - alpha) * all_loss[i]
                # calculate loss on samples of the same class.
                loss_ = criterion(logistic[labels == sample_label], labels[labels == sample_label]).item()
                same_loss[i] = alpha * loss_ + (1 - alpha) * same_loss[i]
                # calculate loss on samples of the different class.
                loss_ = criterion(logistic[labels != sample_label], labels[labels != sample_label]).item()
                diff_loss[i] = alpha * loss_ + (1 - alpha) * diff_loss[i]

    return (sample_index, dict(all=all_loss, same=same_loss, diff=diff_loss))