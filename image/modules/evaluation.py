# NOTE: do not change anything in this file when you code are running!!!
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .shard import DataShard


def train(model, dataloader, optimizer, criterion, device, schedule=None):
    scaler = torch.cuda.amp.GradScaler()
    batch_loss, batch_acc = 0, 0
    model.train()
    for x, y in dataloader:
        x, y = x.to(device), y.to(device) 
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logistic = model(x)
            loss = criterion(logistic, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_loss += loss.item()
        batch_acc  += torch.argmax(logistic, dim=1).eq(y).sum().item() / y.size(0)
    if schedule: schedule.step()


def benchmark_evaluation_nn(
    shard_index: int,
    shards: DataShard, 
    mode: str,
    model_fn: nn.Module, 
    epoch: int, 
    lr: float, 
    bs: int, 
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module, 
    device: str, 
    ema_coef: float,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> Tuple[int, dict]:
    # set model and dataset.
    model = model_fn().to(device)
    optimizer_  = optimizer(model.parameters(), lr=lr)
    scheduler_  = scheduler(optimizer_) if scheduler else None
    trainloader = DataLoader(shards.exclude(shard_index), batch_size=bs, shuffle=True)
    validloader = DataLoader(shards[shard_index], batch_size=bs, shuffle=False)
    labels  = torch.cat([y for _, y in validloader]).to(device)
    classes = labels.unique(sorted=True).cpu().numpy()
    # set initial loss.
    if mode == 'all':
        loss = 0
    else:
        loss = np.zeros(len(classes))
    # evaluate performance.
    for _ in range(epoch):
        train(model=model, dataloader=trainloader, optimizer=optimizer_, criterion=criterion, schedule=scheduler_, device=device)
        # test model.
        model.eval()
        logistic = list()
        with torch.no_grad():
            for x, _ in validloader:
                x = x.to(device)
                logistic.append(F.softmax(model(x), dim=-1))
            logistic = torch.cat(logistic)
            if mode == 'all':
                loss_ = criterion(logistic, labels).item()
                loss = ema_coef * loss_ + (1 - ema_coef) * loss
            elif mode == 'same':
                for i in classes:
                    loss_ = criterion(logistic[labels == i], labels[labels == i]).item()
                    loss[i] = ema_coef * loss_ + (1 - ema_coef) * loss[i]
            # calculate loss of different class.
            elif mode == 'diff':
                for i in classes:
                    loss_ = criterion(logistic[labels != i], labels[labels != i]).item()
                    loss[i] = ema_coef * loss_ + (1 - ema_coef) * loss[i]
            else:
                raise ValueError
    return (shard_index, loss)


def sample_evaluation_nn(
    eval_index: int,
    shards: DataShard, 
    mode: str,
    model_fn: nn.Module, 
    epoch: int, 
    lr: float, 
    bs: int, 
    optimizer: torch.optim.Optimizer, 
    criterion: nn.Module, 
    device: str, 
    ema_coef: float,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> Tuple[int, dict]:
    # obtain dataset and sample feature.
    sample_label = shards.dataset[eval_index][-1]
    sample_shard = shards.get_shard_by_sample_index(eval_index)
    # set initial loss value.
    loss  = list()
    # k-fold validation.
    for i in range(shards.k):
        if i == sample_shard:
            continue
        loss.append(0)
        # set model and dataset.
        model = model_fn().to(device)
        optimizer_  = optimizer(model.parameters(), lr=lr)
        scheduler_  = scheduler(optimizer_) if scheduler else None
        trainloader = DataLoader(shards.exclude(i).remove(eval_index), batch_size=bs, shuffle=True)
        validloader = DataLoader(shards[i], batch_size=bs, shuffle=False)
        labels = torch.cat([y for _, y in validloader]).to(device)
        # evaluate performance.
        for _ in range(epoch):
            train(model=model, dataloader=trainloader, optimizer=optimizer_, criterion=criterion, schedule=scheduler_, device=device)
            # test model.
            model.eval()
            logistic = list()
            for x, _ in validloader:
                with torch.no_grad():
                    x = x.to(device)
                    logistic.append(F.softmax(model(x), dim=-1))
            logistic = torch.cat(logistic)
                
            # calculate total loss.
            if mode == 'all':
                loss_ = criterion(logistic, labels).item()
            elif mode == 'same':
                loss_ = criterion(logistic[labels == sample_label], labels[labels == sample_label]).item()
            # calculate loss of different class.
            elif mode == 'diff':
                loss_ = criterion(logistic[labels != sample_label], labels[labels != sample_label]).item()
            else:
                raise ValueError
            loss[-1] = ema_coef * loss_ + (1 - ema_coef) * loss[-1]
    return (eval_index, loss)