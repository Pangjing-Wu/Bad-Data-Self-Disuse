from collections import Counter, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._utils import get_device


__all__ = ['train', 'test', 'test_each_class']


train_ret = namedtuple('train_result', ['loss', 'acc', 'lr'])
test_ret  = namedtuple('test_result', ['loss', 'acc'])


scaler       = torch.cuda.amp.GradScaler()
default_crit = nn.CrossEntropyLoss()


def train(model, dataloader, optimizer, criterion=None, schedule=None):
    device    = get_device(model)
    criterion = criterion if criterion else default_crit
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
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    if schedule: schedule.step()
    batch_loss /= len(dataloader)
    batch_acc  /= len(dataloader) / 100
    return train_ret(batch_loss, batch_acc, lr)


def test(model, dataloader, criterion=None):
    device = get_device(model)
    criterion = criterion if criterion else default_crit
    batch_loss, batch_acc = 0, 0
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logistic = F.softmax(model(x), dim=-1)
            loss = criterion(logistic, y)
            batch_loss += loss.item()
            batch_acc  += torch.argmax(logistic, dim=1).eq(y).sum().item() / y.size(0)
        batch_loss /= len(dataloader)
        batch_acc  /= len(dataloader) / 100
    return test_ret(batch_loss, batch_acc)


def test_each_class(model, dataloader):
    device = get_device(model)
    correct_pred = Counter()
    total_pred   = Counter()
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = F.softmax(model(x), dim=-1)
            y_hat  = torch.argmax(output, dim=1)
            for y_, y_hat_ in zip(y, y_hat):
                if y_ == y_hat_:
                    correct_pred[dataloader.classes[y_.cpu().item()]] += 1
                total_pred[dataloader.classes[y_.cpu().item()]] += 1
    ret = dict()
    for class_name, correct_count in correct_pred.items():
        ret[class_name] = round(100 * correct_count / total_pred[class_name], 3)
    return ret