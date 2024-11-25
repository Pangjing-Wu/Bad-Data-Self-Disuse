import argparse
import datetime
import sys
import traceback

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append('.')
from configs.metadata import aum as configs
from models.resnet import ResNet18, ResNet50
from utils.logger import set_logger
from utils.io.dataset import load_cv_dataset
from utils.nn import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Test Baseline Performance')
    parser.add_argument('--dataset', required=True, help='dataset name')
    parser.add_argument('--model', default='resnet18', help='model name')
    parser.add_argument('--bs', type=int, default=512, help='batch size')
    parser.add_argument('--epoch', default=100, help='epoch number')
    parser.add_argument('--lr', default=1e-2, help='learning rate')
    parser.add_argument('--min_lr', default=1e-4, help='minimum learning rate')
    parser.add_argument('--cuda', type=int, default=None, help='cuda device id (default: None)')
    return parser.parse_args()


def main(args, config, logger):
    set_seed(0)
    # log parameters.
    logger.info(f'====== Test Baseline Performance ======')
    logger.info('\n'.join([f'{k} = {str(v)}' for k, v in sorted(vars(args).items())]))
    
    # set dataset.
    trainset = load_cv_dataset(args.dataset, train=True, augment=True)
    testset  = load_cv_dataset(args.dataset, train=False, augment=False)
    logger.info(f'load train dataset {args.dataset} with {len(trainset)} samples.')
    
    # set model
    if args.model == 'resnet18':
        model_cls = ResNet18
    elif args.model == 'resnet50':
        model_cls = ResNet50
    else:
        raise ValueError(f"unknown model '{args.model}'.")
    
    device = torch.device(f'cuda:{args.cuda}') if isinstance(args.cuda, int) else torch.device('cpu')
    logger.info(f"criterion = CrossEntropyLoss; optimizer = Adam; use scaler = True.")
    logger.info(f"use learning rate schedule: CosineAnnealingLR(T_max={args.epoch}).")
    
    trainloader = DataLoader(dataset=trainset, batch_size=args.bs, shuffle=True, num_workers=0)
    testloader  = DataLoader(dataset=testset, batch_size=args.bs, shuffle=False, num_workers=0)
    
    # reset model.
    model     = model_cls(config.n_classes, **config.model_kwargs).to(device)
    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler    = torch.cuda.amp.GradScaler()
    schedule  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=args.min_lr)
    
    for e in range(1, args.epoch + 1):
        model.train()
        batch_loss, batch_acc = 0, 0
        for x, y in trainloader:
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
        schedule.step()
        batch_loss /= len(trainloader)
        batch_acc  /= len(trainloader) / 100
        logger.info(f"Epoch {e:3}/{args.epoch}: train loss:{batch_loss:.5f}, train acc: {batch_acc:.3f}, lr: {lr:.3e}.")
        
        model.eval()
        test_loss, test_acc = 0, 0
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(device), y.to(device)
                logistic = model(x)
                loss = criterion(logistic, y)
                test_loss += loss.item()
                test_acc  += torch.argmax(logistic, dim=1).eq(y).sum().item() / y.size(0)
        test_loss /= len(testloader)
        test_acc  /= len(testloader) / 100
        logger.info(f"Epoch {e:3}/{args.epoch}: test loss: {test_loss:.5f}, test acc: {test_acc:.3f}%")
        

# python -u scripts/test/evaluation/scoring.py --dataset cifar10 --score_file xx --score_col 'score-norm' --top_n 45000 --cuda 0
if __name__ == '__main__':
    args   = parse_args()
    config  = getattr(configs, args.dataset)
    date   = datetime.datetime.now().strftime('%y%m%d-%H%M')
    logger = set_logger(name=f'baseline-{date}')
    try:
        main(args, config, logger)
    except Exception:
        logger.error(traceback.format_exc())