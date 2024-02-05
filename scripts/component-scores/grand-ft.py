import argparse
import datetime
import os
import sys
import traceback
from functools import partial

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from transformers import logging

sys.path.append('.')
import configs
from component_scores import GraNd
from utils.dataset import load_cv_dataset
from utils.logger import set_logger
from utils.nn import set_seed
from utils.nn.io import load_embedding
from utils.path import set_component_path, set_component_log_path


logging.set_verbosity_error()
SCORE = 'grand'


def parse_args():
    parser = argparse.ArgumentParser(description='GraNd Evaluation (Finetune Version)')
    parser.add_argument('--dataset', required=True, help='dataset name {cifar10/cifar100/Caltech101/Caltech256}')
    parser.add_argument('--embed_net', required=True, type=str, help='DNN model name {vit/mae/beit}')
    parser.add_argument('--embed_dataset', required=True, type=str, help='pretrain dataset name {imagenet21k/}')
    parser.add_argument('--embed_epoch', type=int, default=-1, help='embedding epoch (default: -1)')
    parser.add_argument('--embed_date', required=True, type=str, help='date of embedding model')
    parser.add_argument('--epoch', type=int, nargs='+', default=[50], help='training epoch (default: [50])')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate (default: 0.001)')
    parser.add_argument('--bs', type=int, default=128, help='training batch size (default: 128)')
    parser.add_argument('--order', type=int, default=2, help='the order of norm (default: 2)')
    parser.add_argument('--gamma', type=float, default=0.95, help='learning schedule parameter')
    parser.add_argument('--fc_name', type=str, default='fc', help="name of full connected layer (default: 'fc')")
    parser.add_argument('--n_jobs', type=int, default=0, help='n workers for dataloader (default: 0)')
    parser.add_argument('--seed', type=int, nargs='+', default=[0], help='random seed (default: [0])')
    parser.add_argument('--cuda', type=int, default=None, help='cuda device id (default: None)')
    parser.add_argument('--debug', action='store_true', help='running without saving model')
    return parser.parse_args()


class Linear(nn.Module):
    
    def __init__(self, in_features, out_features, *args, **kawrgs) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, *args, **kawrgs)
        self.in_features  = in_features
        self.out_features = out_features
        
    def forward(self, x):
        return self.fc(x)
        

def main(args, logger, filepaths):
    # log parameters.
    logger.info(f'====== Calculate GraNd Score (Finetune Version) ======')
    logger.info('\n'.join([f'{k} = {str(v)}' for k, v in sorted(dict(vars(args)).items())]))
    
    # set forgetting event scorer and invariant objects.
    param   = getattr(configs, f'{args.embed_net}_{args.embed_dataset}_params')
    dataset = load_cv_dataset(args.dataset, train=True, augment=False, resize=param['input_size'])
    if args.embed_net in ['vit', 'mae', 'beit']:
        model_fn = partial(Linear, out_features=len(dataset.classes))
    else:
        raise ValueError(f"unknown model '{args.embed_net}'.")
    device  = torch.device(f'cuda:{args.cuda}') if isinstance(args.cuda, int) else torch.device('cpu')
    logger.info(f'load dataset {args.dataset} with batch size {args.bs}.')
    logger.info('data transform:\n' + '\n'.join([f'==> {str(t)}' for t in dataset.transform.transforms]))
    logger.info(f"use device: `{device}`.")
    logger.info(f"finetune model `nn.Linear()`.")
    logger.info(f"criterion = CrossEntropyLoss; optimizer = Adam; use scaler = True.")
    logger.info(f"use learning rate schedule: StepLR(T_max={args.gamma}).")
    logger.info(f"going to save results at epochs: {', '.join(map(str, args.epoch))}.")
    
    # load pre-train model.
    logger.info(f"load pre-train embedding model `{args.embed_net}`.")
    embed = load_embedding(
        model=args.embed_net, 
        dataset=args.embed_dataset, 
        algo='pt',
        date=args.embed_date, 
        epoch=args.embed_epoch, 
        param=param
        ).to(device)
    
    # embedding raw data.
    logger.info(f"calculate data embedding based on pre-trained `{args.embed_net}`.")
    dataloader = DataLoader(dataset=dataset, batch_size=args.bs, shuffle=False, num_workers=args.n_jobs)
    with torch.no_grad():
        x = torch.cat([embed(x.to(device)).cpu() for x, _ in dataloader], dim=0)
        y = torch.LongTensor(dataset.targets) if hasattr(dataset, 'targets') else torch.cat([y for _, y in dataloader], dim=0).long()
    dataset = TensorDataset(x, y)
    logger.info(f"complete calculation of data embedding based on pre-trained `{args.embed_net}`.")
    
    rets = {f'epoch:{e}':{} for e in args.epoch}
    evalloader = DataLoader(dataset=dataset, batch_size=args.bs, shuffle=False, num_workers=args.n_jobs)
    
    for r, seed in enumerate(args.seed):
        logger.info(f'start {r+1}/{len(args.seed)} round score calculation with random seed {seed}.')
        set_seed(seed)
        # reset data loader.
        trainloader = DataLoader(
            dataset=dataset, 
            batch_size=args.bs, 
            shuffle=True, 
            num_workers=args.n_jobs
            )
        # reset model.
        in_ftrs   = dataset[0][0].size(0)
        model     = model_fn(in_features=in_ftrs).to(device)
        criterion = nn.CrossEntropyLoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scaler    = torch.cuda.amp.GradScaler()
        schedule  = torch.optim.lr_scheduler.StepLR(optimizer, 1, args.gamma)
        # reset scorer.
        scorer = GraNd(model=model, n_classes=model.out_features, order=args.order, optim=torch.optim.Adam)
        
        # training model & calculate component scores.
        if 0 in args.epoch:
            rets[f'epoch:0'][f'round:{r}'] = scorer.scores(dataloader=evalloader)
        for e in range(1, max(args.epoch) + 1):
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
            logger.info(f"Epoch {e:3}/{max(args.epoch)}: train loss:{batch_loss:.5f}, train acc: {batch_acc:.3f}, lr: {lr:.3e}.")
            if e in args.epoch:
                rets[f'epoch:{e}'][f'round:{r}'] = scorer.scores(dataloader=evalloader)
        logger.info(f'obtain scores of {r+1}/{len(args.seed)} round calculation of all different {len(args.epoch)} epochs.')
        
        
    # summarize results.
    for e, filepath in zip(args.epoch, filepaths):
        ret = pd.DataFrame(rets[f'epoch:{e}'])
        ret['scores'] = ret.mean(axis=1)
        logger.info(f'summarization of score of epoch {e}:\n{ret.describe()}')
        if not args.debug:
            logger.info(f"save file to '{filepath}'")
            ret.to_csv(filepath, index=False)
        logger.debug(f'results of score of epoch {e}:\n{ret}')
        logger.debug(f"save file to '{filepath}'")
    logger.info('done!')


# python -u ./scripts/component-scores/grand-ft.py --dataset cifar10 --embed_net vit --embed_dataset imagenet21k --embed_date '666666-6666' --epoch 0 1 --lr 1e-2 --bs 128 --seed 0 1 --cuda 0 --debug
if __name__ == '__main__':
    args       = parse_args()
    args.epoch = sorted(args.epoch)
    date       = datetime.datetime.now().strftime('%y%m%d-%H%M')
    logpath    = set_component_log_path(score=SCORE, dataset=args.dataset, model=args.embed_net, date=date)
    filepaths  = [set_component_path(score=SCORE, dataset=args.dataset, model=args.embed_net, date=date, epoch=e) for e in args.epoch]
    logger     = set_logger(name=SCORE, level='DEBUG') if args.debug else set_logger(name=SCORE, logfile=logpath)
    logger.debug(f"save log to: '{logpath}'.")
    try:
        main(args, logger, filepaths)
    except Exception:
        for filepath in filepaths:
            if os.path.exists(filepath):
                logger.warning(f"catch exceptions, delete incomplete results.")
                os.remove(filepath)
        logger.error(traceback.format_exc())
    except KeyboardInterrupt:
        for filepath in filepaths:
            if os.path.exists(filepath):
                os.remove(filepath)
        if os.path.exists(logpath):
            os.remove(logpath)