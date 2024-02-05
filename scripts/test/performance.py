import argparse
import datetime
import os
import sys
import traceback
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.append('.')
import configs
from models.resnet import *
from utils.dataset import load_cv_dataset
from utils.logger import set_logger
from utils.nn import set_seed
from utils.nn.classifier import train, test
from utils.path import set_eval_path, set_performance_path, set_performance_log_path


def parse_args():
    parser = argparse.ArgumentParser(description='Supervised Model Embedding')
    parser.add_argument('--dataset', required=True, help='dataset name {cifar10/cifar100}')
    parser.add_argument('--date', type=str, default=None, help='evaluation results file date')
    parser.add_argument('--net', required=True, type=str, help='DNN model name {resnet18/resnet50}')
    parser.add_argument('--epoch', type=int, default=50, help='training epoch (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate (default: 0.001)')
    parser.add_argument('--bs', type=int, default=128, help='training batch size (default: 128)')
    parser.add_argument('--n_jobs', type=int, default=0, help='n workers for dataloader (default: 0)')
    parser.add_argument('--seed', type=int, nargs='+', default=[0], help='random seed (default: [0])')
    parser.add_argument('--augment', action='store_true', help='augment training data')
    parser.add_argument('--cuda', type=int, default=None, help='cuda device id (default: None)')
    parser.add_argument('--debug', action='store_true', help='running without saving model')
    return parser.parse_args()


def main(args, logger, filepath):
    # log parameters.
    logger.info(f'====== Test All/Good/Bad/Random Data Performance ======')
    logger.info('\n'.join([f'{k} = {str(v)}' for k, v in sorted(dict(vars(args)).items())]))
    
    # load dataset.
    trainset   = load_cv_dataset(args.dataset, train=True, augment=args.augment)
    testset    = load_cv_dataset(args.dataset, train=False, augment=False)
    testloader = DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=args.n_jobs)
    logger.info(f'load dataset {args.dataset} with batch size {args.bs}.') 
    logger.info(f'trainloader shuffle = True, testloader shuffle = False.')
    logger.info('training transform:\n' + '\n'.join([f'==> {str(t)}' for t in trainset.transform.transforms]))
    logger.info('test transform:\n' + '\n'.join([f'==> {str(t)}' for t in testset.transform.transforms]))

    # set model.
    param = getattr(configs, f'{args.net}_{args.dataset}_params')
    if args.net == 'resnet18':
        model = partial(resnet18, **param)
    elif args.net == 'resnet34':
        model = partial(resnet34, **param)
    elif args.net == 'resnet50':
        model = partial(resnet50, **param)
    else:
        raise ValueError(f"unknown model '{args.net}'.")
    logger.info(f"set model {args.net} with {param['n_channel']} input channel and {param['n_classes']} output classes.")
    
    # set scaler and model.
    device    = torch.device(f'cuda:{args.cuda}') if isinstance(args.cuda, int) else torch.device('cpu')
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam
    scheduler = partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=args.epoch)
    logger.info(f"criterion = CrossEntropyLoss; optimizer = Adam; use scaler = True.")
    logger.info(f"use learning rate schedule: CosineAnnealingLR(T_max={args.epoch}).")

    # train & test function.
    def train_test(dataset, seed):
        set_seed(seed)
        best_loss, best_acc = 1e9, 0
        model_      = model().to(device)
        optimizer_  = optimizer(model_.parameters())
        scheduler_  = scheduler(optimizer_)
        trainloader = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=args.n_jobs)
        train_loss, train_acc, test_loss, test_acc = list(), list(), list(), list()
        for e in range(1, args.epoch + 1):
            ret = train(model_, trainloader, optimizer_, criterion, scheduler_)
            train_loss.append(ret.loss)
            train_acc.append(ret.acc)
            logger.info(f"Epoch {e:3}/{args.epoch}: train loss:{ret.loss:.5f}, train acc: {ret.acc:.3f}, lr: {ret.lr:.3e}.")
            ret = test(model_, testloader, criterion)
            test_loss.append(ret.loss)
            test_acc.append(ret.acc)
            logger.info(f"Epoch {e:3}/{args.epoch}: test loss:{ret.loss:.5f}, acc: {ret.acc:.3f}.")
        if ret.loss < best_loss:
            best_loss = ret.loss
            logger.info(f'obtain the best test loss {ret.loss:.5f} at Epoch {e}.')
        if ret.acc > best_acc:
            best_acc = ret.acc
            logger.info(f'obtain the best test accuracy {ret.acc:.3f} at Epoch {e}.')
        return train_loss, train_acc, test_loss, test_acc

    # test good/bad/random data performance.
    if args.date:
        file = set_eval_path(args.dataset, args.date)
        logger.info(f'load `{args.dataset}` data quality evaluation result from `{file}`.')
        flag = pd.read_csv(file)
        flag = flag if flag.columns[-1][-4:] == 'pred' else flag.drop(flag.columns[-1], axis=1)
        flag = flag[flag.columns[-1]]
        good_index = flag[flag == 0].index.values.tolist()
        bad_index  = flag[flag == 1].index.values.tolist()
        
        logger.info('train & test good data.')
        subset = Subset(trainset, good_index)
        good   = dict()
        for seed in args.seed:
            logger.info(f'set random seed {seed}.')
            ret = train_test(subset, seed)
            good[f'train-loss-s{seed}'] = ret[0]
            good[f'train-acc-s{seed}']  = ret[1]
            good[f'test-loss-s{seed}']  = ret[2]
            good[f'test-acc-s{seed}']   = ret[3]
        good = pd.DataFrame(good)
        if args.debug:
            logger.debug(f"save result to: {filepath['good']}.")
            logger.debug(f'good data test result:\n{good}')
        else:
            logger.info(f"save result to: {filepath['good']}.")
            good.to_csv(filepath['good'], index=False)
            
        logger.info('train & test random data with the same size as the good data.')
        good_random = dict()
        for seed in args.seed:
            logger.info(f'set random seed {seed}.')
            np.random.seed(seed)
            index  = np.random.choice(len(trainset), len(good_index), replace=False)
            subset = Subset(trainset, index)
            ret = train_test(subset, seed)
            good_random[f'train-loss-s{seed}'] = ret[0]
            good_random[f'train-acc-s{seed}']  = ret[1]
            good_random[f'test-loss-s{seed}']  = ret[2]
            good_random[f'test-acc-s{seed}']   = ret[3]
        good_random = pd.DataFrame(good_random)
        if args.debug:
            logger.debug(f"save result to: {filepath['good_random']}.")
            logger.debug(f'random good data test result:\n{good_random}')
        else:
            logger.info(f"save result to: {filepath['good_random']}.")
            good_random.to_csv(filepath['good_random'], index=False)
        
        logger.info('train & test bad data.')
        subset = Subset(trainset, bad_index)
        bad    = dict()
        for seed in args.seed:
            logger.info(f'set random seed {seed}.')
            ret = train_test(subset, seed)
            bad[f'train-loss-s{seed}'] = ret[0]
            bad[f'train-acc-s{seed}']  = ret[1]
            bad[f'test-loss-s{seed}']  = ret[2]
            bad[f'test-acc-s{seed}']   = ret[3]
        bad = pd.DataFrame(bad)
        if args.debug:
            logger.debug(f"save result to: {filepath['bad']}.")
            logger.debug(f'bad data test result:\n{bad}')
        else:
            logger.info(f"save result to: {filepath['bad']}.")
            bad.to_csv(filepath['bad'], index=False)
        
        logger.info('train & test random data with the same size as the bad data.')
        bad_random = dict()
        for seed in args.seed:
            logger.info(f'set random seed {seed}.')
            np.random.seed(seed)
            index  = np.random.choice(len(trainset), len(bad_index), replace=False)
            subset = Subset(trainset, index)
            ret = train_test(subset, seed)
            bad_random[f'train-loss-s{seed}'] = ret[0]
            bad_random[f'train-acc-s{seed}']  = ret[1]
            bad_random[f'test-loss-s{seed}']  = ret[2]
            bad_random[f'test-acc-s{seed}']   = ret[3]
        bad_random = pd.DataFrame(bad_random)
        if args.debug:
            logger.debug(f"save result to: {filepath['bad_random']}.")
            logger.debug(f'random bad data test result:\n{bad_random}')
        else:
            logger.info(f"save result to: {filepath['bad_random']}.")
            bad_random.to_csv(filepath['bad_random'], index=False)
        
    # test all data performance.
    else:
        logger.info(f'test all samples in dataset: `{args.dataset}`.')
        all_data = dict()
        for seed in args.seed:
            logger.info(f'set random seed {seed}.')
            ret = train_test(trainset, seed)
            all_data[f'train-loss-s{seed}'] = ret[0]
            all_data[f'train-acc-s{seed}']  = ret[1]
            all_data[f'test-loss-s{seed}']  = ret[2]
            all_data[f'test-acc-s{seed}']   = ret[3]
        all_data = pd.DataFrame(all_data)
        if args.debug:
            logger.debug(f"save result to: {filepath['all']}.")
            logger.debug(f'full dataset test result:\n{all_data}')
        else:
            logger.info(f"save result to: {filepath['all']}.")
            all_data.to_csv(filepath['all'], index=False)
            
    logger.info('done!')
    
    
if __name__ == '__main__':
    # python ./scripts/test/performance.py --dataset cifar100 --date 231015-0917 --net resnet18 --bs 128 --lr 1e-3 --epoch 20 --cuda 0 --debug
    args    = parse_args()
    date    = datetime.datetime.now().strftime('%y%m%d-%H%M')
    logpath = set_performance_log_path(args.dataset, date, args.date)
    logger = set_logger(name='sl-embed', level='DEBUG') if args.debug else set_logger(name='sl-embed', logfile=logpath)
    logger.debug(f"save log to: '{logpath}'.")
    if args.date:
        filepath = dict(
            good=set_performance_path(args.dataset, date, args.date, name='good'),
            good_random=set_performance_path(args.dataset, date, args.date, name='good-random'),
            bad=set_performance_path(args.dataset, date, args.date, name='bad'),
            bad_random=set_performance_path(args.dataset, date, args.date, name='bad-random')
            )
    else:
        filepath = dict(all=set_performance_path(args.dataset, date, args.date))
    
    try:
        main(args, logger, filepath)
    except Exception:
        logger.warning(f"catch exceptions, delete incomplete results.")
        for path in filepath.values():
            if os.path.exists(path):
                os.remove(path)
        logger.error(traceback.format_exc())
    except KeyboardInterrupt:
        for path in filepath.values():
            if os.path.exists(path):
                os.remove(path)