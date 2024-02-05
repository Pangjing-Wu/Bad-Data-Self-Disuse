import argparse
import datetime
import math
import multiprocessing as mp
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import LabelSpreading

sys.path.append('.')
import configs
from models.resnet import *
from utils.dataset import load_cv_dataset, KFoldDataset
from utils.evaluate import baseline_evaluation, sample_evaluation
from utils.logger import set_logger
from utils.nn import set_seed
from utils.path import set_component_collection_path, set_eval_path, set_eval_log_path


def parse_args():
    parser = argparse.ArgumentParser(description='Bad Data Self-Disuse')
    parser.add_argument('--dataset', required=True, help='dataset name {cifar10/cifar100}')
    parser.add_argument('--collection_date', required=True, type=str, help='date of component score collection')
    parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
    parser.add_argument('--max_iter', type=int, default=500, help='maximum evaluation iteration (default: 500)')
    parser.add_argument('--n_eval', type=int, default=100, help='select n samples for evaluation (default: 100)')
    parser.add_argument('--eta', type=float, default=1.0, help='prediction consistency threshold (default: 1.0)')
    parser.add_argument('--alpha', type=float, default=0.9, help='EMA coefficient (default: 0.9)')
    parser.add_argument('--beta', type=float, default=1.0, help='shrink coefficient for sampling number (default: 1.0)')
    parser.add_argument('--tolerate', type=float, default=0.5, help='tolerating rate for bad data (default: 0.5)')
    parser.add_argument('--eval_n_jobs', type=int, default=1, help='n workers for evaluation (default: 1)')
    parser.add_argument('--net', required=True, type=str, help='DNN model name {resnet18/resnet50}')
    parser.add_argument('--epoch', type=int, default=20, help='training epoch (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate (default: 0.001)')
    parser.add_argument('--bs', type=int, default=512, help='training batch size (default: 512)')
    parser.add_argument('--eval_mode', type=str, default='same', help="evaluation mode {all/same/diff} (default: 'all')")
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=None, help='cuda device id (default: None)')
    parser.add_argument('--debug', action='store_true', help='running without saving model')
    return parser.parse_args()

        
def main(args, logger, filepath):
    """
    1. load dataset & component score.
    2. split dataset into k fold.
    3. evaluate the baseline performance.
    >>>>> Loop
    4. sampling good/bad data based on the results of the component score-based decision model (uniformly sample samples from the dataset for the first round). 
    5. create training-validation pair for each sampled data.
    6. multiprocessing evaluation.
    7. fit the component score-based decision model.
    <<<<< Done
    8. Use the component score-based decision model to discriminate bad data.
    """
    # log parameters.
    logger.info(f'====== Bad Data Self-Disuse ======')
    logger.info('\n'.join([f'{k} = {str(v)}' for k, v in sorted(dict(vars(args)).items())]))
    
    set_seed(args.seed)
    
    #check evaluation mode.
    if args.eval_mode in ['all', 'same', 'diff']:
        mode = args.eval_mode
    else:
        raise ValueError(f'unknown evaluation mode {args.eval_mode}')
    
    # set model.
    param = getattr(configs, f"{args.net}_{args.dataset.split('-')[0]}_params")
    if args.net == 'resnet18':
        model_fn = partial(resnet18, **param)
    elif args.net == 'resnet34':
        model_fn = partial(resnet34, **param)
    elif args.net == 'resnet50':
        model_fn = partial(resnet50, **param)
    else:
        raise ValueError(f"unknown model '{args.net}'.")
    device    = torch.device(f'cuda:{args.cuda}') if isinstance(args.cuda, int) else torch.device('cpu')
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam
    scheduler = partial(torch.optim.lr_scheduler.StepLR, step_size=args.gamma)
    logger.info(f"use device: `{device}`.")
    logger.info(f"set model {args.net} with {param['n_channel']} input channel and {param['n_classes']} output classes.")
    logger.info(f"criterion = CrossEntropyLoss; optimizer = Adam; use scaler = True.")
    logger.info(f"use learning rate schedule: StepLR(step_size={args.gamma}).")
    
    # load dataset.
    dataset = load_cv_dataset(args.dataset, train=True, augment=False, resize=param['input_size'])
    logger.info(f'load dataset {args.dataset} with batch size {args.bs}.') 
    logger.info('data transform:\n' + '\n'.join([f'==> {str(t)}' for t in dataset.transform.transforms]))
    folds   = KFoldDataset(dataset=dataset, k=args.k, shuffle=True)
    columns = [f'fold-{i}' for i in range(args.k)]
    logger.info(f'split dataset into {args.k} folds, where each fold contained {len(folds[0])} elements.')
    
    # load component scores.
    collection = set_component_collection_path(args.dataset, args.collection_date)
    logger.info(f'load collection file: {collection}.')
    components = dict()
    with open(collection, 'r') as f:
        for file in f.readlines():
            file   = file.rstrip('\n')
            score = pd.read_csv(file)['scores'].values
            name  = f"{os.path.dirname(file).split('/')[-1]}-{os.path.basename(file).rstrip('.csv')}"
            components[name] = score
    components = pd.DataFrame(components)
    components = components[components.columns.sort_values()]
    logger.info(f'load {components.shape[1]} component scores:\n{components.head()}')
    
    # parallel baseline performance evaluation.
    logger.info(f'start parallel baseline performance evaluation with {args.eval_n_jobs} processes.')
    baselines = dict(all=[None] * args.k, same=[None] * args.k, diff=[None] * args.k)
    eval_fn   = partial(baseline_evaluation, dataset=folds, model_fn=model_fn, epoch=args.epoch, lr=args.lr, 
                        bs=args.bs, optimizer=optimizer, criterion=criterion, device=device, alpha=args.alpha,
                        scheduler=scheduler)
    with ProcessPoolExecutor(max_workers=args.eval_n_jobs, mp_context=mp.get_context('spawn')) as executor:
        for i, loss in executor.map(eval_fn, range(args.k)):
            for k, v in loss.items():
                baselines[k][i] = v
    b_all   = pd.DataFrame(np.array(baselines['all']), index=columns).T     # [1, k-fold]
    b_same  = pd.DataFrame(np.array(baselines['same']), index=columns).T    # [n,class, k-fold]
    b_diff  = pd.DataFrame(np.array(baselines['diff']), index=columns).T    # [n,class, k-fold]
    if mode == 'all':
        baselines = b_all
    elif mode == 'same':
        baselines = b_same
    elif mode == 'diff':
        baselines = b_diff
    logger.info(f"obtain baseline validation set performance under mode {mode}:\n{baselines}.")
    if not args.debug:
        b_all.to_csv(filepath['ba'])
        b_same.to_csv(filepath['bs'])
        b_diff.to_csv(filepath['bd'])
    # set decision model.
    ssl = LabelSpreading(**getattr(configs, f"label_spreading_{args.dataset.split('-')[0]}_params"))
    clf = RandomForestClassifier(**getattr(configs, f"random_forest_{args.dataset.split('-')[0]}_params"))
    logger.info('Label Spreading (SSL) parameters:\n' + '\n'.join([f'{k} = {str(v)}' for k, v in ssl.get_params().items()]))
    logger.info('Random Forest (Explicable) parameters:\n' + '\n'.join([f'{k} = {str(v)}' for k, v in clf.get_params().items()]))
    # set parallel bad data evaluation function.
    eval_fn = partial(sample_evaluation, dataset=folds, model_fn=model_fn, epoch=args.epoch, lr=args.lr, 
                      bs=args.bs, optimizer=optimizer, criterion=criterion, device=device, alpha=args.alpha,
                      scheduler=scheduler)
    # set bad data flag, 0: good data, 1: bad data, -1: have not been evaluated.
    flags   = -np.ones(len(dataset), dtype=int)
    pred   = -np.ones(len(dataset), dtype=int)
    ret    = dict()
    weight = list()
    losses = dict(id=list(), all=list(), same=list(), diff=list())
    
    # start bad data self-disuse evaluation.
    for r in range(args.max_iter):
        logger.info(f'start {r+1}/{args.max_iter} bad data self-disuse evaluation iteration.')
        
        # sampling.
        if r == 0:
            # random sampling.
            sample_indices = np.random.choice(len(dataset), args.n_eval, replace=False)
        else:
            # sample from positive and negative classes respectively to ensure class balance.
            good_index = np.arange(len(dataset))[(pred == 0) & (flags == -1)]
            bad_index  = np.arange(len(dataset))[(pred == 1) & (flags == -1)]
            if len(good_index) >= math.ceil(args.n_eval / 2) and len(bad_index) >= math.ceil(args.n_eval / 2):
                good_index     = np.random.choice(good_index, math.ceil(args.n_eval / 2), replace=False)
                bad_index      = np.random.choice(bad_index, math.ceil(args.n_eval / 2), replace=False)
                sample_indices = np.concatenate([good_index, bad_index], axis=0)                                                 
            else:
                logger.warning(f'no sufficient candidate good/bad samples (good = {len(good_index)}, bad = {len(bad_index)}), imbalanced sampling.')
                sample_indices = np.random.choice(np.arange(len(dataset))[flags == -1], args.n_eval, replace=False)
        logger.info(f'sampling {len(sample_indices)} samples.')
        logger.info('start evaluate each sample.')
        
        # parallel bad data evaluation.
        count  = [0, 0]  # [n good data, n bad data].
        with ProcessPoolExecutor(max_workers=args.eval_n_jobs, mp_context=mp.get_context('spawn')) as executor:
            # loss = dict(key=['all', 'same', 'diff'], value=[k-fold])
            for i, loss in executor.map(eval_fn, sample_indices):
                losses['id'].append(i)
                for k, v in loss.items():
                    losses[k].append(v)
                loss = np.array(loss[mode])
                # evaluate sample quality on each fold.
                baseline = baselines.values.flatten() if mode == 'all' else baselines.loc[dataset[i][-1]].values
                # the sample is bad data if the loss is smaller than baseline after removing the sample
                flag = (loss < baseline).tolist()
                # remove the fold containing the sample.
                if (flag.count(True) - 1) / (len(flag) - 1) > args.tolerate:  
                    flags[i]   = 1
                    count[1] += 1
                else:
                    flags[i]   = 0
                    count[0] += 1
        ret[f'iter:{r}-true'] = flags.copy()
        logger.info(f'obtain {count[0]} new good data, and {count[1]} new bad data.')
        logger.info(f"save round {r} evaluation result to '{filepath['result']}'")
        s_all  = pd.DataFrame(np.array(losses['all']), index=losses['id'], columns=columns)  # [n sample, k-fold]
        s_same = pd.DataFrame(np.array(losses['same']), index=losses['id'], columns=columns) # [n sample, k-fold]
        s_diff = pd.DataFrame(np.array(losses['diff']), index=losses['id'], columns=columns) # [n sample, k-fold]
        if not args.debug:
            pd.DataFrame(ret).to_csv(filepath['result'], index=False)
            s_all.to_csv(filepath['sa'])
            s_same.to_csv(filepath['ss'])
            s_diff.to_csv(filepath['sd'])
        
        # calculate prediction consistency.
        consistency = sum(flags[sample_indices] == pred[sample_indices]) / len(sample_indices)
        logger.info(f'last prediction consistency = {consistency:.5f}, eta = {args.eta}.')
        if consistency < args.eta:
            # fit bad data discriminate model.
            logger.info(f'fit SSL model with {len(flags[flags!=-1])} labeled samples.')
            ssl  = ssl.fit(components.to_numpy(), flags)
            pred = ssl.predict(components.to_numpy())
            ret[f'iter:{r}-pred'] = pred
            logger.info(f'prediction results: {len(pred[pred==0])} good data, {len(pred[pred==1])} bad data.')
            logger.info(f'fit explicable model with SSL results and calculate feature importances.')
            clf = clf.fit(components.to_numpy(), pred)
            weight.append(clf.feature_importances_)
            logger.info(f"save round {r} feature weights to '{filepath['weight']}'")
            if not args.debug: pd.DataFrame(np.array(weight), columns=components.columns).to_csv(filepath['weight'], index=False)
            # update sampling number.
            args.n_eval = math.ceil(args.n_eval * args.beta)
            if args.n_eval > sum(flags == -1):
                args.n_eval = sum(flags == -1)
                logger.warning(f'no sufficient unevaluated data (only {sum(flags == -1)}), reset `args.n_eval` to {sum(flags == -1)}.')
        else:
            logger.info(f'achieving prediction consistency threshold at iteration {r+1}, stop iteration.')
            break
        
    logger.debug(f'bad data evaluation results:\n{pd.DataFrame(ret)}')
    logger.debug(f"save file to '{filepath['result']}'")
    logger.debug(f'baseline performance under mode {mode}:\n{baselines}')
    logger.debug(f"sample performance under mode {mode}:\n{pd.DataFrame(np.array(losses[mode]), index=losses['id'], columns=columns)}")
    logger.debug(f'feature weight:\n{pd.DataFrame(np.array(weight), columns=components.columns)}')
    logger.info('done!')


# python -u ./scripts/evaluate.py --dataset cifar10 --collection_date '230827-0851' --k 3 --max_iter 2 --n_eval 10 --eval_n_jobs 2 --net resnet18 --epoch 1 --bs 512 --cuda 1 --debug
if __name__ == '__main__':
    # set argument and basic paths.
    args    = parse_args()
    date    = datetime.datetime.now().strftime('%y%m%d-%H%M')
    logpath = set_eval_log_path(dataset=args.dataset, date=date)
    filepath = dict(
        result=set_eval_path(dataset=args.dataset, date=date),
        weight=set_eval_path(dataset=args.dataset, date=date, name='weight'),
        sa=set_eval_path(dataset=args.dataset, date=date, name='samples-all'),
        ss=set_eval_path(dataset=args.dataset, date=date, name='samples-same'),
        sd=set_eval_path(dataset=args.dataset, date=date, name='samples-diff'),
        ba=set_eval_path(dataset=args.dataset, date=date, name='baseline-all'),
        bs=set_eval_path(dataset=args.dataset, date=date, name='baseline-same'),
        bd=set_eval_path(dataset=args.dataset, date=date, name='baseline-diff')
        )
    logger = set_logger(name='eval', level='DEBUG') if args.debug else set_logger(name='eval', logfile=logpath)
    logger.debug(f"save log to: '{logpath}'.")
    # start evaluation.
    try:
        main(args, logger, filepath)
    except Exception:
        logger.error(traceback.format_exc())
    except KeyboardInterrupt:
        for path in filepath.values():
            if os.path.exists(path):
                os.remove(path)