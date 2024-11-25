import argparse
import datetime
import logging
import math
import multiprocessing as mp
import sys
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from scipy.stats import mannwhitneyu

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from sklearn.semi_supervised import LabelSpreading

sys.path.append('.')
from configs import evaluate as configs
from models.lightnn import MobileNetV3
from modules.shard import DataShard
from modules.evaluation import benchmark_evaluation_nn, sample_evaluation_nn
from utils.logger import set_logger
from utils.io.dataset import load_cv_dataset
from utils.io.evaluation import EvaluationManager
from utils.nn import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Bad Data Evaluation')
    parser.add_argument('--dataset', required=True, help='dataset name')
    parser.add_argument('--cuda', type=int, default=None, help='cuda device id (default: None)')
    return parser.parse_args()

        
def main(args, config, logger, eval_manager):
    # log parameters.
    logger.info(f'====== Bad Data Evaluation ======')
    logger.info('\n'.join([f'{k} = {v}' for k, v in config._asdict().items()]))
    
    set_seed(config.seed)
    
    #check evaluation mode.
    if config.eval_mode in ['all', 'same', 'diff']:
        mode = config.eval_mode
    else:
        raise ValueError(f'unknown evaluation mode {args.eval_mode}')
    
    # set model.
    model_fn = partial(MobileNetV3, **config.net_kwargs)
    
    device    = torch.device(f'cuda:{args.cuda}') if isinstance(args.cuda, int) else torch.device('cpu')
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam
    scheduler = partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=config.train_kwargs['epoch'], eta_min=config.train_kwargs['min_lr'])
    
    # load dataset.
    dataset = load_cv_dataset(args.dataset, train=True, augment=False)
    shards  = DataShard(dataset=dataset, k=config.n_shard, shuffle=True)
    logger.debug('data transform:\n' + '\n'.join([f'==> {str(t)}' for t in dataset.transform.transforms]))
    logger.debug(f'split dataset into {config.n_shard} shards, where each shard contained {len(shards[0])} elements.')
    
    # load component scores.
    metadata = pd.read_csv(config.metadata)
    logger.info(f'load metadata with shape of {metadata.shape}\n, meta data preview:\n{metadata.head()}')
    
    # parallel baseline performance evaluation.
    logger.info(f'start parallel baseline performance evaluation with {config.n_jobs} processes.')
    benchmarks = [[] for _ in range (config.n_shard)]
    eval_fn    = partial(
        benchmark_evaluation_nn, 
        shards=shards, 
        mode=mode,
        model_fn=model_fn, 
        epoch=config.train_kwargs['epoch'], 
        lr=config.train_kwargs['lr'], 
        bs=config.train_kwargs['bs'], 
        optimizer=optimizer, 
        criterion=criterion, 
        device=device, 
        ema_coef=config.ema_coef,
        scheduler=scheduler
        )
    with ProcessPoolExecutor(max_workers=config.n_jobs, mp_context=mp.get_context('spawn')) as executor:
        for i, loss in executor.map(eval_fn, range(config.n_shard)):
            logger.debug(f"validation loss of shard {i} is {loss}.")
            for j in range(config.n_shard):
                if j != i:
                    # append all loss when shard i in the training set.
                    benchmarks[j].append(loss)
    benchmarks = np.array(benchmarks)
    eval_manager.save_benchmark(benchmarks)
    logger.debug(f"obtain benchmark shard performance (shape: {benchmarks.shape}) under evaluation mode {mode}.")
    
    # set decision model.
    ssl = LabelSpreading(**config.graph_ssl_kwargs)
    logger.info('graph SSL parameters:\n' + '\n'.join([f'{k} = {str(v)}' for k, v in ssl.get_params().items()]))
    
    # set parallel bad data evaluation function.
    eval_fn = partial(
        sample_evaluation_nn, 
        shards=shards, 
        mode=mode,
        model_fn=model_fn, 
        epoch=config.train_kwargs['epoch'], 
        lr=config.train_kwargs['lr'], 
        bs=config.train_kwargs['bs'], 
        optimizer=optimizer, 
        criterion=criterion, 
        device=device, 
        ema_coef=config.ema_coef,
        scheduler=scheduler
        )
    # set bad data flag, 0: good data, 1: bad data, -1: have not been evaluated.
    flags = -np.ones(len(dataset), dtype=int)
    pred = -np.ones(len(dataset), dtype=int)
    
    ret  = dict()
    
    # start bad data self-disuse evaluation.
    warmup_status = True
    pred_status   = False
    num_pre_flag = math.ceil(config.n_eval_per_iter / 2) 
    for r in range(config.max_iter):
        logger.info(f'start {r+1}/{config.max_iter} bad data self-disuse evaluation iteration.')
        if warmup_status:
            # random sampling.
            sample_indices = np.random.choice(len(dataset), config.warmup_num, replace=False)
        else:
            # balanced sampling positive and negative samples.
            candidate_good = np.arange(len(dataset))[(pred == 1) & (flags == -1)]
            candidate_bad  = np.arange(len(dataset))[(pred == 0) & (flags == -1)]
            if len(candidate_good) >= num_pre_flag and len(candidate_bad) >= num_pre_flag:
                num_good, num_bad = num_pre_flag, num_pre_flag
            elif len(candidate_good) < num_pre_flag and len(candidate_bad) >= num_pre_flag:
                num_good, num_bad = len(candidate_good), 2 * num_pre_flag - len(candidate_good)
                logger.warning(f'no sufficient candidate good samples ({len(candidate_good)}/{num_pre_flag}, imbalanced sampling.')
            elif len(candidate_good) >= num_pre_flag and len(candidate_bad) < num_pre_flag:
                num_good, num_bad = 2 * num_pre_flag - len(candidate_bad), len(candidate_bad)
                logger.warning(f'no sufficient candidate bad samples ({len(candidate_bad)}/{num_pre_flag}, imbalanced sampling.')
            else:
                num_good, num_bad = len(candidate_good), len(candidate_bad)
                logger.warning(f'no sufficient candidate good and bad samples ({len(candidate_good)}/{len(candidate_bad)}/{num_pre_flag}.')
            sampled_good = np.random.choice(candidate_good, num_good, replace=False)
            sampled_bad  = np.random.choice(candidate_bad, num_bad, replace=False)
            sample_indices  = np.concatenate([sampled_good, sampled_bad], axis=0)
        sample_id_2_shard_id = {i: shards.get_shard_by_sample_index(i) for i in sample_indices}
        logger.info(f'sampling {len(sample_indices)} samples.')
        logger.info(f'sample shard count: {Counter(sample_id_2_shard_id.values())}.')
        
        logger.info('start evaluate each sample.')
        # parallel bad data evaluation.
        count = [0, 0]  # [n good data, n bad data].
        with ProcessPoolExecutor(max_workers=config.n_jobs, mp_context=mp.get_context('spawn')) as executor:
            # loss = dict(key=['all', 'same', 'diff'], value=[k-fold])
            for i, loss in executor.map(eval_fn, sample_indices):
                # evaluate sample quality on each shard.
                if mode == 'all':
                    benchmark = benchmarks[sample_id_2_shard_id[i]]
                else:
                    benchmark = benchmarks[sample_id_2_shard_id[i], :, dataset[i][-1]]
                # the sample is bad data if the loss is significantly smaller than benchmark after removing the sample
                logger.debug(f'{loss = }')
                logger.debug(f'{benchmark = }')
                _, p_value = mannwhitneyu(loss, benchmark, alternative='less', method='exact')
                logger.debug(f'{p_value = }')
                flag = int(p_value > config.significance)
                flags[i] = flag
                count[flag] += 1
                evaluation_report = {"i": i, "loss": loss, "benchmark": benchmark, "p_value": p_value, "ground_truth": flag, "previous_pred": pred[i]}
                eval_manager.save_eval(evaluation_report)
        logger.info(f'evaluation result: obtain {count[1]} new good data, and {count[0]} new bad data.')
        ret[f'iter:{r}-gt'] = flags.copy()
        eval_manager.save_est(ret)
        
        # stop warmup when obtain positive and negative samples.
        if warmup_status and min(*count) > 0:
            warmup_status = False
        
        if not warmup_status:
            # calculate estimation consistency.
            if pred_status:
                consistency = balanced_accuracy_score(flags[sample_indices], pred[sample_indices])
                logger.info(f'estimation consistency = {consistency:.5f}, budget = {config.budget}.')
            else:
                pred_status = True
                consistency = 0.
            if consistency < config.budget:
                # fit bad data estimation model.
                logger.info(f'fit SSL model with {len(flags[flags!=-1])} labeled samples.')
                ssl  = ssl.fit(metadata.to_numpy(), flags)
                pred = ssl.predict(metadata.to_numpy())
                ret[f'iter:{r}-pred'] = pred
                eval_manager.save_est(ret)
                logger.info(f'estimation results: {len(pred[pred==1])} good data, {len(pred[pred==0])} bad data.')
                if config.n_eval_per_iter > sum(flags == -1):
                    logger.warning(f'no sufficient unevaluated data for further evaluation, stop iteration.')
                    break
            else:
                logger.info(f'achieving budget target at iteration {r+1}, stop iteration.')
                break
    logger.info('done!')


if __name__ == '__main__':
    args   = parse_args()
    config  = getattr(configs, args.dataset)
    date   = datetime.datetime.now().strftime('%y%m%d-%H%M')
    eval_manager = EvaluationManager(dataset=args.dataset, date=date)
    logger = set_logger(name=f'eval-{args.dataset}', logfile=eval_manager.logpath, level=logging.DEBUG)
    try:
        main(args, config, logger, eval_manager)
    except Exception:
        logger.error(traceback.format_exc())