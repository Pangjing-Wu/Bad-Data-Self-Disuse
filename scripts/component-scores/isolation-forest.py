import argparse
import datetime
import os
import sys
import traceback

import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from transformers import logging

sys.path.append('.')
import configs
from component_scores import IsolationForest
from models.resnet import *
from models.vit import *
from utils.dataset import load_cv_dataset
from utils.logger import set_logger
from utils.nn import set_seed
from utils.nn.io import load_embedding
from utils.path import set_component_path, set_component_log_path


logging.set_verbosity_error()
SCORE = 'isolation-forest'


def parse_args():
    parser = argparse.ArgumentParser(description='Isolation Forest Evaluation')
    parser.add_argument('--dataset', required=True, help='dataset name {cifar10/cifar100/Caltech101/Caltech256}')
    parser.add_argument('--embed_net', required=True, type=str, help='DNN model name {resnet18/resnet34/resnet50/vit/mae/beit}')
    parser.add_argument('--embed_dataset', required=True, type=str, help='DNN model name {cifar10/cifar100/imagenet1k/imagenet21k}')
    parser.add_argument('--embed_algo', required=True, type=str, help='embedding model training algorithm {sl/pt}')
    parser.add_argument('--embed_epoch', type=int, default=-1, help='embedding epoch (default: -1)')
    parser.add_argument('--embed_date', required=True, type=str, help='date of embedding model')
    parser.add_argument('--pca_dim', type=int, default=0, help='number of components to keep (default: 0)')
    parser.add_argument('--bs', type=int, default=512, help='dataloader batch size (default: 512)')
    parser.add_argument('--model_n_jobs', type=int, default=1, help='n workers for isolation forest (default: 1)')
    parser.add_argument('--dataloader_n_jobs', type=int, default=0, help='n workers for dataloader (default: 0)')
    parser.add_argument('--n_estimators', type=int, default=100, help='`n_estimators` of isolation forest (default: 100)')
    parser.add_argument('--max_samples', type=str, default='auto', help="`max_samples` of isolation forest (default: 'auto')")
    parser.add_argument('--seed', type=int, nargs='+', default=[0], help='random seed (default: [0])')
    parser.add_argument('--cuda', type=int, default=None, help='cuda device id (default: None)')
    parser.add_argument('--debug', action='store_true', help='running without saving model')
    return parser.parse_args()


def main(args, logger, filepath):
    # log parameters.
    logger.info(f'====== Calculate Isolation Forest Score ======')
    logger.info('\n'.join([f'{k} = {str(v)}' for k, v in sorted(dict(vars(args)).items())]))

    # load embedding model.
    param = getattr(configs, f'{args.embed_net}_{args.embed_dataset}_params')
    model = load_embedding(
        model=args.embed_net, 
        dataset=args.embed_dataset, 
        algo=args.embed_algo,
        date=args.embed_date, 
        epoch=args.embed_epoch, 
        param=param
        )
    device = torch.device(f'cuda:{args.cuda}') if isinstance(args.cuda, int) else torch.device('cpu')
    logger.info(f"use device: `{device}`.")
    model  = model.to(device)
    logger.info(f"load embedding model `{args.embed_net}`.")

    # load dataset.
    dataset = load_cv_dataset(args.dataset, train=True, augment=False, resize=param['input_size'])
    logger.info(f'load dataset {args.dataset} with batch size {args.bs}.') 
    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=args.bs, 
        shuffle=False, 
        num_workers=args.dataloader_n_jobs
        )
    logger.info('data transform:\n' + '\n'.join([f'==> {str(t)}' for t in dataset.transform.transforms]))
    
    # calculate embedding.
    logger.info('start calculating data embedding representation.')
    with torch.no_grad():
        x = torch.cat([model(x.to(device)).cpu() for x, _ in dataloader], dim=0).numpy()
    logger.info(f'totally get {x.shape[0]} samples with dimension {x.shape[1:]}.')
    logger.info('complete calculating data embedding representation.')
    
    # embedding dimensionality reduction.
    if args.pca_dim:
        pca = PCA(n_components=args.pca_dim)
        logger.info(f'start reducing embedding dimension to {pca.n_components}.')
        x = pca.fit_transform(x)
        logger.info('complete PCA dimensionality reduction.')
        
    # calculate component scores.
    ret = dict()
    for i, seed in enumerate(args.seed):
        logger.info(f'start {i+1}/{len(args.seed)} round score calculation with random seed {seed}.')
        set_seed(seed)
        clf = IsolationForest(
            n_estimators=args.n_estimators, 
            max_samples=args.max_samples,
            random_state=seed, 
            n_jobs=args.model_n_jobs
            )
        
        logger.info('start fitting outlier score.')
        clf = clf.fit(x)
        logger.info('complete fitting outlier score.')
        ret[f'round:{i}'] = clf.decision_function(x)

    # summarize results.
    ret = pd.DataFrame(ret)
    ret['scores'] = ret.mean(axis=1)
    logger.info(f'summarization of score:\n{ret.describe()}')
    
    if not args.debug:
        logger.info(f"save file to '{filepath}'")
        ret.to_csv(filepath, index=False)
    logger.debug(f'results of score:\n{ret}')
    logger.debug(f"save file to '{filepath}'")
    logger.info('done!')


# python -u ./scripts/component-scores/isolation-forest.py --dataset cifar10 --embed_net 'vit' --embed_dataset 'imagenet21k' --embed_algo 'pt' --embed_date '666666-6666' --model_n_jobs 10 --cuda 0 --debug
if __name__ == '__main__':
    args    = parse_args()
    date    = datetime.datetime.now().strftime('%y%m%d-%H%M')
    logpath = set_component_log_path(score=SCORE, dataset=args.dataset, model=args.embed_net, date=date)
    filepath = set_component_path(score=SCORE, dataset=args.dataset, model=args.embed_net, date=date, epoch=None)
    logger  = set_logger(name=SCORE, level='DEBUG') if args.debug else set_logger(name=SCORE, logfile=logpath)
    logger.debug(f"save log to: '{logpath}'.")
    try:
        main(args, logger, filepath)
    except Exception:
        if os.path.exists(filepath):
            logger.warning(f"catch exceptions, delete incomplete results.")
            os.remove(filepath)
        logger.error(traceback.format_exc())
    except KeyboardInterrupt:
        if os.path.exists(filepath):
            os.remove(filepath)
        if os.path.exists(logpath):
            os.remove(logpath)