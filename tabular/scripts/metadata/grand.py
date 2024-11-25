import argparse
import datetime
import sys
import traceback

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append('.')
from configs.metadata import grand as configs
from metadata import GraNd
from models.ftnn import FTMLP
from utils.logger import set_logger
from utils.io.dataset import load_tabular_dataset
from utils.io.metadata import MetaDataManager
from utils.nn import set_seed


SCORE = 'grand'


def parse_args():
    parser = argparse.ArgumentParser(description='GraNd')
    parser.add_argument('--dataset', required=True, help='dataset name')
    parser.add_argument('--cuda', type=int, default=None, help='cuda device id (default: None)')
    return parser.parse_args()


def main(args, config, logger, meta_manager):
    # log parameters.
    logger.info(f'====== Calculate GraNd Score ======')
    logger.info('\n'.join([f'{k} = {str(v)}' for k, v in sorted(dict(vars(args)).items())]))
    device = torch.device(f'cuda:{args.cuda}') if isinstance(args.cuda, int) else torch.device('cpu')
    
    # set models.
    X_train, y_train = load_tabular_dataset(args.dataset, train=True)
    X_train   = torch.tensor(X_train.values, dtype=torch.float32)
    y_train   = torch.tensor(y_train.values, dtype=torch.long)
    dataset   = TensorDataset(X_train, y_train)
    logger.info(f'load dataset {args.dataset}.')
    
    score      = pd.DataFrame()
    evalloader = DataLoader(dataset=dataset, batch_size=config.bs, shuffle=False, num_workers=0)
    
    for seed in range(config.repeat):
        set_seed(seed)
        logger.info(f'calculating GraNd score ({seed+1}/{config.repeat}).')
        # reset data loader.
        trainloader = DataLoader(dataset=dataset, batch_size=config.bs, shuffle=True, num_workers=0)
        # reset model.
        model     = FTMLP(config.n_classes, **config.model_kwargs).to(device)
        criterion = nn.CrossEntropyLoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        schedule  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epoch, eta_min=config.min_lr)
        # reset scorer.
        scorer = GraNd(model=model, n_classes=config.n_classes, order=config.order, optim=torch.optim.Adam, fc_name=config.fc_name)
        
        # training model & calculate component scores.
        for e in range(1, config.epoch + 1):
            batch_loss, batch_acc = 0, 0
            for x, y in trainloader:
                x, y = x.to(device), y.to(device) 
                optimizer.zero_grad()
                logistic = model(x)
                loss = criterion(logistic, y)
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                batch_acc  += torch.argmax(logistic, dim=1).eq(y).sum().item() / y.size(0)
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            schedule.step()
            batch_loss /= len(trainloader)
            batch_acc  /= len(trainloader) / 100
            logger.info(f"Epoch {e:3}/{config.epoch}: train loss:{batch_loss:.5f}, train acc: {batch_acc:.3f}, lr: {lr:.3e}.")
        score[f'score-{seed}'] = scorer.scores(dataloader=evalloader)
        
    # summarize results.
    score['score'] = score.mean(axis=1)
    logger.info('normalizing scores')
    lower_bound = score['score'].quantile(0.05)
    upper_bound = score['score'].quantile(0.95)
    score['score-norm'] = ((score['score'] - lower_bound) / (upper_bound - lower_bound)).clip(0, 1) # robust min-max norm
    logger.info(f'summarization of score:\n{score.describe()}')
    logger.info(f"save metadata")
    meta_manager.save(score)
    logger.info('done!')



if __name__ == '__main__':
    args   = parse_args()
    config  = getattr(configs, args.dataset)
    date   = datetime.datetime.now().strftime('%y%m%d-%H%M')
    meta_manager = MetaDataManager('grand', dataset=args.dataset, date=date)
    logger = set_logger(name='grand', logfile=meta_manager.logpath)
    try:
        main(args, config, logger, meta_manager)
    except Exception:
        logger.error(traceback.format_exc())