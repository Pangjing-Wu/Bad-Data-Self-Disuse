import argparse
import datetime
import sys
import traceback

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append('.')
from configs.metadata import centroid as configs
from metadata import Centroid
from utils.logger import set_logger
from utils.io.dataset import load_cv_dataset
from utils.io.embedding import ImageEmbeddingManager
from utils.io.metadata import MetaDataManager
from utils.nn import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Centroid')
    parser.add_argument('--dataset', required=True, help='dataset name')
    parser.add_argument('--cuda', type=int, default=None, help='cuda device id (default: None)')
    return parser.parse_args()


def main(args, config, logger, meta_manager):
    # log parameters.
    logger.info(f'====== Calculate Centroid Score ======')
    logger.info('\n'.join([f'{k} = {v}' for k, v in config._asdict().items()]))

    # load embedding model.
    manager = ImageEmbeddingManager(
        method=config.embed_method, 
        dataset=args.dataset, 
        backbone=config.backbone, 
        backbone_kwargs=config.backbone_kwargs,
        date=config.embed_date
        )
    model  = manager.load(config.embed_epoch)
    device = torch.device(f'cuda:{args.cuda}') if isinstance(args.cuda, int) else torch.device('cpu')
    logger.info(f"use device: `{device}`.")
    model  = model.to(device)

    # load dataset.
    dataset = load_cv_dataset(args.dataset, train=True, augment=False)
    logger.info(f'load dataset {args.dataset} with batch size {config.bs}.') 
    dataloader = DataLoader(dataset=dataset, batch_size=config.bs, shuffle=False, num_workers=0)
    logger.info('data transform:\n' + '\n'.join([f'==> {str(t)}' for t in dataset.transform.transforms]))
    
    # calculate embedding.
    logger.info('start calculating data embedding representation.')
    with torch.no_grad():
        x = torch.cat([model(x.to(device)).cpu() for x, _ in dataloader], dim=0).numpy()
    logger.info(f'totally get {x.shape[0]} samples with dimension {x.shape[1:]}.')
    logger.info('complete calculating data embedding representation.')
        
    # calculate scores.
    score = pd.DataFrame()
    for seed in range(config.repeat):
        set_seed(seed)
        logger.info(f'calculating Centroid score ({seed+1}/{config.repeat}).')
        scorer = Centroid(n_classes=config.n_classes)
        scorer = scorer.fit(x)
        score[f'score-{seed}'] = scorer.scores() 
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
    args  = parse_args()
    config = getattr(configs, args.dataset)
    date  = datetime.datetime.now().strftime('%y%m%d-%H%M')
    meta_manager = MetaDataManager('centroid', dataset=args.dataset, date=date)
    logger = set_logger(name='centroid', logfile=meta_manager.logpath)
    try:
        main(args, config, logger, meta_manager)
    except Exception:
        logger.error(traceback.format_exc())