import argparse
import datetime
import sys
import traceback

import pandas as pd
from sklearn.decomposition import PCA

sys.path.append('.')
from configs.metadata import ocsvm as configs
from metadata import MultiClassesOneClassSVM
from utils.logger import set_logger
from utils.io.dataset import load_tabular_dataset
from utils.io.metadata import MetaDataManager
from utils.nn import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='One Class SVM')
    parser.add_argument('--dataset', required=True, help='dataset name')
    parser.add_argument('--cuda', type=int, default=None, help='cuda device id (default: None)')
    return parser.parse_args()


def main(args, config, logger, meta_manager):
    # log parameters.
    logger.info(f'====== Calculate One Class SVM Score ======')
    logger.info('\n'.join([f'{k} = {v}' for k, v in config._asdict().items()]))

    # load dataset.
    dataset = load_tabular_dataset(args.dataset, train=True)
    logger.info(f'load dataset {args.dataset}.')
    
    # calculate embedding.
    logger.info('start calculating data embedding representation.')
    x = dataset[0].values
    y = dataset[1].values
    logger.info(f'totally get {x.shape[0]} samples with dimension {x.shape[1:]}.')
    logger.info('complete calculating data embedding representation.')

    # embedding dimensionality reduction.
    if config.pca_dim:
        pca = PCA(n_components=config.pca_dim)
        logger.info(f'start reducing embedding dimension to {pca.n_components}.')
        x = pca.fit_transform(x)
        logger.info('complete PCA dimensionality reduction.')
    
    # calculate scores.
    score = pd.DataFrame()
    for seed in range(config.repeat):
        set_seed(seed)
        logger.info(f'calculating one-class SVM score ({seed+1}/{config.repeat}).')
        scorer = MultiClassesOneClassSVM(n_classes=config.n_classes, **config.ocsvm_kwargs)
        scorer = scorer.fit(x, y)
        score[f'score-{seed}'] = scorer.scores(x, y) # the lower, the more abnormal.
    logger.info(f'observed {sum(scorer.labels(x, y) == -1)} outliers.')
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
    args    = parse_args()
    config   = getattr(configs, args.dataset)
    date    = datetime.datetime.now().strftime('%y%m%d-%H%M')
    meta_manager = MetaDataManager('oc-svm', dataset=args.dataset, date=date)
    logger  = set_logger(name='oc-svm', logfile=meta_manager.logpath)
    try:
        main(args, config, logger, meta_manager)
    except Exception:
        logger.error(traceback.format_exc())