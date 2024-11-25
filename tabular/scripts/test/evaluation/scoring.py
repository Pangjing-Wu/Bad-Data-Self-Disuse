import argparse
import datetime
import sys
import traceback

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.append('.')
from utils.logger import set_logger
from utils.io.dataset import load_tabular_dataset
from utils.nn import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Test Score Performance')
    parser.add_argument('--dataset', required=True, help='dataset name')
    parser.add_argument('--score_file', required=True, help='score file path')
    parser.add_argument('--score_col', required=True, help='score column name')
    parser.add_argument('--top_n', required=True, type=int, help='top n samples')
    return parser.parse_args()


def main(args, logger):
    set_seed(0)
    # log parameters.
    logger.info(f'====== Test Score Performance ======')
    logger.info('\n'.join([f'{k} = {str(v)}' for k, v in sorted(vars(args).items())]))
    
    # set dataset.
    index  = pd.read_csv(args.score_file)[args.score_col].nlargest(args.top_n).index.tolist()
    X_train, y_train = load_tabular_dataset(args.dataset, train=True)
    X_test, y_test   = load_tabular_dataset(args.dataset, train=False)
    X_train, y_train = X_train.values[index], y_train.values[index]
    logger.info(f'load train dataset {args.dataset} with {len(X_train)} samples.')
    
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    logger.info(f'{accuracy = }\n{report = }\n{conf_matrix = }')
        

# python -u scripts/test/evaluation/scoring.py --dataset cifar10 --score_file xx --score_col 'score-norm' --top_n 45000 --cuda 0
if __name__ == '__main__':
    args   = parse_args()
    date   = datetime.datetime.now().strftime('%y%m%d-%H%M')
    logger = set_logger(name=f'score-eval-{date}', logfile=args.score_file.replace('.csv', '-eval.log'))
    try:
        main(args, logger)
    except Exception:
        logger.error(traceback.format_exc())