import argparse
import datetime
import glob
import os
import sys
import traceback

import numpy as np

sys.path.append('.')
import configs
from utils.logger import set_logger
from utils.path import set_component_collection_path, set_component_collection_log_path


def parse_args():
    parser = argparse.ArgumentParser(description='Collect Component Scores')
    parser.add_argument('--dataset', required=True, type=str, help='dataset name {cifar10/cifar100}')
    parser.add_argument('--score', type=str, nargs='+', default=None, help='name of component scores')
    parser.add_argument('--model', type=str, nargs='+', default=None, help='name of the backbone model of the component scores')
    parser.add_argument('--date_range', type=str, nargs=2, default=None, help='date range (include boundaries) of component scores: (date1, date2)')
    parser.add_argument('--remark', '-m', type=str, help='date range of component scores')
    parser.add_argument('--debug', action='store_true', help='running without saving model')
    return parser.parse_args()


def date_condition(file, lb, ub):
    name = os.path.basename(file).rstrip('.csv')
    date = int(''.join([s for s in name.split('-') if s.isdigit()]))
    if lb <= date and date <= ub:
        return True
    else:
        return False


def main(args, logger, filepath):
    # log parameters.
    logger.info(f'====== Collect Component Scores ======')
    logger.info('\n'.join([f'{k} = {str(v)}' for k, v in sorted(dict(vars(args)).items())]))
    # set dirname.
    dirname = os.path.join(configs.COMPONENT_DIR, args.dataset)
    
    # set score pattern.
    if args.score is None:
        dirnames = [os.path.join(dirname, '*')]
    else:
        dirnames = [os.path.join(dirnames, score) for score in args.score]
    
    # set model pattern.
    if args.model is None:
        filenames = [os.path.join(dir_, '*.csv') for dir_ in dirnames]
    else:
        filenames = [[os.path.join(dir_, f'{model}-*.csv') for model in args.model] for dir_ in dirnames]
        filenames = np.array(filenames).flatten().tolist()
    
    # search files.
    files = [glob.glob(filename) for filename in filenames]
    files = np.concatenate(files).tolist()

    # set date pattern.
    if args.date_range is not None:
        assert len(args.date_range[0]) == 10 and len(args.date_range[1]) == 10
        args.date_range = list(map(int, args.date_range))
        assert args.date_range[0] <= args.date_range[1]
        files = [file for file in files if date_condition(file, *args.date_range)]
    
    # summarize results.
    if len(files) == 0:
        raise RuntimeError('no component score file is selected, please check your arguments.')
    else:
        files = sorted(files)
        logger.info(f'totally selected {len(files)} score files.')
    
    # save results.
    if args.debug:
        files = '\n'.join(files)
        logger.debug(f'selected component score files:\n{files}')
        logger.debug(f"save file to '{filepath}'")
    else:
        logger.info(f"save file to '{filepath}'")
        with open(filepath, 'w') as f:
            for file in files:
                f.write(file + '\n')
    logger.info(f'done!')


if __name__ == '__main__':
    args    = parse_args()
    date    = datetime.datetime.now().strftime('%y%m%d-%H%M')
    logpath = set_component_collection_log_path(dataset=args.dataset, date=date)
    filepath = set_component_collection_path(dataset=args.dataset, date=date)
    logger  = set_logger(name='collection', level='DEBUG') if args.debug else set_logger(name='collection', logfile=logpath)
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