import argparse
import datetime
import os
import pickle
import sys
import traceback

import numpy as np
import torchvision

sys.path.append('.')
import configs
from _datasets import *
from configs.datasets import noise_transform
from utils.dataset import VisionDataset
from utils.logger import set_logger
from utils.nn import set_seed
from utils.path import set_dataset_path, set_dataset_log_path


def parse_args():
    parser = argparse.ArgumentParser(description='Build Noisy Dataset')
    parser.add_argument('--dataset', required=True, type=str, help='dataset name {cifar10/cifar100/caltech101/caltech256}')
    parser.add_argument('--feature_noise_rate', '-x', type=float, default=0., help='feature noise rate (default: 0.00)')
    parser.add_argument('--label_noise_rate', '-y', type=float, default=0., help='label noise rate (default: 0.00)')
    parser.add_argument('--asymmetric', '-a', action='store_true', help='asymmetric label noise')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--debug', action='store_true', help='running without saving model')
    return parser.parse_args()


def main(args, logger, filepath):
    # log parameters.
    logger.info(f'====== Build Noisy Dataset ======')
    logger.info('\n'.join([f'{k} = {str(v)}' for k, v in sorted(dict(vars(args)).items())]))
    
    assert 0 <= args.feature_noise_rate and args.feature_noise_rate < 1, '`feature_noise_rate` must be in range of [0,1).'
    assert 0 <= args.label_noise_rate and args.label_noise_rate < 1, '`label_noise_rate` must be in range of [0,1).'
    
    set_seed(args.seed)
    
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(configs.cifar10_dir, train=True)
        testset  = torchvision.datasets.CIFAR10(configs.cifar10_dir, train=False)
    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(configs.cifar100_dir, train=True)
        testset  = torchvision.datasets.CIFAR100(configs.cifar100_dir, train=False)
    elif args.dataset == 'caltech101':
        trainset = Caltech101(configs.caltech101_dir, train=True, transform=configs.caltech101_transform)
        testset  = Caltech101(configs.caltech101_dir, train=False, transform=configs.caltech101_transform)
    elif args.dataset == 'caltech256':
        trainset = Caltech256(configs.caltech256_dir, train=True, transform=configs.caltech256_transform)
        testset  = Caltech256(configs.caltech256_dir, train=False, transform=configs.caltech256_transform)
    else:
        raise ValueError(f'unknown dataset "{args.dataset}".')
    
    img_size = trainset[0][0].size
    logger.info(f'feature size = {img_size}, number of classes = {len(trainset.classes)}.')
    
    noise_label_offset = np.random.randint(1, len(trainset.classes))
    logger.info(f'{noise_label_offset = }.')
    
    noise_feature_n = int(len(trainset) * args.feature_noise_rate)
    noise_label_n   = int(len(trainset) * args.label_noise_rate)
    noise_feature_index = np.random.choice(len(trainset), noise_feature_n, replace=False)
    noise_label_index   = np.random.choice(len(trainset), noise_label_n, replace=False)
    logger.info(f'fabricate {noise_feature_n} sample(s) by noising feature and {noise_label_n} sample(s) by noising label.')
    logger.info(f'overlapping number of noise feature samples and noise label samples: {len(set(noise_feature_index) & set(noise_label_index))}.')
    
    feature_noise = noise_transform + [torchvision.transforms.RandomResizedCrop(size=img_size, scale=(0.1, 0.9))]
    feature_noise = torchvision.transforms.Compose(feature_noise)
    logger.info('noise feature transform:\n' + '\n'.join([f'==> {str(t)}' for t in feature_noise.transforms]))
    
    data = list()
    for i, (x, y) in enumerate(trainset):
        if i in noise_feature_index:
            x = feature_noise(x)
        if i in noise_label_index:
            if args.asymmetric:
                y = (y + noise_label_offset) % len(trainset.classes)
            else:
                y = (y + np.random.randint(1, len(trainset.classes))) % len(trainset.classes)
        data.append((x, y))

    assert all([x.size == data[0][0].size for x, _ in data]), 'found different image size.'
    assert all([y < len(trainset.classes) for _, y in data]), 'found label out of range.'
    logger.info(f'after transform, image size = {data[0][0].size}.')
        
    trainset = VisionDataset(data, classes=trainset.classes)
    
    if args.debug:
        logger.debug(f"save trainset to `{filepath['train']}`")
    else:
        with open(filepath['train'], 'wb') as f:
            pickle.dump(trainset, f)
        logger.info(f"save trainset to `{filepath['train']}`")
        
    if noise_feature_n > 0:
        if args.debug:
            logger.debug(f"save noise feature index to `{filepath['feature']}`")
        else:
            with open(filepath['feature'], 'wb') as f:
                pickle.dump(noise_feature_index, f)
            logger.info(f"save noise feature index to `{filepath['feature']}`")
            
    if noise_label_n > 0:
        if args.debug:
            logger.debug(f"save noise label index to `{filepath['label']}`")
        else:
            with open(filepath['label'], 'wb') as f:
                pickle.dump(noise_label_index, f)
            logger.info(f"save noise label index to `{filepath['label']}`")
    
    if not os.path.exists(filepath['test']):
        data = list()
        for x, y in testset:
            data.append((x, y))
        testset = VisionDataset(data, classes=testset.classes)
        if args.debug:
            logger.debug(f"save testset to `{filepath['test']}`")
        else:
            os.makedirs(os.path.dirname(filepath['test']), exist_ok=True)
            with open(filepath['test'], 'wb') as f:
                pickle.dump(testset, f)
            logger.info(f"save testset to `{filepath['test']}`")
            
            
if __name__ == '__main__':
    # set argument and basic paths.
    args    = parse_args()
    date    = datetime.datetime.now().strftime('%y%m%d-%H%M')
    logpath = set_dataset_log_path(dataset=args.dataset, feature_noise_rate=args.feature_noise_rate, label_noise_rate=args.label_noise_rate, asymmetric=args.asymmetric)
    filepath = dict(
        train=set_dataset_path(dataset=args.dataset, feature_noise_rate=args.feature_noise_rate, label_noise_rate=args.label_noise_rate, asymmetric=args.asymmetric, name='train'),
        test=set_dataset_path(dataset=args.dataset, feature_noise_rate=0, label_noise_rate=0,  name='test'),
        feature=set_dataset_path(dataset=args.dataset, feature_noise_rate=args.feature_noise_rate, label_noise_rate=args.label_noise_rate, asymmetric=args.asymmetric, name='feature'),
        label=set_dataset_path(dataset=args.dataset, feature_noise_rate=args.feature_noise_rate, label_noise_rate=args.label_noise_rate, asymmetric=args.asymmetric, name='label')
        )
    logger = set_logger(name='dataset', level='DEBUG') if args.debug else set_logger(name='dataset', logfile=logpath)
    logger.debug(f"save log to: '{logpath}'.")
    # start evaluation.
    try:
        main(args, logger, filepath)
    except Exception:
        logger.warning(f"catch exceptions, delete incomplete results.")
        for key, path in filepath.items():
            if os.path.exists(path) and key != 'test':
                os.remove(path)
        logger.error(traceback.format_exc())
    except KeyboardInterrupt:
        for key, path in filepath.items():
            if os.path.exists(path) and key != 'test':
                os.remove(path)