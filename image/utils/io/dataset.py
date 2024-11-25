import copy
from typing import Union

import torchvision
from torchvision.datasets import VisionDataset

import configs.path as path_config
import configs.datasets as dataset_config
from ..datasets.cv import Caltech101, Caltech256
from ..datasets.utils import IndexedDataset


def load_cv_dataset(dataset:str, train=True, augment=False, resize=False, raw=False, include_index=False) -> Union[VisionDataset, IndexedDataset]:
    if raw: 
        transform = torchvision.transforms.ToTensor()
    else:
        transform = getattr(dataset_config, f'{dataset}_aug_trans' if augment else f'{dataset}_norm')
        # NOTE: without deepcopy, recalling this function will further change the transform.
        transform = copy.deepcopy(transform) 
        if resize and augment: transform.transforms[1] = torchvision.transforms.Resize(resize)
        if resize and not augment: transform.transforms.insert(0, torchvision.transforms.Resize(resize))
    if dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(path_config.cifar10_dir, train=train, transform=transform)
    elif dataset == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(path_config.cifar100_dir, train=train, transform=transform)
    elif dataset == 'caltech101':
        dataset = Caltech101(path_config.caltech101_dir, train=train, transform=transform)
    elif dataset == 'caltech256':
        dataset = Caltech256(path_config.caltech256_dir, train=train, transform=transform)
    else:
        raise ValueError(f'unknown cv dataset "{dataset}".')
    
    if include_index:
        dataset = IndexedDataset(dataset)
    
    return dataset