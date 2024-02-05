import copy
import pickle
import os
from typing import Union

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, Subset

import configs
from _datasets import *
from utils.path import get_dataset_path


class NumpyDataset(Dataset):
    
    def __init__(self, *data):
        assert all(data[0].shape[0] == d.shape[0] for d in data)
        self.data = data
    
    def __getitem__(self, index):
        return tuple(d[index] for d in self.data)

    def __len__(self):
        return self.data[0].shape[0]
    
    
class KFoldSubset(Subset):
    
    def __init__(self, dataset, indices) -> None:
        super().__init__(dataset=dataset, indices=indices)
    
    def remove(self, index):
        assert index in self.indices
        indices = [i for i in self.indices if i != index]
        return KFoldSubset(self.dataset, indices)
    
    def remove_(self, index):
        assert index in self.indices
        self.indices = [i for i in self.indices if i != index]
        return self
    
    
class KFoldDataset(Dataset):
    
    def __init__(self, dataset, k, shuffle=True):
        self.dataset = dataset
        self.k = k
        self.indices = torch.randperm(len(dataset)).long().tolist() if shuffle else list(range(len(dataset)))
        self.fold_element_indices = np.array_split(self.indices, k)
        self.folds = [KFoldSubset(self.dataset, indices=index) for index in self.fold_element_indices]
    
    def __len__(self) -> int:
        return self.k
    
    def __getitem__(self, index) -> KFoldSubset:
        return self.folds[index]
    
    def exclude(self, fold_index) -> KFoldSubset:
        if fold_index >= self.k:
            raise IndexError("Fold index out of range.")
        index = np.concatenate([self.fold_element_indices[i] for i in range(self.k) if i != fold_index]).tolist()
        return KFoldSubset(self.dataset, indices=index)
    
    def get_fold_by_sample_index(self, sample_index) -> int:
        for i, fold_index in enumerate(self.fold_element_indices):
            if sample_index in fold_index:
                return i
        raise ValueError("sample index not found in any fold.")


class IndexedDataset(Dataset):
    
    def __init__(self, dataset: Dataset):
        assert hasattr(dataset, 'classes')
        self.dataset   = dataset
        self.classes   = dataset.classes
        self.transform = dataset.transform
        
    def __getitem__(self, index):
        x, y = self.dataset[index]
        return index, x, y
    
    def __len__(self) -> int:
        return len(self.dataset)


class VisionDataset(Dataset):
    
    def __init__(self, data, classes, transform=None):
        self.data = data
        self.classes = classes
        self.transform = transform
    
    def __getitem__(self, index):
        img, label = self.data[index]
        img = self.transform(img) if self.transform else img
        return img, label

    def __len__(self):
        return len(self.data)
    
    def set_transform(self, transform):
        self.transform = transform
        

def load_cv_dataset(dataset:str, train=True, augment=False, resize=False, include_index=False) -> Union[VisionDataset, IndexedDataset]:
    filename = get_dataset_path(dataset, train=train)
    dataset = dataset.split('-')[0]
    if dataset == 'cifar10':
        transform = configs.cifar100_aug_trans if augment else configs.cifar100_norm
        transform = copy.deepcopy(transform) # or the following steps will change the original transform.
        if resize and augment: transform.transforms[1] = torchvision.transforms.Resize(resize)
        if resize and not augment: transform.transforms.insert(0, torchvision.transforms.Resize(resize))
    elif dataset == 'cifar100':
        transform = configs.cifar100_aug_trans if augment else configs.cifar100_norm
        transform = copy.deepcopy(transform) # or the following steps will change the original transform.
        if resize and augment: transform.transforms[1] = torchvision.transforms.Resize(resize)
        if resize and not augment: transform.transforms.insert(0, torchvision.transforms.Resize(resize))
    elif dataset == 'caltech101':
        transform = configs.caltech101_aug_trans if augment else configs.caltech101_norm
        transform = copy.deepcopy(transform) # or the following steps will change the original transform.
        if resize and augment: transform.transforms[3] = torchvision.transforms.Resize(resize)
        if resize and not augment: transform.transforms.insert(0, torchvision.transforms.Resize(resize))
    elif dataset == 'caltech256':
        transform = configs.caltech256_aug_trans if augment else configs.caltech256_norm
        transform = copy.deepcopy(transform) # or the following steps will change the original transform.
        if resize and augment: transform.transforms[3] = torchvision.transforms.Resize(resize)
        if resize and not augment: transform.transforms.insert(0, torchvision.transforms.Resize(resize))
    else:
        raise ValueError(f'unknown cv dataset "{dataset}".')
    
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
        dataset.set_transform(transform)
    if include_index:
        dataset = IndexedDataset(dataset)
    
    return dataset