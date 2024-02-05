import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append('.')
from utils.dataset import KFoldDataset


def test():
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    dataset = TensorDataset(torch.randn(20,2,3), torch.randn(20))

    print('Test creating a `KFoldDataset`: ', end='')
    try:
        k_fold_dataset = KFoldDataset(dataset, k=5)
    except Exception as e:
        print(f'[ERROR]\n >> ({e})')
        return
    else:
        print('[DONE]')
        print(f'  >> results: indices = {k_fold_dataset.indices}'.replace('\n', ''))
        print(f'  >> results: fold indices = {k_fold_dataset.fold_element_indices}'.replace('\n', ''))
        print(f'  >> results: fold-0 indices = {k_fold_dataset[0].indices}'.replace('\n', ''))

    print('Test get `get_fold_by_sample_index`: ', end='')
    try:
        ret = k_fold_dataset.get_fold_by_sample_index(10)
    except Exception as e:
        print(f'[ERROR]\n >> ({e})')
        return
    else:
        print('[DONE]')
        print(f'  >> results: fold index = {ret}'.replace('\n', ''))
    
    print('Test the `exclude` function by excluding fold-0: ', end='')
    try:
        ret = k_fold_dataset.exclude(0)
    except Exception as e:
        print(f'[ERROR]\n >> ({e})')
        return
    else:
        print('[DONE]')
        print(f'  >> results: indices of remaining samples = {ret.indices}'.replace('\n', ''))

    print('Test the `remove` function of `KFoldSubset` class by removing index `5` in a random `KFoldSubset`: ', end='')
    try:
        ret = k_fold_dataset.exclude(0).remove(5)
    except Exception as e:
        print(f'[ERROR]\n >> ({e})')
        return
    else:
        print('[DONE]')
        print(f'  >> results: indices = {ret.indices}'.replace('\n', ''))
        
    print('Test the `remove` function of `KFoldSubset` class by removing index `1` that not in the random `KFoldSubset`: ', end='')
    try:
        ret = k_fold_dataset.exclude(0).remove(1)
    except AssertionError:
        print('[DONE]')
        print(f'  >> results: raise assertion error.')
    except Exception as e:
        print(f'[ERROR]\n >> ({e})')
        return
    else:
        print('[ERROR]\n >> (unexpected result).')
        return

    print('Test data loader: ', end='')
    try:
        subset = k_fold_dataset.exclude(0).remove(5)
        dataloader = DataLoader(subset, batch_size=1, shuffle=False)
        batch = next(iter(dataloader))
    except Exception as e:
        print(f'[ERROR]\n >> ({e})')
        return
    else:
        print('[DONE]')
        print(f'  >> results: first batch: {batch}.')

if __name__ == '__main__':
    test()