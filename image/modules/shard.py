import numpy as np
import torch
from torch.utils.data import Dataset, Subset


class Shard(Subset):
    
    def __init__(self, dataset, indices) -> None:
        super().__init__(dataset=dataset, indices=indices)
    
    def remove(self, index):
        assert index in self.indices
        indices = [i for i in self.indices if i != index]
        return Shard(self.dataset, indices)
    
    def remove_(self, index):
        assert index in self.indices
        self.indices = [i for i in self.indices if i != index]
        return self
    

class DataShard(Dataset):
    
    def __init__(self, dataset, k, shuffle=True):
        self.dataset = dataset
        self.k = k
        self.indices = torch.randperm(len(dataset)).long().tolist() if shuffle else list(range(len(dataset)))
        self.fold_element_indices = np.array_split(self.indices, k)
        self.folds = [Shard(self.dataset, indices=index) for index in self.fold_element_indices]
    
    def __len__(self) -> int:
        return self.k
    
    def __getitem__(self, index) -> Shard:
        return self.folds[index]
    
    def exclude(self, fold_index) -> Shard:
        if fold_index >= self.k:
            raise IndexError("Fold index out of range.")
        index = np.concatenate([self.fold_element_indices[i] for i in range(self.k) if i != fold_index]).tolist()
        return Shard(self.dataset, indices=index)
    
    def get_shard_by_sample_index(self, sample_index) -> int:
        for i, fold_index in enumerate(self.fold_element_indices):
            if sample_index in fold_index:
                return i
        raise ValueError("sample index not found in any fold.")