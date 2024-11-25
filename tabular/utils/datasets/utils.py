from torch.utils.data import Dataset


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