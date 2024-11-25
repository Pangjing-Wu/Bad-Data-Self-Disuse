import configs.path as path_config
from utils.datasets.tabular import Diabetes130, Adults


def load_tabular_dataset(dataset:str, train=True):
    if dataset == 'diabetes130':
        dataset = Diabetes130(path_config.diabetes130_dir)
    elif dataset == 'adults':
        dataset = Adults(path_config.adult_dir)
    else:
        raise ValueError(f'unknown cv dataset "{dataset}".')
    
    return dataset.trainset if train else dataset.testset