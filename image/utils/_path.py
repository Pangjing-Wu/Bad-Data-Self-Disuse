import os

from configs.path import *


# >>>>>>>>>> Dataset Path
def set_dataset_path(dataset: str, feature_noise_rate: float,  label_noise_rate: float, 
                     asymmetric: bool = False, name: str = None) -> str:
    dataset = dataset + '-' if feature_noise_rate or label_noise_rate else dataset
    dataset = dataset + f'x{str(int(feature_noise_rate * 100))}' if feature_noise_rate else dataset
    dataset = dataset + f'y{str(int(label_noise_rate * 100))}' if label_noise_rate else dataset
    dataset = dataset + 'a' if label_noise_rate and asymmetric else dataset
    filepath = os.path.join(DATASET_DIR, dataset, f'{name}.pkl')
    return filepath

def get_dataset_path(dataset: str, train: bool = True) -> str:
    name    = 'train' if train else 'test'
    filepath = os.path.join(DATASET_DIR, dataset, f'{name}.pkl')
    return filepath

def set_dataset_log_path(dataset: str, feature_noise_rate: float, label_noise_rate: float, asymmetric: bool = False) -> str:
    dataset = dataset + f'x{str(int(feature_noise_rate * 100))}' if feature_noise_rate else dataset
    dataset = dataset + f'y{str(int(label_noise_rate * 100))}' if label_noise_rate else dataset
    dataset = dataset + 'a' if label_noise_rate and asymmetric else dataset
    logpath = os.path.join(DATASET_DIR, dataset, 'main.log')
    return logpath
# <<<<<<<<<< Dataset Path


# >>>>>>>>>> Embedding Path
def set_embedding_path(model: str, dataset: str, algo: str, date: str, 
                       epoch: str, metric: str, best: bool = False) -> str:
    filename = f'best-e{epoch}-{metric}.pth' if best else f'e{epoch}-{metric}.pth'
    filepath  = os.path.join(EMBEDDING_DIR, model, f'{algo}-{dataset}', date, filename)
    return filepath

def set_embedding_log_path(model, dataset, algo, date) -> str:
    logpath = os.path.join(EMBEDDING_DIR, model, f'{algo}-{dataset}', date, 'main.log')
    return logpath
# <<<<<<<<<< Embedding Path

# >>>>>>>>>> Component Score Path
def set_component_path(score: str, dataset: str, model:str, date: str, epoch: str = None) -> str:
    epoch   = 'best' if epoch is -1 else epoch
    filename =  f'{model}-{date}.csv' if epoch is None else f'{model}-{date}-e{epoch}.csv'
    filepath = os.path.join(COMPONENT_DIR, dataset, score, filename)
    return filepath

def set_component_log_path(score: str, dataset: str, model:str, date: str) -> str:
    filename = f'{model}-{date}.log'
    logpath = os.path.join(COMPONENT_DIR, dataset, score, filename)
    return logpath

def set_component_collection_path(dataset: str, date: str) -> str:
    filepath = os.path.join(COMPONENT_DIR, 'collection', dataset, f'{date}.txt')
    return filepath

def set_component_collection_log_path(dataset: str, date: str) -> str:
    logpath = os.path.join(COMPONENT_DIR, 'collection', dataset, f'{date}.log')
    return logpath
# <<<<<<<<<< Component Score Path

# >>>>>>>>>> Evaluation Path
def set_eval_path(dataset: str, date: str, name: str = None) -> str:
    filepath = os.path.join(EVAL_DIR, f'{dataset}-{date}-{name}.csv' if name else f'{dataset}-{date}.csv')
    return filepath

def set_eval_log_path(dataset: str, date: str) -> str:
    logpath = os.path.join(EVAL_DIR, f'{dataset}-{date}.log')
    return logpath
# <<<<<<<<<< Evaluation Path


# >>>>>>>>>> Performance Test Path
def set_performance_path(dataset: str, date: str, eval_date: str = None, name: str = None) -> str:
    if eval_date:
        filepath = os.path.join(PERFORMANCE_DIR, dataset, f'eval-{eval_date}', f'{date}-{name}.csv' if name else f'{date}.csv')
    else:
        filepath = os.path.join(PERFORMANCE_DIR, dataset, f'{date}.csv')
    return filepath

def set_performance_log_path(dataset: str, date: str, eval_date: str = None) -> str:
    if eval_date:
        logpath = os.path.join(PERFORMANCE_DIR, dataset, f'eval-{eval_date}', f'{date}.log')
    else:
        logpath = os.path.join(PERFORMANCE_DIR, dataset, f'{date}.log')
    return logpath
# <<<<<<<<<< Performance Test Path