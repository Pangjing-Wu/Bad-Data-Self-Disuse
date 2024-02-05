import glob
import os

import torch
import torch.nn as nn

import configs
from models.resnet import *
from models.vit import *
from utils.path import set_embedding_path



# >>>>>>>>>> Embedding
def save_embedding_state(state: dict, model: str, dataset: str, algo: str, date: str, 
                         epoch: int, metric:str, best=False, override=False) -> None:
    filepath = set_embedding_path(model, dataset, algo, date, epoch, metric, best)
    if not override and os.path.exists(filepath):
        raise FileExistsError('model state exists, please set `override` to `True` if you want to restore model state.')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)

def load_embedding_state(model:str, dataset:str, algo:str, date:str, epoch=-1) -> dict:
    metric  = '*'
    best  = True if epoch == -1 else False
    epoch = '*'  if epoch == -1 else epoch
    filepath = set_embedding_path(model, dataset, algo, date, epoch, metric, best)
    file  = glob.glob(filepath)
    if len(file) == 0:
        raise FileNotFoundError(f"not found embedding file '{filepath}'.")
    elif len(file) > 1:
        raise RuntimeError(f"found multiple embedding files '{filepath}', please give a precise condition.")
    else:
        file = file[0]
    state = torch.load(file)
    return state

def load_embedding(model:str, dataset:str, algo:str, date:str, epoch=-1, param: dict = None) -> nn.Module:
    model_name = model
    param = param if param else eval(f'configs.{model}_{dataset}_params')
    if model_name == 'resnet18':
        model = resnet18_embedding(**param)
    elif model_name == 'resnet34':
        model = resnet34_embedding(**param)
    elif model_name == 'resnet50':
        model = resnet50_embedding(**param)
    elif model_name == 'vit':
        model = vit_embedding(**param)
    elif model_name == 'mae':
        model = mae_embedding(**param)
    elif model_name == 'beit':
        model = beit_embedding(**param)
    else:
        raise ValueError(f"unknown model '{model_name}'.")
    state = load_embedding_state(model_name, dataset, algo, date, epoch)
    model.load_state_dict(state, strict=False)
    return model

# <<<<<<<<<< Embedding