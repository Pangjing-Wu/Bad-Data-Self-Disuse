import random

import numpy as np
import torch


__all__ = ['set_seed']


def get_device(model):
    return next(iter(model.state_dict().values())).device


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)