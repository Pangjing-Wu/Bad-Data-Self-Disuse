import abc
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class BasicScore(abc.ABC):
    
    @abc.abstractmethod
    def score():
        raise NotImplementedError()
    
    @abc.abstractmethod
    def scores():
        raise NotImplementedError()
    
    def set_device(self, model:nn.Module):
        assert isinstance(model, nn.Module)
        self.device = next(iter(model.parameters())).device


class TrainingScore(BasicScore, abc.ABC):

    def __init__(self, model: nn.Module) -> None:
        self.set_device(model)
        self.model = model
            
    @abc.abstractmethod
    def score(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> np.ndarray:
        raise NotImplementedError()
    
    def scores(self, dataloader: DataLoader) -> np.ndarray:
        score = list()
        for data in dataloader:
            assert len(data) == 2 or len(data) == 3
            x = data[0] if len(data) == 2 else data[1]
            y = data[1] if len(data) == 2 else data[2]
            score.append(self.score(x, y))
        return np.concatenate(score, axis=-1)
        

class IterUpdatedScore(BasicScore, abc.ABC):

    def __init__(self) -> None:
        raise NotImplementedError
            
    @abc.abstractmethod
    def score(self, index: int) -> float:
        raise NotImplementedError()
    
    @abc.abstractmethod
    def scores(self) -> np.ndarray:
        raise NotImplementedError()
    
    @abc.abstractmethod
    def init(self) -> None:
        raise NotImplementedError()
    
    @abc.abstractmethod
    def update(self) -> None:
        raise NotImplementedError()
    
    
    
class EmbeddingScore(BasicScore, abc.ABC):
    
    def __init__(self) -> None:
        self.score_ = None
        raise NotImplementedError
        
    @abc.abstractmethod
    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None):
        raise NotImplementedError()
        
    def score(self, index: int) -> np.ndarray:
        if self.score_ is None: self.fit()
        return self.score_[index]
    
    def scores(self, slice: Optional[List[int]] = None) -> np.ndarray:
        if self.score_ is None: self.fit()
        return self.score_[slice] if slice else self.score_