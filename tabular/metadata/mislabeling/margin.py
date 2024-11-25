import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ..base import TrainingScore, IterUpdatedScore


def cal_margin(logistic: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    assert logistic.dim() == 2 and label.dim() == 1
    rank    = torch.argsort(logistic, dim=-1, descending=True)
    margin1 = torch.gather(logistic, 1, label.unsqueeze(-1)) - torch.gather(logistic, 1, rank[:,1].unsqueeze(-1))
    margin2 = torch.gather(logistic, 1, label.unsqueeze(-1)) - torch.gather(logistic, 1, rank[:,0].unsqueeze(-1))
    return torch.where(rank[:,0] == label, margin1.squeeze(-1), margin2.squeeze(-1))


class Margin(TrainingScore):
    
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model=model)
        
    def score(self, x: torch.Tensor, y: torch.Tensor) -> np.ndarray:
        """x: batch of features, [batch size, feature]. y: batch of labels, [batch size]."""
        assert x.size(0) == y.size(0)
        with torch.no_grad():
            x        = x.to(self.device)
            logistic = torch.softmax(self.model(x), dim=-1).cpu()
        margin = cal_margin(logistic, y)
        return margin.numpy()


class AreaUnderMargin(IterUpdatedScore):
    
    def __init__(self) -> None:
        self.done = False
        
    def init(self) -> None:
        self.done = False
        self.aum  = list() # size = [epoch, n_sample]
        
    def update(self, model: nn.Module, dataloader: DataLoader) -> None:
        """update after each epoch."""
        self.set_device(model)
        if self.done:
            raise RuntimeError('step process is terminated when calling `score()` or `scores()`, please run `init()` to reset.')
        margins = list()
        for data in dataloader:
            assert len(data) == 2 or len(data) == 3
            x = data[0] if len(data) == 2 else data[1]
            y = data[1] if len(data) == 2 else data[2]
            with torch.no_grad():
                x        = x.to(self.device)
                logistic = torch.softmax(model(x).cpu(), dim=-1)
                margin   = cal_margin(logistic, y)
            margins.extend(margin.numpy().tolist())
        self.aum.append(margins)
    
    def score(self, index) -> np.ndarray:
        self.__calculate_aum()
        return self.aum[index]
    
    def scores(self) -> np.ndarray:
        self.__calculate_aum()
        return self.aum
    
    def __calculate_aum(self) -> np.ndarray:
        if not self.done:
            self.done = True
            self.aum  = np.mean(self.aum, axis=0)