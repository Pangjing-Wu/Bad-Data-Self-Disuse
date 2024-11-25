import numpy as np
import torch
from torch.utils.data import Dataset

from ..base import IterUpdatedScore


class ForgettingEvent(IterUpdatedScore):
    
    def __init__(self, n_samples) -> None:
        self.n_samples = n_samples
        
    def init(self) -> None:
        self.done       = False
        self.forgetting = torch.zeros(self.n_samples).long()
        self.prev_acc   = torch.zeros(self.n_samples).long()
        self.unlearned  = torch.ones(self.n_samples).bool()
        
    def update(self, model, index, x, y) -> None:
        """update after each batch iteration.
        batch: batch data, tuple of (index, features, labels)."""
        self.set_device(model)
        if self.done:
            raise RuntimeError('step process is terminated when calling `score()` or `scores()`, please run `init()` to reset.')
        assert x.size(0) == index.size(0) and x.size(0) == y.size(0)
        with torch.no_grad():
            x    = x.to(self.device)
            pred = torch.argmax(torch.softmax(model(x).cpu(), dim=-1), dim=-1)
        acc = torch.zeros_like(y).masked_fill(pred.eq(y), 1)
        forget_index = self.prev_acc[index] > acc
        self.forgetting[index[forget_index]] += 1
        self.unlearned[index[acc.bool()]] = False
        self.prev_acc[index] = acc
    
    def score(self, index) -> float:
        if not self.done:
            self.__post_process()
            self.done = True
        return self.forgetting[index]
    
    def scores(self) -> np.ndarray:
        if not self.done:
            self.__post_process()
            self.done = True
        return self.forgetting
    
    def __post_process(self):
        # label never learned.
        self.forgetting[self.unlearned] = self.forgetting.max() + 1
        # norm.
        self.forgetting = (self.forgetting - self.forgetting.min()) / (self.forgetting.max() - self.forgetting.min())
        self.forgetting = 1 - self.forgetting