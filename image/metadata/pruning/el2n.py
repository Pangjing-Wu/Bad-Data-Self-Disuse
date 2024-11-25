from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from ..base import TrainingScore


class EL2N(TrainingScore):

    def __init__(self, model: nn.Module, n_classes: int, order: Union[int,str] = 2) -> None:
        super().__init__(model=model)
        self.order = order
        self.n_classes = n_classes

    def score(self, x: torch.Tensor, y: torch.Tensor) -> np.ndarray:
        """x: batch of features, [batch size, feature]. y: batch of labels, [batch size]."""
        assert x.size(0) == y.size(0)
        with torch.no_grad():
            x        = x.to(self.device)
            logistic = torch.softmax(self.model(x), dim=-1).cpu().numpy()
            y        = self.onehot_map(y).numpy()
        assert logistic.shape == y.shape
        return np.linalg.norm((logistic-y), ord=self.order, axis=-1)
    
    def onehot_map(self, y: torch.Tensor) -> torch.Tensor:
        return torch.eye(self.n_classes)[y]


class GraNd(TrainingScore):

    def __init__(
        self, 
        model: nn.Module, 
        n_classes: int, 
        order: Union[int,str] = 2, 
        optim: Optional[torch.optim.Optimizer] = None, 
        fc_name: str = 'fc'
        ) -> None:
        super().__init__(model=model)
        self.order      = order
        self.n_classes  = n_classes
        self.optimizer  = optim(model.parameters(), lr=0) if optim else torch.optim.SGD(model.parameters(), lr=0)
        self.criterion  = nn.CrossEntropyLoss()
        self.fc_name    = fc_name

    def score(self, x: torch.Tensor, y: torch.Tensor) -> np.ndarray:
        """x: batch of features, [batch size, feature]. y: batch of labels, [batch size]."""
        assert x.size(0) == y.size(0)
        x, y  = x.to(self.device), y.to(self.device)
        fc    = getattr(self.model, self.fc_name)
        setattr(self.model, self.fc_name, nn.Identity())
        with torch.no_grad():
            x = self.model(x).detach()
        setattr(self.model, self.fc_name, fc)
        grand = list()
        for x_, y_ in zip(x, y):
            self.optimizer.zero_grad()
            x_, y_   = x_.unsqueeze(0), y_.unsqueeze(0)
            logistic = torch.softmax(fc(x_), dim=-1)
            loss = self.criterion(logistic, y_)
            loss.backward()
            grad = fc.weight.grad.cpu().numpy()
            logistic = logistic.detach().cpu().numpy()
            y_   = self.onehot_map(y_.detach().cpu()).numpy()
            grad = (logistic - y_).reshape(-1,1) * grad
            grad = np.sum(grad, axis=0)
            grad = np.linalg.norm(grad, ord=self.order)
            grand.append(grad)
        self.optimizer.zero_grad()
        return np.array(grand)
    
    def onehot_map(self, y: torch.Tensor) -> torch.Tensor:
        return torch.eye(self.n_classes)[y]