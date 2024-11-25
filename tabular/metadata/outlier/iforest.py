import numpy as np
from sklearn.ensemble import IsolationForest


class MultiClassesIsolationForest(object):
    
    def __init__(self, n_classes, **iforest_kwargs) -> None:
        self.n_classes = n_classes
        self.forests = [IsolationForest(**iforest_kwargs) for _ in range(n_classes)]
        
    def fit(self, x, y):
        for i in range(self.n_classes):
            self.forests[i] = self.forests[i].fit(x[y == i])
        return self
    
    def label(self, x, y):
        return self.forests[y].predict(x.reshape(1, -1))[0]
    
    def labels(self, x, y):
        return np.array([self.label(x_, y_) for x_, y_ in zip(x, y)])
    
    def score(self, x, y) -> float:
        return self.forests[y].decision_function(x.reshape(1, -1))[0]
    
    def scores(self, x, y):
        return np.array([self.score(x_, y_) for x_, y_ in zip(x, y)])