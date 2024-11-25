import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

import configs.path as path_config


class EvaluationManager(object):
    
    def __init__(self, dataset: str, date: Optional[str] = None) -> None:
        self.dataset   = dataset
        self.date      = date
        self.directory = os.path.join(path_config.EVAL_DIR, self.dataset, self.date)
        self.est_path  = os.path.join(self.directory, 'estimation.csv')
        self.eval_path = os.path.join(self.directory, 'evaluation.json')
        self.bench_path = os.path.join(self.directory, 'benchmark.npy')
        os.makedirs(self.directory, exist_ok=True)
        
    def save_est(self, ret: Dict) -> None:
        pd.DataFrame(ret).to_csv(self.est_path, index=False, float_format='%.4f')

    def save_eval(self, report: Dict):
        with open(self.eval_path, 'a') as f:
            f.write(str(report) + '\n')
    
    def save_benchmark(self, arr):
        np.save(self.bench_path, arr)
        
    @property
    def logpath(self):
        return os.path.join(self.directory, 'main.log')