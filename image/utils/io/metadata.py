import os
from typing import Optional

import pandas as pd

import configs.path as path_config


class MetaDataManager(object):
    SUPPORT_METADATA = set(['i-forest', 'oc-svm', 'aum', 'consistence', 'centroid', 'el2n', 'grand', 'forgetting'])
    
    def __init__(self, metadata: str, dataset: str, date: Optional[str] = None) -> None:
        assert metadata.lower() in self.SUPPORT_METADATA
        self.metadata  = metadata
        self.dataset   = dataset
        self.date      = date
        self.directory = os.path.join(path_config.METADATA_DIR, self.dataset)
        self.path      = os.path.join(self.directory, f'{self.metadata}-{date}.csv')
        os.makedirs(self.directory, exist_ok=True)
    
    def save(self, df: pd.DataFrame) -> None:
        df.to_csv(self.path, index=False, float_format='%.4f')
    
    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.path)
    
    @property
    def logpath(self):
        return self.path.replace('.csv', '.log')