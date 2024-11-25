import os
from typing import Any, Optional, Union

import torch

import configs.path as path_config
from models import ResNet18Embedding, ResNet50Embedding, MixedBTEmbedding, MAEEmbedding, ViTEmbedding


class ImageEmbeddingManager(object):
    SUPPORT_EMBEDDING_METHODS = set(['mixed-bt', 'pre-trained'])
    SUPPORT_EMBEDDING_MODELS  = {
        'mixed-bt': ['resnet18', 'resnet50'],
        'pre-trained': ['imagenet1k-resnet18', 'imagenet1k-resnet50', 'vit', 'mae']
    }
    
    def __init__(self, method: str, dataset: str, backbone: str, backbone_kwargs:dict, date: Optional[str] = None) -> None:
        assert method.lower() in self.SUPPORT_EMBEDDING_METHODS
        assert backbone.lower() in self.SUPPORT_EMBEDDING_MODELS[method.lower()]
        self.method    = method
        self.dataset   = dataset
        self.backbone  = backbone
        self.date      = date
        self.directory = os.path.join(path_config.EMBEDDING_DIR, method)
        self.backbone_kwargs = backbone_kwargs
        os.makedirs(self.directory, exist_ok=True)
    
    def save(self, obj: Any, epoch: Optional[Union[int, str]] = None) -> None:
        path = self.parse_path(epoch)
        torch.save(obj, path)
    
    def load(self, epoch: Optional[Union[int, str]] = None):
        path = self.parse_path(epoch)
        if self.method == 'mixed-bt':
            if self.backbone == 'resnet18':
                backbone = ResNet18Embedding(**self.backbone_kwargs)
            elif self.backbone == 'resnet50':
                backbone = ResNet50Embedding(**self.backbone_kwargs)
            model = MixedBTEmbedding(backbone)
            model.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
        elif self.method == 'pre-trained':
            if self.backbone == 'imagenet1k-resnet18':
                model = ResNet18Embedding(**self.backbone_kwargs)
                model.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
            elif self.backbone == 'imagenet1k-resnet50':
                model = ResNet50Embedding(**self.backbone_kwargs)
                model.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
            elif self.backbone == 'vit':
                model = ViTEmbedding()
            elif self.backbone == 'mae':
                model = MAEEmbedding()
            else:
                raise ValueError(f'unknown backbone {self.backbone}')
        return model
    
    def parse_path(self, epoch: Optional[Union[int, str]] = None):
        name = f'{self.dataset}-{self.backbone}'
        name += f'-{self.date}' if self.date else ''
        name += f'-{epoch}' if epoch else ''
        return os.path.join(self.directory, f'{name}.pth')
    
    @property
    def logpath(self):
        return os.path.join(self.directory, 'main.log')