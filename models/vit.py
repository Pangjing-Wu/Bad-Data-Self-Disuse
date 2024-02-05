import torch.nn as nn
from transformers import BeitForImageClassification, ViTForImageClassification, ViTMAEModel


__all__ = ['vit_embedding', 'mae_embedding', 'beit_embedding']


class ViTEncoder(nn.Module):
    def __init__(self, name) -> None:
        super().__init__()
        self.model = ViTForImageClassification.from_pretrained(name, return_dict=False)
        self.model.classifier = nn.Identity()

    def forward(self, x):
        return self.model(x)[0]
    
    def load_state_dict(self, state_dict, strict: bool = False):
        return self.model.load_state_dict(state_dict, strict)
    
    def state_dict(self, **kawrgs) -> dict:
        return self.model.state_dict(**kawrgs)


class ViTMAEEncoder(nn.Module):

    def __init__(self, name) -> None:
        super().__init__()
        self.model = ViTMAEModel.from_pretrained(name, return_dict=False)

    def forward(self, x):
        return self.model(x)[0][:,0]
    
    def load_state_dict(self, state_dict, strict: bool = False):
        return self.model.load_state_dict(state_dict, strict)
    
    def state_dict(self, **kawrgs) -> dict:
        return self.model.state_dict(**kawrgs)
    

class BeitEncoder(nn.Module):
    def __init__(self, name) -> None:
        super().__init__()
        self.model = BeitForImageClassification.from_pretrained(name, return_dict=False)
        self.model.classifier = nn.Identity()

    def forward(self, x):
        return self.model(x)[0]
    
    def load_state_dict(self, state_dict, strict: bool = False):
        return self.model.load_state_dict(state_dict, strict)
    
    def state_dict(self, **kawrgs) -> dict:
        return self.model.state_dict(**kawrgs)


def vit_embedding(name: str, **kawrgs):
    return ViTEncoder(name)


def mae_embedding(name: str, **kawrgs):
    return ViTMAEEncoder(name)


def beit_embedding(name: str, **kawrgs):
    return BeitEncoder(name)