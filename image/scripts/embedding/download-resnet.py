import sys

import torchvision
import torch.nn as nn
import torchvision.models.resnet as resnet


sys.path.append('.')
from utils.nn.embedding import EmbeddingManager


def main():
    # download resnet50
    model  = resnet.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    state  = model.state_dict()
    EmbeddingManager(method='pre-trained', dataset='imagenet1k', model='resnet18').save(state)
    # download resnet50
    model  = resnet.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()
    state  = model.state_dict()
    EmbeddingManager(method='pre-trained', dataset='imagenet1k', model='resnet50').save(state)
    
if  __name__ == '__main__':
    main()