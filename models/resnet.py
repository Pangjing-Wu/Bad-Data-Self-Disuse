import torch.nn as nn
import torchvision.models.resnet as resnet


__all__ = ['resnet18_embedding', 'resnet34_embedding', 'resnet50_embedding', 
           'resnet18', 'resnet34', 'resnet50']


def resnet18_embedding(n_channel=3, kernel=7, stride=2, padding=3, maxpool=True, **kawrgs):
    model = resnet.resnet18()
    model.conv1 = nn.Conv2d(n_channel, 64, kernel_size=kernel, 
                            stride=stride, padding=padding, bias=False)
    model.maxpool = model.maxpool if maxpool else nn.Identity()
    model.fc = nn.Identity()
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    return model

def resnet34_embedding(n_channel=3, kernel=7, stride=2, padding=3, maxpool=True, **kawrgs):
    model = resnet.resnet34()
    model.conv1 = nn.Conv2d(n_channel, 64, kernel_size=kernel, 
                            stride=stride, padding=padding, bias=False)
    model.maxpool = model.maxpool if maxpool else nn.Identity()
    model.fc = nn.Identity()
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    return model

def resnet50_embedding(n_channel=3, kernel=7, stride=2, padding=3, maxpool=True, **kawrgs):
    model = resnet.resnet50()
    model.conv1 = nn.Conv2d(n_channel, 64, kernel_size=kernel, 
                            stride=stride, padding=padding, bias=False)
    model.maxpool = model.maxpool if maxpool else nn.Identity()
    model.fc = nn.Identity()
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    return model


def resnet18(n_classes=1000, n_channel=3, kernel=7, stride=2, padding=3, maxpool=True, **kawrgs):
    model = resnet.resnet18()
    model.conv1 = nn.Conv2d(n_channel, 64, kernel_size=kernel, 
                            stride=stride, padding=padding, bias=False)
    model.maxpool = model.maxpool if maxpool else nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    return model

def resnet34(n_classes=1000, n_channel=3, kernel=7, stride=2, padding=3, maxpool=True, **kawrgs):
    model = resnet.resnet34()
    model.conv1 = nn.Conv2d(n_channel, 64, kernel_size=kernel, 
                            stride=stride, padding=padding, bias=False)
    model.maxpool = model.maxpool if maxpool else nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    return model


def resnet50(n_classes=1000, n_channel=3, kernel=7, stride=2, padding=3, maxpool=True, **kawrgs):
    model = resnet.resnet50()
    model.conv1 = nn.Conv2d(n_channel, 64, kernel_size=kernel, 
                            stride=stride, padding=padding, bias=False)
    model.maxpool = model.maxpool if maxpool else nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    return model