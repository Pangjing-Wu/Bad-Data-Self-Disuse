import torch
import torch.nn as nn
import torchvision.models.resnet as resnet


class _BaseResNet(nn.Module):
    
    def __init__(self, backbone, n_classes=1000, in_channel=3, out_channel=64, kernel=7, stride=2, padding=3, maxpool=True, **kawrgs):
        super().__init__()
        self.backbone = getattr(resnet, backbone)
        for name, layer in self.backbone().named_children():
            if 'conv1' in name:
                layer = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding, bias=False)
            elif not maxpool and 'maxpool' in name:
                continue
            elif 'fc' in name:
                self.add_module('flatten', nn.Flatten(1))
                layer = nn.Linear(layer.in_features, n_classes)
            self.add_module(name, layer)
        self.n_classes = n_classes
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.children():
            if isinstance(m, resnet.Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, resnet.Bottleneck):
                nn.init.constant_(m.bn2.weight, 0)
                
    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResNet18(_BaseResNet):
    
    def __init__(self, n_classes=1000, in_channel=3, out_channel=64, kernel=7, stride=2, padding=3, maxpool=True, **kawrgs):
        super().__init__(
            'resnet18', 
            n_classes=n_classes, 
            in_channel=in_channel, 
            out_channel=out_channel, 
            kernel=kernel, stride=stride, 
            padding=padding, 
            maxpool=maxpool, 
            **kawrgs
            )


class ResNet50(_BaseResNet):
    
    def __init__(self, n_classes=1000, in_channel=3, out_channel=64, kernel=7, stride=2, padding=3, maxpool=True, **kawrgs):
        super().__init__(
            'resnet50', 
            n_classes=n_classes, 
            in_channel=in_channel, 
            out_channel=out_channel, 
            kernel=kernel, stride=stride, 
            padding=padding, 
            maxpool=maxpool, 
            **kawrgs
            )
        
        
class _BaseResNetEmbedding(nn.Module):
    
    def __init__(self, backbone, in_channel=3, out_channel=64, kernel=7, stride=2, padding=3, maxpool=True, **kawrgs):
        super().__init__()
        self.backbone = getattr(resnet, backbone)
        for name, layer in self.backbone().named_children():
            if 'conv1' in name:
                layer = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding, bias=False)
            elif not maxpool and 'maxpool' in name:
                continue
            elif 'fc' in name:
                self.embedding_dim = layer.in_features
                continue
            self.add_module(name, layer)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.children():
            if isinstance(m, resnet.Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, resnet.Bottleneck):
                nn.init.constant_(m.bn2.weight, 0)
        self.eval()
        
    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return torch.flatten(x, start_dim=1)


class ResNet18Embedding(_BaseResNetEmbedding):
    
    def __init__(self, in_channel=3, out_channel=64, kernel=7, stride=2, padding=3, maxpool=True, **kawrgs):
        super().__init__(
            'resnet18',
            in_channel=in_channel, 
            out_channel=out_channel, 
            kernel=kernel, stride=stride, 
            padding=padding, 
            maxpool=maxpool, 
            **kawrgs
            )
        
class ResNet50Embedding(_BaseResNetEmbedding):
    
    def __init__(self, in_channel=3, out_channel=64, kernel=7, stride=2, padding=3, maxpool=True, **kawrgs):
        super().__init__(
            'resnet50',
            in_channel=in_channel, 
            out_channel=out_channel, 
            kernel=kernel, stride=stride, 
            padding=padding, 
            maxpool=maxpool, 
            **kawrgs
            )