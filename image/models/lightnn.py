import torch.nn as nn
from torchvision.models import mobilenet_v3_small, shufflenet_v2_x0_5
        
        
class MobileNetV3(nn.Module):
    
    def __init__(self, n_classes=1000, stride=2, **kawrgs):
        super().__init__()
        self.layer = mobilenet_v3_small(weights=None)
        self.layer.features[0][0].stride = (stride, stride)
        self.layer.classifier[3] = nn.Linear(self.layer.classifier[3].in_features, n_classes)
        
    def forward(self, x):
        return self.layer(x)
    

class ShuffleNet(nn.Module):
    
    def __init__(self, n_classes=1000, stride=2, **kawrgs):
        super().__init__()
        self.layer = shufflenet_v2_x0_5(weights=None)
        self.layer.conv1[0].stride = (stride, stride)
        self.layer.fc = nn.Linear(self.layer.fc.in_features, n_classes)
        
    def forward(self, x):
        return self.layer(x)
    