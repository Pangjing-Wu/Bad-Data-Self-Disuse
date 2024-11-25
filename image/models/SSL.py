import torch
import torch.nn as nn


class MixedBT(nn.Module):
    def __init__(self, pretrained_path, backbone, embedding_size, n_classes):
        super().__init__()
        self.f = nn.Sequential(*[layer for layer in backbone.children() if not isinstance(layer, nn.Identity)])
        self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)
        self.fc = nn.Linear(embedding_size, n_classes, bias=True)
        
    def forward(self, x):
        with torch.no_grad():
            embedding = self.f(x)
            embedding = torch.flatten(embedding, start_dim=1)
        out = self.fc(embedding)
        return out
    
    def embedding(self, x):
        with torch.no_grad():
            x = self.f(x)
            embedding = torch.flatten(x, start_dim=1)
        return embedding
    
    
class MixedBTEmbedding(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.embedding_dim = backbone.embedding_dim
        self.f = nn.Sequential(*[layer for layer in backbone.children() if not isinstance(layer, nn.Identity)])
        self.eval()
    
    def forward(self, x):
        with torch.no_grad():
            x = self.f(x)
            embedding = torch.flatten(x, start_dim=1)
        return embedding