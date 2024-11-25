import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTMAEModel


class MAEEmbedding(nn.Module):
    
    def __init__(self, ):
        super().__init__()
        self.embed = ViTMAEModel.from_pretrained('facebook/vit-mae-base')
        self.embedding_dim = 768
        self.eval()
        
    def forward(self, x):
        with torch.no_grad():
            x = self.embed(x)['last_hidden_state']
            x = x.mean(dim=1)
        return torch.flatten(x, start_dim=1)
        
        
class ViTEmbedding(nn.Module):
    
    def __init__(self, ):
        super().__init__()
        self.embed = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
        self.embed.classifier = nn.Identity()
        self.embedding_dim = 768
        self.eval()
    
    def forward(self, x):
        with torch.no_grad():
            x = self.embed(x)['logits']
        return torch.flatten(x, start_dim=1)