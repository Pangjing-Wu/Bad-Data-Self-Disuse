import torch.nn as nn
from rtdl_revisiting_models import MLP
from rtdl_num_embeddings import LinearReLUEmbeddings


class FTMLP(nn.Module):
    
    def __init__(self, n_classes, input_dim, embedding_dim=24, n_blocks=2, d_block=256, dropout=0.1) -> None:
        super().__init__()
        self.mlp  = MLP(d_in=input_dim*embedding_dim, d_out=n_classes, n_blocks=n_blocks, d_block=d_block, dropout=dropout)
        self.fc   = nn.Linear(self.mlp.output.in_features, self.mlp.output.out_features)
        self.mlp.output = nn.Identity()
        self.model = nn.Sequential(
            LinearReLUEmbeddings(input_dim, embedding_dim), 
            nn.Flatten(), 
            self.mlp
        )

    def forward(self, x):
        return self.fc(self.model(x))